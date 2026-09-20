import { z } from "zod";
import type { BundleSpec, FieldSpec, ManifoldDefinitionSpec, NetSpec, PointSetSpec } from "@tensatory/schema";
import { BUNDLE_VERSION } from "@tensatory/schema";
import { SpecError, TensatoryError } from "../errors";
import { Codomain } from "../fields/codomain";
import type { ScalarFieldData, VectorFieldData } from "../fields/fieldData";
import { FieldSchema, buildScalarFieldData, buildVectorFieldData, type FieldResolver } from "../fields/spec";
import { compileNet, evaluate, type Program, type ProgramResolver } from "../nets/program";
import { inferNet, type NetSignature } from "../nets/shapes";
import { NetSchema } from "../nets/spec";
import type { NdArray } from "../arrays/ndarray";

/*******************************************************/
/* schema */

export const ManifoldDefinitionSchema: z.ZodType<ManifoldDefinitionSpec> = z.object({
  name: z.string().optional(),
  summary: z.string().optional(),
  details: z.string().optional(),
  numDims: z.number().int().positive(),
  dimNames: z.array(z.string()).optional(),
  dimWeights: z.array(z.number()).optional(),
  origin: z.array(z.number()).optional(),
  flow: z.string().optional(),
});

export const PointSetSchema: z.ZodType<PointSetSpec> = z.object({
  domain: z.string().optional(),
  points: z.array(z.array(z.number())),
  labels: z.array(z.string()).optional(),
  ordered: z.boolean().optional(),
  name: z.string().optional(),
});

export const BundleSchema: z.ZodType<BundleSpec> = z.object({
  tensatory: z.literal(BUNDLE_VERSION),
  name: z.string().optional(),
  summary: z.string().optional(),
  details: z.string().optional(),
  manifolds: z.record(z.string(), ManifoldDefinitionSchema).optional(),
  defaultManifold: z.string().optional(),
  fields: z.record(z.string(), FieldSchema),
  pointSets: z.record(z.string(), PointSetSchema).optional(),
  nets: z.record(z.string(), NetSchema).optional(),
});

/*******************************************************/
/* runtime */

/** what a bundle / space / field / net says about itself: `summary` is meant to be one line, `details` any length */
export interface Info {
  name: string;
  summary?: string | undefined;
  details?: string | undefined;
}
/** the Info of a spec with optional summary / details, or undefined when it has neither */
export function infoOf(name: string, spec: { summary?: string; details?: string }): Info | undefined {
  const summary = spec.summary?.trim() || undefined, details = spec.details?.trim() || undefined;
  return summary || details ? { name, summary, details } : undefined;
}

export class Manifold {
  readonly name: string;
  readonly dimNames: readonly string[];
  constructor(readonly id: string, readonly spec: ManifoldDefinitionSpec) {
    this.name = spec.name ?? id;
    if (spec.dimNames && spec.dimNames.length !== spec.numDims)
      throw new SpecError(`dimNames has ${spec.dimNames.length} entries for ${spec.numDims} dims`, ["manifolds", id]);
    if (spec.dimWeights && spec.dimWeights.length !== spec.numDims)
      throw new SpecError(`dimWeights has ${spec.dimWeights.length} entries for ${spec.numDims} dims`, ["manifolds", id]);
    if (spec.origin && spec.origin.length !== spec.numDims)
      throw new SpecError(`origin has ${spec.origin.length} entries for ${spec.numDims} dims`, ["manifolds", id]);
    this.dimNames = spec.dimNames ?? Array.from({ length: spec.numDims }, (_, i) => `x${i}`);
  }
  get numDims(): number { return this.spec.numDims; }
  /** the id of the vector field declared as this space's dynamical system (ẋ = flow(x)), if any */
  get flow(): string | undefined { return this.spec.flow; }
  get info(): Info | undefined { return infoOf(this.name, this.spec); }
}

export class PointSet {
  readonly name: string;
  constructor(readonly id: string, readonly spec: PointSetSpec, readonly domain: Manifold) {
    this.name = spec.name ?? id;
    for (const [i, p] of spec.points.entries())
      if (p.length !== domain.numDims) throw new SpecError(`point ${i} has ${p.length} coords, manifold has ${domain.numDims}`, ["pointSets", id]);
    if (spec.labels && spec.labels.length !== spec.points.length)
      throw new SpecError(`labels has ${spec.labels.length} entries for ${spec.points.length} points`, ["pointSets", id]);
  }
  get points(): readonly number[][] { return this.spec.points; }
  get ordered(): boolean { return this.spec.ordered ?? false; }
}

export class ScalarField {
  readonly kind = "scalar" as const;
  readonly name: string;
  readonly codomain: Codomain;
  constructor(readonly id: string, readonly spec: Extract<FieldSpec, { kind: "scalar" }>, readonly domain: Manifold, readonly data: ScalarFieldData, private readonly bundle: Bundle) {
    this.name = spec.name ?? id;
    this.codomain = new Codomain(spec.codomain);
  }
  /** the exact gradient field, when the bundle has one */
  get exactGradient(): VectorField | undefined {
    return this.spec.exactGradient === undefined ? undefined : this.bundle.vectorField(this.spec.exactGradient);
  }
  get info(): Info | undefined { return infoOf(this.name, this.spec); }
}

export class VectorField {
  readonly kind = "vector" as const;
  readonly name: string;
  constructor(readonly id: string, readonly spec: Extract<FieldSpec, { kind: "vector" }>, readonly domain: Manifold, readonly data: VectorFieldData) {
    this.name = spec.name ?? id;
  }
  get info(): Info | undefined { return infoOf(this.name, this.spec); }
}

export type Field = ScalarField | VectorField;

/** a parsed net: its spec, inferred signature, and (lazily) its compiled program */
export class Net {
  readonly name: string;
  private _program: Program | undefined;
  constructor(readonly id: string, readonly spec: NetSpec, readonly signature: NetSignature, private readonly nets: ProgramResolver) {
    this.name = spec.name ?? id;
  }
  /** the compiled program (CPU reference evaluator); throws NotSupportedError for grad nets */
  get program(): Program {
    return (this._program ??= compileNet(this.spec, this.nets, ["nets", this.id]));
  }
  /** evaluate on the CPU: inputs with their batch prefixes -> outputs */
  evaluate(inputs: Record<string, NdArray>): Record<string, NdArray> {
    return evaluate(this.program, inputs, { nets: this.nets }, ["nets", this.id]);
  }
}

/**
 * A parsed bundle. Fields are built lazily on first access (so a field may
 * refer to another by id regardless of declaration order); reference cycles
 * are detected.
 */
export class Bundle {
  readonly name: string;
  readonly manifolds: ReadonlyMap<string, Manifold>;
  /** the manifold of fields / point sets without `domain`; undefined when several manifolds exist and none is declared default */
  readonly defaultManifold: Manifold | undefined;
  readonly pointSets: ReadonlyMap<string, PointSet>;
  private readonly built = new Map<string, Field>();
  private readonly building = new Set<string>();
  private readonly builtNets = new Map<string, Net>();
  private readonly buildingNets = new Set<string>();
  private readonly netResolver: ProgramResolver = {
    net: (id, path) => this.net(id, path).signature,
    program: (id, path) => this.net(id, path).program,
  };

  constructor(readonly spec: BundleSpec) {
    this.name = spec.name ?? "untitled";
    const manifolds = new Map<string, Manifold>();
    for (const [id, m] of Object.entries(spec.manifolds ?? {})) manifolds.set(id, new Manifold(id, m));
    if (manifolds.size === 0) {
      // implicit default manifold: dimension inferred from the first field that says
      manifolds.set("default", new Manifold("default", { numDims: inferDims(spec) }));
    }
    this.manifolds = manifolds;
    // the default manifold is only needed by fields / point sets that omit `domain`
    const dflt = spec.defaultManifold ?? (manifolds.size === 1 ? [...manifolds.keys()][0]! : undefined);
    if (dflt !== undefined && !manifolds.has(dflt)) throw new SpecError(`defaultManifold "${dflt}" is not defined`);
    this.defaultManifold = dflt === undefined ? undefined : manifolds.get(dflt);
    const pointSets = new Map<string, PointSet>();
    for (const [id, ps] of Object.entries(spec.pointSets ?? {})) pointSets.set(id, new PointSet(id, ps, this.manifoldOf(ps.domain, ["pointSets", id])));
    this.pointSets = pointSets;
  }

  /** parse + validate a JSON value */
  static parse(json: unknown): Bundle {
    const r = BundleSchema.safeParse(json);
    if (!r.success) {
      const issue = r.error.issues[0]!;
      throw new SpecError(`${issue.message}${r.error.issues.length > 1 ? ` (+${r.error.issues.length - 1} more issues)` : ""}`, issue.path.map(String));
    }
    return new Bundle(r.data);
  }

  get info(): Info | undefined { return infoOf(this.name, this.spec); }
  get fieldIds(): string[] { return Object.keys(this.spec.fields); }
  get scalarFieldIds(): string[] { return this.fieldIds.filter((id) => this.spec.fields[id]!.kind === "scalar"); }
  get vectorFieldIds(): string[] { return this.fieldIds.filter((id) => this.spec.fields[id]!.kind === "vector"); }
  get netIds(): string[] { return Object.keys(this.spec.nets ?? {}); }

  private manifoldOf(id: string | undefined, path: string[]): Manifold {
    if (id === undefined) {
      if (!this.defaultManifold) throw new SpecError("no `domain` and the bundle has several manifolds but no `defaultManifold`", path);
      return this.defaultManifold;
    }
    const m = this.manifolds.get(id);
    if (!m) throw new SpecError(`unknown manifold "${id}"`, [...path, "domain"]);
    return m;
  }

  field(id: string, path: string[] = []): Field {
    const done = this.built.get(id);
    if (done) return done;
    const spec = this.spec.fields[id];
    if (!spec) throw new SpecError(`unknown field "${id}"`, path);
    if (this.building.has(id)) throw new SpecError(`field "${id}" refers to itself (cycle: ${[...this.building, id].join(" -> ")})`, path);
    this.building.add(id);
    try {
      const domain = this.manifoldOf(spec.domain, ["fields", id]);
      const resolver: FieldResolver = {
        scalar: (ref, p) => this.scalarField(ref, p).data,
        vector: (ref, p) => this.vectorField(ref, p).data,
        nets: this.netResolver,
      };
      const fpath = ["fields", id, "data"];
      const field: Field =
        spec.kind === "scalar"
          ? new ScalarField(id, spec, domain, buildScalarFieldData(spec.data, domain.numDims, resolver, fpath), this)
          : new VectorField(id, spec, domain, buildVectorFieldData(spec.data, domain.numDims, resolver, fpath));
      this.built.set(id, field);
      return field;
    } finally {
      this.building.delete(id);
    }
  }

  scalarField(id: string, path: string[] = []): ScalarField {
    const f = this.field(id, path);
    if (f.kind !== "scalar") throw new SpecError(`field "${id}" is a vector field, expected a scalar field`, path);
    return f;
  }

  vectorField(id: string, path: string[] = []): VectorField {
    const f = this.field(id, path);
    if (f.kind !== "vector") throw new SpecError(`field "${id}" is a scalar field, expected a vector field`, path);
    return f;
  }

  /** a net by id, with its signature inferred (lazily, so nets may refer to each other in any order) */
  net(id: string, path: string[] = []): Net {
    const done = this.builtNets.get(id);
    if (done) return done;
    const spec = this.spec.nets?.[id];
    if (!spec) throw new SpecError(`unknown net "${id}"`, path);
    if (this.buildingNets.has(id)) throw new SpecError(`net "${id}" refers to itself (cycle: ${[...this.buildingNets, id].join(" -> ")})`, path);
    this.buildingNets.add(id);
    try {
      const net = new Net(id, spec, inferNet(spec, this.netResolver, ["nets", id]), this.netResolver);
      this.builtNets.set(id, net);
      return net;
    } finally {
      this.buildingNets.delete(id);
    }
  }

  /** build every field and net, collecting errors per id (nets keyed "nets.<id>") */
  buildAll(): Map<string, TensatoryError> {
    const errors = new Map<string, TensatoryError>();
    for (const id of this.netIds) {
      try { this.net(id); } catch (e) {
        if (e instanceof TensatoryError) errors.set(`nets.${id}`, e); else throw e;
      }
    }
    for (const id of this.fieldIds) {
      try { this.field(id); } catch (e) {
        if (e instanceof TensatoryError) errors.set(id, e); else throw e;
      }
    }
    // a manifold's `flow` must name a vector field living on that manifold
    for (const m of this.manifolds.values()) {
      if (m.flow === undefined) continue;
      try {
        const f = this.vectorField(m.flow, ["manifolds", m.id, "flow"]);
        if (f.domain !== m) throw new SpecError(`flow field "${m.flow}" lives on manifold "${f.domain.id}"`, ["manifolds", m.id, "flow"]);
      } catch (e) {
        if (e instanceof TensatoryError) errors.set(`manifolds.${m.id}`, e); else throw e;
      }
    }
    return errors;
  }
}

/** without explicit manifolds, take the dimension from the first field that makes it evident */
export function inferDims(spec: BundleSpec): number {
  for (const f of Object.values(spec.fields)) {
    const d = f.data;
    if ("box" in d && d.box) return Array.isArray(d.box) ? d.box.length : d.box.a.length;
    if (d.type === "dense" && "shape" in d.samples) return d.samples.shape.length;
    if (d.type === "densev" && "shape" in d.samples) return d.samples.shape.length - 1;
  }
  throw new SpecError("cannot infer the manifold dimension; declare `manifolds`");
}

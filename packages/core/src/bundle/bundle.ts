import { z } from "zod";
import type { BundleSpec, FieldSpec, ManifoldDefinitionSpec, PointSetSpec } from "@tensatory/schema";
import { BUNDLE_VERSION } from "@tensatory/schema";
import { SpecError, TensatoryError } from "../errors";
import { Codomain } from "../fields/codomain";
import type { ScalarFieldData, VectorFieldData } from "../fields/fieldData";
import { FieldSchema, buildScalarFieldData, buildVectorFieldData, type FieldResolver } from "../fields/spec";

/*******************************************************/
/* schema */

export const ManifoldDefinitionSchema: z.ZodType<ManifoldDefinitionSpec> = z.object({
  name: z.string().optional(),
  numDims: z.number().int().positive(),
  dimNames: z.array(z.string()).optional(),
  dimWeights: z.array(z.number()).optional(),
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
  description: z.string().optional(),
  manifolds: z.record(z.string(), ManifoldDefinitionSchema).optional(),
  defaultManifold: z.string().optional(),
  fields: z.record(z.string(), FieldSchema),
  pointSets: z.record(z.string(), PointSetSchema).optional(),
});

/*******************************************************/
/* runtime */

export class Manifold {
  readonly name: string;
  readonly dimNames: readonly string[];
  constructor(readonly id: string, readonly spec: ManifoldDefinitionSpec) {
    this.name = spec.name ?? id;
    if (spec.dimNames && spec.dimNames.length !== spec.numDims)
      throw new SpecError(`dimNames has ${spec.dimNames.length} entries for ${spec.numDims} dims`, ["manifolds", id]);
    if (spec.dimWeights && spec.dimWeights.length !== spec.numDims)
      throw new SpecError(`dimWeights has ${spec.dimWeights.length} entries for ${spec.numDims} dims`, ["manifolds", id]);
    this.dimNames = spec.dimNames ?? Array.from({ length: spec.numDims }, (_, i) => `x${i}`);
  }
  get numDims(): number { return this.spec.numDims; }
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
}

export class VectorField {
  readonly kind = "vector" as const;
  readonly name: string;
  constructor(readonly id: string, readonly spec: Extract<FieldSpec, { kind: "vector" }>, readonly domain: Manifold, readonly data: VectorFieldData) {
    this.name = spec.name ?? id;
  }
}

export type Field = ScalarField | VectorField;

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

  get fieldIds(): string[] { return Object.keys(this.spec.fields); }
  get scalarFieldIds(): string[] { return this.fieldIds.filter((id) => this.spec.fields[id]!.kind === "scalar"); }
  get vectorFieldIds(): string[] { return this.fieldIds.filter((id) => this.spec.fields[id]!.kind === "vector"); }

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

  /** build every field, collecting errors per field */
  buildAll(): Map<string, TensatoryError> {
    const errors = new Map<string, TensatoryError>();
    for (const id of this.fieldIds) {
      try { this.field(id); } catch (e) {
        if (e instanceof TensatoryError) errors.set(id, e); else throw e;
      }
    }
    return errors;
  }
}

/** without explicit manifolds, take the dimension from the first field that makes it evident */
function inferDims(spec: BundleSpec): number {
  for (const f of Object.values(spec.fields)) {
    const d = f.data;
    if ("box" in d && d.box) return Array.isArray(d.box) ? d.box.length : d.box.a.length;
    if (d.type === "dense" && "shape" in d.samples) return d.samples.shape.length;
    if (d.type === "densev" && "shape" in d.samples) return d.samples.shape.length - 1;
  }
  throw new SpecError("cannot infer the manifold dimension; declare `manifolds`");
}

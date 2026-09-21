// zod schemas + builders for curve specs (schema/curves.ts).

import { z } from "zod";
import type { ArraySpec, CurveDataSpec, CurveSpec, FlowMethodSpec, ParamSpec, PointSpec, ScalarFieldDataSpec, VectorFieldDataSpec } from "@tensatory/schema";
import { ArraySchema, buildAnyArray, type ArrayResolver } from "../arrays/spec";
import { SpecError } from "../errors";
import { CodomainSchema } from "../fields/codomain";
import { SymbolicVectorFieldData, type ScalarFieldData, type VectorFieldData } from "../fields/fieldData";
import { BoxSchema, ScalarFieldDataSchema, VectorFieldDataSchema, buildScalarFieldData, buildVectorFieldData, type FieldResolver } from "../fields/spec";
import { emptyEnv, normalizeVector } from "../symbolic";
import { SymbolicVectorSchema } from "../symbolic/spec";
import { CurveData, FlowCurveData, SampledCurveData, SymbolicCurveData, scaleCurve, translateCurve } from "./curveData";

/*******************************************************/
/* schemas */

const interval = z.tuple([z.number(), z.number()]);
const point = z.array(z.number());
const pointSpec: z.ZodType<PointSpec> = z.union([point, ArraySchema]);
const curveId = z.string().min(1);
const consts = z.record(z.string(), z.number()).optional();

export const FlowMethodSchema: z.ZodType<FlowMethodSpec> = z.object({
  integrator: z.enum(["rk4", "euler"]).optional(),
  step: z.number().positive().optional(),
  tol: z.number().positive().optional(),
});

export const CurveDataSchema: z.ZodType<CurveDataSpec> = z.lazy(() =>
  z.discriminatedUnion("type", [
    z.object({ type: z.literal("symbolic"), expr: SymbolicVectorSchema, interval, consts }),
    z.object({
      type: z.literal("sampled"),
      points: ArraySchema,
      times: z.union([ArraySchema, interval]).optional(),
      velocities: ArraySchema.optional(),
      interp: z.enum(["linear", "cubic", "step"]).optional(),
      labels: z.array(z.string()).optional(),
      closed: z.boolean().optional(),
    }),
    z.object({
      type: z.literal("flow"),
      field: z.union([curveId, VectorFieldDataSchema, ScalarFieldDataSchema]),
      start: pointSpec,
      interval,
      dir: z.enum(["ascending", "descending"]).optional(),
      method: FlowMethodSchema.optional(),
    }),
    z.object({ type: z.literal("translate"), arg: z.union([curveId, CurveDataSchema]), vec: pointSpec }),
    z.object({ type: z.literal("scale"), arg: z.union([curveId, CurveDataSchema]), origin: pointSpec.optional(), scale: z.union([z.number(), z.array(z.number())]) }),
  ]),
);

export const ParamSchema: z.ZodType<ParamSpec> = z.object({
  name: z.string().optional(),
  unit: z.string().nullable().optional(),
  codomain: CodomainSchema.optional(),
});

export const CurveSchema: z.ZodType<CurveSpec> = z.object({
  data: CurveDataSchema,
  domain: z.string().optional(),
  param: ParamSchema.optional(),
  name: z.string().optional(),
  summary: z.string().optional(),
  details: z.string().optional(),
});

// BoxSchema is imported so the field-data schemas above are the same instances as fields/spec.ts uses
void BoxSchema;

/*******************************************************/
/* builders */

/** what a curve builder needs from the bundle: fields (for `flow`), curves (for pullback args), arrays */
export interface CurveResolver {
  fields: FieldResolver;
  curve(id: string, path: string[]): CurveData;
  arrays?: ArrayResolver | undefined;
}

function resolvePoint(spec: PointSpec, dimCount: number, arrays: ArrayResolver | undefined, path: string[]): number[] {
  if (Array.isArray(spec) && spec.every((x) => typeof x === "number")) {
    if (spec.length !== dimCount) throw new SpecError(`expected ${dimCount} components, got ${spec.length}`, path);
    return spec as number[];
  }
  const arr = buildAnyArray(spec as ArraySpec, path, arrays);
  if (arr.ndim !== 1 || arr.size !== dimCount) throw new SpecError(`expected a vector of ${dimCount} components, got shape [${arr.shape}]`, path);
  return Array.from(arr.data);
}

/** the vector field a `flow` follows: a vector field as is, a scalar field's gradient (ascending) or its negative */
function flowField(field: string | ScalarFieldData | VectorFieldData, dir: "ascending" | "descending", dimCount: number, resolver: CurveResolver, path: string[]): VectorFieldData {
  let fd: ScalarFieldData | VectorFieldData;
  if (typeof field === "string") {
    try { fd = resolver.fields.vector(field, path); }
    catch (e) { if (!(e instanceof SpecError) || !/is a scalar field/.test(e.message)) throw e; fd = resolver.fields.scalar(field, path); }
  } else fd = field;
  if (fd.dimCount !== dimCount) throw new SpecError(`the field has ${fd.dimCount} dims, the curve's manifold ${dimCount}`, path);
  if (fd.rank === "vector") return fd;
  const g: VectorFieldData = new SymbolicVectorFieldData({ k: "grad", s: { k: "arg", name: "f" } }, dimCount, { scalars: { f: fd }, vectors: {} });
  if (dir === "ascending") return g;
  return new SymbolicVectorFieldData({ k: "scalev", v: { k: "argv", name: "g" }, s: { k: "const", value: -1 } }, dimCount, { scalars: {}, vectors: { g } });
}

export function buildCurveData(spec: CurveDataSpec, dimCount: number, resolver: CurveResolver, path: string[] = ["data"]): CurveData {
  const arrays = resolver.arrays;
  switch (spec.type) {
    case "symbolic": {
      const env = emptyEnv(dimCount, spec.consts ?? {}); // D-vectors of t (coordinate 0); other coordinates are rejected
      return new SymbolicCurveData(normalizeVector(spec.expr, env, [...path, "expr"]), dimCount, spec.interval, path);
    }
    case "sampled": {
      const pts = buildAnyArray(spec.points, [...path, "points"], arrays);
      if (pts.ndim !== 2 || pts.shape[1] !== dimCount) throw new SpecError(`points must have shape [N, ${dimCount}], got [${pts.shape}]`, [...path, "points"]);
      const N = pts.shape[0]!;
      let times: Float64Array | undefined;
      if (spec.times !== undefined) {
        if (Array.isArray(spec.times)) {
          // an interval (an ArraySpec is an object or a path, never a bare number pair): uniform times over it
          const [a, b] = spec.times;
          times = Float64Array.from({ length: N }, (_, i) => (N === 1 ? a : a + ((b - a) * i) / (N - 1)));
        } else {
          const t = buildAnyArray(spec.times as ArraySpec, [...path, "times"], arrays);
          if (t.ndim !== 1) throw new SpecError(`times must be a vector, got shape [${t.shape}]`, [...path, "times"]);
          times = Float64Array.from(t.data);
        }
      }
      const vel = spec.velocities === undefined ? undefined : buildAnyArray(spec.velocities, [...path, "velocities"], arrays);
      if (vel && (vel.ndim !== 2 || vel.shape[0] !== N || vel.shape[1] !== dimCount)) throw new SpecError(`velocities must have shape [${N}, ${dimCount}], got [${vel.shape}]`, [...path, "velocities"]);
      return new SampledCurveData(dimCount, Float64Array.from(pts.data), times, spec.interp ?? "linear", vel ? Float64Array.from(vel.data) : undefined, spec.closed ?? false, spec.labels, path);
    }
    case "flow": {
      const fieldPath = [...path, "field"];
      const given: string | ScalarFieldData | VectorFieldData = typeof spec.field === "string"
        ? spec.field
        : isVectorPullback(spec.field)
          ? buildVectorFieldData(spec.field as VectorFieldDataSpec, dimCount, resolver.fields, fieldPath)
          : buildScalarFieldData(spec.field as ScalarFieldDataSpec, dimCount, resolver.fields, fieldPath);
      const field = flowField(given, spec.dir ?? "descending", dimCount, resolver, fieldPath);
      const start = resolvePoint(spec.start, dimCount, arrays, [...path, "start"]);
      const span = spec.interval[1] - spec.interval[0];
      const step = spec.method?.step ?? Math.min(span, Math.max(Math.abs(spec.interval[0]), Math.abs(spec.interval[1]))) / 1000;
      if (spec.method?.tol !== undefined) throw new SpecError("adaptive stepping (method.tol) is not supported yet; give method.step", [...path, "method"]);
      return new FlowCurveData(field, start, spec.interval, { integrator: spec.method?.integrator ?? "rk4", step }, path);
    }
    case "translate": {
      const inner = typeof spec.arg === "string" ? resolver.curve(spec.arg, [...path, "arg"]) : buildCurveData(spec.arg, dimCount, resolver, [...path, "arg"]);
      return translateCurve(inner, resolvePoint(spec.vec, dimCount, arrays, [...path, "vec"]));
    }
    case "scale": {
      const inner = typeof spec.arg === "string" ? resolver.curve(spec.arg, [...path, "arg"]) : buildCurveData(spec.arg, dimCount, resolver, [...path, "arg"]);
      const scale = spec.scale;
      if (Array.isArray(scale) && scale.length !== dimCount) throw new SpecError(`scale has ${scale.length} factors for ${dimCount} dims`, [...path, "scale"]);
      return scaleCurve(inner, spec.origin === undefined ? undefined : resolvePoint(spec.origin, dimCount, arrays, [...path, "origin"]), scale);
    }
  }
}

/**
 * Whether an inline field-data spec is vector-valued: by its type, through translate / scale wrappers. A wrapper
 * chain ending in a field ID is taken as scalar (write the id directly as `field` to have it resolved either way).
 */
function isVectorPullback(d: ScalarFieldDataSpec | VectorFieldDataSpec): boolean {
  let cur: ScalarFieldDataSpec | VectorFieldDataSpec | string = d;
  while (typeof cur === "object" && (cur.type === "translate" || cur.type === "scale")) cur = cur.arg;
  if (typeof cur === "string") return false;
  const t = cur.type;
  return t === "densev" || t === "sparsev" || t === "pointwisev" || t === "symbolicv" || t === "netv";
}

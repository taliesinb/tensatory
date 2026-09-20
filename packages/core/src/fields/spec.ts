// zod schemas + builders for field data specs and field specs.

import { z } from "zod";
import type {
  BoxSpec,
  FieldSpec,
  NetScalarFieldDataSpec,
  NetVectorFieldDataSpec,
  PointSpec,
  ScalarFieldDataSpec,
  ScalarFieldSpec,
  VectorFieldDataSpec,
  VectorFieldSpec,
  VectorSpec,
} from "@tensatory/schema";
import { ArraySchema, SizedArraySchema, buildArray } from "../arrays/spec";
import { NotSupportedError, SpecError } from "../errors";
import { Box } from "../geometry/box";
import { DenseGrid } from "../geometry/grid";
import { checkNamespaces, normalizeScalar, normalizeVector, type NameEnv } from "../symbolic/normalize";
import { SymbolicScalarSchema, SymbolicVectorSchema } from "../symbolic/spec";
import { NetScalarFieldData, NetVectorFieldData, fieldProgram, type NetField } from "../nets/fieldData";
import type { Val } from "../nets/ops";
import { compileNet, type ProgramResolver } from "../nets/program";
import { inferNetField } from "../nets/shapes";
import { NetScalarFieldDataObject, NetVectorFieldDataObject } from "../nets/spec";
import { CodomainSchema } from "./codomain";
import {
  DenseScalarFieldData,
  DenseVectorFieldData,
  SymbolicScalarFieldData,
  SymbolicVectorFieldData,
  scaleField,
  translateField,
  type FieldArgs,
  type ScalarFieldData,
  type VectorFieldData,
} from "./fieldData";

/*******************************************************/
/* schemas */

const point = z.array(z.number());
const interval = z.tuple([z.number(), z.number()]);
export const BoxSchema: z.ZodType<BoxSpec> = z.union([z.array(interval), z.object({ a: point, b: point })]);
const pointSpec: z.ZodType<PointSpec> = z.union([point, ArraySchema]);
const vectorSpec: z.ZodType<VectorSpec> = pointSpec;
const consts = z.record(z.string(), z.number()).optional();
const fieldId = z.string().min(1);

export const ScalarStatisticsSchema = z.object({
  extrema: z.object({ min: z.number(), max: z.number() }).optional(),
  counts: z.object({
    total: z.number().int().optional(), pos: z.number().int().optional(), neg: z.number().int().optional(),
    zero: z.number().int().optional(), nan: z.number().int().optional(), posInf: z.number().int().optional(), negInf: z.number().int().optional(),
  }).optional(),
  moments: z.object({ mean: z.number(), variance: z.number().optional(), skewness: z.number().optional(), kurtosis: z.number().optional() }).optional(),
  quantiles: z.union([z.array(z.number()), ArraySchema]).optional(),
  histogram: z.object({ counts: z.union([z.array(z.number()), ArraySchema]), edges: z.union([interval, z.array(z.number()), ArraySchema]) }).optional(),
});

export const ScalarFieldDataSchema: z.ZodType<ScalarFieldDataSpec> = z.lazy(() =>
  z.discriminatedUnion("type", [
    z.object({ type: z.literal("dense"), samples: SizedArraySchema, stats: ScalarStatisticsSchema.optional(), box: BoxSchema.optional() }),
    z.object({ type: z.literal("sparse"), points: ArraySchema, samples: ArraySchema, stats: ScalarStatisticsSchema.optional(), box: BoxSchema.optional() }),
    z.object({ type: z.literal("translate"), arg: z.union([fieldId, ScalarFieldDataSchema]), vec: vectorSpec }),
    z.object({ type: z.literal("scale"), arg: z.union([fieldId, ScalarFieldDataSchema]), origin: pointSpec.optional(), scale: z.union([z.number(), z.array(z.number())]) }),
    z.object({
      type: z.literal("pointwise"),
      expr: SymbolicScalarSchema,
      consts,
      scalars: z.record(z.string(), z.union([fieldId, ScalarFieldDataSchema])).optional(),
      vectors: z.record(z.string(), z.union([fieldId, VectorFieldDataSchema])).optional(),
    }),
    z.object({ type: z.literal("symbolic"), expr: SymbolicScalarSchema, box: BoxSchema.optional(), consts }),
    NetScalarFieldDataObject,
  ]),
);

export const VectorFieldDataSchema: z.ZodType<VectorFieldDataSpec> = z.lazy(() =>
  z.discriminatedUnion("type", [
    z.object({ type: z.literal("densev"), samples: SizedArraySchema, box: BoxSchema.optional() }),
    z.object({ type: z.literal("sparsev"), points: ArraySchema, samples: ArraySchema, box: BoxSchema.optional() }),
    z.object({ type: z.literal("translate"), arg: z.union([fieldId, VectorFieldDataSchema]), vec: vectorSpec }),
    z.object({ type: z.literal("scale"), arg: z.union([fieldId, VectorFieldDataSchema]), origin: pointSpec.optional(), scale: z.union([z.number(), z.array(z.number())]) }),
    z.object({
      type: z.literal("pointwisev"),
      expr: SymbolicVectorSchema,
      consts,
      scalars: z.record(z.string(), z.union([fieldId, ScalarFieldDataSchema])).optional(),
      vectors: z.record(z.string(), z.union([fieldId, VectorFieldDataSchema])).optional(),
    }),
    z.object({ type: z.literal("symbolicv"), expr: SymbolicVectorSchema, box: BoxSchema.optional(), consts }),
    NetVectorFieldDataObject,
  ]),
);

const fieldCommon = {
  domain: z.string().optional(),
  name: z.string().optional(),
  summary: z.string().optional(),
  details: z.string().optional(),
};

const ScalarFieldObject = z.object({
  kind: z.literal("scalar"),
  data: ScalarFieldDataSchema,
  codomain: CodomainSchema.optional(),
  exactGradient: fieldId.optional(),
  ...fieldCommon,
});

const VectorFieldObject = z.object({
  kind: z.literal("vector"),
  data: VectorFieldDataSchema,
  ...fieldCommon,
});

export const ScalarFieldSchema: z.ZodType<ScalarFieldSpec> = ScalarFieldObject;
export const VectorFieldSchema: z.ZodType<VectorFieldSpec> = VectorFieldObject;
export const FieldSchema: z.ZodType<FieldSpec> = z.discriminatedUnion("kind", [ScalarFieldObject, VectorFieldObject]);

/*******************************************************/
/* builders */

/** resolves field ids (and net ids) referenced from within field data specs */
export interface FieldResolver {
  scalar(id: string, path: string[]): ScalarFieldData;
  vector(id: string, path: string[]): VectorFieldData;
  nets?: ProgramResolver;
}

export const noNets: ProgramResolver = {
  net: (id, path) => { throw new SpecError(`cannot resolve net reference "${id}" outside a bundle`, path); },
  program: (id, path) => { throw new SpecError(`cannot resolve net reference "${id}" outside a bundle`, path); },
};

export const noResolver: FieldResolver = {
  scalar: (id, path) => { throw new SpecError(`cannot resolve field reference "${id}" outside a bundle`, path); },
  vector: (id, path) => { throw new SpecError(`cannot resolve field reference "${id}" outside a bundle`, path); },
};

/** shape-check a net-backed field spec and fold it into one single-input program */
function netField(spec: NetScalarFieldDataSpec | NetVectorFieldDataSpec, dimCount: number, resolver: FieldResolver, path: string[]): NetField {
  const nets = resolver.nets ?? noNets;
  inferNetField(spec, dimCount, nets, path);
  const base = typeof spec.net === "string" ? nets.program(spec.net, [...path, "net"]) : compileNet(spec.net, nets, [...path, "net"]);
  const arrays = new Map<string, Val>();
  for (const [n, a] of Object.entries(spec.arrays ?? {})) {
    if (typeof a === "string" || !("shape" in a)) throw new NotSupportedError(`external arrays are not loaded yet`, [...path, "arrays", n]);
    arrays.set(n, { arr: buildArray(a, [...path, "arrays", n]), rank: a.shape.length });
  }
  return { program: fieldProgram(base, spec.inputs, arrays, dimCount, path), output: spec.output, nets };
}

function resolvePoint(spec: PointSpec, dimCount: number, path: string[]): number[] {
  if (Array.isArray(spec) && spec.every((x) => typeof x === "number")) {
    if (spec.length !== dimCount) throw new SpecError(`expected ${dimCount} components, got ${spec.length}`, path);
    return spec as number[];
  }
  if (typeof spec === "string" || (typeof spec === "object" && spec !== null && "type" in spec && spec.type === "handle"))
    throw new NotSupportedError(`array-backed points are not supported yet`, path);
  const arr = buildArray(spec as Parameters<typeof buildArray>[0], path);
  if (arr.ndim !== 1 || arr.size !== dimCount) throw new SpecError(`expected a vector of ${dimCount} components, got shape [${arr.shape}]`, path);
  return Array.from(arr.data);
}

function boxOrUnit(spec: BoxSpec | undefined, dimCount: number, path: string[]): Box {
  const box = spec ? Box.fromSpec(spec, path) : Box.unit(dimCount);
  if (box.dimCount !== dimCount) throw new SpecError(`box has ${box.dimCount} dims, expected ${dimCount}`, path);
  return box;
}

function buildArgs(
  spec: { consts?: Record<string, number>; scalars?: Record<string, ScalarFieldDataSpec | string>; vectors?: Record<string, VectorFieldDataSpec | string> },
  dimCount: number,
  resolver: FieldResolver,
  path: string[],
): { env: NameEnv; args: FieldArgs } {
  checkNamespaces(spec.consts, spec.scalars, spec.vectors, path);
  const scalars: Record<string, ScalarFieldData> = {};
  const vectors: Record<string, VectorFieldData> = {};
  for (const [name, s] of Object.entries(spec.scalars ?? {}))
    scalars[name] = typeof s === "string" ? resolver.scalar(s, [...path, "scalars", name]) : buildScalarFieldData(s, dimCount, resolver, [...path, "scalars", name]);
  for (const [name, s] of Object.entries(spec.vectors ?? {}))
    vectors[name] = typeof s === "string" ? resolver.vector(s, [...path, "vectors", name]) : buildVectorFieldData(s, dimCount, resolver, [...path, "vectors", name]);
  const env: NameEnv = { dimCount, consts: spec.consts ?? {}, scalarArgs: new Set(Object.keys(scalars)), vectorArgs: new Set(Object.keys(vectors)) };
  return { env, args: { scalars, vectors } };
}

export function buildScalarFieldData(spec: ScalarFieldDataSpec, dimCount: number, resolver: FieldResolver = noResolver, path: string[] = ["data"]): ScalarFieldData {
  switch (spec.type) {
    case "dense": {
      const arr = buildArray(spec.samples, [...path, "samples"]);
      if (arr.ndim !== dimCount) throw new SpecError(`samples have ${arr.ndim} axes but the field has ${dimCount} dims`, path);
      return new DenseScalarFieldData(new DenseGrid([...arr.shape], boxOrUnit(spec.box, dimCount, [...path, "box"])), arr.data, spec.stats);
    }
    case "sparse":
      throw new NotSupportedError("sparse sampled fields are not supported yet", path);
    case "translate": {
      const inner = typeof spec.arg === "string" ? resolver.scalar(spec.arg, [...path, "arg"]) : buildScalarFieldData(spec.arg, dimCount, resolver, [...path, "arg"]);
      return translateField(inner, resolvePoint(spec.vec, dimCount, [...path, "vec"]));
    }
    case "scale": {
      const inner = typeof spec.arg === "string" ? resolver.scalar(spec.arg, [...path, "arg"]) : buildScalarFieldData(spec.arg, dimCount, resolver, [...path, "arg"]);
      return scaleField(inner, spec.origin === undefined ? undefined : resolvePoint(spec.origin, dimCount, [...path, "origin"]), spec.scale);
    }
    case "pointwise": {
      const { env, args } = buildArgs(spec, dimCount, resolver, path);
      return new SymbolicScalarFieldData(normalizeScalar(spec.expr, env, [...path, "expr"]), dimCount, args, undefined, path);
    }
    case "symbolic": {
      const env: NameEnv = { dimCount, consts: spec.consts ?? {}, scalarArgs: new Set(), vectorArgs: new Set() };
      return new SymbolicScalarFieldData(normalizeScalar(spec.expr, env, [...path, "expr"]), dimCount, undefined, boxOrUnit(spec.box, dimCount, [...path, "box"]), path);
    }
    case "net":
      return new NetScalarFieldData(dimCount, boxOrUnit(spec.box, dimCount, [...path, "box"]), netField(spec, dimCount, resolver, path));
  }
}

export function buildVectorFieldData(spec: VectorFieldDataSpec, dimCount: number, resolver: FieldResolver = noResolver, path: string[] = ["data"]): VectorFieldData {
  switch (spec.type) {
    case "densev": {
      const arr = buildArray(spec.samples, [...path, "samples"]);
      if (arr.ndim !== dimCount + 1 || arr.shape[dimCount] !== dimCount)
        throw new SpecError(`vector samples must have shape [S_0..S_${dimCount - 1}, ${dimCount}], got [${arr.shape}]`, path);
      return new DenseVectorFieldData(new DenseGrid(arr.shape.slice(0, dimCount), boxOrUnit(spec.box, dimCount, [...path, "box"])), arr.data);
    }
    case "sparsev":
      throw new NotSupportedError("sparse sampled fields are not supported yet", path);
    case "translate": {
      const inner = typeof spec.arg === "string" ? resolver.vector(spec.arg, [...path, "arg"]) : buildVectorFieldData(spec.arg, dimCount, resolver, [...path, "arg"]);
      return translateField(inner, resolvePoint(spec.vec, dimCount, [...path, "vec"]));
    }
    case "scale": {
      const inner = typeof spec.arg === "string" ? resolver.vector(spec.arg, [...path, "arg"]) : buildVectorFieldData(spec.arg, dimCount, resolver, [...path, "arg"]);
      return scaleField(inner, spec.origin === undefined ? undefined : resolvePoint(spec.origin, dimCount, [...path, "origin"]), spec.scale);
    }
    case "pointwisev": {
      const { env, args } = buildArgs(spec, dimCount, resolver, path);
      return new SymbolicVectorFieldData(normalizeVector(spec.expr, env, [...path, "expr"]), dimCount, args, undefined, path);
    }
    case "symbolicv": {
      const env: NameEnv = { dimCount, consts: spec.consts ?? {}, scalarArgs: new Set(), vectorArgs: new Set() };
      return new SymbolicVectorFieldData(normalizeVector(spec.expr, env, [...path, "expr"]), dimCount, undefined, boxOrUnit(spec.box, dimCount, [...path, "box"]), path);
    }
    case "netv":
      return new NetVectorFieldData(dimCount, boxOrUnit(spec.box, dimCount, [...path, "box"]), netField(spec, dimCount, resolver, path));
  }
}

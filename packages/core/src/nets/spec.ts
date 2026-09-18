// zod schemas validating the SYNTAX of array expressions and net specs (see
// @tensatory/schema nets.ts). Names, shapes and axes are checked in shapes.ts.

import { z } from "zod";
import type {
  ArrayExpr,
  ArrayReduceFn,
  NetScalarFieldDataSpec,
  NetSpec,
  NetVectorFieldDataSpec,
  ShapeSpec,
} from "@tensatory/schema";
import { ArraySchema, SizedArraySchema } from "../arrays/spec";
import { RandomWidgetSchema } from "../arrays/random";
import { SCALAR_BINARY_OPS, SCALAR_NARY_OPS, SCALAR_UNARY_OPS } from "../symbolic/spec";

export const ARRAY_REDUCE_FNS = ["sum", "mean", "prod", "max", "min", "logsumexp"] as const satisfies readonly ArrayReduceFn[];
export const ARRAY_COMPARE_OPS = ["lt", "le", "gt", "ge", "eq", "ne"] as const;

const name = z.string().min(1);
const int = z.number().int();
const dimIndex = int.nonnegative();
const axisSize = int.nonnegative();

export const ShapeSchema: z.ZodType<ShapeSpec> = z.array(z.union([axisSize, name]));
const reshapeShape = z.array(z.union([axisSize, name, z.literal(-1)]));

export const ArrayExprSchema: z.ZodType<ArrayExpr> = z.lazy(() =>
  z.union([
    z.number(),
    name,
    z.discriminatedUnion("op", [
      z.object({ op: z.literal("arg"), name }),
      z.object({ op: z.literal("coord"), index: dimIndex }),
      z.object({ op: z.literal("coordv") }),
      z.object({ op: z.enum(SCALAR_UNARY_OPS), val: ArrayExprSchema }),
      z.object({ op: z.enum(SCALAR_NARY_OPS), vals: z.array(ArrayExprSchema).min(1) }),
      z.object({ op: z.enum(SCALAR_BINARY_OPS), vals: z.tuple([ArrayExprSchema, ArrayExprSchema]) }),
      z.object({ op: z.literal("clamp"), val: ArrayExprSchema, min: ArrayExprSchema, max: ArrayExprSchema }),
      z.object({ op: z.enum(ARRAY_COMPARE_OPS), vals: z.tuple([ArrayExprSchema, ArrayExprSchema]) }),
      z.object({ op: z.literal("where"), cond: ArrayExprSchema, vals: z.tuple([ArrayExprSchema, ArrayExprSchema]) }),
      z.object({ op: z.literal("matmul"), vals: z.tuple([ArrayExprSchema, ArrayExprSchema]) }),
      z.object({ op: z.literal("einsum"), subscripts: z.string().min(1), vals: z.array(ArrayExprSchema).min(1) }),
      z.object({ op: z.literal("reduce"), fn: z.enum(ARRAY_REDUCE_FNS), val: ArrayExprSchema, axes: z.array(int).optional(), keepDims: z.boolean().optional() }),
      z.object({ op: z.enum(["argmax", "argmin"]), val: ArrayExprSchema, axis: int.optional() }),
      z.object({ op: z.enum(["softmax", "logSoftmax"]), val: ArrayExprSchema, axis: int.optional() }),
      z.object({ op: z.literal("reshape"), val: ArrayExprSchema, shape: reshapeShape }),
      z.object({ op: z.literal("transpose"), val: ArrayExprSchema, perm: z.array(dimIndex).optional() }),
      z.object({ op: z.literal("concat"), vals: z.array(ArrayExprSchema).min(1), axis: int }),
      z.object({ op: z.literal("slice"), val: ArrayExprSchema, axis: int, start: int.optional(), stop: int.optional(), step: int.optional() }),
      z.object({ op: z.literal("oneHot"), val: ArrayExprSchema, size: z.union([axisSize, name]) }),
      z.object({ op: z.literal("takeAlong"), val: ArrayExprSchema, indices: ArrayExprSchema, axis: int }),
      z.object({ op: z.literal("stopGradient"), val: ArrayExprSchema }),
      z.object({ op: z.literal("call"), net: z.union([name, NetSchema]), inputs: z.record(name, ArrayExprSchema), output: name }),
    ]),
  ]),
);

const netCommon = { name: z.string().optional(), description: z.string().optional() };

export const DirectionSchema = z.object({
  arrays: z.record(name, ArraySchema),
  norm: z.union([z.number().positive(), z.literal("origin")]).optional(),
  scale: z.number().positive().optional(),
  name: z.string().optional(),
  widget: RandomWidgetSchema.nullable().optional(),
});

export const NetSchema: z.ZodType<NetSpec> = z.lazy(() =>
  z.discriminatedUnion("type", [
    z.object({
      type: z.literal("def"),
      inputs: z.record(name, ShapeSchema),
      nodes: z.record(name, ArrayExprSchema).optional(),
      arrays: z.record(name, SizedArraySchema).optional(),
      outputs: z.record(name, ShapeSchema),
      ...netCommon,
    }),
    z.object({
      type: z.literal("bind"),
      net: z.union([name, NetSchema]),
      bind: z.record(name, ArraySchema),
      ...netCommon,
    }),
    z.object({
      type: z.literal("displace"),
      net: z.union([name, NetSchema]),
      coeffs: name.optional(),
      directions: z.array(DirectionSchema).min(1),
      ...netCommon,
    }),
    z.object({
      type: z.literal("grad"),
      net: z.union([name, NetSchema]),
      bind: z.record(name, ArraySchema).optional(),
      outputs: z.record(name, z.object({ of: name, wrt: name, seed: z.union([name, ArraySchema]).optional() })),
      keep: z.array(name).optional(),
      ...netCommon,
    }),
  ]),
);

const netFieldCommon = {
  net: z.union([name, NetSchema]),
  output: name,
  inputs: z.record(name, ArrayExprSchema).optional(),
  arrays: z.record(name, ArraySchema).optional(),
};

// BoxSchema lives in fields/spec.ts, which imports this module; take the box
// as a structural schema here to avoid the cycle (the field builder re-checks it).
const interval = z.tuple([z.number(), z.number()]);
const point = z.array(z.number());
const box = z.union([z.array(interval), z.object({ a: point, b: point })]);

// the plain objects are what the field-data discriminated unions splice in
export const NetScalarFieldDataObject = z.object({ type: z.literal("net"), ...netFieldCommon, box: box.optional() });
export const NetVectorFieldDataObject = z.object({ type: z.literal("netv"), ...netFieldCommon, box: box.optional() });
export const NetScalarFieldDataSchema: z.ZodType<NetScalarFieldDataSpec> = NetScalarFieldDataObject;
export const NetVectorFieldDataSchema: z.ZodType<NetVectorFieldDataSpec> = NetVectorFieldDataObject;

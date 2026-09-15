// zod schemas validating the SYNTAX of symbolic expressions (see
// @tensatory/schema symbolic.ts). Name resolution and typing happen in
// normalize.ts.

import { z } from "zod";
import type {
  ScalarBinaryOp,
  ScalarNaryOp,
  ScalarUnaryOp,
  SymbolicScalar,
  SymbolicVector,
} from "@tensatory/schema";

export const SCALAR_UNARY_OPS = [
  "sin", "cos", "tan", "sinh", "cosh", "tanh",
  "asin", "acos", "atan", "asinh", "acosh", "atanh",
  "relu", "sigmoid", "gelu", "silu", "softplus", "elu", "erf",
  "floor", "ceil", "round", "sign", "abs",
  "exp", "exp2", "exp10", "log", "log2", "log10", "log1p", "expm1",
  "plogp", "sqrt", "square", "negate", "reciprocal", "gauss",
] as const satisfies readonly ScalarUnaryOp[];

export const SCALAR_NARY_OPS = ["add", "mul", "min", "max", "mean", "rms"] as const satisfies readonly ScalarNaryOp[];

export const SCALAR_BINARY_OPS = ["sub", "div", "pow", "logBase", "atan2", "mod"] as const satisfies readonly ScalarBinaryOp[];

const dimIndex = z.number().int().nonnegative();
const name = z.string().min(1);

export const SymbolicScalarSchema: z.ZodType<SymbolicScalar> = z.lazy(() =>
  z.union([
    z.number(),
    name,
    z.discriminatedUnion("op", [
      z.object({ op: z.literal("const"), name }),
      z.object({ op: z.literal("coord"), index: dimIndex }),
      z.object({ op: z.literal("arg"), name }),
      z.object({ op: z.literal("argvi"), name, index: dimIndex }),
      z.object({ op: z.enum(SCALAR_UNARY_OPS), val: SymbolicScalarSchema }),
      z.object({ op: z.enum(SCALAR_NARY_OPS), vals: z.array(SymbolicScalarSchema).min(1) }),
      z.object({ op: z.enum(SCALAR_BINARY_OPS), vals: z.tuple([SymbolicScalarSchema, SymbolicScalarSchema]) }),
      z.object({ op: z.literal("clamp"), val: SymbolicScalarSchema, min: SymbolicScalarSchema, max: SymbolicScalarSchema }),
      z.object({ op: z.enum(["gaussKernel", "normalPDF"]), val: SymbolicScalarSchema, mu: SymbolicScalarSchema, sigma: SymbolicScalarSchema }),
      z.object({ op: z.enum(["dot", "cosineSim"]), vecs: z.tuple([SymbolicVectorSchema, SymbolicVectorSchema]) }),
      z.object({ op: z.literal("norm"), vec: SymbolicVectorSchema }),
      z.object({ op: z.literal("comp"), vec: SymbolicVectorSchema, index: dimIndex }),
    ]),
  ]),
);

export const SymbolicVectorSchema: z.ZodType<SymbolicVector> = z.lazy(() =>
  z.union([
    name,
    z.discriminatedUnion("op", [
      z.object({ op: z.literal("constv"), value: z.array(z.number()).min(1) }),
      z.object({ op: z.literal("basisv"), index: dimIndex }),
      z.object({ op: z.literal("coordv") }),
      z.object({ op: z.literal("argv"), name }),
      z.object({ op: z.literal("scalev"), vec: SymbolicVectorSchema, by: SymbolicScalarSchema }),
      z.object({ op: z.enum(["addv", "meanv"]), vecs: z.array(SymbolicVectorSchema).min(1) }),
      z.object({ op: z.literal("subv"), vecs: z.tuple([SymbolicVectorSchema, SymbolicVectorSchema]) }),
      z.object({ op: z.literal("sumv"), vecs: z.array(SymbolicVectorSchema).min(1), coeffs: z.array(SymbolicScalarSchema).min(1) }),
      z.object({ op: z.literal("compv"), coeffs: z.array(SymbolicScalarSchema).min(1) }),
      z.object({ op: z.literal("normalize"), vec: SymbolicVectorSchema }),
      z.object({ op: z.literal("grad"), val: SymbolicScalarSchema }),
    ]),
  ]),
);

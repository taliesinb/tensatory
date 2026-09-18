// zod schemas + builders for array specs.

import { z } from "zod";
import type { ArraySpec, SizedArraySpec } from "@tensatory/schema";
import { NotSupportedError, SpecError } from "../errors";
import { compileScalar, compileVector, emptyEnv, normalizeScalar, normalizeVector, pureContext } from "../symbolic";
import { SymbolicScalarSchema, SymbolicVectorSchema } from "../symbolic/spec";
import { NdArray } from "./ndarray";
import { RandomArraySchema, buildRandomArray } from "./random";

const axisSize = z.number().int().nonnegative();
const axisPos = z.number().int();
const shape = z.array(axisSize);
const consts = z.record(z.string(), z.number()).optional();

export const InlineArraySchema = z.object({ type: z.literal("inline"), shape, data: z.array(z.number()) });
export const ConstantArraySchema = z.object({ type: z.literal("constant"), shape, value: z.number() });
export const OneHotArraySchema = z.object({
  type: z.literal("oneHot"),
  shape: z.union([z.tuple([axisSize]), z.tuple([axisSize, axisSize])]),
  pos: z.union([axisPos, z.array(axisPos)]),
  hot: z.number().optional(),
  cold: z.number().optional(),
});
export const ManyHotArraySchema = z.object({
  type: z.literal("manyHot"),
  shape: z.union([z.tuple([axisSize]), z.tuple([axisSize, axisSize])]),
  pos: z.union([z.array(axisPos), z.array(z.array(axisPos))]),
  hot: z.number().optional(),
  cold: z.number().optional(),
});
export const SymbolicScalarArraySchema = z.object({
  type: z.literal("symbolic"),
  shape,
  origin: z.array(axisPos).optional(),
  expr: SymbolicScalarSchema,
  consts,
});
export const SymbolicVectorArraySchema = z.object({
  type: z.literal("symbolicv"),
  shape,
  origin: z.array(axisPos).optional(),
  expr: SymbolicVectorSchema,
  consts,
});
export const SizedArrayHandleSchema = z.object({
  type: z.literal("handle"),
  shape,
  path: z.string(),
  part: z.array(axisPos).optional(),
});
export const ArrayHandleSchema = z.object({
  type: z.literal("handle"),
  path: z.string(),
  part: z.array(axisPos).optional(),
});

export const SizedArraySchema: z.ZodType<SizedArraySpec> = z.discriminatedUnion("type", [
  InlineArraySchema,
  ConstantArraySchema,
  OneHotArraySchema,
  ManyHotArraySchema,
  SymbolicScalarArraySchema,
  SymbolicVectorArraySchema,
  RandomArraySchema,
  SizedArrayHandleSchema,
]);

export const ArraySchema: z.ZodType<ArraySpec> = z.union([z.string(), SizedArraySchema, ArrayHandleSchema]);

/*******************************************************/

const product = (xs: readonly number[]) => xs.reduce((a, b) => a * b, 1);

/** Materialize a sized array spec. External handles are not supported in phase 1. */
export function buildArray(spec: SizedArraySpec, path: string[] = []): NdArray {
  switch (spec.type) {
    case "inline": {
      const n = product(spec.shape);
      if (spec.data.length !== n) throw new SpecError(`inline data has ${spec.data.length} values but shape [${spec.shape}] needs ${n}`, path);
      return new NdArray(spec.shape, Float64Array.from(spec.data));
    }
    case "constant": {
      const arr = new NdArray(spec.shape);
      (arr.data as Float64Array).fill(spec.value);
      return arr;
    }
    case "oneHot":
    case "manyHot": {
      const hot = spec.hot ?? 1, cold = spec.cold ?? 0;
      const arr = new NdArray(spec.shape);
      (arr.data as Float64Array).fill(cold);
      // normalize to one list of hot positions per row
      let rows: number[][];
      if (spec.shape.length === 1) {
        const pos = spec.pos as number | number[];
        rows = [Array.isArray(pos) ? pos : [pos]];
        if (spec.type === "oneHot" && rows[0]!.length !== 1) throw new SpecError(`oneHot vector needs exactly one position`, path);
      } else {
        const pos = spec.pos as number[] | number[][];
        if (pos.length !== spec.shape[0]) throw new SpecError(`expected one pos entry per row (${spec.shape[0]}), got ${pos.length}`, path);
        rows = pos.map((p) => (Array.isArray(p) ? p : [p]));
      }
      const width = spec.shape[spec.shape.length - 1]!;
      rows.forEach((hots, r) => {
        for (const h of hots) {
          const i = h < 0 ? width + h : h;
          if (!Number.isInteger(i) || i < 0 || i >= width) throw new SpecError(`hot position ${h} out of range for width ${width}`, path);
          arr.data[(spec.shape.length === 1 ? 0 : r * width) + i] = hot;
        }
      });
      return arr;
    }
    case "symbolic":
    case "symbolicv": {
      const isVec = spec.type === "symbolicv";
      const cellShape = isVec ? spec.shape.slice(0, -1) : spec.shape;
      const D = cellShape.length;
      if (isVec && spec.shape[spec.shape.length - 1] !== D)
        throw new SpecError(`symbolicv last axis must have size ${D} (the number of other axes), got ${spec.shape[spec.shape.length - 1]}`, path);
      const origin = spec.origin ?? new Array<number>(D).fill(0);
      if (origin.length !== D) throw new SpecError(`origin has ${origin.length} entries, expected ${D}`, path);
      const env = emptyEnv(D, spec.consts ?? {});
      const arr = new NdArray(spec.shape);
      const cells = product(cellShape);
      const idx = new Float64Array(D);
      const cellStrides = new Array<number>(D);
      for (let d = D - 1, acc = 1; d >= 0; d--) { cellStrides[d] = acc; acc *= cellShape[d]!; }
      const fillIdx = (c: number) => {
        let rem = c;
        for (let d = 0; d < D; d++) { const i = Math.floor(rem / cellStrides[d]!); rem -= i * cellStrides[d]!; idx[d] = i - origin[d]!; }
      };
      if (!isVec) {
        const f = compileScalar(normalizeScalar(spec.expr, env, [...path, "expr"]), pureContext(D));
        for (let c = 0; c < cells; c++) { fillIdx(c); arr.data[c] = f(idx, c); }
      } else {
        const f = compileVector(normalizeVector(spec.expr, env, [...path, "expr"]), pureContext(D));
        const out = new Float64Array(D);
        for (let c = 0; c < cells; c++) { fillIdx(c); f(idx, c, out); arr.data.set(out, c * D); }
      }
      return arr;
    }
    case "random":
      return buildRandomArray(spec, path);
    case "handle":
      throw new NotSupportedError(`external array handles ("${spec.path}") are not supported yet`, path);
  }
}

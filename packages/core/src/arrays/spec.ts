// zod schemas + builders for array specs.

import { z } from "zod";
import type { ArrayAxes, ArrayHandleSpec, ArrayPart, ArraySpec, Dtype, SizedArrayHandleSpec, SizedArraySpec } from "@tensatory/schema";
import { SpecError } from "../errors";
import { compileScalar, compileVector, emptyEnv, normalizeScalar, normalizeVector, pureContext } from "../symbolic";
import { SymbolicScalarSchema, SymbolicVectorSchema } from "../symbolic/spec";
import type { ArrayHint, RawArray } from "./load";
import { NdArray, sameType } from "./ndarray";
import { RandomArraySchema, buildRandomArray } from "./random";

const axisSize = z.number().int().nonnegative();
const axisPos = z.number().int();
const shape = z.array(axisSize);
const consts = z.record(z.string(), z.number()).optional();
const part: z.ZodType<ArrayPart> = z.array(z.union([axisPos, z.null()]));
export const DTYPES = ["float32", "float64", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64", "bool"] as const satisfies readonly Dtype[];
export const DtypeSchema: z.ZodType<Dtype> = z.enum(DTYPES);

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
  path: z.string().min(1),
  part: part.optional(),
  axes: z.array(axisSize).optional(),
  dtype: DtypeSchema.optional(),
});
export const ArrayHandleSchema = z.object({
  type: z.literal("handle"),
  path: z.string().min(1),
  part: part.optional(),
  axes: z.array(axisSize).optional(),
  dtype: DtypeSchema.optional(),
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
/* external arrays: the environment loads them up front (see load.ts / Bundle.load), the build looks them up synchronously */

/** resolves the stored arrays a bundle refers to by path (already decoded, before `part`) */
export interface ArrayResolver {
  raw(path: string, at: string[]): RawArray;
}

/** the resolver of a bundle built without sidecars: every handle fails with a clear message */
export const noArrays: ArrayResolver = {
  raw: (path, at) => { throw new SpecError(`external array "${path}" was not loaded (the bundle was built without its sidecar files)`, at); },
};

/** a resolver over decoded arrays by path */
export function mapResolver(arrays: ReadonlyMap<string, RawArray>): ArrayResolver {
  return {
    raw: (path, at) => {
      const r = arrays.get(path);
      if (!r) throw new SpecError(`external array "${path}" was not loaded`, at);
      return r;
    },
  };
}

/** the hint a handle gives the loader */
export const handleHint = (spec: SizedArrayHandleSpec | ArrayHandleSpec): ArrayHint => ({
  ...("shape" in spec ? { shape: spec.shape } : {}),
  ...(spec.dtype !== undefined ? { dtype: spec.dtype } : {}),
});

/** the permutation `axes` checked against a rank (identity when absent) */
function permutation(axes: ArrayAxes | undefined, rank: number, path: string[]): number[] {
  if (!axes) return Array.from({ length: rank }, (_, i) => i);
  if (axes.length !== rank || new Set(axes).size !== rank || axes.some((a) => !Number.isInteger(a) || a < 0 || a >= rank))
    throw new SpecError(`axes [${axes}] is not a permutation of the ${rank} kept axes`, path);
  return [...axes];
}

/** the shape that remains after `part` fixes some axes of an array of `stored` shape, permuted by `axes` */
export function partShape(stored: readonly number[], part: ArrayPart | undefined, axes?: ArrayAxes, path: string[] = []): number[] {
  const p = part ?? [];
  if (p.length > stored.length) throw new SpecError(`part [${p}] has more entries than the array has axes (${stored.length})`, path);
  const kept = stored.filter((_, d) => p[d] === undefined || p[d] === null);
  return permutation(axes, kept.length, path).map((a) => kept[a]!);
}

/** the shape of a sized array spec's result, without materializing it */
export function sizedShape(spec: SizedArraySpec, path: string[] = []): number[] {
  return spec.type === "handle" ? partShape(spec.shape, spec.part, spec.axes, path) : [...spec.shape];
}

/** apply a `part` (and an axis permutation) to a decoded array: a strided gather keeping the `null` axes (a copy) */
export function applyPart(raw: RawArray, part: ArrayPart | undefined, axes?: ArrayAxes, path: string[] = []): NdArray {
  const src = new NdArray(raw.shape, raw.data);
  const p = part ?? [];
  const outShape = partShape(raw.shape, p, axes, path);
  const perm = permutation(axes, outShape.length, path);
  const identity = perm.every((a, i) => a === i);
  if (identity && (!p.length || p.every((x) => x === null))) return src;
  const kept: number[] = [];
  let base = 0;
  for (let d = 0; d < src.ndim; d++) {
    const x = p[d];
    if (x === undefined || x === null) kept.push(d);
    else {
      try { base += src.axisPos(d, x) * src.strides[d]!; }
      catch (e) { throw new SpecError(`part [${p}]: ${(e as Error).message}`, path); }
    }
  }
  // the leading-fixed, unpermuted case is contiguous
  if (identity && kept.every((d, i) => d === src.ndim - kept.length + i)) {
    const len = outShape.reduce((a, b) => a * b, 1);
    return new NdArray(outShape, src.data.slice(base, base + len));
  }
  // output axis i walks source axis kept[perm[i]]; the gather keeps the element type
  const srcStride = perm.map((a) => src.strides[kept[a]!]!);
  const out = new NdArray(outShape, sameType(src.data, outShape.reduce((a, b) => a * b, 1)));
  const idx = new Array<number>(outShape.length).fill(0);
  for (let i = 0; i < out.size; i++) {
    let o = base;
    for (let k = 0; k < idx.length; k++) o += idx[k]! * srcStride[k]!;
    out.data[i] = src.data[o]!;
    for (let k = idx.length - 1; k >= 0; k--) { idx[k]!++; if (idx[k]! < outShape[k]!) break; idx[k] = 0; }
  }
  return out;
}

/*******************************************************/

const product = (xs: readonly number[]) => xs.reduce((a, b) => a * b, 1);

/** Materialize a sized array spec; `handle` specs look their stored array up in `arrays`. */
export function buildArray(spec: SizedArraySpec, path: string[] = [], arrays: ArrayResolver = noArrays): NdArray {
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
      return buildHandle(spec, path, arrays);
  }
}

function buildHandle(spec: SizedArrayHandleSpec | ArrayHandleSpec, path: string[], arrays: ArrayResolver): NdArray {
  const raw = arrays.raw(spec.path, path);
  if ("shape" in spec && (spec.shape.length !== raw.shape.length || spec.shape.some((s, i) => s !== raw.shape[i])))
    throw new SpecError(`array "${spec.path}" has shape [${raw.shape}] but the spec declares [${spec.shape}]`, path);
  if (spec.dtype !== undefined && spec.dtype !== raw.dtype) throw new SpecError(`array "${spec.path}" is ${raw.dtype} but the spec declares ${spec.dtype}`, path);
  return applyPart(raw, spec.part, spec.axes, path);
}

/** Materialize any array spec: sized specs as `buildArray`, a bare path or an unsized handle by loading its shape. */
export function buildAnyArray(spec: ArraySpec, path: string[] = [], arrays: ArrayResolver = noArrays): NdArray {
  if (typeof spec === "string") return buildHandle({ type: "handle", path: spec }, path, arrays);
  if (!("shape" in spec)) return buildHandle(spec, path, arrays);
  return buildArray(spec, path, arrays);
}

/** the shape of any array spec's result: declared, or from the loaded stored array */
export function anyShape(spec: ArraySpec, path: string[] = [], arrays: ArrayResolver = noArrays): number[] {
  if (typeof spec === "string") return [...arrays.raw(spec, path).shape];
  if (!("shape" in spec)) return partShape(arrays.raw(spec.path, path).shape, spec.part, spec.axes, path);
  return sizedShape(spec, path);
}

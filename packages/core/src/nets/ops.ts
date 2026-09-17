// Array operations for the CPU reference evaluator of nets.
//
// A value is an NdArray plus its DECLARED rank; any extra leading axes are the
// implicit batch (see notes/nets.md). Every op here acts on the declared axes
// and broadcasts the batch prefixes numpy-style, so `vmap` semantics come for
// free: an op never needs to know whether it is batched.
//
// This is a reference implementation: straightforward strided iteration,
// Float64, no blocking. It exists to define the semantics and to check the
// WebGPU evaluator against; it is not meant to be fast.

import type { ArrayReduceFn } from "@tensatory/schema";
import { NdArray, type ArrayData } from "../arrays/ndarray";
import { EvalError } from "../errors";

export interface Val {
  readonly arr: NdArray;
  /** declared (per-example) rank; arr.ndim - rank is the batch rank */
  readonly rank: number;
}

export const batchRank = (v: Val): number => v.arr.ndim - v.rank;
export const scalar = (x: number): Val => ({ arr: new NdArray([], Float64Array.of(x)), rank: 0 });
const product = (xs: readonly number[]) => xs.reduce((a, b) => a * b, 1);

/*******************************************************/
/* strided views and iteration */

interface View {
  shape: number[];
  strides: number[];
  data: ArrayData;
  base: number;
}

const viewOf = (a: NdArray): View => ({ shape: [...a.shape], strides: [...a.strides], data: a.data, base: 0 });

/** view of v with its batch padded to B axes and its declared part padded to R axes (size-1 axes, stride 0) */
function aligned(v: Val, B: number, R: number): View {
  const b = batchRank(v);
  const shape: number[] = [], strides: number[] = [];
  for (let i = 0; i < B - b; i++) { shape.push(1); strides.push(0); }
  for (let i = 0; i < b; i++) { shape.push(v.arr.shape[i]!); strides.push(v.arr.strides[i]!); }
  for (let i = 0; i < R - v.rank; i++) { shape.push(1); strides.push(0); }
  for (let i = b; i < v.arr.ndim; i++) { shape.push(v.arr.shape[i]!); strides.push(v.arr.strides[i]!); }
  return { shape, strides, data: v.arr.data, base: 0 };
}

/** broadcast views of equal rank to a common shape (strides 0 where stretched); returns that shape */
function broadcast(views: View[], what: string): number[] {
  const n = views[0]!.shape.length;
  const out = new Array<number>(n).fill(1);
  for (const v of views) for (let d = 0; d < n; d++) {
    const s = v.shape[d]!;
    if (s === out[d] || s === 1) continue;
    if (out[d] === 1) out[d] = s;
    else throw new EvalError(`${what}: shapes do not broadcast (${views.map((w) => `[${w.shape}]`).join(" vs ")})`);
  }
  for (const v of views) for (let d = 0; d < n; d++) if (v.shape[d] !== out[d]) { v.shape[d] = out[d]!; v.strides[d] = 0; }
  return out;
}

/** visit every multi-index of `shape` in row-major order with each view's offset */
function forEach(shape: readonly number[], views: readonly View[], fn: (offs: number[], i: number) => void): void {
  const n = shape.length, total = product(shape), K = views.length;
  const idx = new Array<number>(n).fill(0);
  const offs = views.map((v) => v.base);
  for (let i = 0; i < total; i++) {
    fn(offs, i);
    for (let d = n - 1; d >= 0; d--) {
      idx[d]!++;
      for (let k = 0; k < K; k++) offs[k]! += views[k]!.strides[d]!;
      if (idx[d]! < shape[d]!) break;
      for (let k = 0; k < K; k++) offs[k]! -= views[k]!.strides[d]! * shape[d]!;
      idx[d] = 0;
    }
  }
}

/**
 * Like forEach, but calls `fn` once per RUN along the last axis with the
 * views' offsets at the start of the run; `fn` loops over the `n` elements
 * itself using each view's last stride. This is where the hot ops spend
 * their time, so it avoids a call per element.
 */
function forEachRow(shape: readonly number[], views: readonly View[], fn: (offs: number[], n: number, i0: number) => void): void {
  const nd = shape.length;
  if (nd === 0) { fn(views.map((v) => v.base), 1, 0); return; }
  const n = shape[nd - 1]!;
  const outer = shape.slice(0, -1);
  const outerViews = views.map((v) => ({ ...v, shape: v.shape.slice(0, -1), strides: v.strides.slice(0, -1) }));
  forEach(outer, outerViews, (o, i) => fn(o, n, i * n));
}

/** copy a view into a fresh contiguous array */
function materialize(view: View): NdArray {
  const out = new NdArray(view.shape);
  const d = out.data, src = view.data, s = view.strides[view.strides.length - 1] ?? 0;
  forEachRow(view.shape, [view], (o, n, i0) => {
    let p = o[0]!;
    for (let j = 0; j < n; j++, p += s) d[i0 + j] = src[p]!;
  });
  return out;
}

/*******************************************************/
/* elementwise */

/** apply `fn` elementwise with broadcasting over batch and declared axes */
export function mapN(vals: Val[], fn: (xs: number[]) => number, what = "elementwise"): Val {
  const B = Math.max(...vals.map(batchRank)), R = Math.max(...vals.map((v) => v.rank));
  const views = vals.map((v) => aligned(v, B, R));
  const shape = broadcast(views, what);
  const out = new NdArray(shape);
  const d = out.data, K = vals.length, xs = new Array<number>(K);
  const last = views.map((v) => v.strides[v.strides.length - 1] ?? 0);
  const offs = new Array<number>(K);
  forEachRow(shape, views, (o, n, i0) => {
    for (let k = 0; k < K; k++) offs[k] = o[k]!;
    for (let j = 0; j < n; j++) {
      for (let k = 0; k < K; k++) { xs[k] = views[k]!.data[offs[k]!]!; offs[k]! += last[k]!; }
      d[i0 + j] = fn(xs);
    }
  });
  return { arr: out, rank: R };
}

export const map1 = (v: Val, fn: (x: number) => number): Val => {
  const out = new NdArray(v.arr.shape);
  const src = v.arr.data, d = out.data;
  for (let i = 0; i < d.length; i++) d[i] = fn(src[i]!);
  return { arr: out, rank: v.rank };
};

export function map2(a: Val, b: Val, fn: (x: number, y: number) => number, what = "elementwise"): Val {
  const B = Math.max(batchRank(a), batchRank(b)), R = Math.max(a.rank, b.rank);
  const va = aligned(a, B, R), vb = aligned(b, B, R);
  const shape = broadcast([va, vb], what);
  const out = new NdArray(shape);
  const d = out.data, da = va.data, db = vb.data;
  const sa = va.strides[va.strides.length - 1] ?? 0, sb = vb.strides[vb.strides.length - 1] ?? 0;
  forEachRow(shape, [va, vb], (o, n, i0) => {
    let pa = o[0]!, pb = o[1]!;
    for (let j = 0; j < n; j++, pa += sa, pb += sb) d[i0 + j] = fn(da[pa]!, db[pb]!);
  });
  return { arr: out, rank: R };
}

/*******************************************************/
/* contractions */

const BATCH_LETTERS = Array.from({ length: 16 }, (_, i) => String.fromCharCode(0x2460 + i)); // private letters for batch axes

/**
 * Generalized einsum over DECLARED axes; batch prefixes are prepended
 * automatically and broadcast. A letter of size 1 in one operand broadcasts
 * against the same letter's size elsewhere.
 */
export function einsum(terms: string[], out: string, vals: Val[]): Val {
  const B = Math.max(...vals.map(batchRank));
  if (B > BATCH_LETTERS.length) throw new EvalError(`einsum: batch rank ${B} exceeds ${BATCH_LETTERS.length}`);
  const batch = BATCH_LETTERS.slice(0, B).join("");
  const full = vals.map((v, i) => {
    const t = [...terms[i]!];
    if (t.length !== v.rank) throw new EvalError(`einsum: term "${terms[i]}" has ${t.length} letters, operand has declared rank ${v.rank}`);
    return [...batch.slice(B - batchRank(v)), ...t];
  });
  const outLetters = [...batch, ...out];
  // sizes: max over operands; 1 broadcasts
  const sizes = new Map<string, number>();
  full.forEach((letters, k) => letters.forEach((l, d) => {
    const s = vals[k]!.arr.shape[d]!;
    const prev = sizes.get(l);
    if (prev === undefined || prev === 1) sizes.set(l, s);
    else if (s !== 1 && s !== prev) throw new EvalError(`einsum: letter "${l}" has sizes ${prev} and ${s}`);
  }));
  const summed = [...new Set(full.flat())].filter((l) => !outLetters.includes(l));
  // iteration order: output letters, then summed letters, with the LARGEST letter moved innermost; a letter repeated
  // in the output is iterated once (its strides add up: only the diagonal is written, the rest stays zero)
  let all = [...new Set([...outLetters, ...summed])];
  const biggest = all.reduce((b, l) => ((sizes.get(l) ?? 1) > (sizes.get(b) ?? 1) ? l : b), all[0] ?? "");
  all = [...all.filter((l) => l !== biggest), ...(all.length ? [biggest] : [])];
  const shape = all.map((l) => sizes.get(l) ?? 1);
  const outArr = new NdArray(outLetters.map((l) => sizes.get(l)!));
  // one view per operand over `all`; strides sum over repeated letters, 0 for absent / broadcast
  const strideOver = (letters: string[], shapeOf: readonly number[], stridesOf: readonly number[]) =>
    all.map((l) => {
      let s = 0;
      letters.forEach((m, d) => { if (m === l && shapeOf[d] !== 1) s += stridesOf[d]!; });
      return s;
    });
  const views: View[] = full.map((letters, k) => ({ shape: [...shape], strides: strideOver(letters, vals[k]!.arr.shape, vals[k]!.arr.strides), data: vals[k]!.arr.data, base: 0 }));
  const outView: View = { shape: [...shape], strides: strideOver(outLetters, outArr.shape, outArr.strides), data: outArr.data, base: 0 };
  const d = outArr.data, K = vals.length;
  const last = [outView, ...views].map((v) => v.strides[v.strides.length - 1] ?? 0);
  if (K === 2) {
    const [da, db] = [views[0]!.data, views[1]!.data], [so, sa, sb] = [last[0]!, last[1]!, last[2]!];
    forEachRow(shape, [outView, ...views], (o, n) => {
      let po = o[0]!, pa = o[1]!, pb = o[2]!;
      if (so === 0) { let acc = 0; for (let j = 0; j < n; j++, pa += sa, pb += sb) acc += da[pa]! * db[pb]!; d[po]! += acc; }
      else for (let j = 0; j < n; j++, po += so, pa += sa, pb += sb) d[po]! += da[pa]! * db[pb]!;
    });
  } else {
    const offs = new Array<number>(K + 1);
    forEachRow(shape, [outView, ...views], (o, n) => {
      for (let k = 0; k <= K; k++) offs[k] = o[k]!;
      for (let j = 0; j < n; j++) {
        let p = 1;
        for (let k = 1; k <= K; k++) { p *= views[k - 1]!.data[offs[k]!]!; offs[k]! += last[k]!; }
        d[offs[0]!]! += p;
        offs[0]! += last[0]!;
      }
    });
  }
  return { arr: outArr, rank: out.length };
}

/** numpy matmul on the declared axes */
export function matmul(a: Val, b: Val): Val {
  if (a.rank === 0 || b.rank === 0) throw new EvalError("matmul operands must have rank >= 1");
  if (a.rank === 1 && b.rank === 1) return einsum(["k", "k"], "", [a, b]);
  const L = Math.max(a.rank, b.rank) - 2;
  const lead = Array.from({ length: Math.max(L, 0) }, (_, i) => String.fromCharCode(0x41 + i)); // A, B, C ...
  const la = lead.slice(lead.length - Math.max(a.rank - 2, 0)).join(""), lb = lead.slice(lead.length - Math.max(b.rank - 2, 0)).join("");
  if (b.rank === 1) return einsum([la + "ik", "k"], lead.join("") + "i", [a, b]);
  if (a.rank === 1) return einsum(["k", lb + "kj"], lead.join("") + "j", [a, b]);
  return einsum([la + "ik", lb + "kj"], lead.join("") + "ij", [a, b]);
}

/*******************************************************/
/* reductions */

/** reduce over declared axes (already normalized, unique) */
export function reduce(v: Val, fn: ArrayReduceFn, axes: number[], keepDims: boolean): Val {
  const b = batchRank(v);
  const full = new Set(axes.map((a) => a + b));
  const inShape = [...v.arr.shape];
  const outShape = inShape.map((s, d) => (full.has(d) ? 1 : s));
  const count = axes.reduce((n, a) => n * inShape[a + b]!, 1);
  const outArr = new NdArray(outShape);
  const outView: View = { shape: [...inShape], strides: outArr.strides.map((s, d) => (full.has(d) ? 0 : s)), data: outArr.data, base: 0 };
  const inView = viewOf(v.arr);
  const src = v.arr.data, d = outArr.data;
  const init = fn === "prod" ? 1 : fn === "max" || fn === "logsumexp" ? -Infinity : fn === "min" ? Infinity : 0;
  d.fill(init);
  const step =
    fn === "sum" || fn === "mean" ? (acc: number, x: number) => acc + x
    : fn === "prod" ? (acc: number, x: number) => acc * x
    : fn === "min" ? Math.min : Math.max; // max, logsumexp (first pass)
  const si = inView.strides[inView.strides.length - 1] ?? 0, so = outView.strides[outView.strides.length - 1] ?? 0;
  forEachRow(inShape, [inView, outView], (o, n) => {
    let pi = o[0]!, po = o[1]!;
    for (let j = 0; j < n; j++, pi += si, po += so) d[po] = step(d[po]!, src[pi]!);
  });
  if (fn === "mean") for (let i = 0; i < d.length; i++) d[i]! /= count;
  if (fn === "logsumexp") {
    const sums = new Float64Array(d.length);
    forEachRow(inShape, [inView, outView], (o, n) => {
      let pi = o[0]!, po = o[1]!;
      for (let j = 0; j < n; j++, pi += si, po += so) sums[po]! += Math.exp(src[pi]! - d[po]!);
    });
    for (let i = 0; i < d.length; i++) d[i] = d[i]! + Math.log(sums[i]!);
  }
  const finalShape = keepDims ? outShape : inShape.filter((_, i) => !full.has(i));
  return { arr: new NdArray(finalShape, outArr.data), rank: keepDims ? v.rank : v.rank - axes.length };
}

/** index of the max / min along a declared axis (first occurrence) */
export function argReduce(v: Val, axis: number, max: boolean): Val {
  const b = batchRank(v), ax = axis + b;
  const inShape = [...v.arr.shape];
  const outShape = inShape.filter((_, d) => d !== ax);
  const outArr = new NdArray(outShape);
  const best = new Float64Array(outArr.size).fill(max ? -Infinity : Infinity);
  const outStrides = [...outArr.strides]; outStrides.splice(ax, 0, 0);
  const outView: View = { shape: [...inShape], strides: outStrides, data: outArr.data, base: 0 };
  const idxView: View = { shape: [...inShape], strides: inShape.map((_, d) => (d === ax ? 1 : 0)), data: outArr.data, base: 0 };
  const src = v.arr.data, d = outArr.data;
  forEach(inShape, [viewOf(v.arr), outView, idxView], (o) => {
    const x = src[o[0]!]!, j = o[1]!;
    if (max ? x > best[j]! : x < best[j]!) { best[j] = x; d[j] = o[2]!; }
  });
  return { arr: outArr, rank: v.rank - 1 };
}

export function softmax(v: Val, axis: number, log: boolean): Val {
  const m = reduce(v, "max", [axis], true);
  const shifted = map2(v, m, (x, mx) => x - mx);
  const e = map1(shifted, Math.exp);
  const s = reduce(e, "sum", [axis], true);
  return log ? map2(shifted, s, (x, sum) => x - Math.log(sum)) : map2(e, s, (x, sum) => x / sum);
}

/*******************************************************/
/* shape ops */

/** reshape the declared part to `target` (fully resolved); the batch prefix is kept */
export function reshape(v: Val, target: number[]): Val {
  const b = batchRank(v);
  const shape = [...v.arr.shape.slice(0, b), ...target];
  if (product(shape) !== v.arr.size) throw new EvalError(`cannot reshape [${v.arr.shape}] to [${shape}]`);
  return { arr: new NdArray(shape, v.arr.data), rank: target.length };
}

/** permute the declared axes */
export function transpose(v: Val, perm: number[]): Val {
  const b = batchRank(v);
  const full = [...Array.from({ length: b }, (_, i) => i), ...perm.map((p) => p + b)];
  const view: View = { shape: full.map((p) => v.arr.shape[p]!), strides: full.map((p) => v.arr.strides[p]!), data: v.arr.data, base: 0 };
  return { arr: materialize(view), rank: v.rank };
}

export function concat(vals: Val[], axis: number): Val {
  const B = Math.max(...vals.map(batchRank)), R = vals[0]!.rank, ax = B + axis;
  const views = vals.map((v) => aligned(v, B, R));
  // broadcast every axis but `ax`
  const probe = views.map((v) => ({ ...v, shape: v.shape.map((s, d) => (d === ax ? 1 : s)), strides: [...v.strides] }));
  const shape = broadcast(probe, "concat");
  const total = views.reduce((n, v) => n + v.shape[ax]!, 0);
  shape[ax] = total;
  const out = new NdArray(shape);
  let at = 0;
  views.forEach((v, k) => {
    const part = [...probe[k]!.shape]; part[ax] = v.shape[ax]!;
    const src: View = { shape: part, strides: probe[k]!.strides.map((s, d) => (d === ax ? v.strides[d]! : s)), data: v.data, base: 0 };
    const dst: View = { shape: part, strides: [...out.strides], data: out.data, base: at * out.strides[ax]! };
    forEach(part, [src, dst], (o) => { out.data[o[1]!] = src.data[o[0]!]!; });
    at += v.shape[ax]!;
  });
  return { arr: out, rank: R };
}

/** python slice along a declared axis */
export function slice(v: Val, axis: number, start: number | undefined, stop: number | undefined, step = 1): Val {
  const b = batchRank(v), ax = axis + b, n = v.arr.shape[ax]!;
  const clampIdx = (i: number | undefined, dflt: number, lo: number, hi: number) => (i === undefined ? dflt : Math.min(hi, Math.max(lo, i < 0 ? n + i : i)));
  const a = step > 0 ? clampIdx(start, 0, 0, n) : clampIdx(start, n - 1, -1, n - 1);
  const e = step > 0 ? clampIdx(stop, n, 0, n) : clampIdx(stop, -1, -1, n - 1);
  const len = Math.max(0, Math.ceil((e - a) / step));
  const shape = [...v.arr.shape]; shape[ax] = len;
  const strides = [...v.arr.strides]; strides[ax] = v.arr.strides[ax]! * step;
  return { arr: materialize({ shape, strides, data: v.arr.data, base: len ? a * v.arr.strides[ax]! : 0 }), rank: v.rank };
}

export function oneHot(v: Val, size: number): Val {
  const out = new NdArray([...v.arr.shape, size]);
  const src = v.arr.data, d = out.data;
  for (let i = 0; i < src.length; i++) {
    const j = Math.round(src[i]!);
    if (j >= 0 && j < size) d[i * size + j] = 1;
  }
  return { arr: out, rank: v.rank + 1 };
}

/** out[.., i, ..] = val[.., round(indices[.., i, ..]), ..] along a declared axis (torch.gather) */
export function takeAlong(v: Val, idx: Val, axis: number): Val {
  const B = Math.max(batchRank(v), batchRank(idx)), R = v.rank, ax = B + axis;
  const vv = aligned(v, B, R), iv = aligned(idx, B, R);
  const axisStride = vv.shape[ax] === 1 ? 0 : vv.strides[ax]!, axisSize = vv.shape[ax]!;
  // broadcast every axis but `ax`; along `ax` the output follows the indices
  const vProbe: View = { ...vv, shape: vv.shape.map((s, d) => (d === ax ? 1 : s)), strides: vv.strides.map((s, d) => (d === ax ? 0 : s)) };
  const iProbe: View = { ...iv, shape: iv.shape.map((s, d) => (d === ax ? 1 : s)), strides: [...iv.strides] };
  const shape = broadcast([vProbe, iProbe], "takeAlong");
  shape[ax] = iv.shape[ax]!;
  iProbe.shape[ax] = iv.shape[ax]!; iProbe.strides[ax] = iv.strides[ax]!;
  vProbe.shape[ax] = iv.shape[ax]!;
  const out = new NdArray(shape);
  forEach(shape, [vProbe, iProbe], (o, i) => {
    const j = Math.round(iv.data[o[1]!]!);
    if (j < 0 || j >= axisSize) throw new EvalError(`takeAlong: index ${j} out of range for axis of size ${axisSize}`);
    out.data[i] = vv.data[o[0]! + j * axisStride]!;
  });
  return { arr: out, rank: R };
}

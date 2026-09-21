// Slices: restrict the fields of an N-dimensional manifold to an axis-aligned
// k-dimensional subspace through the manifold's `origin`, as a SPEC REWRITE.
//
// `sliceSpec(spec, manifold, dims)` returns a new BundleSpec in which the
// manifold has `dims.length` dimensions and every field / point set on it is
// an ordinary k-D field / point set: the rest of the pipeline (zod, field data,
// the GPU transpiler, the viewer's 2D / 3D arms) never sees a mixed-dimension
// object. Coordinates along `dims` become the slice's coordinates (in the order
// of `dims`, ascending), every other coordinate is baked in as the origin's
// component.
//
// Semantics are those of the N-D field evaluated on the slice:
//   * scalar fields: f'(p) = f(embed(p));
//   * vector fields: the components along `dims` of v(embed(p)) (the projection
//     of the vector onto the slice);
//   * expressions are rewritten on the normalized AST. N-D vector-valued
//     subexpressions are UNROLLED into their N components (as k-D scalar
//     expressions), so |∇f|, dot products and cosine similarities keep their
//     full N-D meaning; `grad` is differentiated in N-D first (`diffScalar`),
//     then sliced. A derivative along a fixed dimension, or a fixed component
//     of a vector argument, needs the argument's N-D expression: field
//     arguments are then INLINED on demand (symbolic / pointwise / pullback
//     data, following ids); sampled or net arguments cannot be, and such a
//     field is dropped with a reason.
//   * dense data: the grid is sliced at the sample nearest to the origin in
//     each fixed dimension (an error when the origin is outside the box);
//   * scalar net fields: the point input becomes `E·p + o` (a selection
//     matrix and the origin, as inline arrays in the field's `arrays`), any
//     explicit `inputs` get their coordinate leaves substituted; vector net
//     fields are not sliced yet (dropped);
//   * point sets: points are projected onto the slice (their shadow);
//   * the manifold keeps its id (the viewer's per-space options stay valid),
//     gets `dims.length` dimensions, the sliced `dimNames`, no `origin`, and
//     its `flow` if that field survived.
//
// Like `adjustSpec` / `zoomBoxes` this is applied to the base spec before a
// Bundle is built, so a changed slice is a rebuild — nothing is pipelined.

import type {
  ArrayExpr,
  BundleSpec,
  FieldSpec,
  ManifoldDefinitionSpec,
  NetScalarFieldDataSpec,
  PointSetSpec,
  ScalarFieldDataSpec,
  SymbolicScalar,
  SymbolicVector,
  VectorFieldDataSpec,
} from "@tensatory/schema";
import { buildArray, noArrays, type ArrayResolver } from "../arrays/spec";
import { NdArray } from "../arrays/ndarray";
import { SpecError } from "../errors";
import { Box } from "../geometry/box";
import { DenseGrid } from "../geometry/grid";
import { add, C, mul, un, type SExpr, type VExpr } from "../symbolic/ast";
import { diffScalar } from "../symbolic/diff";
import { emptyEnv, normalizeScalar, normalizeVector, type NameEnv } from "../symbolic/normalize";
import { Bundle } from "./bundle";
import { domainOf } from "./zoom";

export interface SliceResult {
  spec: BundleSpec;
  /** fields (and point sets) of the manifold that could not be sliced, with the reason */
  dropped: Record<string, string>;
}

/** raised inside the rewrite when a field needs something the slice cannot provide; caught per field */
class NotSliceable extends Error {}

type ScalarData = ScalarFieldDataSpec;
type VectorData = VectorFieldDataSpec;

/*******************************************************/
/* AST -> spec */

/** ∂^n f / ∂x_{dims} of an argument as spec: nested comp(grad(·)) */
function argDerivativeSpec(base: SymbolicScalar, dims: number[]): SymbolicScalar {
  return dims.reduce<SymbolicScalar>((e, d) => ({ op: "comp", vec: { op: "grad", val: e }, index: d }), base);
}

export function astToScalarSpec(e: SExpr): SymbolicScalar {
  const s = astToScalarSpec, v = astToVectorSpec;
  switch (e.k) {
    case "const": return e.value;
    case "coord": return { op: "coord", index: e.index };
    case "arg": return { op: "arg", name: e.name };
    case "argvi": return { op: "argvi", name: e.name, index: e.index };
    case "argd": return argDerivativeSpec({ op: "arg", name: e.name }, e.dims);
    case "argvid": return argDerivativeSpec({ op: "argvi", name: e.name, index: e.index }, e.dims);
    case "un": return { op: e.op, val: s(e.a) };
    case "nary": return { op: e.op, vals: e.args.map(s) };
    case "bin": return { op: e.op, vals: [s(e.a), s(e.b)] };
    case "clamp": return { op: "clamp", val: s(e.a), min: s(e.lo), max: s(e.hi) };
    case "gaussKernel": case "normalPDF": return { op: e.k, val: s(e.a), mu: s(e.mu), sigma: s(e.sigma) };
    case "dot": case "cosineSim": return { op: e.k, vecs: [v(e.a), v(e.b)] };
    case "norm": return { op: "norm", vec: v(e.v) };
    case "comp": return { op: "comp", vec: v(e.v), index: e.index };
    case "where": {
      // lt / eq / gt by the sign σ of (test − ref): (σ² − σ)/2, 1 − σ², (σ² + σ)/2 select exactly one branch
      const sigma: SymbolicScalar = { op: "sign", val: { op: "sub", vals: [s(e.test), s(e.ref)] } };
      const sq: SymbolicScalar = { op: "square", val: sigma };
      return {
        op: "add",
        vals: [
          { op: "mul", vals: [0.5, { op: "sub", vals: [sq, sigma] }, s(e.lt)] },
          { op: "mul", vals: [{ op: "sub", vals: [1, sq] }, s(e.eq)] },
          { op: "mul", vals: [0.5, { op: "add", vals: [sq, sigma] }, s(e.gt)] },
        ],
      };
    }
  }
}

export function astToVectorSpec(e: VExpr): SymbolicVector {
  const s = astToScalarSpec, v = astToVectorSpec;
  switch (e.k) {
    case "constv": return { op: "constv", value: [...e.value] };
    case "basisv": return { op: "basisv", index: e.index };
    case "coordv": return { op: "coordv" };
    case "argv": return { op: "argv", name: e.name };
    case "scalev": return { op: "scalev", vec: v(e.v), by: s(e.s) };
    case "naryv": return { op: e.op, vecs: e.args.map(v) };
    case "subv": return { op: "subv", vecs: [v(e.a), v(e.b)] };
    case "sumv": return { op: "sumv", vecs: e.vecs.map(v), coeffs: e.coeffs.map(s) };
    case "compv": return { op: "compv", coeffs: e.comps.map(s) };
    case "normalize": return { op: "normalize", vec: v(e.v) };
    case "grad": return { op: "grad", val: s(e.s) };
  }
}

/*******************************************************/
/* N-D -> N-D: substitute the coordinates (pullbacks), substitute arguments (inlining) */

type CoordMap = (index: number) => SExpr;

function mapS(e: SExpr, coord: CoordMap, args: ArgSubst): SExpr {
  const s = (x: SExpr) => mapS(x, coord, args), v = (x: VExpr) => mapV(x, coord, args);
  switch (e.k) {
    case "const": return e;
    case "coord": return coord(e.index);
    case "arg": return args.scalar(e.name, []);
    case "argd": return args.scalar(e.name, e.dims);
    case "argvi": return args.vectorComp(e.name, e.index, []);
    case "argvid": return args.vectorComp(e.name, e.index, e.dims);
    case "un": return { ...e, a: s(e.a) };
    case "nary": return { ...e, args: e.args.map(s) };
    case "bin": return { ...e, a: s(e.a), b: s(e.b) };
    case "clamp": return { ...e, a: s(e.a), lo: s(e.lo), hi: s(e.hi) };
    case "gaussKernel": case "normalPDF": return { ...e, a: s(e.a), mu: s(e.mu), sigma: s(e.sigma) };
    case "dot": case "cosineSim": return { ...e, a: v(e.a), b: v(e.b) };
    case "norm": return { ...e, v: v(e.v) };
    case "comp": return { ...e, v: v(e.v) };
    case "where": return { ...e, test: s(e.test), ref: s(e.ref), lt: s(e.lt), eq: s(e.eq), gt: s(e.gt) };
  }
}

function mapV(e: VExpr, coord: CoordMap, args: ArgSubst): VExpr {
  const s = (x: SExpr) => mapS(x, coord, args), v = (x: VExpr) => mapV(x, coord, args);
  switch (e.k) {
    case "constv": case "basisv": return e;
    case "coordv": return { k: "compv", comps: Array.from({ length: args.dimCount }, (_, i) => coord(i)) };
    case "argv": return args.vector(e.name);
    case "scalev": return { ...e, v: v(e.v), s: s(e.s) };
    case "naryv": return { ...e, args: e.args.map(v) };
    case "subv": return { ...e, a: v(e.a), b: v(e.b) };
    case "sumv": return { ...e, vecs: e.vecs.map(v), coeffs: e.coeffs.map(s) };
    case "compv": return { ...e, comps: e.comps.map(s) };
    case "normalize": return { ...e, v: v(e.v) };
    case "grad": return { ...e, s: s(e.s) };
  }
}

/** how argument nodes are replaced during an N-D -> N-D rewrite */
interface ArgSubst {
  readonly dimCount: number;
  scalar(name: string, dims: number[]): SExpr;
  vector(name: string): VExpr;
  vectorComp(name: string, index: number, dims: number[]): SExpr;
}

const identityCoord: CoordMap = (index) => ({ k: "coord", index });
/** arguments left as they are */
const keepArgs = (dimCount: number): ArgSubst => ({
  dimCount,
  scalar: (name, dims) => (dims.length ? { k: "argd", name, dims } : { k: "arg", name }),
  vector: (name) => ({ k: "argv", name }),
  vectorComp: (name, index, dims) => (dims.length ? { k: "argvid", name, index, dims } : { k: "argvi", name, index }),
});

const diffN = (e: SExpr, dims: number[], D: number): SExpr => dims.reduce((acc, d) => diffScalar(acc, d, D), e);

/*******************************************************/
/* inlining: an N-D field datum as one argument-free N-D expression */

class Inliner {
  private readonly scalarMemo = new Map<string, SExpr>();
  private readonly vectorMemo = new Map<string, VExpr>();
  constructor(private readonly spec: BundleSpec, private readonly manifold: string, readonly D: number) {}

  private fieldSpec(id: string, kind: "scalar" | "vector"): FieldSpec {
    const f = this.spec.fields[id];
    if (!f) throw new NotSliceable(`unknown field "${id}"`);
    if (f.kind !== kind) throw new NotSliceable(`field "${id}" is a ${f.kind} field, expected ${kind}`);
    if (domainOf(this.spec, f.domain) !== this.manifold) throw new NotSliceable(`field "${id}" lives on another manifold`);
    return f;
  }

  /** the substitution that inlines a pointwise datum's arguments */
  private argsOf(scalars: Record<string, ScalarData | string> | undefined, vectors: Record<string, VectorData | string> | undefined): ArgSubst {
    const D = this.D;
    return {
      dimCount: D,
      scalar: (name, dims) => diffN(this.scalar(scalars![name]!), dims, D),
      vector: (name) => this.vector(vectors![name]!),
      vectorComp: (name, index, dims) => diffN({ k: "comp", v: this.vector(vectors![name]!), index }, dims, D),
    };
  }

  private pullbackCoord(d: { type: "translate"; vec: unknown } | { type: "scale"; origin?: unknown; scale: unknown }): CoordMap {
    const D = this.D;
    const vec = (x: unknown, what: string): number[] => {
      if (!Array.isArray(x) || x.length !== D || !x.every((c) => typeof c === "number")) throw new NotSliceable(`${what} of a pullback must be an inline vector of ${D} numbers`);
      return x as number[];
    };
    if (d.type === "translate") { const v = vec(d.vec, "vec"); return (i) => add({ k: "coord", index: i }, C(-v[i]!)); }
    const o = d.origin === undefined ? new Array<number>(D).fill(0) : vec(d.origin, "origin");
    const f = typeof d.scale === "number" ? new Array<number>(D).fill(d.scale) : vec(d.scale, "scale");
    return (i) => add(C(o[i]!), mul(C(1 / f[i]!), add({ k: "coord", index: i }, C(-o[i]!))));
  }

  scalar(ref: ScalarData | string): SExpr {
    if (typeof ref === "string") {
      const memo = this.scalarMemo.get(ref); if (memo) return memo;
      const out = this.scalar(this.fieldSpec(ref, "scalar").data as ScalarData);
      this.scalarMemo.set(ref, out); return out;
    }
    switch (ref.type) {
      case "symbolic": return normalizeScalar(ref.expr, emptyEnv(this.D, ref.consts));
      case "pointwise": {
        const env = pointwiseEnv(this.D, ref.consts, ref.scalars, ref.vectors);
        return mapS(normalizeScalar(ref.expr, env), identityCoord, this.argsOf(ref.scalars, ref.vectors));
      }
      case "translate": case "scale": return mapS(this.scalar(ref.arg), this.pullbackCoord(ref), keepArgs(this.D));
      default: throw new NotSliceable(`a ${ref.type} field cannot be evaluated off the slice (derivative along, or component of, a fixed dimension)`);
    }
  }

  vector(ref: VectorData | string): VExpr {
    if (typeof ref === "string") {
      const memo = this.vectorMemo.get(ref); if (memo) return memo;
      const out = this.vector(this.fieldSpec(ref, "vector").data as VectorData);
      this.vectorMemo.set(ref, out); return out;
    }
    switch (ref.type) {
      case "symbolicv": return normalizeVector(ref.expr, emptyEnv(this.D, ref.consts));
      case "pointwisev": {
        const env = pointwiseEnv(this.D, ref.consts, ref.scalars, ref.vectors);
        return mapV(normalizeVector(ref.expr, env), identityCoord, this.argsOf(ref.scalars, ref.vectors));
      }
      case "translate": case "scale": return mapV(this.vector(ref.arg), this.pullbackCoord(ref), keepArgs(this.D));
      default: throw new NotSliceable(`a ${ref.type} field cannot be evaluated off the slice (component along a fixed dimension)`);
    }
  }
}

function pointwiseEnv(D: number, consts: Record<string, number> | undefined, scalars: Record<string, unknown> | undefined, vectors: Record<string, unknown> | undefined): NameEnv {
  return { dimCount: D, consts: consts ?? {}, scalarArgs: new Set(Object.keys(scalars ?? {})), vectorArgs: new Set(Object.keys(vectors ?? {})) };
}

/*******************************************************/
/* N-D -> k-D: the slice itself */

class Slicer {
  readonly k: number;
  /** position of an N-D dimension in the slice, or -1 when fixed */
  private readonly pos: number[];
  constructor(readonly D: number, readonly dims: readonly number[], readonly origin: readonly number[], private readonly inliner: Inliner) {
    this.k = dims.length;
    this.pos = Array.from({ length: D }, (_, i) => dims.indexOf(i));
  }

  /** scalar arguments the current expression may keep referring to (along the slice), by name */
  private scalars: Record<string, ScalarData | string> = {};
  private vectors: Record<string, VectorData | string> = {};
  /** names of arguments the sliced expression still refers to */
  readonly used = { scalars: new Set<string>(), vectors: new Set<string>() };

  withArgs(scalars: Record<string, ScalarData | string> | undefined, vectors: Record<string, VectorData | string> | undefined): this {
    this.scalars = scalars ?? {}; this.vectors = vectors ?? {};
    this.used.scalars.clear(); this.used.vectors.clear();
    return this;
  }

  private mapDims(dims: number[]): number[] | undefined {
    const out = dims.map((d) => this.pos[d]!);
    return out.every((p) => p >= 0) ? out : undefined;
  }

  /** k-D expression of an N-D scalar expression on the slice */
  scalar(e: SExpr): SExpr {
    const s = (x: SExpr) => this.scalar(x);
    switch (e.k) {
      case "const": return e;
      case "coord": { const p = this.pos[e.index]!; return p >= 0 ? { k: "coord", index: p } : C(this.origin[e.index]!); }
      case "arg": this.used.scalars.add(e.name); return e;
      case "argd": {
        const dims = this.mapDims(e.dims);
        if (dims) { this.used.scalars.add(e.name); return { k: "argd", name: e.name, dims }; }
        // a derivative along a fixed dimension: differentiate the argument's own N-D expression, then slice
        return s(diffN(this.inliner.scalar(this.scalars[e.name]!), e.dims, this.D));
      }
      case "argvi": {
        const p = this.pos[e.index]!;
        if (p >= 0) { this.used.vectors.add(e.name); return { k: "argvi", name: e.name, index: p }; }
        return s({ k: "comp", v: this.inliner.vector(this.vectors[e.name]!), index: e.index });
      }
      case "argvid": {
        const p = this.pos[e.index]!, dims = this.mapDims(e.dims);
        if (p >= 0 && dims) { this.used.vectors.add(e.name); return { k: "argvid", name: e.name, index: p, dims }; }
        return s(diffN({ k: "comp", v: this.inliner.vector(this.vectors[e.name]!), index: e.index }, e.dims, this.D));
      }
      case "un": return un(e.op, s(e.a));
      case "nary": return { k: "nary", op: e.op, args: e.args.map(s) };
      case "bin": return { k: "bin", op: e.op, a: s(e.a), b: s(e.b) };
      case "clamp": return { k: "clamp", a: s(e.a), lo: s(e.lo), hi: s(e.hi) };
      case "gaussKernel": case "normalPDF": return { k: e.k, a: s(e.a), mu: s(e.mu), sigma: s(e.sigma) };
      case "dot": { const a = this.unroll(e.a), b = this.unroll(e.b); return add(...a.map((x, i) => mul(x, b[i]!))); }
      case "cosineSim": {
        const a = this.unroll(e.a), b = this.unroll(e.b);
        const nrm = (c: SExpr[]) => un("sqrt", add(...c.map((x) => un("square", x))));
        return { k: "bin", op: "div", a: add(...a.map((x, i) => mul(x, b[i]!))), b: mul(nrm(a), nrm(b)) };
      }
      case "norm": return un("sqrt", add(...this.unroll(e.v).map((x) => un("square", x))));
      case "comp": return this.component(e.v, e.index);
      case "where": return { k: "where", test: s(e.test), ref: s(e.ref), lt: s(e.lt), eq: s(e.eq), gt: s(e.gt) };
    }
  }

  /** the N components of an N-D vector expression, each a k-D scalar expression on the slice */
  unroll(v: VExpr): SExpr[] {
    return Array.from({ length: this.D }, (_, i) => this.component(v, i));
  }

  /** component i (an N-D index) of an N-D vector expression, as a k-D scalar expression on the slice. Lazy per
   *  component: `comp(grad g, 1)` of a sampled g along the slice never asks for g's fixed-dimension derivatives. */
  component(v: VExpr, i: number): SExpr {
    const s = (x: SExpr) => this.scalar(x), c = (x: VExpr) => this.component(x, i);
    switch (v.k) {
      case "constv": return C(v.value[i]!);
      case "basisv": return C(i === v.index ? 1 : 0);
      case "coordv": return s({ k: "coord", index: i });
      case "argv": return s({ k: "argvi", name: v.name, index: i });
      case "scalev": return mul(c(v.v), s(v.s));
      case "naryv": { const sum = add(...v.args.map(c)); return v.op === "meanv" ? mul(C(1 / v.args.length), sum) : sum; }
      case "subv": return add(c(v.a), mul(C(-1), c(v.b)));
      case "sumv": return add(...v.vecs.map((x, j) => mul(s(v.coeffs[j]!), c(x))));
      case "compv": return s(v.comps[i]!);
      case "normalize": {
        // v / |v|; the runtime op maps the zero vector to itself, this form is NaN there (documented difference)
        const all = this.unroll(v.v);
        return { k: "bin", op: "div", a: all[i]!, b: un("sqrt", add(...all.map((x) => un("square", x)))) };
      }
      case "grad": return s(diffScalar(v.s, i, this.D));
    }
  }

  /** the components of an N-D vector expression along the slice: the sliced vector field */
  project(v: VExpr): SExpr[] {
    const all = this.unroll(v);
    return this.dims.map((d) => all[d]!);
  }
}

/*******************************************************/
/* per-field rewrite */

interface Ctx {
  spec: BundleSpec;
  manifold: string;
  D: number;
  dims: number[];
  origin: number[];
  inliner: Inliner;
  bundle: () => Bundle;
  arrays: ArrayResolver;
}

const sliceBox = (box: Box, dims: number[]): [number, number][] => dims.map((d) => [box.a[d]!, box.b[d]!]);
const boxOf = (spec: { box?: unknown } | undefined, D: number): Box => (spec?.box ? Box.fromSpec(spec.box as Parameters<typeof Box.fromSpec>[0]) : Box.unit(D));

/** the expression's remaining arguments, as (sliced) argument specs for the pointwise datum */
function keptArgs<T extends ScalarData | VectorData>(all: Record<string, T | string> | undefined, used: Set<string>, slice: (d: T) => T): Record<string, T | string> | undefined {
  if (!all) return undefined;
  const out: Record<string, T | string> = {};
  for (const [n, d] of Object.entries(all)) if (used.has(n)) out[n] = typeof d === "string" ? d : slice(d);
  return Object.keys(out).length ? out : undefined;
}

/** a dense grid sliced at the sample nearest the origin in each fixed dimension */
function sliceDense(arr: NdArray, box: Box, ctx: Ctx, channels: number): { data: number[]; shape: number[]; box: [number, number][] } {
  const D = ctx.D, size = arr.shape.slice(0, D);
  const fixed = new Array<number>(D).fill(0);
  for (let d = 0; d < D; d++) {
    if (ctx.dims.includes(d)) continue;
    const n = size[d]!, a = box.a[d]!, b = box.b[d]!, o = ctx.origin[d]!;
    if (o < a - 1e-9 || o > b + 1e-9) throw new NotSliceable(`origin component ${o} along dimension ${d} is outside the sampled box [${a}, ${b}]`);
    fixed[d] = n <= 1 ? 0 : Math.max(0, Math.min(n - 1, Math.round(((o - a) / (b - a)) * (n - 1))));
  }
  const shape = ctx.dims.map((d) => size[d]!);
  const out = new DenseGrid(shape, new Box(ctx.dims.map((d) => box.a[d]!), ctx.dims.map((d) => box.b[d]!)));
  const data: number[] = [];
  const idx = new Array<number>(D + (channels > 1 ? 1 : 0)).fill(0);
  for (let i = 0; i < out.sampleCount; i++) {
    const gp = out.gridPos(i);
    for (let d = 0; d < D; d++) idx[d] = fixed[d]!;
    ctx.dims.forEach((d, a) => { idx[d] = gp[a]!; });
    if (channels > 1) for (const d of ctx.dims) { idx[D] = d; data.push(arr.get(...idx)); }
    else data.push(arr.get(...idx));
  }
  return { data, shape: channels > 1 ? [...shape, ctx.dims.length] : shape, box: sliceBox(box, ctx.dims) };
}

/** the sampled array and box of dense data, through translate / scale pullbacks */
function denseSource(d: ScalarData | VectorData, ctx: Ctx, vector: boolean): { arr: NdArray; box: Box } {
  switch (d.type) {
    case "dense": case "densev": {
      const arr = buildArray(d.samples, [], ctx.arrays);
      if (arr.ndim !== ctx.D + (vector ? 1 : 0)) throw new NotSliceable(`dense array of rank ${arr.ndim} on a ${ctx.D}D manifold`);
      return { arr, box: boxOf(d, ctx.D) };
    }
    case "translate": case "scale": {
      const inner = typeof d.arg === "string" ? (ctx.spec.fields[d.arg]?.data as ScalarData | VectorData | undefined) : d.arg;
      if (!inner) throw new NotSliceable(`unknown field "${String(d.arg)}"`);
      const src = denseSource(inner, ctx, vector);
      const vec = (x: unknown): number[] => { if (!Array.isArray(x) || x.length !== ctx.D) throw new NotSliceable("pullback vectors must be inline"); return x as number[]; };
      const box = d.type === "translate" ? src.box.translate(vec(d.vec)) : src.box.scale(d.origin === undefined ? new Array<number>(ctx.D).fill(0) : vec(d.origin), typeof d.scale === "number" ? new Array<number>(ctx.D).fill(d.scale) : vec(d.scale));
      return { arr: src.arr, box };
    }
    default: throw new NotSliceable(`cannot slice ${d.type} data`);
  }
}

/** whether a datum (following ids) bottoms out in sampled data — sliced as a grid rather than an expression */
function isSampled(d: ScalarData | VectorData, spec: BundleSpec, seen = new Set<string>()): boolean {
  switch (d.type) {
    case "dense": case "densev": case "sparse": case "sparsev": return true;
    case "translate": case "scale": {
      if (typeof d.arg !== "string") return isSampled(d.arg, spec, seen);
      if (seen.has(d.arg)) return false; seen.add(d.arg);
      const f = spec.fields[d.arg]; return !!f && isSampled(f.data, spec, seen);
    }
    default: return false;
  }
}

function sliceScalarData(d: ScalarData, ctx: Ctx): ScalarData {
  const slicer = new Slicer(ctx.D, ctx.dims, ctx.origin, ctx.inliner);
  switch (d.type) {
    case "symbolic": {
      const e = slicer.scalar(normalizeScalar(d.expr, emptyEnv(ctx.D, d.consts)));
      return { type: "symbolic", box: sliceBox(boxOf(d, ctx.D), ctx.dims), expr: astToScalarSpec(e) };
    }
    case "pointwise": {
      const env = pointwiseEnv(ctx.D, d.consts, d.scalars, d.vectors);
      const e = slicer.withArgs(d.scalars, d.vectors).scalar(normalizeScalar(d.expr, env));
      const scalars = keptArgs(d.scalars, slicer.used.scalars, (x) => sliceScalarData(x, ctx));
      const vectors = keptArgs(d.vectors, slicer.used.vectors, (x) => sliceVectorData(x, ctx));
      if (!scalars && !vectors) return { type: "symbolic", box: sliceBox(argBox(d, ctx), ctx.dims), expr: astToScalarSpec(e) };
      return { type: "pointwise", expr: astToScalarSpec(e), ...(scalars ? { scalars } : {}), ...(vectors ? { vectors } : {}) };
    }
    case "translate": case "scale": {
      if (isSampled(d, ctx.spec)) { const { arr, box } = denseSource(d, ctx, false); const s = sliceDense(arr, box, ctx, 1); return { type: "dense", samples: { type: "inline", shape: s.shape, data: s.data }, box: s.box }; }
      const e = slicer.scalar(ctx.inliner.scalar(d));
      return { type: "symbolic", box: sliceBox(pullbackBox(d, ctx), ctx.dims), expr: astToScalarSpec(e) };
    }
    case "dense": {
      const { arr, box } = denseSource(d, ctx, false);
      const s = sliceDense(arr, box, ctx, 1);
      return { type: "dense", samples: { type: "inline", shape: s.shape, data: s.data }, box: s.box };
    }
    case "net": return sliceNet(d, ctx);
    default: throw new NotSliceable(`${d.type} data is not sliceable yet`);
  }
}

function sliceVectorData(d: VectorData, ctx: Ctx): VectorData {
  const slicer = new Slicer(ctx.D, ctx.dims, ctx.origin, ctx.inliner);
  const compv = (comps: SExpr[]): SymbolicVector => ({ op: "compv", coeffs: comps.map(astToScalarSpec) });
  switch (d.type) {
    case "symbolicv": {
      const comps = slicer.project(normalizeVector(d.expr, emptyEnv(ctx.D, d.consts)));
      return { type: "symbolicv", box: sliceBox(boxOf(d, ctx.D), ctx.dims), expr: compv(comps) };
    }
    case "pointwisev": {
      const env = pointwiseEnv(ctx.D, d.consts, d.scalars, d.vectors);
      const comps = slicer.withArgs(d.scalars, d.vectors).project(normalizeVector(d.expr, env));
      const scalars = keptArgs(d.scalars, slicer.used.scalars, (x) => sliceScalarData(x, ctx));
      const vectors = keptArgs(d.vectors, slicer.used.vectors, (x) => sliceVectorData(x, ctx));
      if (!scalars && !vectors) return { type: "symbolicv", box: sliceBox(argBox(d, ctx), ctx.dims), expr: compv(comps) };
      return { type: "pointwisev", expr: compv(comps), ...(scalars ? { scalars } : {}), ...(vectors ? { vectors } : {}) };
    }
    case "translate": case "scale": {
      if (isSampled(d, ctx.spec)) { const { arr, box } = denseSource(d, ctx, true); const s = sliceDense(arr, box, ctx, ctx.D); return { type: "densev", samples: { type: "inline", shape: s.shape, data: s.data }, box: s.box }; }
      const comps = slicer.project(ctx.inliner.vector(d));
      return { type: "symbolicv", box: sliceBox(pullbackBox(d, ctx), ctx.dims), expr: compv(comps) };
    }
    case "densev": {
      const { arr, box } = denseSource(d, ctx, true);
      const s = sliceDense(arr, box, ctx, ctx.D);
      return { type: "densev", samples: { type: "inline", shape: s.shape, data: s.data }, box: s.box };
    }
    default: throw new NotSliceable(`${d.type} data is not sliceable yet`);
  }
}

/** the box of a pointwise datum: the intersection of its arguments' boxes (as the runtime does), via the built bundle when it refers to fields */
function argBox(d: Extract<ScalarData, { type: "pointwise" }> | Extract<VectorData, { type: "pointwisev" }>, ctx: Ctx): Box {
  let box: Box | undefined;
  const meet = (b: Box) => { box = box ? (box.intersect(b) ?? b) : b; };
  for (const [kind, args] of [["scalar", d.scalars], ["vector", d.vectors]] as const) {
    for (const a of Object.values(args ?? {})) {
      if (typeof a === "string") meet(kind === "scalar" ? ctx.bundle().scalarField(a).data.box : ctx.bundle().vectorField(a).data.box);
      else if (a.type === "pointwise" || a.type === "pointwisev") meet(argBox(a as never, ctx));
      else if (a.type === "translate" || a.type === "scale") meet(pullbackBox(a, ctx));
      else meet(boxOf(a as { box?: unknown }, ctx.D));
    }
  }
  return box ?? Box.unit(ctx.D);
}

type PullbackData = Extract<ScalarData, { type: "translate" | "scale" }> | Extract<VectorData, { type: "translate" | "scale" }>;
function pullbackBox(d: PullbackData, ctx: Ctx): Box {
  const inner = typeof d.arg === "string" ? ctx.bundle().field(d.arg).data.box : d.arg.type === "pointwise" || d.arg.type === "pointwisev" ? argBox(d.arg as never, ctx) : d.arg.type === "translate" || d.arg.type === "scale" ? pullbackBox(d.arg as never, ctx) : boxOf(d.arg as { box?: unknown }, ctx.D);
  const vec = (x: unknown): number[] => { if (!Array.isArray(x)) throw new NotSliceable("pullback vectors must be inline"); return x as number[]; };
  return d.type === "translate" ? inner.translate(vec(d.vec)) : inner.scale(d.origin === undefined ? new Array<number>(ctx.D).fill(0) : vec(d.origin), typeof d.scale === "number" ? new Array<number>(ctx.D).fill(d.scale) : vec(d.scale));
}

/** a scalar net field: the point input becomes E·p + o; explicit inputs get their coordinate leaves substituted */
function sliceNet(d: NetScalarFieldDataSpec, ctx: Ctx): ScalarData {
  const D = ctx.D, k = ctx.dims.length;
  const E: number[] = []; for (let i = 0; i < D; i++) for (let a = 0; a < k; a++) E.push(ctx.dims[a] === i ? 1 : 0);
  const arrays = { ...(d.arrays ?? {}), __sliceE: { type: "inline" as const, shape: [D, k], data: E }, __sliceO: { type: "inline" as const, shape: [D], data: [...ctx.origin] } };
  const point: ArrayExpr = { op: "add", vals: [{ op: "matmul", vals: ["__sliceE", { op: "coordv" }] }, "__sliceO"] };
  const subst = (e: ArrayExpr): ArrayExpr => {
    if (typeof e !== "object" || e === null) return e;
    if (e.op === "coordv") return point;
    if (e.op === "coord") { const p = ctx.dims.indexOf(e.index); return p >= 0 ? { op: "coord", index: p } : ctx.origin[e.index]!; }
    // generic: substitute in every ArrayExpr-valued property (vals / val / inputs / cond / indices)
    const out: Record<string, unknown> = { ...e };
    for (const [key, val] of Object.entries(e)) {
      if (key === "op") continue;
      if (Array.isArray(val)) out[key] = val.map((x) => (isArrayExpr(x) ? subst(x as ArrayExpr) : x));
      else if (key === "inputs" && val && typeof val === "object") out[key] = Object.fromEntries(Object.entries(val as Record<string, ArrayExpr>).map(([n, x]) => [n, subst(x)]));
      else if (isArrayExpr(val) && key !== "net") out[key] = subst(val as ArrayExpr);
    }
    return out as ArrayExpr;
  };
  let inputs: Record<string, ArrayExpr>;
  if (d.inputs) inputs = Object.fromEntries(Object.entries(d.inputs).map(([n, e]) => [n, subst(e)]));
  else {
    // DEFAULT: the sole remaining input is the point
    const sig = typeof d.net === "string" ? ctx.bundle().net(d.net).signature : undefined;
    if (!sig) throw new NotSliceable("an inline net spec with implicit inputs is not sliceable yet (name the net or give `inputs`)");
    const names = Object.keys(sig.inputs);
    if (names.length !== 1) throw new NotSliceable(`net has ${names.length} remaining inputs; give \`inputs\``);
    inputs = { [names[0]!]: point };
  }
  return { ...d, inputs, arrays, box: sliceBox(boxOf(d, D), ctx.dims) };
}
const isArrayExpr = (x: unknown): boolean => typeof x === "number" || typeof x === "string" || (typeof x === "object" && x !== null && "op" in x);

/*******************************************************/

/** whether a manifold could be sliced by the viewer: 3 < D <= 8 */
export const SLICE_MAX_DIMS = 8;
export function sliceable(spec: BundleSpec, manifold: string): boolean {
  const D = spec.manifolds?.[manifold]?.numDims;
  return D !== undefined && D > 3 && D <= SLICE_MAX_DIMS;
}

/**
 * A new spec with `manifold` (N-D) restricted to the axis-aligned subspace along `dims` (2 or 3 of them, any
 * order — they are sorted) through the manifold's `origin` (default 0). Fields and point sets on other manifolds
 * are untouched. Fields that cannot be sliced are left out and listed in `dropped`.
 */
export function sliceSpec(spec: BundleSpec, manifold: string, dims: readonly number[], originOverride?: readonly number[], arrays: ArrayResolver = noArrays): SliceResult {
  const m = spec.manifolds?.[manifold];
  if (!m) throw new SpecError(`unknown manifold "${manifold}"`, ["manifolds", manifold]);
  const D = m.numDims;
  const sorted = [...new Set(dims)].sort((a, b) => a - b);
  for (const d of sorted) if (!Number.isInteger(d) || d < 0 || d >= D) throw new SpecError(`slice dimension ${d} out of range for ${D} dimensions`, ["manifolds", manifold]);
  if (sorted.length < 1 || sorted.length >= D) throw new SpecError(`a slice needs between 1 and ${D - 1} dimensions, got ${sorted.length}`, ["manifolds", manifold]);
  const origin = [...(originOverride ?? m.origin ?? new Array<number>(D).fill(0))];
  if (origin.length !== D) throw new SpecError(`origin has ${origin.length} components for ${D} dims`, ["manifolds", manifold, "origin"]);

  let built: Bundle | undefined;
  const ctx: Ctx = { spec, manifold, D, dims: sorted, origin, inliner: new Inliner(spec, manifold, D), bundle: () => (built ??= new Bundle(spec, arrays)), arrays };
  const dropped: Record<string, string> = {};
  const fields: Record<string, FieldSpec> = {};
  for (const [id, f] of Object.entries(spec.fields)) {
    if (domainOf(spec, f.domain) !== manifold) { fields[id] = f; continue; }
    try {
      const data = f.kind === "scalar" ? sliceScalarData(f.data, ctx) : sliceVectorData(f.data, ctx);
      fields[id] = { ...f, data } as FieldSpec;
    } catch (e) {
      if (e instanceof NotSliceable || e instanceof SpecError) dropped[id] = e.message; else throw e;
    }
  }
  // a scalar field's exactGradient must have survived too
  for (const f of Object.values(fields)) if (f.kind === "scalar" && f.exactGradient !== undefined && !(f.exactGradient in fields)) delete (f as { exactGradient?: string }).exactGradient;

  const pointSets: Record<string, PointSetSpec> = {};
  for (const [id, ps] of Object.entries(spec.pointSets ?? {})) {
    if (domainOf(spec, ps.domain) !== manifold) { pointSets[id] = ps; continue; }
    pointSets[id] = { ...ps, points: ps.points.map((p) => sorted.map((d) => p[d]!)) };
  }

  const { origin: _o, ...rest } = m;
  const manifoldSpec: ManifoldDefinitionSpec = {
    ...rest,
    numDims: sorted.length,
    ...(m.dimNames ? { dimNames: sorted.map((d) => m.dimNames![d]!) } : {}),
    ...(m.dimWeights ? { dimWeights: sorted.map((d) => m.dimWeights![d]!) } : {}),
    ...(m.flow !== undefined && !(m.flow in fields) ? { flow: undefined } : {}),
  };
  if (manifoldSpec.flow === undefined) delete manifoldSpec.flow;
  return { spec: { ...spec, manifolds: { ...spec.manifolds, [manifold]: manifoldSpec }, fields, ...(spec.pointSets ? { pointSets } : {}) }, dropped };
}

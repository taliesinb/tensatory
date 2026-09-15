// AST -> JS closures. Evaluators receive the point `p` and, when the point is
// a sample of the enclosing discrete support, its sample position `pos`
// (otherwise -1). Field arguments use `pos` when sampled and `p` when symbolic.

import { EvalError } from "../errors";
import { evalBinary, type SExpr, type VExpr } from "./ast";
import { gradient } from "./diff";
import { UNARY, normalPDF } from "./functions";

export type ScalarFn = (p: ArrayLike<number>, pos: number) => number;
/** writes the result into `out` and returns it */
export type VectorFn = (p: ArrayLike<number>, pos: number, out: Float64Array) => Float64Array;

export interface CompileContext {
  readonly dimCount: number;
  scalarArg(name: string): ScalarFn;
  vectorArg(name: string): VectorFn;
  /** ∂^n (scalar arg) / ∂x_dims[0] ... ∂x_dims[n-1] */
  scalarArgPartial(name: string, dims: number[]): ScalarFn;
  /** ∂^n (vector arg)[index] / ∂x_dims... */
  vectorArgPartial(name: string, index: number, dims: number[]): ScalarFn;
}

/** a context for expressions without field arguments */
export function pureContext(dimCount: number): CompileContext {
  const fail = (what: string) => (): never => {
    throw new EvalError(`expression refers to ${what} but no arguments are available`);
  };
  return {
    dimCount,
    scalarArg: fail("a scalar argument"),
    vectorArg: fail("a vector argument"),
    scalarArgPartial: fail("a scalar argument"),
    vectorArgPartial: fail("a vector argument"),
  };
}

/*******************************************************/
/* common-subexpression elimination
 *
 * Trees produced by differentiation repeat subtrees heavily (the same
 * exponential appears in a value, its derivative and its Hessian entries).
 * A Compilation hash-conses nodes by structural key: every distinct subtree is
 * compiled once, and subtrees referenced more than once are wrapped in a memo
 * that returns the cached value when re-evaluated at the same point (same
 * coordinates and sample position). Memo hits are exact, so this is safe
 * across nested argument evaluation, which passes the same point down. */

const keyCache = new WeakMap<object, string>();
/** structural key of a node (memoized per object; shared object references are common) */
function keyOf(e: SExpr | VExpr): string {
  let k = keyCache.get(e);
  if (k) return k;
  const parts: string[] = [e.k];
  for (const [name, v] of Object.entries(e)) {
    if (name === "k") continue;
    if (Array.isArray(v)) parts.push(`${name}=[${v.map((x) => (typeof x === "object" && x !== null ? keyOf(x as SExpr) : String(x))).join(",")}]`);
    else if (typeof v === "object" && v !== null) parts.push(`${name}=${keyOf(v as SExpr)}`);
    else parts.push(`${name}=${String(v)}`);
  }
  k = `(${parts.join(" ")})`;
  keyCache.set(e, k);
  return k;
}

const LEAF_KINDS = new Set(["const", "coord", "arg", "argv", "constv", "basisv", "coordv"]);

class Compilation {
  private readonly uses = new Map<string, number>();
  private readonly scalars = new Map<string, ScalarFn>();
  private readonly vectors = new Map<string, VectorFn>();
  constructor(readonly ctx: CompileContext) {}

  /** count how often each subtree occurs (by structure) */
  count(e: SExpr | VExpr): void {
    const k = keyOf(e);
    const n = (this.uses.get(k) ?? 0) + 1;
    this.uses.set(k, n);
    if (n > 1) return; // children already counted through the first occurrence
    for (const v of Object.values(e as Record<string, unknown>)) {
      if (Array.isArray(v)) { for (const x of v) if (x && typeof x === "object" && "k" in (x as object)) this.count(x as SExpr); }
      else if (v && typeof v === "object" && "k" in (v as object)) this.count(v as SExpr);
    }
  }

  scalar(e: SExpr): ScalarFn {
    const k = keyOf(e);
    let f = this.scalars.get(k);
    if (f) return f;
    f = compileScalarRaw(e, this);
    if ((this.uses.get(k) ?? 1) > 1 && !LEAF_KINDS.has(e.k)) f = memoScalar(f, this.ctx.dimCount);
    this.scalars.set(k, f);
    return f;
  }

  vector(e: VExpr): VectorFn {
    const k = keyOf(e);
    let f = this.vectors.get(k);
    if (f) return f;
    f = compileVectorRaw(e, this);
    if ((this.uses.get(k) ?? 1) > 1 && !LEAF_KINDS.has(e.k)) f = memoVector(f, this.ctx.dimCount);
    this.vectors.set(k, f);
    return f;
  }
}

function memoScalar(f: ScalarFn, D: number): ScalarFn {
  const last = new Float64Array(D).fill(NaN);
  let lastPos = -2, value = 0;
  return (p, pos) => {
    let same = pos === lastPos;
    for (let d = 0; same && d < D; d++) same = p[d] === last[d];
    if (same) return value;
    for (let d = 0; d < D; d++) last[d] = p[d]!;
    lastPos = pos;
    return (value = f(p, pos));
  };
}

function memoVector(f: VectorFn, D: number): VectorFn {
  const last = new Float64Array(D).fill(NaN), value = new Float64Array(D);
  let lastPos = -2;
  return (p, pos, out) => {
    let same = pos === lastPos;
    for (let d = 0; same && d < D; d++) same = p[d] === last[d];
    if (!same) {
      for (let d = 0; d < D; d++) last[d] = p[d]!;
      lastPos = pos;
      value.set(f(p, pos, out));
      return out;
    }
    out.set(value);
    return out;
  };
}

/** compile a scalar expression (with common-subexpression elimination) */
export function compileScalar(e: SExpr, ctx: CompileContext): ScalarFn {
  const c = new Compilation(ctx);
  c.count(e);
  return c.scalar(e);
}

/** compile a vector expression (with common-subexpression elimination) */
export function compileVector(e: VExpr, ctx: CompileContext): VectorFn {
  const c = new Compilation(ctx);
  c.count(e);
  return c.vector(e);
}

function compileScalarRaw(e: SExpr, c: Compilation): ScalarFn {
  const ctx = c.ctx;
  const D = ctx.dimCount;
  const compileScalar = (x: SExpr) => c.scalar(x);
  const compileVector = (x: VExpr) => c.vector(x);
  switch (e.k) {
    case "const": { const v = e.value; return () => v; }
    case "coord": { const i = e.index; return (p) => p[i]!; }
    case "arg": return ctx.scalarArg(e.name);
    case "argvi": {
      const f = ctx.vectorArg(e.name), tmp = new Float64Array(D), i = e.index;
      return (p, pos) => f(p, pos, tmp)[i]!;
    }
    case "argd": return ctx.scalarArgPartial(e.name, e.dims);
    case "argvid": return ctx.vectorArgPartial(e.name, e.index, e.dims);
    case "un": { const f = UNARY[e.op], a = compileScalar(e.a); return (p, pos) => f(a(p, pos)); }
    case "nary": {
      const fs = e.args.map((a) => compileScalar(a));
      const n = fs.length;
      switch (e.op) {
        case "add": return (p, pos) => { let s = 0; for (let i = 0; i < n; i++) s += fs[i]!(p, pos); return s; };
        case "mul": return (p, pos) => { let s = 1; for (let i = 0; i < n; i++) s *= fs[i]!(p, pos); return s; };
        case "min": return (p, pos) => { let s = Infinity; for (let i = 0; i < n; i++) s = Math.min(s, fs[i]!(p, pos)); return s; };
        case "max": return (p, pos) => { let s = -Infinity; for (let i = 0; i < n; i++) s = Math.max(s, fs[i]!(p, pos)); return s; };
        case "mean": return (p, pos) => { let s = 0; for (let i = 0; i < n; i++) s += fs[i]!(p, pos); return s / n; };
        case "rms": return (p, pos) => { let s = 0; for (let i = 0; i < n; i++) { const v = fs[i]!(p, pos); s += v * v; } return Math.sqrt(s / n); };
      }
      break;
    }
    case "bin": {
      const a = compileScalar(e.a), b = compileScalar(e.b), op = e.op;
      switch (op) {
        case "sub": return (p, pos) => a(p, pos) - b(p, pos);
        case "div": return (p, pos) => a(p, pos) / b(p, pos);
        case "pow": return (p, pos) => Math.pow(a(p, pos), b(p, pos));
        default: return (p, pos) => evalBinary(op, a(p, pos), b(p, pos));
      }
    }
    case "clamp": {
      const a = compileScalar(e.a), lo = compileScalar(e.lo), hi = compileScalar(e.hi);
      return (p, pos) => Math.min(Math.max(a(p, pos), lo(p, pos)), hi(p, pos));
    }
    case "gaussKernel": case "normalPDF": {
      const a = compileScalar(e.a), mu = compileScalar(e.mu), sigma = compileScalar(e.sigma);
      if (e.k === "gaussKernel") return (p, pos) => { const s = sigma(p, pos), z = (a(p, pos) - mu(p, pos)) / s; return Math.exp(-0.5 * z * z); };
      return (p, pos) => { const s = sigma(p, pos); return normalPDF((a(p, pos) - mu(p, pos)) / s, s); };
    }
    case "dot": case "cosineSim": {
      const a = compileVector(e.a), b = compileVector(e.b);
      const ta = new Float64Array(D), tb = new Float64Array(D);
      if (e.k === "dot") return (p, pos) => { a(p, pos, ta); b(p, pos, tb); let s = 0; for (let i = 0; i < D; i++) s += ta[i]! * tb[i]!; return s; };
      return (p, pos) => {
        a(p, pos, ta); b(p, pos, tb);
        let s = 0, na = 0, nb = 0;
        for (let i = 0; i < D; i++) { s += ta[i]! * tb[i]!; na += ta[i]! * ta[i]!; nb += tb[i]! * tb[i]!; }
        const n = Math.sqrt(na * nb);
        return n === 0 ? 0 : s / n;
      };
    }
    case "norm": {
      const v = compileVector(e.v), t = new Float64Array(D);
      return (p, pos) => { v(p, pos, t); let s = 0; for (let i = 0; i < D; i++) s += t[i]! * t[i]!; return Math.sqrt(s); };
    }
    case "comp": {
      const v = compileVector(e.v), t = new Float64Array(D), i = e.index;
      return (p, pos) => v(p, pos, t)[i]!;
    }
    case "where": {
      const test = compileScalar(e.test), ref = compileScalar(e.ref);
      const lt = compileScalar(e.lt), eq = compileScalar(e.eq), gt = compileScalar(e.gt);
      return (p, pos) => { const t = test(p, pos), r = ref(p, pos); return t < r ? lt(p, pos) : t > r ? gt(p, pos) : eq(p, pos); };
    }
  }
  throw new Error(`unreachable: cannot compile ${(e as SExpr).k}`);
}

function compileVectorRaw(e: VExpr, c: Compilation): VectorFn {
  const ctx = c.ctx;
  const D = ctx.dimCount;
  const compileScalar = (x: SExpr) => c.scalar(x);
  const compileVector = (x: VExpr) => c.vector(x);
  switch (e.k) {
    case "constv": { const v = Float64Array.from(e.value); return (_p, _pos, out) => { out.set(v); return out; }; }
    case "basisv": { const i = e.index; return (_p, _pos, out) => { out.fill(0); out[i] = 1; return out; }; }
    case "coordv": return (p, _pos, out) => { for (let i = 0; i < D; i++) out[i] = p[i]!; return out; };
    case "argv": return ctx.vectorArg(e.name);
    case "scalev": {
      const v = compileVector(e.v), s = compileScalar(e.s);
      return (p, pos, out) => { v(p, pos, out); const k = s(p, pos); for (let i = 0; i < D; i++) out[i]! *= k; return out; };
    }
    case "naryv": {
      const fs = e.args.map((a) => compileVector(a)), n = fs.length, t = new Float64Array(D);
      const scale = e.op === "meanv" ? 1 / n : 1;
      return (p, pos, out) => {
        out.fill(0);
        for (let j = 0; j < n; j++) { fs[j]!(p, pos, t); for (let i = 0; i < D; i++) out[i]! += t[i]!; }
        if (scale !== 1) for (let i = 0; i < D; i++) out[i]! *= scale;
        return out;
      };
    }
    case "subv": {
      const a = compileVector(e.a), b = compileVector(e.b), t = new Float64Array(D);
      return (p, pos, out) => { a(p, pos, out); b(p, pos, t); for (let i = 0; i < D; i++) out[i]! -= t[i]!; return out; };
    }
    case "sumv": {
      const vs = e.vecs.map((v) => compileVector(v)), cs = e.coeffs.map((c) => compileScalar(c));
      const n = vs.length, t = new Float64Array(D);
      return (p, pos, out) => {
        out.fill(0);
        for (let j = 0; j < n; j++) { const c = cs[j]!(p, pos); vs[j]!(p, pos, t); for (let i = 0; i < D; i++) out[i]! += c * t[i]!; }
        return out;
      };
    }
    case "compv": {
      const cs = e.comps.map((c) => compileScalar(c));
      return (p, pos, out) => { for (let i = 0; i < D; i++) out[i] = cs[i]!(p, pos); return out; };
    }
    case "normalize": {
      const v = compileVector(e.v);
      return (p, pos, out) => {
        v(p, pos, out);
        let s = 0; for (let i = 0; i < D; i++) s += out[i]! * out[i]!;
        if (s > 0) { const k = 1 / Math.sqrt(s); for (let i = 0; i < D; i++) out[i]! *= k; }
        return out;
      };
    }
    case "grad": { const g = gradient(e.s, D); c.count(g); return compileVector(g); }
  }
}

// Normalized expression AST -> WGSL.
//
// Expressions are emitted in SSA form inside a function body: every distinct
// subtree (by structural key) becomes one `let`, so common subexpressions are
// computed once — the GPU counterpart of the CPU compiler's memoization.
// Field arguments are calls to functions supplied by the program builder
// (program.ts), which decides how each argument is realized (transpiled,
// buffer-backed, ...).

import type { SExpr, VExpr } from "@tensatory/core";

/** how the emitter refers to field arguments */
export interface ArgBindings {
  /** WGSL function name evaluating the named scalar argument, differentiated along `dims` (empty = value) */
  scalar(name: string, dims: number[]): string;
  /** WGSL function name evaluating the named vector argument (returns vecD) */
  vector(name: string): string;
  /** WGSL function name evaluating component `index` of the named vector argument, differentiated along `dims` */
  vectorComponent(name: string, index: number, dims: number[]): string;
}

export const vecType = (D: number): string => (D === 1 ? "f32" : `vec${D}<f32>`);

/** a finite f32 literal */
export function f32(v: number): string {
  if (!Number.isFinite(v)) throw new Error(`cannot emit non-finite constant ${v}`);
  if (Math.abs(v) < 1e-37) return "0.0";
  const c = Math.sign(v) * Math.min(Math.abs(v), 3.4e38);
  let s = c.toPrecision(9);
  if (!/[.e]/.test(s)) s += ".0";
  return s;
}

/** structural key (mirrors core's keyOf; kept local so the emitter has no private imports) */
const keyCache = new WeakMap<object, string>();
export function keyOf(e: SExpr | VExpr): string {
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

/** helper functions shared by every generated shader */
export const PRELUDE = `
const LN10: f32 = 2.302585092994046;
const INV_LN10: f32 = 0.4342944819032518;
const INV_LN2: f32 = 1.4426950408889634;
const SQRT_2PI: f32 = 2.5066282746310002;
var<private> zero_: f32 = 0.0;
fn nan_() -> f32 { return zero_ / zero_; }
fn erf_(x: f32) -> f32 {
  let s = sign(x); let a = abs(x); let t = 1.0 / (1.0 + 0.3275911 * a);
  let y = 1.0 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-a * a);
  return s * y;
}
fn sigmoid_(x: f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }
fn softplus_(x: f32) -> f32 { return select(log(1.0 + exp(x)), x, x > 30.0); }
fn gelu_(x: f32) -> f32 { return 0.5 * x * (1.0 + erf_(x * 0.7071067811865476)); }
fn silu_(x: f32) -> f32 { return x * sigmoid_(x); }
fn elu_(x: f32) -> f32 { return select(exp(x) - 1.0, x, x > 0.0); }
fn relu_(x: f32) -> f32 { return max(x, 0.0); }
fn plogp_(x: f32) -> f32 { return select(x * log(x), 0.0, x == 0.0); }
fn gauss_(x: f32) -> f32 { return exp(-0.5 * x * x); }
fn round_(x: f32) -> f32 { return floor(x + 0.5); }
fn mod_(a: f32, b: f32) -> f32 { return a - b * floor(a / b); }
fn pow_(a: f32, b: f32) -> f32 {
  // JS semantics for negative bases with integral exponents
  if (a < 0.0 && b == floor(b)) { let m = pow(-a, b); return select(m, -m, mod_(b, 2.0) == 1.0); }
  return pow(a, b);
}
fn logbase_(a: f32, b: f32) -> f32 { return log(a) / log(b); }
fn normalpdf_(z: f32, s: f32) -> f32 { return exp(-0.5 * z * z) / (s * SQRT_2PI); }
`;

const UNARY_WGSL: Record<string, (x: string) => string> = {
  sin: (x) => `sin(${x})`, cos: (x) => `cos(${x})`, tan: (x) => `tan(${x})`,
  sinh: (x) => `sinh(${x})`, cosh: (x) => `cosh(${x})`, tanh: (x) => `tanh(${x})`,
  asin: (x) => `asin(${x})`, acos: (x) => `acos(${x})`, atan: (x) => `atan(${x})`,
  asinh: (x) => `asinh(${x})`, acosh: (x) => `acosh(${x})`, atanh: (x) => `atanh(${x})`,
  relu: (x) => `relu_(${x})`, sigmoid: (x) => `sigmoid_(${x})`, gelu: (x) => `gelu_(${x})`, silu: (x) => `silu_(${x})`,
  softplus: (x) => `softplus_(${x})`, elu: (x) => `elu_(${x})`, erf: (x) => `erf_(${x})`,
  floor: (x) => `floor(${x})`, ceil: (x) => `ceil(${x})`, round: (x) => `round_(${x})`, sign: (x) => `sign(${x})`, abs: (x) => `abs(${x})`,
  exp: (x) => `exp(${x})`, exp2: (x) => `exp2(${x})`, exp10: (x) => `exp(${x} * LN10)`,
  log: (x) => `log(${x})`, log2: (x) => `log2(${x})`, log10: (x) => `(log(${x}) * INV_LN10)`, log1p: (x) => `log(1.0 + ${x})`, expm1: (x) => `(exp(${x}) - 1.0)`,
  plogp: (x) => `plogp_(${x})`, sqrt: (x) => `sqrt(${x})`, square: (x) => `(${x} * ${x})`, negate: (x) => `(-${x})`, reciprocal: (x) => `(1.0 / ${x})`, gauss: (x) => `gauss_(${x})`,
};

/**
 * Emits one WGSL function `fn <name>(p: vecD, pos: i32) -> f32|vecD` for an
 * expression. Shared subtrees are emitted once as `let`s.
 */
export class FunctionEmitter {
  private readonly lines: string[] = [];
  private readonly temps = new Map<string, string>();
  private n = 0;

  constructor(readonly D: number, readonly args: ArgBindings) {}

  private tmp(type: string, expr: string, key: string): string {
    const t = `t${this.n++}`;
    this.lines.push(`  let ${t}: ${type} = ${expr};`);
    this.temps.set(key, t);
    return t;
  }

  scalar(e: SExpr): string {
    const key = keyOf(e);
    const have = this.temps.get(key);
    if (have) return have;
    const S = (x: SExpr) => this.scalar(x);
    const V = (x: VExpr) => this.vector(x);
    let expr: string;
    switch (e.k) {
      case "const": return f32(e.value); // literals inline
      case "coord": expr = this.D === 1 ? "p" : `p[${e.index}]`; break;
      case "arg": expr = `${this.args.scalar(e.name, [])}(p, pos)`; break;
      case "argd": expr = `${this.args.scalar(e.name, e.dims)}(p, pos)`; break;
      case "argvi": expr = `${this.args.vectorComponent(e.name, e.index, [])}(p, pos)`; break;
      case "argvid": expr = `${this.args.vectorComponent(e.name, e.index, e.dims)}(p, pos)`; break;
      case "un": expr = UNARY_WGSL[e.op]!(S(e.a)); break;
      case "nary": {
        const xs = e.args.map(S);
        switch (e.op) {
          case "add": expr = xs.join(" + "); break;
          case "mul": expr = xs.join(" * "); break;
          case "min": expr = xs.reduce((a, b) => `min(${a}, ${b})`); break;
          case "max": expr = xs.reduce((a, b) => `max(${a}, ${b})`); break;
          case "mean": expr = `(${xs.join(" + ")}) / ${f32(xs.length)}`; break;
          case "rms": expr = `sqrt((${xs.map((x) => `${x} * ${x}`).join(" + ")}) / ${f32(xs.length)})`; break;
        }
        break;
      }
      case "bin": {
        const a = S(e.a), b = S(e.b);
        switch (e.op) {
          case "sub": expr = `${a} - ${b}`; break;
          case "div": expr = `${a} / ${b}`; break;
          case "pow": expr = `pow_(${a}, ${b})`; break;
          case "logBase": expr = `logbase_(${a}, ${b})`; break;
          case "atan2": expr = `atan2(${a}, ${b})`; break;
          case "mod": expr = `mod_(${a}, ${b})`; break;
        }
        break;
      }
      case "clamp": expr = `clamp(${S(e.a)}, ${S(e.lo)}, ${S(e.hi)})`; break;
      case "gaussKernel": { const s = S(e.sigma); expr = `gauss_((${S(e.a)} - ${S(e.mu)}) / ${s})`; break; }
      case "normalPDF": { const s = S(e.sigma); expr = `normalpdf_((${S(e.a)} - ${S(e.mu)}) / ${s}, ${s})`; break; }
      case "dot": expr = this.D === 1 ? `${V(e.a)} * ${V(e.b)}` : `dot(${V(e.a)}, ${V(e.b)})`; break;
      case "cosineSim": {
        const a = V(e.a), b = V(e.b);
        const d = this.D === 1 ? `${a} * ${b}` : `dot(${a}, ${b})`;
        const n = this.D === 1 ? `abs(${a}) * abs(${b})` : `length(${a}) * length(${b})`;
        expr = `select(${d} / ${n}, 0.0, ${n} == 0.0)`; break;
      }
      case "norm": expr = this.D === 1 ? `abs(${V(e.v)})` : `length(${V(e.v)})`; break;
      case "comp": expr = this.D === 1 ? V(e.v) : `${V(e.v)}[${e.index}]`; break;
      case "where": {
        const t = S(e.test), r = S(e.ref), lt = S(e.lt), eq = S(e.eq), gt = S(e.gt);
        expr = `select(select(${eq}, ${gt}, ${t} > ${r}), ${lt}, ${t} < ${r})`; break;
      }
    }
    return this.tmp("f32", expr!, key);
  }

  vector(e: VExpr): string {
    const key = keyOf(e);
    const have = this.temps.get(key);
    if (have) return have;
    const D = this.D, T = vecType(D);
    const S = (x: SExpr) => this.scalar(x);
    const V = (x: VExpr) => this.vector(x);
    let expr: string;
    switch (e.k) {
      case "constv": expr = D === 1 ? f32(e.value[0]!) : `${T}(${e.value.map(f32).join(", ")})`; break;
      case "basisv": expr = D === 1 ? "1.0" : `${T}(${Array.from({ length: D }, (_, i) => (i === e.index ? "1.0" : "0.0")).join(", ")})`; break;
      case "coordv": expr = "p"; break;
      case "argv": expr = `${this.args.vector(e.name)}(p, pos)`; break;
      case "scalev": expr = `${V(e.v)} * ${S(e.s)}`; break;
      case "naryv": { const xs = e.args.map(V); expr = xs.join(" + "); if (e.op === "meanv") expr = `(${expr}) / ${f32(xs.length)}`; break; }
      case "subv": expr = `${V(e.a)} - ${V(e.b)}`; break;
      case "sumv": expr = e.vecs.map((v, i) => `${V(v)} * ${S(e.coeffs[i]!)}`).join(" + "); break;
      case "compv": expr = D === 1 ? S(e.comps[0]!) : `${T}(${e.comps.map(S).join(", ")})`; break;
      case "normalize": {
        const v = V(e.v);
        const len = D === 1 ? `abs(${v})` : `length(${v})`;
        expr = `select(${v} / ${len}, ${v}, ${len} == 0.0)`; break;
      }
      case "grad": throw new Error("grad nodes must be expanded (gradient()) before WGSL emission");
    }
    return this.tmp(T, expr!, key);
  }

  /** the complete function text */
  functionText(name: string, returnType: string, resultTemp: string): string {
    return `fn ${name}(p: ${vecType(this.D)}, pos: i32) -> ${returnType} {\n${this.lines.join("\n")}\n  return ${resultTemp};\n}`;
  }
}

/** replace every `grad` node by its explicit gradient so the emitter never sees one */
export function expandGrad(e: SExpr, gradient: (s: SExpr, D: number) => VExpr, D: number): SExpr;
export function expandGrad(e: VExpr, gradient: (s: SExpr, D: number) => VExpr, D: number): VExpr;
export function expandGrad(e: SExpr | VExpr, gradient: (s: SExpr, D: number) => VExpr, D: number): SExpr | VExpr {
  const rec = (x: unknown): unknown => {
    if (Array.isArray(x)) return x.map(rec);
    if (x && typeof x === "object" && "k" in (x as object)) {
      const node = x as SExpr | VExpr;
      if (node.k === "grad") return rec(gradient(expandGrad(node.s, gradient, D), D));
      const out: Record<string, unknown> = {};
      for (const [k, v] of Object.entries(node)) out[k] = rec(v);
      return out;
    }
    return x;
  };
  return rec(e) as SExpr | VExpr;
}

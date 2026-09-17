// Reverse-mode automatic differentiation as a PROGRAM REWRITE.
//
// A `grad` net is compiled into an ordinary Program: the forward program in
// A-normal form (every intermediate is a named node — these are the "state"
// a backward rule may need: the pre-activation of a relu, the output of a
// softmax, the max of a logsumexp) followed by adjoint nodes emitted while
// walking the forward nodes in reverse. Every adjoint is expressed with the
// EXISTING array ops, so both evaluators (CPU ops.ts, WGSL gpu/nets.ts) get
// gradients without knowing about differentiation, and the result composes:
// a grad program can be bound, displaced, called and differentiated again
// (Hessian-vector products are seeded rewrites of a rewrite).
//
// The set of ops is CLOSED under taking adjoints: the two adjoints without a
// direct op — scattering for `slice` and `takeAlong` — are written with a baked
// selection matrix + `einsum`, and with `oneHot` + `reduce` + `transpose`; the
// adjoint of an einsum that reads a diagonal ("ii->i") is an einsum whose
// output repeats the letter ("i->ii", writes the diagonal), and vice versa.
// Every adjoint expression is itself differentiable, so higher orders compose.
//
// Everything works on DECLARED shapes; the implicit batch is invisible here
// and comes out as per-batch-element gradients by construction (all
// reductions in adjoint rules are over declared axes).

import type { ArrayExpr, ArrayReduceFn, NetSpec, ScalarUnaryOp } from "@tensatory/schema";
import { NdArray } from "../arrays/ndarray";
import { NotSupportedError, SpecError } from "../errors";
import { SCALAR_BINARY_OPS, SCALAR_NARY_OPS, SCALAR_UNARY_OPS } from "../symbolic/spec";
import type { Program } from "./program";
import { inferExpr, noNetResolver, type ArrayEnv, type Dim, type NetResolver, type Shape } from "./shapes";
import { ARRAY_COMPARE_OPS } from "./spec";

/** one requested gradient: d(of)/d(wrt), optionally seeded (vector–Jacobian product) */
export interface GradRequest {
  /** output name of the gradient program */
  name: string;
  of: string;
  wrt: string;
  /** a seed: an input name (existing, or new with the shape of `of`) or a fixed array */
  seed?: string | NdArray;
}

/*******************************************************/
/* expression builders */

type E = ArrayExpr;
const un = (op: ScalarUnaryOp, val: E): E => ({ op, val }) as E;
const add = (...vals: E[]): E => (vals.length === 1 ? vals[0]! : { op: "add", vals });
const mul = (...vals: E[]): E => (vals.length === 1 ? vals[0]! : { op: "mul", vals });
const sub = (a: E, b: E): E => ({ op: "sub", vals: [a, b] });
const div = (a: E, b: E): E => ({ op: "div", vals: [a, b] });
const neg = (a: E): E => un("negate", a);
const where = (cond: E, a: E, b: E): E => ({ op: "where", cond, vals: [a, b] });
const cmp = (op: "lt" | "le" | "gt" | "ge" | "eq" | "ne", a: E, b: E): E => ({ op, vals: [a, b] });
const reduce = (fn: ArrayReduceFn, val: E, axes?: number[], keepDims?: boolean): E => ({ op: "reduce", fn, val, ...(axes ? { axes } : {}), ...(keepDims ? { keepDims } : {}) });
const reshape = (val: E, shape: readonly Dim[]): E => ({ op: "reshape", val, shape: [...shape] });
const transpose = (val: E, perm: number[]): E => ({ op: "transpose", val, perm });
/** zeros with the shape of `x` (broadcasting partner) */
const zerosLike = (x: E): E => mul(x, 0);
const onesLike = (x: E): E => add(mul(x, 0), 1);

const isUnary = (op: string): op is ScalarUnaryOp => (SCALAR_UNARY_OPS as readonly string[]).includes(op);
const isNary = (op: string) => (SCALAR_NARY_OPS as readonly string[]).includes(op);
const isBinary = (op: string) => (SCALAR_BINARY_OPS as readonly string[]).includes(op);
const isCompare = (op: string) => (ARRAY_COMPARE_OPS as readonly string[]).includes(op);

const LN2 = Math.LN2, LN10 = Math.LN10, SQRT2 = Math.SQRT2, INV_SQRT_2PI = 1 / Math.sqrt(2 * Math.PI), TWO_OVER_SQRT_PI = 2 / Math.sqrt(Math.PI);

/** d/dx of a unary op as an expression of x (the input) and y (the output), times g */
const UNARY_D: Record<ScalarUnaryOp, (x: E, y: E, g: E) => E | null> = {
  sin: (x, _y, g) => mul(g, un("cos", x)),
  cos: (x, _y, g) => neg(mul(g, un("sin", x))),
  tan: (_x, y, g) => mul(g, add(1, mul(y, y))),
  sinh: (x, _y, g) => mul(g, un("cosh", x)),
  cosh: (x, _y, g) => mul(g, un("sinh", x)),
  tanh: (_x, y, g) => mul(g, sub(1, mul(y, y))),
  asin: (x, _y, g) => div(g, un("sqrt", sub(1, mul(x, x)))),
  acos: (x, _y, g) => neg(div(g, un("sqrt", sub(1, mul(x, x))))),
  atan: (x, _y, g) => div(g, add(1, mul(x, x))),
  asinh: (x, _y, g) => div(g, un("sqrt", add(mul(x, x), 1))),
  acosh: (x, _y, g) => div(g, un("sqrt", sub(mul(x, x), 1))),
  atanh: (x, _y, g) => div(g, sub(1, mul(x, x))),
  relu: (_x, y, g) => where(cmp("gt", y, 0), g, 0), // y > 0 iff x > 0: reading y lets the pre-activation die early
  sigmoid: (_x, y, g) => mul(g, y, sub(1, y)),
  gelu: (x, _y, g) => mul(g, add(mul(0.5, add(1, un("erf", div(x, SQRT2)))), mul(x, INV_SQRT_2PI, un("gauss", x)))),
  silu: (x, _y, g) => { const s = un("sigmoid", x); return mul(g, add(s, mul(x, s, sub(1, s)))); },
  softplus: (x, _y, g) => mul(g, un("sigmoid", x)),
  elu: (x, _y, g) => mul(g, where(cmp("gt", x, 0), 1, un("exp", x))),
  erf: (x, _y, g) => mul(g, TWO_OVER_SQRT_PI, un("exp", neg(mul(x, x)))),
  floor: () => null, ceil: () => null, round: () => null, sign: () => null,
  abs: (x, _y, g) => mul(g, un("sign", x)),
  exp: (_x, y, g) => mul(g, y),
  exp2: (_x, y, g) => mul(g, y, LN2),
  exp10: (_x, y, g) => mul(g, y, LN10),
  log: (x, _y, g) => div(g, x),
  log2: (x, _y, g) => div(g, mul(x, LN2)),
  log10: (x, _y, g) => div(g, mul(x, LN10)),
  log1p: (x, _y, g) => div(g, add(1, x)),
  expm1: (x, _y, g) => mul(g, un("exp", x)),
  plogp: (x, _y, g) => mul(g, add(un("log", x), 1)),
  sqrt: (_x, y, g) => div(g, mul(2, y)),
  square: (x, _y, g) => mul(g, 2, x),
  negate: (_x, _y, g) => neg(g),
  reciprocal: (_x, y, g) => neg(mul(g, y, y)),
  gauss: (x, y, g) => neg(mul(g, x, y)),
};

/*******************************************************/

const isName = (e: E): e is string => typeof e === "string";
const isNode = (e: E): e is Exclude<E, string | number> => typeof e === "object";

/** letters for einsum terms, avoiding a set */
function letters(n: number, avoid: Set<string>): string[] {
  const out: string[] = [];
  for (let c = 0x61; out.length < n && c < 0x7b; c++) { const l = String.fromCharCode(c); if (!avoid.has(l)) out.push(l); }
  for (let c = 0x41; out.length < n && c < 0x5b; c++) { const l = String.fromCharCode(c); if (!avoid.has(l)) out.push(l); }
  if (out.length < n) throw new NotSupportedError("too many axes for einsum letters");
  return out;
}

const dimEq = (a: Dim, b: Dim) => a === b;

class Rewriter {
  readonly nodes: { name: string; expr: E }[] = [];
  readonly consts: Program["consts"];
  readonly shapes: Record<string, Shape>;
  private n = 0;
  private readonly env: ArrayEnv;

  constructor(prog: Program, nets: NetResolver, private readonly path: string[]) {
    this.consts = { ...prog.consts };
    this.shapes = { ...prog.shapes };
    for (const [k, c] of Object.entries(prog.consts)) this.shapes[k] ??= c.shape;
    const axisNames = new Set<string>();
    for (const s of Object.values(this.shapes)) for (const d of s) if (typeof d === "string") axisNames.add(d);
    const names = new Map<string, Shape>();
    this.env = { names, axisNames, nets };
    for (const [k, s] of Object.entries(this.shapes)) names.set(k, s);
  }

  private fresh(prefix: string): string {
    let nm: string;
    do nm = `${prefix}${this.n++}`; while (nm in this.shapes);
    return nm;
  }

  /** add a node, inferring its shape */
  emit(expr: E, prefix = "_g"): string {
    if (isName(expr)) return expr;
    const name = this.fresh(prefix);
    const shape = inferExpr(expr, this.env, [...this.path, name]);
    this.nodes.push({ name, expr });
    this.shapes[name] = shape;
    (this.env.names as Map<string, Shape>).set(name, shape);
    return name;
  }

  /** declare a (new) input of the gradient program */
  declareInput(name: string, shape: Shape): void {
    this.shapes[name] ??= shape;
    (this.env.names as Map<string, Shape>).set(name, this.shapes[name]!);
  }

  shape(name: string): Shape {
    const s = this.shapes[name];
    if (!s) throw new SpecError(`autodiff: unknown array "${name}"`, this.path);
    return s;
  }

  /** a baked constant; `shape` is its declared shape when the array carries a batch prefix */
  constant(arr: NdArray, prefix = "_c", shape: Shape = [...arr.shape]): string {
    const name = this.fresh(prefix);
    this.consts[name] = { arr, shape };
    this.shapes[name] = shape;
    (this.env.names as Map<string, Shape>).set(name, this.shapes[name]!);
    return name;
  }

  /*******************************************************/
  /* A-normal form: every non-leaf subexpression becomes a node */

  anf(expr: E): E {
    if (!isNode(expr)) return expr;
    const rec = (x: E): E => (isNode(x) ? this.emit(this.anf(x), "_t") : x);
    if (expr.op === "call") return { ...expr, inputs: Object.fromEntries(Object.entries(expr.inputs).map(([k, v]) => [k, rec(v)])) };
    const out = { ...expr } as unknown as Record<string, unknown>;
    for (const k of ["val", "vals", "min", "max", "cond", "indices"]) {
      const v = out[k];
      if (v === undefined) continue;
      out[k] = Array.isArray(v) ? (v as E[]).map(rec) : rec(v as E);
    }
    // matmul -> einsum, so only einsum needs an adjoint rule
    if (expr.op === "matmul") {
      const [a, b] = (out as { vals: [E, E] }).vals;
      const ra = this.shapeOf(a).length, rb = this.shapeOf(b).length;
      if (ra === 0 || rb === 0) throw new SpecError("matmul operands must have rank >= 1", this.path);
      if (ra === 1 && rb === 1) return { op: "einsum", subscripts: "k,k->", vals: [a, b] };
      const L = Math.max(ra, rb) - 2;
      const lead = Array.from({ length: Math.max(L, 0) }, (_, i) => String.fromCharCode(0x41 + i));
      const la = lead.slice(lead.length - Math.max(ra - 2, 0)).join(""), lb = lead.slice(lead.length - Math.max(rb - 2, 0)).join("");
      if (rb === 1) return { op: "einsum", subscripts: `${la}ik,k->${lead.join("")}i`, vals: [a, b] };
      if (ra === 1) return { op: "einsum", subscripts: `k,${lb}kj->${lead.join("")}j`, vals: [a, b] };
      return { op: "einsum", subscripts: `${la}ik,${lb}kj->${lead.join("")}ij`, vals: [a, b] };
    }
    return out as unknown as E;
  }

  private shapeOf(e: E): Shape {
    if (typeof e === "number") return [];
    if (isName(e)) return this.shape(e);
    return inferExpr(e, this.env, this.path);
  }

  /*******************************************************/
  /* adjoints */

  /** sum `g` (shaped like `from`) down to the shape of `to`, undoing numpy broadcasting */
  unbroadcast(g: E, from: Shape, to: Shape): E {
    if (from.length === to.length && from.every((d, i) => dimEq(d, to[i]!))) return g;
    let e = g, cur = from;
    const lead = from.length - to.length;
    if (lead > 0) { e = reduce("sum", e, Array.from({ length: lead }, (_, i) => i)); cur = from.slice(lead); }
    const axes = to.map((d, i) => (d === 1 && cur[i] !== 1 ? i : -1)).filter((i) => i >= 0);
    if (axes.length) e = reduce("sum", e, axes, true);
    return e;
  }

  /** contribution of node `y = expr` (with adjoint `g`, shaped like y) to each name operand: [operand, contribution] */
  backward(y: string, expr: E, g: string): [string, E][] {
    if (!isNode(expr)) return [];
    const out: [string, E][] = [];
    const yShape = this.shape(y);
    const push = (x: E, contrib: E | null, xShape?: Shape) => {
      if (!isName(x) || contrib === null) return; // literals and non-differentiable paths
      const xs = xShape ?? this.shape(x);
      out.push([x, this.unbroadcast(contrib, yShape, xs)]);
    };
    const op = expr.op;
    switch (op) {
      case "arg": push(expr.name, g); return out;
      case "coord": case "coordv": throw new SpecError(`autodiff: coordinate leaves must be folded into the program first`, this.path);
      case "clamp": {
        const { val: x, min: lo, max: hi } = expr;
        push(x, mul(g, cmp("ge", x, lo), cmp("le", x, hi)));
        push(lo, mul(g, cmp("lt", x, lo)));
        push(hi, mul(g, cmp("gt", x, hi)));
        return out;
      }
      case "where": {
        const [a, b] = expr.vals;
        push(a, where(expr.cond, g, 0));
        push(b, where(expr.cond, 0, g));
        return out;
      }
      case "matmul": throw new SpecError("autodiff: matmul must be lowered to einsum (anf)", this.path);
      case "einsum": {
        const text = expr.subscripts.replace(/\s+/g, "");
        const [lhs, rhs] = text.split("->");
        const terms = lhs!.split(",");
        let outL = rhs;
        if (outL === undefined) {
          const count = new Map<string, number>();
          for (const l of lhs!.replace(/,/g, "")) count.set(l, (count.get(l) ?? 0) + 1);
          outL = [...count.entries()].filter(([, c]) => c === 1).map(([l]) => l).sort().join("");
        }
        expr.vals.forEach((x, k) => {
          if (!isName(x)) return;
          const tk = terms[k]!; // a repeated letter here (a diagonal) makes the adjoint's OUTPUT repeat it: the diagonal embedding
          // letters of x that neither the output nor the other operands carry need a ones operand to reappear
          const others = terms.filter((_, j) => j !== k).join("") + outL!;
          const vals: E[] = [g], ts: string[] = [outL!];
          expr.vals.forEach((v, j) => { if (j !== k) { vals.push(v); ts.push(terms[j]!); } });
          if ([...tk].some((l) => !others.includes(l))) { vals.push(onesLike(x)); ts.push(tk); }
          const ge: E = { op: "einsum", subscripts: `${ts.join(",")}->${tk}`, vals };
          // size-1 (broadcast) letters of x come back at full size: sum them
          const full = this.shapeOf(ge);
          out.push([x, this.unbroadcast(ge, full, this.shape(x))]);
        });
        return out;
      }
      case "reduce": {
        const x = expr.val;
        if (!isName(x)) return out;
        const xs = this.shape(x), r = xs.length;
        const axes = expr.axes === undefined ? xs.map((_, i) => i) : [...new Set(expr.axes.map((a) => (a < 0 ? r + a : a)))];
        const keepShape = xs.map((d, i) => (axes.includes(i) ? 1 : d));
        const gk: E = expr.keepDims ? g : reshape(g, keepShape);
        const yk: E = expr.keepDims ? y : reshape(y, keepShape);
        let c: E;
        switch (expr.fn) {
          case "sum": c = mul(gk, onesLike(x)); break;
          case "mean": c = div(mul(gk, onesLike(x)), reduce("sum", onesLike(x), axes, true)); break;
          case "max": case "min": c = mul(gk, cmp("eq", x, yk)); break;
          case "prod": c = mul(gk, div(yk, x)); break;
          case "logsumexp": c = mul(gk, un("exp", sub(x, yk))); break;
        }
        out.push([x, c]);
        return out;
      }
      case "argmax": case "argmin": case "oneHot": return out; // piecewise constant
      case "softmax": case "logSoftmax": {
        const x = expr.val;
        if (!isName(x)) return out;
        const r = this.shape(x).length, ax = (expr.axis ?? -1) < 0 ? r + (expr.axis ?? -1) : expr.axis!;
        if (op === "softmax") out.push([x, mul(y, sub(g, reduce("sum", mul(g, y), [ax], true)))]);
        else out.push([x, sub(g, mul(un("exp", y), reduce("sum", g, [ax], true)))]);
        return out;
      }
      case "reshape": { const x = expr.val; if (isName(x)) out.push([x, reshape(g, this.shape(x))]); return out; }
      case "transpose": {
        const x = expr.val;
        if (!isName(x)) return out;
        const r = this.shape(x).length;
        const perm = expr.perm ?? Array.from({ length: r }, (_, i) => r - 1 - i);
        const inv = new Array<number>(r);
        perm.forEach((p, i) => { inv[p] = i; });
        out.push([x, transpose(g, inv)]);
        return out;
      }
      case "concat": {
        const ax = expr.axis < 0 ? yShape.length + expr.axis : expr.axis;
        let at = 0;
        for (const x of expr.vals) {
          const xs = this.shapeOf(x);
          const n = xs[ax];
          if (typeof n !== "number") throw new SpecError("autodiff: concat along a symbolic axis", this.path);
          if (isName(x)) {
            const part: E = { op: "slice", val: g, axis: ax, start: at, stop: at + n };
            const partShape = yShape.map((d, i) => (i === ax ? n : d));
            out.push([x, this.unbroadcast(part, partShape, xs)]);
          }
          at += n;
        }
        return out;
      }
      case "slice": {
        const x = expr.val;
        if (!isName(x)) return out;
        const xs = this.shape(x), r = xs.length, ax = expr.axis < 0 ? r + expr.axis : expr.axis;
        const n = xs[ax], m = yShape[ax];
        if (typeof n !== "number" || typeof m !== "number") throw new SpecError("autodiff: slice along a symbolic axis", this.path);
        // selection matrix S[j, start + j*step] = 1: y = einsum(x, S) along ax, so dx = einsum(g, S) with S transposed
        const step = expr.step ?? 1;
        const clampIdx = (i: number | undefined, dflt: number, lo: number, hi: number) => (i === undefined ? dflt : Math.min(hi, Math.max(lo, i < 0 ? n + i : i)));
        const start = step > 0 ? clampIdx(expr.start, 0, 0, n) : clampIdx(expr.start, n - 1, -1, n - 1);
        const S = new NdArray([m, n]);
        for (let j = 0; j < m; j++) S.data[j * n + start + j * step] = 1;
        const Sn = this.constant(S, "_S");
        const ls = letters(r + 1, new Set());
        const gL = ls.slice(0, r), a = gL[ax]!, b = ls[r]!;
        const xL = [...gL]; xL[ax] = b;
        out.push([x, { op: "einsum", subscripts: `${gL.join("")},${a}${b}->${xL.join("")}`, vals: [g, Sn] }]);
        return out;
      }
      case "takeAlong": {
        const x = expr.val, idx = expr.indices;
        if (!isName(x)) return out;
        const xs = this.shape(x), r = xs.length, ax = expr.axis < 0 ? r + expr.axis : expr.axis;
        const n = xs[ax];
        if (typeof n !== "number") throw new SpecError("autodiff: takeAlong along a symbolic axis", this.path);
        // dx[.., k, ..] = sum_j g[.., j, ..] [idx[.., j, ..] == k]: oneHot appends k as a trailing axis
        const gE = reshape(g, [...yShape, 1]);
        const prod = mul(gE, { op: "oneHot", val: idx, size: n }); // [..y.., n]
        const summed = reduce("sum", prod, [ax]); // [..y without ax.., n]
        const perm: number[] = []; // move the trailing axis to ax
        for (let i = 0, j = 0; i < r; i++) perm.push(i === ax ? r - 1 : j++);
        const scattered = transpose(summed, perm);
        const full = yShape.map((d, i) => (i === ax ? n : d));
        out.push([x, this.unbroadcast(scattered, full, xs)]);
        return out;
      }
      case "stopGradient": return out;
      case "call": {
        // the gradient of the callee is itself a grad net: call it with the same inputs and the seed
        const outputs: Record<string, { of: string; wrt: string; seed: string }> = {};
        for (const k of Object.keys(expr.inputs)) outputs[`__g_${k}`] = { of: expr.output, wrt: k, seed: "__seed" };
        const gradNet: NetSpec = { type: "grad", net: expr.net, outputs };
        for (const [k, x] of Object.entries(expr.inputs)) {
          if (!isName(x)) continue;
          const ge: E = { op: "call", net: gradNet, inputs: { ...expr.inputs, __seed: g }, output: `__g_${k}` };
          out.push([x, ge]); // shapes match the callee's declared input (batched by vmap like the call itself)
        }
        return out;
      }
      default: {
        if (isUnary(op)) {
          const x = (expr as { val: E }).val;
          if (isName(x)) push(x, UNARY_D[op](x, y, g));
          return out;
        }
        if (isNary(op)) {
          const vals = (expr as { vals: E[] }).vals;
          vals.forEach((x, i) => {
            if (!isName(x)) return;
            const others = vals.filter((_, j) => j !== i);
            switch (op) {
              case "add": push(x, g); break;
              case "mul": push(x, mul(g, ...(others.length ? others : [1]))); break;
              case "min": case "max": push(x, mul(g, cmp("eq", x, y))); break;
              case "mean": push(x, div(g, vals.length)); break;
              case "rms": push(x, mul(g, div(x, mul(vals.length, y)))); break;
            }
          });
          return out;
        }
        if (isBinary(op)) {
          const [a, b] = (expr as { vals: [E, E] }).vals;
          switch (op) {
            case "sub": push(a, g); push(b, neg(g)); break;
            case "div": push(a, div(g, b)); push(b, neg(mul(g, div(y, b)))); break;
            case "pow": push(a, mul(g, b, { op: "pow", vals: [a, sub(b, 1)] })); push(b, mul(g, un("log", a), y)); break;
            case "logBase": push(a, div(g, mul(a, un("log", b)))); push(b, neg(div(mul(g, y), mul(b, un("log", b))))); break;
            case "atan2": { const r2 = add(mul(a, a), mul(b, b)); push(a, mul(g, div(b, r2))); push(b, neg(mul(g, div(a, r2)))); break; }
            case "mod": push(a, g); push(b, neg(mul(g, un("floor", div(a, b))))); break;
          }
          return out;
        }
        if (isCompare(op)) return out;
        throw new SpecError(`autodiff: unknown op "${String(op)}"`, this.path);
      }
    }
  }
}

/*******************************************************/

/**
 * The program in A-normal form: every non-leaf subexpression is its own node
 * (`matmul` lowered to `einsum`). Node names of the original are kept. The
 * WGSL emitter works on this form so per-node liveness sees every intermediate.
 */
export function anfProgram(prog: Program, nets: NetResolver = noNetResolver, path: string[] = ["anf"]): Program {
  const rw = new Rewriter(prog, nets, path);
  for (const n of prog.nodes) rw.nodes.push({ name: n.name, expr: rw.anf(n.expr) });
  return { inputs: prog.inputs, consts: rw.consts, nodes: rw.nodes, outputs: prog.outputs, shapes: rw.shapes };
}

/**
 * Build the gradient program of `prog`: its inputs (plus new seed inputs)
 * to the requested gradients (plus `keep` forward outputs).
 */
export function gradProgram(prog: Program, requests: GradRequest[], keep: string[] = [], nets: NetResolver = noNetResolver, path: string[] = ["grad"]): Program {
  const rw = new Rewriter(prog, nets, path);
  // forward, in A-normal form (node names are kept so `wrt` / `keep` still resolve)
  for (const n of prog.nodes) {
    const expr = rw.anf(n.expr);
    rw.nodes.push({ name: n.name, expr });
    // shapes of the original nodes are known; anf's temporaries were inferred as emitted
  }
  const forward = [...rw.nodes];
  const inputs: Record<string, Shape> = { ...prog.inputs };
  const outputs: Record<string, string> = {};

  // group requests by (of, seed): one backward pass each
  const groups = new Map<string, GradRequest[]>();
  for (const r of requests) {
    const key = `${r.of}|${r.seed === undefined ? "" : typeof r.seed === "string" ? `in:${r.seed}` : `arr:${requests.indexOf(r)}`}`;
    (groups.get(key) ?? groups.set(key, []).get(key)!).push(r);
  }
  for (const group of groups.values()) {
    const { of, seed } = group[0]!;
    const ofNode = prog.outputs[of];
    if (ofNode === undefined) throw new SpecError(`"${of}" is not an output of the net`, path);
    const ofShape = rw.shape(ofNode);
    // the seed
    let g0: E;
    if (seed === undefined) {
      if (ofShape.length !== 0) throw new SpecError(`"${of}" has shape [${ofShape}]; only a scalar output can be differentiated without a seed`, path);
      g0 = 1;
    } else if (typeof seed === "string") {
      if (!(seed in inputs)) inputs[seed] = ofShape;
      rw.declareInput(seed, ofShape);
      g0 = seed;
    } else g0 = rw.constant(seed, "_seed", ofShape);
    // adjoint contributions per name
    const adj = new Map<string, E[]>();
    adj.set(ofNode, [g0]);
    const gradOf = new Map<string, string>(); // name -> node holding its full-shape adjoint
    const finalize = (name: string): string | undefined => {
      const parts = adj.get(name);
      if (!parts || parts.length === 0) return undefined;
      const sum = rw.emit(add(...parts));
      // make the adjoint full-shaped (contributions may have broadcast from smaller shapes)
      const s = rw.shape(sum), want = rw.shape(name);
      const full = s.length === want.length && s.every((d, i) => dimEq(d, want[i]!)) ? sum : rw.emit(add(zerosLike(name), sum));
      gradOf.set(name, full);
      return full;
    };
    for (let i = forward.length - 1; i >= 0; i--) {
      const { name, expr } = forward[i]!;
      const g = finalize(name);
      if (g === undefined) continue;
      for (const [x, contrib] of rw.backward(name, expr, g)) (adj.get(x) ?? adj.set(x, []).get(x)!).push(contrib);
    }
    for (const r of group) {
      if (!(r.wrt in rw.shapes)) throw new SpecError(`"${r.wrt}" is neither an input nor an internal array of the net`, path);
      const g = gradOf.get(r.wrt) ?? finalize(r.wrt) ?? rw.emit(zerosLike(r.wrt));
      outputs[r.name] = g;
    }
  }
  for (const k of keep) {
    const n = prog.outputs[k];
    if (n === undefined) throw new SpecError(`keep: "${k}" is not an output of the net`, path);
    outputs[k] = n;
  }
  return { inputs, consts: rw.consts, nodes: rw.nodes, outputs, shapes: rw.shapes };
}

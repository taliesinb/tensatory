// Normalized expression trees. Names are resolved, named constants are inlined,
// and smart constructors fold constants and drop identities so that the trees
// produced by differentiation stay small.

import type { ScalarBinaryOp, ScalarNaryOp, ScalarUnaryOp } from "@tensatory/schema";
import { UNARY } from "./functions";

export type SExpr =
  | { k: "const"; value: number }
  | { k: "coord"; index: number }
  | { k: "arg"; name: string }
  | { k: "argvi"; name: string; index: number }
  /** (higher) partial derivative of a scalar argument wrt the coordinates `dims`, in order (produced by diff) */
  | { k: "argd"; name: string; dims: number[] }
  /** (higher) partial derivative of a vector argument's component wrt `dims` (produced by diff) */
  | { k: "argvid"; name: string; index: number; dims: number[] }
  | { k: "un"; op: ScalarUnaryOp; a: SExpr }
  | { k: "nary"; op: ScalarNaryOp; args: SExpr[] }
  | { k: "bin"; op: ScalarBinaryOp; a: SExpr; b: SExpr }
  | { k: "clamp"; a: SExpr; lo: SExpr; hi: SExpr }
  | { k: "gaussKernel" | "normalPDF"; a: SExpr; mu: SExpr; sigma: SExpr }
  | { k: "dot" | "cosineSim"; a: VExpr; b: VExpr }
  | { k: "norm"; v: VExpr }
  | { k: "comp"; v: VExpr; index: number }
  /** internal (produced by diff): lt / eq / gt depending on how `test` compares to `ref` */
  | { k: "where"; test: SExpr; ref: SExpr; lt: SExpr; eq: SExpr; gt: SExpr };

export type VExpr =
  | { k: "constv"; value: number[] }
  | { k: "basisv"; index: number }
  | { k: "coordv" }
  | { k: "argv"; name: string }
  | { k: "scalev"; v: VExpr; s: SExpr }
  | { k: "naryv"; op: "addv" | "meanv"; args: VExpr[] }
  | { k: "subv"; a: VExpr; b: VExpr }
  | { k: "sumv"; vecs: VExpr[]; coeffs: SExpr[] }
  | { k: "compv"; comps: SExpr[] }
  | { k: "normalize"; v: VExpr }
  | { k: "grad"; s: SExpr };

/*******************************************************/
/* smart constructors */

export const C = (value: number): SExpr => ({ k: "const", value });
export const ZERO = C(0);
export const ONE = C(1);

export const isConst = (e: SExpr): e is { k: "const"; value: number } => e.k === "const";
/** is `e` the constant `value`? (a plain boolean, so it does not narrow) */
export const isConstVal = (e: SExpr, value: number): boolean => e.k === "const" && e.value === value;

export function un(op: ScalarUnaryOp, a: SExpr): SExpr {
  if (isConst(a)) return C(UNARY[op](a.value));
  if (op === "negate" && a.k === "un" && a.op === "negate") return a.a;
  return { k: "un", op, a };
}

export function add(...args: SExpr[]): SExpr {
  const flat: SExpr[] = [];
  let c = 0;
  for (const a of args) {
    if (isConst(a)) c += a.value;
    else if (a.k === "nary" && a.op === "add") {
      for (const x of a.args) if (isConst(x)) c += x.value; else flat.push(x);
    } else flat.push(a);
  }
  if (c !== 0) flat.push(C(c));
  if (flat.length === 0) return ZERO;
  if (flat.length === 1) return flat[0]!;
  return { k: "nary", op: "add", args: flat };
}

export function mul(...args: SExpr[]): SExpr {
  const flat: SExpr[] = [];
  let c = 1;
  for (const a of args) {
    if (isConst(a)) c *= a.value;
    else if (a.k === "nary" && a.op === "mul") {
      for (const x of a.args) if (isConst(x)) c *= x.value; else flat.push(x);
    } else flat.push(a);
  }
  if (c === 0) return ZERO;
  if (c !== 1) flat.unshift(C(c));
  if (flat.length === 0) return ONE;
  if (flat.length === 1) return flat[0]!;
  return { k: "nary", op: "mul", args: flat };
}

export function nary(op: ScalarNaryOp, args: SExpr[]): SExpr {
  if (op === "add") return add(...args);
  if (op === "mul") return mul(...args);
  if (args.length === 1) return op === "rms" ? un("abs", args[0]!) : args[0]!;
  if (args.every((a) => a.k === "const")) {
    const vs = args.map((a) => (a as { k: "const"; value: number }).value);
    switch (op) {
      case "min": return C(Math.min(...vs));
      case "max": return C(Math.max(...vs));
      case "mean": return C(vs.reduce((s, v) => s + v, 0) / vs.length);
      case "rms": return C(Math.sqrt(vs.reduce((s, v) => s + v * v, 0) / vs.length));
    }
  }
  return { k: "nary", op, args };
}

export const neg = (a: SExpr): SExpr => (isConst(a) ? C(-a.value) : mul(C(-1), a));
export const sub = (a: SExpr, b: SExpr): SExpr => add(a, neg(b));

export function div(a: SExpr, b: SExpr): SExpr {
  if (isConstVal(b, 1)) return a;
  if (isConstVal(a, 0)) return ZERO;
  if (isConst(a) && isConst(b)) return C(a.value / b.value);
  if (isConst(b)) return mul(C(1 / b.value), a);
  return { k: "bin", op: "div", a, b };
}

export function pow(a: SExpr, b: SExpr): SExpr {
  if (isConstVal(b, 0)) return ONE;
  if (isConstVal(b, 1)) return a;
  if (isConstVal(b, 2)) return un("square", a);
  if (isConstVal(b, 0.5)) return un("sqrt", a);
  if (isConstVal(b, -1)) return un("reciprocal", a);
  if (isConst(a) && isConst(b)) return C(Math.pow(a.value, b.value));
  return { k: "bin", op: "pow", a, b };
}

export function bin(op: ScalarBinaryOp, a: SExpr, b: SExpr): SExpr {
  switch (op) {
    case "sub": return sub(a, b);
    case "div": return div(a, b);
    case "pow": return pow(a, b);
    default:
      if (isConst(a) && isConst(b)) return C(evalBinary(op, a.value, b.value));
      return { k: "bin", op, a, b };
  }
}

export function evalBinary(op: ScalarBinaryOp, a: number, b: number): number {
  switch (op) {
    case "sub": return a - b;
    case "div": return a / b;
    case "pow": return Math.pow(a, b);
    case "logBase": return Math.log(a) / Math.log(b);
    case "atan2": return Math.atan2(a, b);
    case "mod": return a - b * Math.floor(a / b);
  }
}

/** does the expression depend on any field argument? */
export function hasArgs(e: SExpr | VExpr): boolean {
  switch (e.k) {
    case "const": case "coord": case "constv": case "basisv": case "coordv": return false;
    case "arg": case "argvi": case "argd": case "argvid": case "argv": return true;
    case "un": return hasArgs(e.a);
    case "nary": return e.args.some(hasArgs);
    case "bin": return hasArgs(e.a) || hasArgs(e.b);
    case "clamp": return hasArgs(e.a) || hasArgs(e.lo) || hasArgs(e.hi);
    case "gaussKernel": case "normalPDF": return hasArgs(e.a) || hasArgs(e.mu) || hasArgs(e.sigma);
    case "dot": case "cosineSim": return hasArgs(e.a) || hasArgs(e.b);
    case "norm": return hasArgs(e.v);
    case "comp": return hasArgs(e.v);
    case "where": return hasArgs(e.test) || hasArgs(e.ref) || hasArgs(e.lt) || hasArgs(e.eq) || hasArgs(e.gt);
    case "scalev": return hasArgs(e.v) || hasArgs(e.s);
    case "naryv": return e.args.some(hasArgs);
    case "subv": return hasArgs(e.a) || hasArgs(e.b);
    case "sumv": return e.vecs.some(hasArgs) || e.coeffs.some(hasArgs);
    case "compv": return e.comps.some(hasArgs);
    case "normalize": return hasArgs(e.v);
    case "grad": return hasArgs(e.s);
  }
}

/** names of the field arguments an expression refers to */
export function argNames(e: SExpr | VExpr, out = { scalars: new Set<string>(), vectors: new Set<string>() }): typeof out {
  const rec = (x: SExpr | VExpr) => argNames(x, out);
  switch (e.k) {
    case "arg": case "argd": out.scalars.add(e.name); break;
    case "argvi": case "argvid": case "argv": out.vectors.add(e.name); break;
    case "un": rec(e.a); break;
    case "nary": e.args.forEach(rec); break;
    case "bin": rec(e.a); rec(e.b); break;
    case "clamp": rec(e.a); rec(e.lo); rec(e.hi); break;
    case "gaussKernel": case "normalPDF": rec(e.a); rec(e.mu); rec(e.sigma); break;
    case "dot": case "cosineSim": rec(e.a); rec(e.b); break;
    case "norm": case "comp": case "normalize": rec(e.v); break;
    case "where": rec(e.test); rec(e.ref); rec(e.lt); rec(e.eq); rec(e.gt); break;
    case "scalev": rec(e.v); rec(e.s); break;
    case "naryv": e.args.forEach(rec); break;
    case "subv": rec(e.a); rec(e.b); break;
    case "sumv": e.vecs.forEach(rec); e.coeffs.forEach(rec); break;
    case "compv": e.comps.forEach(rec); break;
    case "grad": rec(e.s); break;
    default: break;
  }
  return out;
}

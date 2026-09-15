// Spec -> normalized AST: resolve names against the three namespaces, check
// dimension indices, inline named constants.

import type { ScalarBinary, ScalarNary, ScalarUnary, SymbolicScalar, SymbolicVector } from "@tensatory/schema";
import { SpecError } from "../errors";
import { C, bin, nary, un, type SExpr, type VExpr } from "./ast";
import { SCALAR_BINARY_OPS, SCALAR_NARY_OPS, SCALAR_UNARY_OPS } from "./spec";

/** what an expression may refer to */
export interface NameEnv {
  readonly dimCount: number;
  readonly consts: Readonly<Record<string, number>>;
  readonly scalarArgs: ReadonlySet<string>;
  readonly vectorArgs: ReadonlySet<string>;
}

export const emptyEnv = (dimCount: number, consts: Record<string, number> = {}): NameEnv => ({
  dimCount,
  consts,
  scalarArgs: new Set(),
  vectorArgs: new Set(),
});

/** ensure a name is bound in at most one namespace */
export function checkNamespaces(
  consts: Record<string, unknown> = {},
  scalars: Record<string, unknown> = {},
  vectors: Record<string, unknown> = {},
  path: string[] = [],
): void {
  const seen = new Map<string, string>();
  for (const [ns, names] of [
    ["consts", consts],
    ["scalars", scalars],
    ["vectors", vectors],
  ] as const) {
    for (const n of Object.keys(names)) {
      const prev = seen.get(n);
      if (prev) throw new SpecError(`name "${n}" is bound in both ${prev} and ${ns}`, path);
      seen.set(n, ns);
    }
  }
}

const isUnary = (op: string): op is (typeof SCALAR_UNARY_OPS)[number] => (SCALAR_UNARY_OPS as readonly string[]).includes(op);
const isNary = (op: string): op is (typeof SCALAR_NARY_OPS)[number] => (SCALAR_NARY_OPS as readonly string[]).includes(op);
const isBinary = (op: string): op is (typeof SCALAR_BINARY_OPS)[number] => (SCALAR_BINARY_OPS as readonly string[]).includes(op);

function checkDim(index: number, env: NameEnv, path: string[]): number {
  if (!Number.isInteger(index) || index < 0 || index >= env.dimCount)
    throw new SpecError(`dimension index ${index} out of range for ${env.dimCount} dimensions`, path);
  return index;
}

function constByName(name: string, env: NameEnv, path: string[]): SExpr {
  const v = env.consts[name];
  if (v === undefined) throw new SpecError(`unknown constant "${name}"`, path);
  return C(v);
}

export function normalizeScalar(e: SymbolicScalar, env: NameEnv, path: string[] = ["expr"]): SExpr {
  if (typeof e === "number") return C(e);
  if (typeof e === "string") {
    if (env.scalarArgs.has(e)) return { k: "arg", name: e };
    if (e in env.consts) return constByName(e, env, path);
    if (env.vectorArgs.has(e)) throw new SpecError(`"${e}" is a vector argument but a scalar was expected`, path);
    throw new SpecError(`unknown name "${e}" (not a scalar argument or constant)`, path);
  }
  const sub = (x: SymbolicScalar, key: string) => normalizeScalar(x, env, [...path, key]);
  const subv = (x: SymbolicVector, key: string) => normalizeVector(x, env, [...path, key]);
  const op = e.op;
  switch (op) {
    case "const": return constByName(e.name, env, path);
    case "coord": return { k: "coord", index: checkDim(e.index, env, path) };
    case "arg":
      if (!env.scalarArgs.has(e.name)) throw new SpecError(`unknown scalar argument "${e.name}"`, path);
      return { k: "arg", name: e.name };
    case "argvi":
      if (!env.vectorArgs.has(e.name)) throw new SpecError(`unknown vector argument "${e.name}"`, path);
      return { k: "argvi", name: e.name, index: checkDim(e.index, env, path) };
    case "clamp": return { k: "clamp", a: sub(e.val, "val"), lo: sub(e.min, "min"), hi: sub(e.max, "max") };
    case "gaussKernel": case "normalPDF":
      return { k: op, a: sub(e.val, "val"), mu: sub(e.mu, "mu"), sigma: sub(e.sigma, "sigma") };
    case "dot": case "cosineSim": return { k: op, a: subv(e.vecs[0], "vecs.0"), b: subv(e.vecs[1], "vecs.1") };
    case "norm": return { k: "norm", v: subv(e.vec, "vec") };
    case "comp": return { k: "comp", v: subv(e.vec, "vec"), index: checkDim(e.index, env, path) };
    default:
      if (isUnary(op)) return un(op, sub((e as ScalarUnary<SymbolicScalar>).val, "val"));
      if (isNary(op)) return nary(op, (e as ScalarNary<SymbolicScalar>).vals.map((x, i) => sub(x, `vals.${i}`)));
      if (isBinary(op)) {
        const { vals } = e as ScalarBinary<SymbolicScalar>;
        return bin(op, sub(vals[0], "vals.0"), sub(vals[1], "vals.1"));
      }
      throw new SpecError(`unknown scalar op "${String(op)}"`, path);
  }
}

export function normalizeVector(e: SymbolicVector, env: NameEnv, path: string[] = ["expr"]): VExpr {
  if (typeof e === "string") {
    if (env.vectorArgs.has(e)) return { k: "argv", name: e };
    if (env.scalarArgs.has(e) || e in env.consts) throw new SpecError(`"${e}" is not a vector argument`, path);
    throw new SpecError(`unknown vector argument "${e}"`, path);
  }
  const sub = (x: SymbolicScalar, key: string) => normalizeScalar(x, env, [...path, key]);
  const subv = (x: SymbolicVector, key: string) => normalizeVector(x, env, [...path, key]);
  switch (e.op) {
    case "constv":
      if (e.value.length !== env.dimCount)
        throw new SpecError(`constant vector has ${e.value.length} components, expected ${env.dimCount}`, path);
      return { k: "constv", value: [...e.value] };
    case "basisv": return { k: "basisv", index: checkDim(e.index, env, path) };
    case "coordv": return { k: "coordv" };
    case "argv":
      if (!env.vectorArgs.has(e.name)) throw new SpecError(`unknown vector argument "${e.name}"`, path);
      return { k: "argv", name: e.name };
    case "scalev": return { k: "scalev", v: subv(e.vec, "vec"), s: sub(e.by, "by") };
    case "addv": case "meanv": return { k: "naryv", op: e.op, args: e.vecs.map((x, i) => subv(x, `vecs.${i}`)) };
    case "subv": return { k: "subv", a: subv(e.vecs[0], "vecs.0"), b: subv(e.vecs[1], "vecs.1") };
    case "sumv":
      if (e.vecs.length !== e.coeffs.length)
        throw new SpecError(`sumv has ${e.vecs.length} vectors but ${e.coeffs.length} coefficients`, path);
      return { k: "sumv", vecs: e.vecs.map((x, i) => subv(x, `vecs.${i}`)), coeffs: e.coeffs.map((x, i) => sub(x, `coeffs.${i}`)) };
    case "compv":
      if (e.coeffs.length !== env.dimCount)
        throw new SpecError(`compv has ${e.coeffs.length} components, expected ${env.dimCount}`, path);
      return { k: "compv", comps: e.coeffs.map((x, i) => sub(x, `coeffs.${i}`)) };
    case "normalize": return { k: "normalize", v: subv(e.vec, "vec") };
    case "grad": return { k: "grad", s: sub(e.val, "val") };
  }
}

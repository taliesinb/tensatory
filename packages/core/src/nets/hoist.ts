// Hoisting: the point-independent part of a net field's program, computed
// once instead of once per point (notes/nets.md "Step 0, measured").
//
// A displaced weight is a node `W__disp = W + einsum("k,k…->…", t, Dd)` (t the
// coefficients — the point — W and the stacked directions Dd constants). A
// contraction that is LINEAR in it against a CONSTANT operand distributes:
//
//   matmul(X, W + Σ tₖ Dₖ)  =  X·W + Σ tₖ (X·Dₖ)  =  A + einsum("k,k…->…", t, XD)
//
// with A = X·W and XD = stack_k X·Dₖ constants folded here on the CPU. For the
// MNIST MLP the first layer (x·W₁, 75 % of the MACs) becomes a read of A and
// XD per example; the input to the rest of the net is 256 wide instead of
// 784, which is what lets the cooperative kernel's tiles fit. Every other node
// whose operands are all constants is folded too (up to FOLD_MAX_ELEMENTS),
// then dead nodes and constants are pruned.
//
// Applied to FIELD programs (whose sole input is the point): nobody asks such
// a program for gradients with respect to an internal array, so folding a
// weight into A loses nothing; `grad` of the field with respect to the point
// differentiates the hoisted program (through the einsum with t). Semantics
// are unchanged up to floating-point summation order.

import type { ArrayExpr } from "@tensatory/schema";
import type { NdArray } from "../arrays/ndarray";
import { evaluate, pruneProgram, type Program, type ProgramResolver } from "./program";
import { exprNames, inferExpr, type Shape } from "./shapes";

const EXPR_KEYS = ["val", "vals", "min", "max", "cond", "indices"] as const;

/** largest constant (elements) folding may create */
export const FOLD_MAX_ELEMENTS = 1 << 22;

const hoisted = new WeakMap<Program, Program>();

/** the hoisted program (memoized per program object) */
export function hoistProgram(prog: Program, nets: ProgramResolver): Program {
  let h = hoisted.get(prog);
  if (!h) hoisted.set(prog, (h = hoist(prog, nets)));
  return h;
}

const nameOf = (x: ArrayExpr): string | undefined => (typeof x === "string" ? x : typeof x === "object" && x.op === "arg" ? x.name : undefined);

/** `add(W, einsum("k,kL->L", t, Dd))` with W, Dd constants and t not: a displaced constant */
interface Disp { base: string; coeffs: ArrayExpr; dirs: string }

function matchDisp(e: ArrayExpr, isConst: (n: string) => boolean): Disp | undefined {
  if (typeof e !== "object" || e.op !== "add" || e.vals.length !== 2) return undefined;
  for (const [a, b] of [[e.vals[0]!, e.vals[1]!], [e.vals[1]!, e.vals[0]!]] as const) {
    const base = nameOf(a);
    if (base === undefined || !isConst(base)) continue;
    if (typeof b !== "object" || b.op !== "einsum" || b.vals.length !== 2) continue;
    const m = /^([a-zA-Z]),([a-zA-Z])([a-zA-Z]*)->([a-zA-Z]*)$/.exec(b.subscripts.replace(/\s+/g, ""));
    if (!m || m[1] !== m[2] || m[3] !== m[4] || m[3]!.includes(m[1]!)) continue;
    const dirs = nameOf(b.vals[1]!);
    if (dirs === undefined || !isConst(dirs)) continue;
    const tn = nameOf(b.vals[0]!);
    if (tn !== undefined && isConst(tn)) continue; // the whole node folds instead
    return { base, coeffs: b.vals[0]!, dirs };
  }
  return undefined;
}

/** a two-operand contraction as einsum terms / output letters (matmul converted as the evaluator does) */
function contraction(e: ArrayExpr, rankOf: (x: ArrayExpr) => number | undefined): { terms: [string, string]; out: string; vals: [ArrayExpr, ArrayExpr] } | undefined {
  if (typeof e !== "object") return undefined;
  if (e.op === "matmul") {
    const [a, b] = e.vals;
    const ra = rankOf(a), rb = rankOf(b);
    if (ra === undefined || rb === undefined || ra === 0 || rb === 0) return undefined;
    if (ra === 1 && rb === 1) return { terms: ["k", "k"], out: "", vals: [a, b] };
    const L = Math.max(ra, rb) - 2;
    const lead = Array.from({ length: Math.max(L, 0) }, (_, i) => String.fromCharCode(0x41 + i));
    const la = lead.slice(lead.length - Math.max(ra - 2, 0)).join(""), lb = lead.slice(lead.length - Math.max(rb - 2, 0)).join("");
    if (rb === 1) return { terms: [la + "ik", "k"], out: lead.join("") + "i", vals: [a, b] };
    if (ra === 1) return { terms: ["k", lb + "kj"], out: lead.join("") + "j", vals: [a, b] };
    return { terms: [la + "ik", lb + "kj"], out: lead.join("") + "ij", vals: [a, b] };
  }
  if (e.op === "einsum" && e.vals.length === 2) {
    const text = e.subscripts.replace(/\s+/g, "");
    const [lhs, rhs] = text.split("->");
    const terms = lhs!.split(",");
    if (terms.length !== 2) return undefined;
    let out = rhs;
    if (out === undefined) {
      const count = new Map<string, number>();
      for (const l of lhs!.replace(/,/g, "")) count.set(l, (count.get(l) ?? 0) + 1);
      out = [...count.entries()].filter(([, c]) => c === 1).map(([l]) => l).sort().join("");
    }
    return { terms: [terms[0]!, terms[1]!], out, vals: [e.vals[0]!, e.vals[1]!] };
  }
  return undefined;
}

function hoist(prog: Program, nets: ProgramResolver): Program {
  const consts: Program["consts"] = { ...prog.consts };
  const shapes: Record<string, Shape> = { ...prog.shapes };
  const isConst = (n: string) => n in consts;
  const taken = new Set([...Object.keys(prog.inputs), ...Object.keys(prog.consts), ...prog.nodes.map((n) => n.name), ...Object.keys(prog.shapes)]);
  const fresh = (base: string) => { let n = base; for (let k = 1; taken.has(n); k++) n = `${base}_${k}`; taken.add(n); return n; };
  // concrete sizes of symbolic axes, from the constants
  const sizes = new Map<string, number>();
  for (const c of Object.values(consts)) c.shape.forEach((d, i) => { if (typeof d === "string" && !sizes.has(d)) sizes.set(d, c.arr.shape[i]!); });
  const elements = (shape: Shape | undefined): number | undefined => {
    if (!shape) return undefined;
    let n = 1;
    for (const d of shape) { const s = typeof d === "number" ? d : sizes.get(d); if (s === undefined) return undefined; n *= s; }
    return n;
  };
  /** evaluate an expression over the constants (a mini program) */
  const fold = (expr: ArrayExpr): NdArray => evaluate({ inputs: {}, consts, nodes: [{ name: "__fold", expr }], outputs: { __fold: "__fold" }, shapes: {} }, {}, { nets }, ["hoist"])["__fold"]!;
  const addConst = (base: string, expr: ArrayExpr, shape: Shape): string => {
    const name = fresh(base);
    consts[name] = { arr: fold(expr), shape };
    shapes[name] = shape;
    return name;
  };
  const byName = new Map<string, ArrayExpr>();
  const rankOf = (x: ArrayExpr): number | undefined => { const n = nameOf(x); return n === undefined ? undefined : shapes[n]?.length; };
  const axisNames = new Set(sizes.keys());
  const shapeOf = (e: ArrayExpr): Shape | undefined => {
    try { return inferExpr(e, { names: new Map(Object.entries(shapes)), axisNames, nets }, ["hoist"]); } catch { return undefined; }
  };

  /** distributivity, applied to every contraction SUBEXPRESSION of a constant with a displaced constant (the matmul
   *  usually sits inside `relu(add(matmul(x, W), b))`) */
  const distribute = (e: ArrayExpr, node: string): ArrayExpr => {
    if (typeof e !== "object") return e;
    if (e.op === "call") return { ...e, inputs: Object.fromEntries(Object.entries(e.inputs).map(([k, v]) => [k, distribute(v, node)])) };
    const out = { ...e } as unknown as Record<string, unknown>;
    for (const key of EXPR_KEYS) {
      const v = out[key];
      if (v === undefined) continue;
      out[key] = Array.isArray(v) ? (v as ArrayExpr[]).map((x) => distribute(x, node)) : distribute(v as ArrayExpr, node);
    }
    const expr = out as unknown as ArrayExpr;
    const c = contraction(expr, rankOf);
    if (!c) return expr;
    const names = c.vals.map(nameOf);
    for (const i of [0, 1] as const) {
      const w = names[i], x = names[1 - i];
      if (w === undefined || x === undefined || !isConst(x)) continue;
      const dispExpr = byName.get(w);
      const disp = dispExpr && matchDisp(dispExpr, isConst);
      if (!disp) continue;
      const outShape = shapeOf(expr);
      const K = consts[disp.dirs]!.arr.shape[0]!;
      if (!outShape || (elements(outShape) ?? Infinity) * (K + 1) > FOLD_MAX_ELEMENTS) continue;
      const used = new Set([...c.terms[0], ...c.terms[1], ...c.out]);
      const k = [..."abcdefghijklmnopqrstuvwxyz"].find((l) => !used.has(l))!;
      const withW = (val: string, term: string, o: string): ArrayExpr => {
        const terms = [...c.terms] as [string, string], vals = [...c.vals] as [ArrayExpr, ArrayExpr];
        terms[i] = term; vals[i] = val;
        return { op: "einsum", subscripts: `${terms[0]},${terms[1]}->${o}`, vals };
      };
      const A = addConst(`${node}__h`, withW(disp.base, c.terms[i], c.out), outShape);
      const XD = addConst(`${node}__hd`, withW(disp.dirs, k + c.terms[i], k + c.out), [K, ...outShape]);
      return { op: "add", vals: [A, { op: "einsum", subscripts: `${k},${k}${c.out}->${c.out}`, vals: [disp.coeffs, XD] }] };
    }
    return expr;
  };

  const nodes: Program["nodes"] = [];
  for (const node of prog.nodes) {
    const expr = distribute(node.expr, node.name);
    // constant folding: a node over constants only
    const deps = [...exprNames(expr)];
    if (deps.length > 0 && deps.every(isConst) && typeof expr === "object" && expr.op !== "coord" && expr.op !== "coordv") {
      const n = elements(shapes[node.name]);
      if (n !== undefined && n <= FOLD_MAX_ELEMENTS) {
        consts[node.name] = { arr: fold(expr), shape: shapes[node.name]! };
        continue;
      }
    }
    byName.set(node.name, expr);
    nodes.push({ name: node.name, expr });
  }
  return pruneProgram({ inputs: prog.inputs, consts, nodes, outputs: prog.outputs, shapes }, Object.keys(prog.outputs));
}

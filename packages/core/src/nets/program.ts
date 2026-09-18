// Compile a NetSpec into a flat PROGRAM (inputs, constant arrays, nodes in
// evaluation order, outputs) and evaluate it on the CPU (see ops.ts).
//
// `def`      -> the program as written
// `bind`     -> the inner program with the bound inputs turned into constants
// `displace` -> the inner program with, for every displaced array A, a node
//               A__disp = A + einsum("k,k...->...", coeffs, directions_A) and
//               every later use of A renamed to A__disp
// `grad`     -> the forward program in A-normal form plus adjoint nodes
//               (autodiff.ts): an ordinary program made of ordinary ops
// `call`     -> evaluated recursively; the callee sees the caller's batch as
//               extra leading axes, so vmap composes by itself

import type { ArrayExpr, ArraySpec, DisplacedNetSpec, NetDefinitionSpec, NetSpec } from "@tensatory/schema";
import { NdArray } from "../arrays/ndarray";
import { buildArray } from "../arrays/spec";
import { EvalError, NotSupportedError, SpecError } from "../errors";
import { SCALAR_BINARY_OPS, SCALAR_NARY_OPS, SCALAR_UNARY_OPS } from "../symbolic/spec";
import { UNARY } from "../symbolic/functions";
import * as ops from "./ops";
import { type Val, scalar } from "./ops";
import { ARRAY_COMPARE_OPS } from "./spec";
import { gradProgram, type GradRequest } from "./autodiff";
import { coeffsName, exprNames, inferNet, type NetResolver, type Shape } from "./shapes";

export interface Program {
  /** declared per-example shapes of the inputs */
  readonly inputs: Record<string, Shape>;
  /** constant arrays: baked `arrays` and bound inputs, with their declared shapes */
  readonly consts: Record<string, { arr: NdArray; shape: Shape }>;
  /** internal nodes in evaluation order */
  readonly nodes: { name: string; expr: ArrayExpr }[];
  /** output name -> internal name */
  readonly outputs: Record<string, string>;
  /** declared shape of every input, const and node */
  readonly shapes: Record<string, Shape>;
}

/** resolves nets by id to programs (and signatures) */
export interface ProgramResolver extends NetResolver {
  program(id: string, path: string[]): Program;
}

/*******************************************************/
/* compile */

export function compileNet(spec: NetSpec, nets: ProgramResolver, path: string[] = ["net"]): Program {
  switch (spec.type) {
    case "def": return compileDef(spec, nets, path);
    case "bind": return bindProgram(resolveProgram(spec.net, nets, [...path, "net"]), spec.bind, [...path, "bind"]);
    case "displace": return displaceProgram(resolveProgram(spec.net, nets, [...path, "net"]), spec, nets, path);
    case "grad": {
      let inner = resolveProgram(spec.net, nets, [...path, "net"]);
      if (spec.bind) inner = bindProgram(inner, spec.bind, [...path, "bind"]);
      const requests: GradRequest[] = Object.entries(spec.outputs).map(([name, g]) => ({
        name, of: g.of, wrt: g.wrt,
        ...(g.seed === undefined ? {} : { seed: typeof g.seed === "string" ? g.seed : buildSized(g.seed, [...path, "outputs", name, "seed"]) }),
      }));
      return gradProgram(inner, requests, spec.keep ?? [], nets, path);
    }
  }
}

const resolveProgram = (net: string | NetSpec, nets: ProgramResolver, path: string[]): Program =>
  typeof net === "string" ? nets.program(net, path) : compileNet(net, nets, path);

function compileDef(spec: NetDefinitionSpec, nets: ProgramResolver, path: string[]): Program {
  const sig = inferNet(spec, nets, path); // validates names, shapes, acyclicity
  const consts: Program["consts"] = {};
  for (const [n, a] of Object.entries(spec.arrays ?? {})) consts[n] = { arr: buildArray(a, [...path, "arrays", n]), shape: a.shape };
  // evaluation order: depth-first over dependencies
  const nodeSpecs = spec.nodes ?? {};
  const order: Program["nodes"] = [];
  const done = new Set<string>();
  const visit = (n: string) => {
    if (done.has(n)) return;
    done.add(n);
    for (const dep of exprNames(nodeSpecs[n]!)) if (dep in nodeSpecs) visit(dep);
    order.push({ name: n, expr: nodeSpecs[n]! });
  };
  for (const n of Object.keys(nodeSpecs)) visit(n);
  const shapes: Record<string, Shape> = { ...sig.inputs, ...sig.nodes };
  return { inputs: { ...sig.inputs }, consts, nodes: order, outputs: Object.fromEntries(Object.keys(spec.outputs).map((o) => [o, o])), shapes };
}

function bindProgram(inner: Program, bind: Record<string, ArraySpec>, path: string[]): Program {
  const inputs = { ...inner.inputs };
  const consts = { ...inner.consts };
  for (const [n, a] of Object.entries(bind)) {
    const shape = inputs[n];
    if (!shape) throw new SpecError(`"${n}" is not an input of the net`, [...path, n]);
    consts[n] = { arr: buildSized(a, [...path, n]), shape };
    delete inputs[n];
  }
  return { ...inner, inputs, consts };
}

function buildSized(a: ArraySpec, path: string[]): NdArray {
  if (typeof a === "string" || !("shape" in a)) throw new NotSupportedError(`external arrays are not loaded yet`, path);
  return buildArray(a, path);
}

const DISP = "__disp";

function displaceProgram(inner: Program, spec: DisplacedNetSpec, nets: ProgramResolver, path: string[]): Program {
  inferNet(spec, nets, path); // validates targets and direction shapes
  const t = coeffsName(spec);
  const K = spec.directions.length;
  const targets = [...new Set(spec.directions.flatMap((d) => Object.keys(d.arrays)))];
  const consts = { ...inner.consts };
  const shapes: Record<string, Shape> = { ...inner.shapes, [t]: [K] };
  for (const a of targets) {
    const shape = inner.shapes[a]!;
    if (shape.some((d) => typeof d === "string")) throw new SpecError(`cannot displace "${a}": its shape ${JSON.stringify(shape)} has symbolic sizes`, [...path, "directions"]);
  }
  // per direction: its arrays, built once, and the factor `norm` / `scale` apply to the direction as ONE vector
  const built = spec.directions.map((dir, k) => {
    const p = [...path, "directions", String(k)];
    const arrays = new Map(Object.entries(dir.arrays).map(([a, d]) => [a, buildSized(d, [...p, "arrays", a])] as const));
    let factor = dir.scale ?? 1;
    if (dir.norm !== undefined) {
      const sq = (arr: NdArray) => { let s = 0; for (const v of arr.data) s += v * v; return s; };
      let target: number;
      if (dir.norm === "origin") {
        // the joint norm of the displaced arrays' own values: constants only (a node has no value before evaluation)
        let s = 0;
        for (const a of arrays.keys()) {
          const c = inner.consts[a];
          if (!c) throw new SpecError(`norm "origin" needs "${a}" to be a bound input or a baked array of the net, not an input or a node`, [...p, "norm"]);
          s += sq(c.arr);
        }
        target = Math.sqrt(s);
      } else target = dir.norm;
      let s = 0;
      for (const arr of arrays.values()) s += sq(arr);
      const have = Math.sqrt(s);
      if (!(have > 0)) throw new SpecError(`direction ${k} is zero, it cannot be normalized`, [...p, "norm"]);
      if (!(target > 0)) throw new SpecError(`direction ${k}: the target norm is ${target}`, [...p, "norm"]);
      factor *= target / have;
    }
    return { arrays, factor };
  });
  // stacked directions per target: [K, ...shape], zero where a direction omits the target
  for (const a of targets) {
    const dims = inner.shapes[a] as number[];
    const n = dims.reduce((p, s) => p * s, 1);
    const stacked = new NdArray([K, ...dims]);
    built.forEach(({ arrays, factor }, k) => {
      const d = arrays.get(a);
      if (!d) return;
      const off = k * n;
      for (let i = 0; i < n; i++) stacked.data[off + i] = factor * d.data[i]!;
    });
    const dn = `${a}${DISP}d`;
    if (dn in shapes) throw new SpecError(`internal name "${dn}" is taken`, path);
    consts[dn] = { arr: stacked, shape: [K, ...dims] };
    shapes[dn] = [K, ...dims];
    shapes[`${a}${DISP}`] = dims;
  }
  const dispExpr = (a: string): ArrayExpr => {
    const letters = Array.from({ length: inner.shapes[a]!.length }, (_, i) => String.fromCharCode(0x61 + i)).join(""); // a b c ...
    return { op: "add", vals: [a, { op: "einsum", subscripts: `k,k${letters}->${letters}`, vals: [t, `${a}${DISP}d`] }] };
  };
  const rename = new Map(targets.map((a) => [a, `${a}${DISP}`] as const));
  const nodes: Program["nodes"] = [];
  // displaced inputs / constants first
  for (const a of targets) if (!inner.nodes.some((n) => n.name === a)) nodes.push({ name: `${a}${DISP}`, expr: dispExpr(a) });
  for (const n of inner.nodes) {
    nodes.push({ name: n.name, expr: renameExpr(n.expr, rename) });
    if (rename.has(n.name)) nodes.push({ name: `${n.name}${DISP}`, expr: dispExpr(n.name) });
  }
  const outputs = Object.fromEntries(Object.entries(inner.outputs).map(([o, n]) => [o, rename.get(n) ?? n]));
  return { inputs: { ...inner.inputs, [t]: [K] }, consts, nodes, outputs, shapes };
}

const EXPR_KEYS = ["val", "vals", "min", "max", "cond", "indices"] as const;

/** substitute array names */
export function renameExpr(e: ArrayExpr, map: ReadonlyMap<string, string>): ArrayExpr {
  if (typeof e === "number") return e;
  if (typeof e === "string") return map.get(e) ?? e;
  if (e.op === "arg") return { op: "arg", name: map.get(e.name) ?? e.name };
  if (e.op === "call") return { ...e, inputs: Object.fromEntries(Object.entries(e.inputs).map(([k, v]) => [k, renameExpr(v, map)])) };
  const out = { ...e } as unknown as Record<string, unknown>;
  for (const k of EXPR_KEYS) {
    const v = out[k];
    if (v === undefined) continue;
    out[k] = Array.isArray(v) ? (v as ArrayExpr[]).map((x) => renameExpr(x, map)) : renameExpr(v as ArrayExpr, map);
  }
  return out as unknown as ArrayExpr;
}

/*******************************************************/
/* evaluate */

const isUnary = (op: string): op is (typeof SCALAR_UNARY_OPS)[number] => (SCALAR_UNARY_OPS as readonly string[]).includes(op);
const isNary = (op: string): op is (typeof SCALAR_NARY_OPS)[number] => (SCALAR_NARY_OPS as readonly string[]).includes(op);
const isBinary = (op: string): op is (typeof SCALAR_BINARY_OPS)[number] => (SCALAR_BINARY_OPS as readonly string[]).includes(op);
const isCompare = (op: string): op is (typeof ARRAY_COMPARE_OPS)[number] => (ARRAY_COMPARE_OPS as readonly string[]).includes(op);

const NARY: Record<(typeof SCALAR_NARY_OPS)[number], (xs: number[]) => number> = {
  add: (xs) => xs.reduce((a, b) => a + b, 0),
  mul: (xs) => xs.reduce((a, b) => a * b, 1),
  min: (xs) => Math.min(...xs),
  max: (xs) => Math.max(...xs),
  mean: (xs) => xs.reduce((a, b) => a + b, 0) / xs.length,
  rms: (xs) => Math.sqrt(xs.reduce((a, b) => a + b * b, 0) / xs.length),
};
const BINARY: Record<(typeof SCALAR_BINARY_OPS)[number], (a: number, b: number) => number> = {
  sub: (a, b) => a - b,
  div: (a, b) => a / b,
  pow: Math.pow,
  logBase: (a, b) => Math.log(a) / Math.log(b),
  atan2: Math.atan2,
  mod: (a, b) => ((a % b) + b) % b,
};
const COMPARE: Record<(typeof ARRAY_COMPARE_OPS)[number], (a: number, b: number) => number> = {
  lt: (a, b) => (a < b ? 1 : 0), le: (a, b) => (a <= b ? 1 : 0), gt: (a, b) => (a > b ? 1 : 0),
  ge: (a, b) => (a >= b ? 1 : 0), eq: (a, b) => (a === b ? 1 : 0), ne: (a, b) => (a !== b ? 1 : 0),
};

/** what an evaluation may refer to beyond the program's own names */
export interface EvalContext {
  readonly nets: ProgramResolver;
  /** the sample points [G, D] when `coord` / `coordv` are allowed (net-backed field inputs) */
  readonly coords?: NdArray;
}

/**
 * Evaluate a program. Inputs are given with their batch prefixes; symbolic
 * sizes are bound from the actual arrays. Returns the outputs (with batch).
 */
export function evaluate(prog: Program, inputs: Record<string, NdArray>, ctx: EvalContext, path: string[] = []): Record<string, NdArray> {
  const env = new Map<string, Val>();
  const sizes = new Map<string, number>();
  const bindShape = (name: string, arr: NdArray, declared: Shape) => {
    if (arr.ndim < declared.length) throw new EvalError(`"${name}" needs rank >= ${declared.length}, got shape [${arr.shape}]`, path);
    const off = arr.ndim - declared.length;
    declared.forEach((d, i) => {
      const s = arr.shape[off + i]!;
      if (typeof d === "number") { if (d !== s) throw new EvalError(`"${name}": axis ${i} declared ${d}, got ${s}`, path); }
      else {
        const prev = sizes.get(d);
        if (prev === undefined) sizes.set(d, s);
        else if (prev !== s) throw new EvalError(`"${name}": axis "${d}" is ${prev} elsewhere but ${s} here`, path);
      }
    });
    env.set(name, { arr, rank: declared.length });
  };
  for (const [n, shape] of Object.entries(prog.inputs)) {
    const arr = inputs[n];
    if (!arr) throw new EvalError(`input "${n}" not given`, path);
    bindShape(n, arr, shape);
  }
  for (const [n, extra] of Object.entries(inputs)) if (!(n in prog.inputs)) throw new EvalError(`"${n}" is not an input of the net (${Object.keys(prog.inputs).join(", ") || "none"}); got shape [${extra.shape}]`, path);
  for (const [n, c] of Object.entries(prog.consts)) bindShape(n, c.arr, c.shape);
  const size = (d: number | string, p: string[]): number => {
    if (typeof d === "number") return d;
    const s = sizes.get(d);
    if (s === undefined) throw new EvalError(`symbolic size "${d}" is not bound`, p);
    return s;
  };
  const ev = (e: ArrayExpr, p: string[]): Val => evalExpr(e, env, size, ctx, p);
  for (const n of prog.nodes) env.set(n.name, ev(n.expr, [...path, "nodes", n.name]));
  return Object.fromEntries(Object.entries(prog.outputs).map(([o, n]) => [o, env.get(n)!.arr]));
}

/** evaluate one expression against named values (exported for net-backed field inputs) */
export function evalExpr(e: ArrayExpr, env: ReadonlyMap<string, Val>, size: (d: number | string, p: string[]) => number, ctx: EvalContext, path: string[]): Val {
  if (typeof e === "number") return scalar(e);
  if (typeof e === "string") return lookup(e, env, path);
  const sub = (x: ArrayExpr, key: string) => evalExpr(x, env, size, ctx, [...path, key]);
  const norm = (axis: number, rank: number) => (axis < 0 ? rank + axis : axis);
  const op = e.op;
  switch (op) {
    case "arg": return lookup(e.name, env, path);
    case "coord": {
      if (!ctx.coords) throw new EvalError(`"coord" outside a field context`, path);
      const G = ctx.coords.shape[0]!, D = ctx.coords.shape[1]!;
      const out = new NdArray([G]);
      for (let g = 0; g < G; g++) out.data[g] = ctx.coords.data[g * D + e.index]!;
      return { arr: out, rank: 0 };
    }
    case "coordv":
      if (!ctx.coords) throw new EvalError(`"coordv" outside a field context`, path);
      return { arr: ctx.coords, rank: 1 };
    case "clamp": return ops.mapN([sub(e.val, "val"), sub(e.min, "min"), sub(e.max, "max")], (x) => Math.min(x[2]!, Math.max(x[1]!, x[0]!)), "clamp");
    case "where": return ops.mapN([sub(e.cond, "cond"), sub(e.vals[0], "vals.0"), sub(e.vals[1], "vals.1")], (x) => (x[0] !== 0 ? x[1]! : x[2]!), "where");
    case "matmul": return ops.matmul(sub(e.vals[0], "vals.0"), sub(e.vals[1], "vals.1"));
    case "einsum": {
      const text = e.subscripts.replace(/\s+/g, "");
      const [lhs, rhs] = text.split("->");
      const terms = lhs!.split(",");
      const vals = e.vals.map((v, i) => sub(v, `vals.${i}`));
      let out = rhs;
      if (out === undefined) {
        const count = new Map<string, number>();
        for (const l of lhs!.replace(/,/g, "")) count.set(l, (count.get(l) ?? 0) + 1);
        out = [...count.entries()].filter(([, c]) => c === 1).map(([l]) => l).sort().join("");
      }
      return ops.einsum(terms, out, vals);
    }
    case "reduce": {
      const v = sub(e.val, "val");
      const axes = e.axes === undefined ? v.arr.shape.slice(v.arr.ndim - v.rank).map((_, i) => i) : [...new Set(e.axes.map((a) => norm(a, v.rank)))];
      return ops.reduce(v, e.fn, axes, e.keepDims ?? false);
    }
    case "argmax": case "argmin": {
      const v = sub(e.val, "val");
      return ops.argReduce(v, norm(e.axis ?? -1, v.rank), op === "argmax");
    }
    case "softmax": case "logSoftmax": {
      const v = sub(e.val, "val");
      return ops.softmax(v, norm(e.axis ?? -1, v.rank), op === "logSoftmax");
    }
    case "reshape": {
      const v = sub(e.val, "val");
      const known = e.shape.filter((d) => d !== -1).map((d) => size(d, [...path, "shape"]));
      const n = v.arr.shape.slice(v.arr.ndim - v.rank).reduce((a, b) => a * b, 1);
      const k = known.reduce((a, b) => a * b, 1);
      const target = e.shape.map((d) => (d === -1 ? n / k : size(d, [...path, "shape"])));
      return ops.reshape(v, target);
    }
    case "transpose": {
      const v = sub(e.val, "val");
      return ops.transpose(v, e.perm ?? Array.from({ length: v.rank }, (_, i) => v.rank - 1 - i));
    }
    case "concat": {
      const vals = e.vals.map((v, i) => sub(v, `vals.${i}`));
      return ops.concat(vals, norm(e.axis, vals[0]!.rank));
    }
    case "slice": {
      const v = sub(e.val, "val");
      return ops.slice(v, norm(e.axis, v.rank), e.start, e.stop, e.step ?? 1);
    }
    case "oneHot": return ops.oneHot(sub(e.val, "val"), size(e.size, [...path, "size"]));
    case "takeAlong": {
      const v = sub(e.val, "val");
      return ops.takeAlong(v, sub(e.indices, "indices"), norm(e.axis, v.rank));
    }
    case "stopGradient": return sub(e.val, "val");
    case "call": {
      const callee = typeof e.net === "string" ? ctx.nets.program(e.net, [...path, "net"]) : compileNet(e.net, ctx.nets, [...path, "net"]);
      const vals = Object.fromEntries(Object.entries(e.inputs).map(([k, v]) => [k, sub(v, `inputs.${k}`)]));
      const outs = evaluate(callee, Object.fromEntries(Object.entries(vals).map(([k, v]) => [k, v.arr])), { nets: ctx.nets }, [...path, "call"]);
      const internal = callee.outputs[e.output];
      if (internal === undefined) throw new EvalError(`called net has no output "${e.output}"`, path);
      // axes the caller passes beyond the callee's declared rank are a CALL-SITE batch: in the caller they are part of
      // the result's declared shape (shape inference folds them in), only the caller's own batch stays implicit
      const callSiteBatch = Math.max(0, ...Object.entries(vals).map(([k, v]) => v.rank - (callee.inputs[k]?.length ?? v.rank)));
      return { arr: outs[e.output]!, rank: callee.shapes[internal]!.length + callSiteBatch };
    }
    default:
      if (isUnary(op)) return ops.map1(sub((e as { val: ArrayExpr }).val, "val"), UNARY[op]);
      if (isNary(op)) return ops.mapN((e as { vals: ArrayExpr[] }).vals.map((v, i) => sub(v, `vals.${i}`)), NARY[op], op);
      if (isBinary(op)) { const [a, b] = (e as { vals: [ArrayExpr, ArrayExpr] }).vals; return ops.map2(sub(a, "vals.0"), sub(b, "vals.1"), BINARY[op], op); }
      if (isCompare(op)) { const [a, b] = (e as { vals: [ArrayExpr, ArrayExpr] }).vals; return ops.map2(sub(a, "vals.0"), sub(b, "vals.1"), COMPARE[op], op); }
      throw new EvalError(`unknown array op "${String(op)}"`, path);
  }
}

function lookup(name: string, env: ReadonlyMap<string, Val>, path: string[]): Val {
  const v = env.get(name);
  if (!v) throw new EvalError(`unknown name "${name}"`, path);
  return v;
}

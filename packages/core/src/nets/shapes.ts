// Shape inference for nets (see @tensatory/schema nets.ts and notes/nets.md).
//
// A net's SIGNATURE is the static part of its meaning: the per-example shape
// of every remaining input and output, the shapes of its internal nodes (the
// arrays `grad` may differentiate with respect to), and the batch prefix its
// bound arrays contribute. Shapes are lists of Dim = number | AxisName; an
// AxisName is a symbolic size local to the net that declared it. Binding an
// array or calling a net unifies declared symbolic sizes with actual ones.
//
// Symbolic sizes are opaque: two different names are never assumed equal, and a
// symbolic size is never assumed to equal a number other than through
// broadcasting with 1. Whatever cannot be proven is an error; nets are meant
// to be checked once, at parse time, not to fail with NaNs at sample time.

import type {
  ArrayExpr,
  ArraySpec,
  BoundNetSpec,
  DisplacedNetSpec,
  GradNetSpec,
  NetDefinitionSpec,
  NetScalarFieldDataSpec,
  NetSpec,
  NetVectorFieldDataSpec,
} from "@tensatory/schema";
import { NotSupportedError, SpecError } from "../errors";
import { SCALAR_BINARY_OPS, SCALAR_NARY_OPS, SCALAR_UNARY_OPS } from "../symbolic/spec";
import { ARRAY_COMPARE_OPS } from "./spec";

export type Dim = number | string;
export type Shape = readonly Dim[];

export interface NetSignature {
  /** remaining inputs, per-example shapes */
  readonly inputs: Readonly<Record<string, Shape>>;
  /** per-example output shapes (the batch prefix goes in front) */
  readonly outputs: Readonly<Record<string, Shape>>;
  /** internal arrays (forward nodes; for grad nets also the forward outputs) — valid `wrt` targets */
  readonly nodes: Readonly<Record<string, Shape>>;
  /** batch prefix contributed by bound arrays; outputs are batch + declared */
  readonly batch: Shape;
}

/** resolves net ids referenced from specs (by id) */
export interface NetResolver {
  net(id: string, path: string[]): NetSignature;
}

export const noNetResolver: NetResolver = {
  net: (id, path) => { throw new SpecError(`cannot resolve net reference "${id}" outside a bundle`, path); },
};

/*******************************************************/
/* dims and shapes */

export const fmtShape = (s: Shape): string => `[${s.map((d) => (typeof d === "string" ? `"${d}"` : d)).join(", ")}]`;

const dimEq = (a: Dim, b: Dim): boolean => a === b;

function broadcastDim(a: Dim, b: Dim, path: string[], what: () => string): Dim {
  if (dimEq(a, b)) return a;
  if (a === 1) return b;
  if (b === 1) return a;
  throw new SpecError(`${what()}: sizes ${fmt(a)} and ${fmt(b)} do not broadcast`, path);
}
const fmt = (d: Dim) => (typeof d === "string" ? `"${d}"` : String(d));

/** numpy broadcasting: right-aligned, 1 stretches */
export function broadcastShapes(shapes: Shape[], path: string[], what = () => "broadcast"): Dim[] {
  const rank = Math.max(0, ...shapes.map((s) => s.length));
  const out = new Array<Dim>(rank).fill(1);
  for (const s of shapes) {
    const off = rank - s.length;
    for (let i = 0; i < s.length; i++) out[off + i] = broadcastDim(out[off + i]!, s[i]!, path, () => `${what()} of ${shapes.map(fmtShape).join(" and ")}`);
  }
  return out;
}

function normAxis(axis: number, rank: number, path: string[]): number {
  const a = axis < 0 ? rank + axis : axis;
  if (!Number.isInteger(a) || a < 0 || a >= rank) throw new SpecError(`axis ${axis} out of range for rank ${rank}`, path);
  return a;
}

/** symbolic-size bindings made while unifying declared shapes with actual ones */
export type Subst = Map<string, Dim>;

/** unify a declared per-example shape with an actual shape of the same rank */
function unifyShape(declared: Shape, actual: Shape, subst: Subst, path: string[], what: string): void {
  if (declared.length !== actual.length)
    throw new SpecError(`${what}: expected rank ${declared.length} ${fmtShape(declared)}, got ${fmtShape(actual)}`, path);
  for (let i = 0; i < declared.length; i++) {
    const d = declared[i]!, a = actual[i]!;
    if (typeof d === "number") {
      if (!dimEq(d, a)) throw new SpecError(`${what}: axis ${i} declared ${d}, got ${fmt(a)} (${fmtShape(declared)} vs ${fmtShape(actual)})`, path);
    } else {
      const bound = subst.get(d);
      if (bound === undefined) subst.set(d, a);
      else if (!dimEq(bound, a)) throw new SpecError(`${what}: axis "${d}" is ${fmt(bound)} elsewhere but ${fmt(a)} here`, path);
    }
  }
}

/** split an actual shape into (batch prefix, per-example suffix) against a declared rank */
function splitBatch(declared: Shape, actual: Shape, path: string[], what: string): { batch: Shape; example: Shape } {
  if (actual.length < declared.length)
    throw new SpecError(`${what}: expected at least rank ${declared.length} ${fmtShape(declared)}, got ${fmtShape(actual)}`, path);
  const cut = actual.length - declared.length;
  return { batch: actual.slice(0, cut), example: actual.slice(cut) };
}

const applySubst = (shape: Shape, subst: Subst): Dim[] => shape.map((d) => (typeof d === "string" ? subst.get(d) ?? d : d));
const mapShapes = (rec: Readonly<Record<string, Shape>>, f: (s: Shape) => Shape): Record<string, Shape> =>
  Object.fromEntries(Object.entries(rec).map(([k, s]) => [k, f(s)]));

/** the shape of an array spec, when it is known without loading */
function arraySpecShape(spec: ArraySpec, path: string[]): Shape {
  if (typeof spec === "string") throw new NotSupportedError(`array "${spec}" has no declared shape; external arrays are not loaded yet`, path);
  if (!("shape" in spec)) throw new NotSupportedError(`array handle "${spec.path}" has no declared shape; external arrays are not loaded yet`, path);
  return spec.shape;
}

/*******************************************************/
/* array expressions */

/** what an array expression may refer to */
export interface ArrayEnv {
  /** shape of every name in scope (inputs, nodes, constant arrays) */
  readonly names: ReadonlyMap<string, Shape>;
  /** symbolic axis names that may be used in reshape / oneHot */
  readonly axisNames: ReadonlySet<string>;
  /** dimension of the field's manifold when `coord` / `coordv` are allowed; undefined inside a net */
  readonly coordDims?: number;
  readonly nets: NetResolver;
}

const isUnary = (op: string) => (SCALAR_UNARY_OPS as readonly string[]).includes(op);
const isNary = (op: string) => (SCALAR_NARY_OPS as readonly string[]).includes(op);
const isBinary = (op: string) => (SCALAR_BINARY_OPS as readonly string[]).includes(op);
const isCompare = (op: string) => (ARRAY_COMPARE_OPS as readonly string[]).includes(op);

/** infer the (per-example) shape of an array expression */
export function inferExpr(e: ArrayExpr, env: ArrayEnv, path: string[] = ["expr"]): Shape {
  if (typeof e === "number") return [];
  if (typeof e === "string") return lookup(e, env, path);
  const sub = (x: ArrayExpr, key: string) => inferExpr(x, env, [...path, key]);
  const bc = (shapes: Shape[]) => broadcastShapes(shapes, path, () => `"${e.op}"`);
  const op = e.op;
  switch (op) {
    case "arg": return lookup(e.name, env, path);
    case "coord":
      if (env.coordDims === undefined) throw new SpecError(`"coord" is only allowed in the inputs of a net-backed field, not inside a net`, path);
      if (!Number.isInteger(e.index) || e.index < 0 || e.index >= env.coordDims) throw new SpecError(`coordinate index ${e.index} out of range for ${env.coordDims} dimensions`, path);
      return [];
    case "coordv":
      if (env.coordDims === undefined) throw new SpecError(`"coordv" is only allowed in the inputs of a net-backed field, not inside a net`, path);
      return [env.coordDims];
    case "clamp": return bc([sub(e.val, "val"), sub(e.min, "min"), sub(e.max, "max")]);
    case "where": return bc([sub(e.cond, "cond"), sub(e.vals[0], "vals.0"), sub(e.vals[1], "vals.1")]);
    case "matmul": return matmulShape(sub(e.vals[0], "vals.0"), sub(e.vals[1], "vals.1"), path);
    case "einsum": return einsumShape(e.subscripts, e.vals.map((v, i) => sub(v, `vals.${i}`)), path);
    case "reduce": {
      const s = sub(e.val, "val");
      const axes = e.axes === undefined ? s.map((_, i) => i) : [...new Set(e.axes.map((a) => normAxis(a, s.length, [...path, "axes"])))];
      return e.keepDims ? s.map((d, i) => (axes.includes(i) ? 1 : d)) : s.filter((_, i) => !axes.includes(i));
    }
    case "argmax": case "argmin": {
      const s = sub(e.val, "val");
      const a = normAxis(e.axis ?? -1, s.length, [...path, "axis"]);
      return s.filter((_, i) => i !== a);
    }
    case "softmax": case "logSoftmax": {
      const s = sub(e.val, "val");
      normAxis(e.axis ?? -1, s.length, [...path, "axis"]);
      return s;
    }
    case "reshape": return reshapeShape(sub(e.val, "val"), e.shape, env, path);
    case "transpose": {
      const s = sub(e.val, "val");
      const perm = e.perm ?? s.map((_, i) => s.length - 1 - i);
      if (perm.length !== s.length || [...perm].sort((a, b) => a - b).some((p, i) => p !== i))
        throw new SpecError(`perm [${perm}] is not a permutation of ${s.length} axes`, [...path, "perm"]);
      return perm.map((p) => s[p]!);
    }
    case "concat": {
      const shapes = e.vals.map((v, i) => sub(v, `vals.${i}`));
      const rank = shapes[0]!.length;
      const a = normAxis(e.axis, rank, [...path, "axis"]);
      let total: Dim = 0;
      for (const [i, s] of shapes.entries()) {
        if (s.length !== rank) throw new SpecError(`concat operands must have the same rank: ${fmtShape(shapes[0]!)} vs ${fmtShape(s)}`, [...path, `vals.${i}`]);
        for (let d = 0; d < rank; d++)
          if (d !== a && !dimEq(s[d]!, shapes[0]![d]!)) throw new SpecError(`concat operands differ on axis ${d}: ${fmtShape(shapes[0]!)} vs ${fmtShape(s)}`, [...path, `vals.${i}`]);
        const n = s[a]!;
        if (typeof n === "string") throw new SpecError(`cannot concat along symbolic axis "${n}"`, [...path, `vals.${i}`]);
        total = (total as number) + n;
      }
      return shapes[0]!.map((d, i) => (i === a ? total : d));
    }
    case "slice": {
      const s = sub(e.val, "val");
      const a = normAxis(e.axis, s.length, [...path, "axis"]);
      const n = s[a]!;
      if (typeof n === "string") throw new SpecError(`cannot slice along symbolic axis "${n}"`, path);
      return s.map((d, i) => (i === a ? sliceLength(n, e.start, e.stop, e.step, path) : d));
    }
    case "oneHot": {
      if (typeof e.size === "string" && !env.axisNames.has(e.size)) throw new SpecError(`unknown axis name "${e.size}"`, [...path, "size"]);
      return [...sub(e.val, "val"), e.size];
    }
    case "takeAlong": {
      const s = sub(e.val, "val"), idx = sub(e.indices, "indices");
      if (s.length !== idx.length) throw new SpecError(`takeAlong: val ${fmtShape(s)} and indices ${fmtShape(idx)} must have the same rank`, path);
      const a = normAxis(e.axis, s.length, [...path, "axis"]);
      const rest = broadcastShapes([s.filter((_, i) => i !== a), idx.filter((_, i) => i !== a)], path, () => `takeAlong`);
      rest.splice(a, 0, idx[a]!);
      return rest;
    }
    case "stopGradient": return sub(e.val, "val");
    case "call": return callShape(e.net, e.inputs, e.output, env, path);
    default:
      if (isUnary(op)) return sub((e as { val: ArrayExpr }).val, "val");
      if (isNary(op)) return bc((e as { vals: ArrayExpr[] }).vals.map((v, i) => sub(v, `vals.${i}`)));
      if (isBinary(op) || isCompare(op)) return bc((e as { vals: [ArrayExpr, ArrayExpr] }).vals.map((v, i) => sub(v, `vals.${i}`)));
      throw new SpecError(`unknown array op "${String(op)}"`, path);
  }
}

function lookup(name: string, env: ArrayEnv, path: string[]): Shape {
  const s = env.names.get(name);
  if (!s) throw new SpecError(`unknown name "${name}"`, path);
  return s;
}

function matmulShape(a: Shape, b: Shape, path: string[]): Shape {
  if (a.length === 0 || b.length === 0) throw new SpecError(`matmul operands must have rank >= 1, got ${fmtShape(a)} and ${fmtShape(b)}`, path);
  const a2 = a.length === 1 ? [1, a[0]!] : a, b2 = b.length === 1 ? [b[0]!, 1] : b;
  const k1 = a2[a2.length - 1]!, k2 = b2[b2.length - 2]!;
  if (!dimEq(k1, k2)) throw new SpecError(`matmul: inner sizes ${fmt(k1)} and ${fmt(k2)} differ (${fmtShape(a)} · ${fmtShape(b)})`, path);
  const lead = broadcastShapes([a2.slice(0, -2), b2.slice(0, -2)], path, () => "matmul leading axes");
  const out: Dim[] = [...lead];
  if (a.length > 1) out.push(a2[a2.length - 2]!);
  if (b.length > 1) out.push(b2[b2.length - 1]!);
  return out;
}

function einsumShape(subscripts: string, shapes: Shape[], path: string[]): Shape {
  const p = [...path, "subscripts"];
  const text = subscripts.replace(/\s+/g, "");
  if (text.includes("...")) throw new SpecError(`einsum "..." is not supported: batch axes are implicit`, p);
  const parts = text.split("->");
  if (parts.length > 2) throw new SpecError(`einsum has more than one "->"`, p);
  const ins = parts[0]!.split(",");
  if (ins.length !== shapes.length) throw new SpecError(`einsum has ${ins.length} operand terms but ${shapes.length} operands`, p);
  const sizes = new Map<string, Dim>();
  const count = new Map<string, number>();
  ins.forEach((term, i) => {
    if (!/^[A-Za-z]*$/.test(term)) throw new SpecError(`einsum term "${term}" must be letters only`, p);
    const s = shapes[i]!;
    if (term.length !== s.length) throw new SpecError(`einsum term "${term}" has ${term.length} letters for operand ${i} of shape ${fmtShape(s)}`, p);
    for (let j = 0; j < term.length; j++) {
      const l = term[j]!, d = s[j]!;
      const prev = sizes.get(l);
      if (prev !== undefined && !dimEq(prev, d)) throw new SpecError(`einsum letter "${l}" is ${fmt(prev)} and ${fmt(d)}`, p);
      sizes.set(l, d);
      count.set(l, (count.get(l) ?? 0) + 1);
    }
  });
  let out: string;
  if (parts.length === 2) {
    out = parts[1]!;
    if (!/^[A-Za-z]*$/.test(out)) throw new SpecError(`einsum output "${out}" must be letters only`, p);
    for (const l of out) if (!sizes.has(l)) throw new SpecError(`einsum output letter "${l}" does not appear in the operands`, p);
  } else {
    out = [...count.entries()].filter(([, c]) => c === 1).map(([l]) => l).sort().join("");
  }
  return [...out].map((l) => sizes.get(l)!);
}

/** product of a shape as (numeric factor, sorted symbolic factors) */
function factor(shape: Shape): { num: number; sym: string[] } {
  let num = 1;
  const sym: string[] = [];
  for (const d of shape) typeof d === "number" ? (num *= d) : sym.push(d);
  return { num, sym: sym.sort() };
}

function reshapeShape(s: Shape, target: (number | string)[], env: ArrayEnv, path: string[]): Shape {
  const p = [...path, "shape"];
  const holes = target.filter((d) => d === -1).length;
  if (holes > 1) throw new SpecError(`reshape has more than one -1`, p);
  for (const d of target) if (typeof d === "string" && !env.axisNames.has(d)) throw new SpecError(`unknown axis name "${d}"`, p);
  const src = factor(s), dst = factor(target.filter((d) => d !== -1));
  if (src.sym.join() !== dst.sym.join())
    throw new SpecError(`cannot reshape ${fmtShape(s)} to ${fmtShape(target)}: symbolic sizes differ${holes ? " (-1 can only absorb numeric sizes)" : ""}`, p);
  if (holes === 0) {
    if (src.num !== dst.num) throw new SpecError(`cannot reshape ${fmtShape(s)} (${src.num} elements) to ${fmtShape(target)} (${dst.num})`, p);
    return target;
  }
  if (dst.num === 0 || src.num % dst.num !== 0) throw new SpecError(`cannot reshape ${fmtShape(s)} to ${fmtShape(target)}: ${src.num} is not a multiple of ${dst.num}`, p);
  return target.map((d) => (d === -1 ? src.num / dst.num : d));
}

/** length of a python slice over an axis of size n */
function sliceLength(n: number, start: number | undefined, stop: number | undefined, step: number | undefined, path: string[]): number {
  const st = step ?? 1;
  if (st === 0) throw new SpecError(`slice step must not be 0`, path);
  const clampIdx = (i: number | undefined, dflt: number, lo: number, hi: number) => {
    if (i === undefined) return dflt;
    const j = i < 0 ? n + i : i;
    return Math.min(hi, Math.max(lo, j));
  };
  let a: number, b: number;
  if (st > 0) { a = clampIdx(start, 0, 0, n); b = clampIdx(stop, n, 0, n); }
  else { a = clampIdx(start, n - 1, -1, n - 1); b = clampIdx(stop, -1, -1, n - 1); }
  const len = st > 0 ? Math.ceil((b - a) / st) : Math.ceil((a - b) / -st);
  return Math.max(0, len);
}

/** shape of `output` of a net called with the given per-example input shapes (extra leading axes batch) */
function callShape(net: string | NetSpec, inputs: Record<string, ArrayExpr>, output: string, env: ArrayEnv, path: string[]): Shape {
  const sig = typeof net === "string" ? env.nets.net(net, [...path, "net"]) : inferNet(net, env.nets, [...path, "net"]);
  const shapes = Object.fromEntries(Object.entries(inputs).map(([k, v]) => [k, inferExpr(v, env, [...path, "inputs", k])]));
  const { batch, subst } = applyInputs(sig, shapes, [...path, "inputs"], "call");
  const missing = Object.keys(sig.inputs).filter((k) => !(k in shapes));
  if (missing.length) throw new SpecError(`call is missing inputs ${missing.map((m) => `"${m}"`).join(", ")}`, [...path, "inputs"]);
  const out = sig.outputs[output];
  if (!out) throw new SpecError(`net has no output "${output}" (outputs: ${Object.keys(sig.outputs).join(", ")})`, [...path, "output"]);
  return [...batch, ...applySubst(out, subst)];
}

/**
 * Match actual shapes against a signature's declared inputs: every name must
 * be an input, the actual shape must end with the declared one (unifying
 * symbolic sizes), and the extra leading axes broadcast into a batch prefix
 * together with the signature's own batch.
 */
function applyInputs(sig: NetSignature, shapes: Record<string, Shape>, path: string[], what: string): { batch: Dim[]; subst: Subst } {
  const subst: Subst = new Map();
  const batches: Shape[] = [sig.batch];
  for (const [name, actual] of Object.entries(shapes)) {
    const declared = sig.inputs[name];
    if (!declared) throw new SpecError(`"${name}" is not an input of the net (inputs: ${Object.keys(sig.inputs).join(", ") || "none"})`, [...path, name]);
    const { batch, example } = splitBatch(declared, actual, [...path, name], `${what} "${name}"`);
    unifyShape(declared, example, subst, [...path, name], `${what} "${name}"`);
    batches.push(batch);
  }
  return { batch: broadcastShapes(batches, path, () => `${what} batch axes`), subst };
}

/*******************************************************/
/* nets */

/** infer the signature of a net spec */
export function inferNet(spec: NetSpec, nets: NetResolver = noNetResolver, path: string[] = ["net"]): NetSignature {
  switch (spec.type) {
    case "def": return inferDef(spec, nets, path);
    case "bind": return inferBind(spec, nets, path);
    case "displace": return inferDisplace(spec, nets, path);
    case "grad": return inferGrad(spec, nets, path);
  }
}

/** the name of the coefficient input of a displaced net */
export const coeffsName = (spec: DisplacedNetSpec): string => spec.coeffs ?? "t";

function inferDisplace(spec: DisplacedNetSpec, nets: NetResolver, path: string[]): NetSignature {
  const sig = resolve(spec.net, nets, [...path, "net"]);
  const t = coeffsName(spec);
  if (t in sig.inputs || t in sig.nodes || t in sig.outputs)
    throw new SpecError(`coefficient input "${t}" collides with an array of the net; choose another \`coeffs\` name`, [...path, "coeffs"]);
  const K = spec.directions.length;
  spec.directions.forEach((dir, k) => {
    for (const [n, a] of Object.entries(dir)) {
      const p = [...path, "directions", String(k), n];
      const target = sig.inputs[n] ?? sig.nodes[n];
      if (!target) throw new SpecError(`"${n}" is neither an input nor an internal array of the net`, p);
      const s = arraySpecShape(a, p);
      if (s.length !== target.length || s.some((d, i) => !dimEq(d, target[i]!)))
        throw new SpecError(`direction ${k} of "${n}" has shape ${fmtShape(s)}, "${n}" has ${fmtShape(target)}`, p);
    }
  });
  return { inputs: { ...sig.inputs, [t]: [K] }, outputs: sig.outputs, nodes: sig.nodes, batch: sig.batch };
}

function inferDef(spec: NetDefinitionSpec, nets: NetResolver, path: string[]): NetSignature {
  const names = new Map<string, Shape>();
  const seen = new Map<string, string>();
  const claim = (n: string, ns: string) => {
    const prev = seen.get(n);
    if (prev) throw new SpecError(`name "${n}" is both ${prev === ns ? `declared twice in ${ns}` : `in ${prev} and ${ns}`}`, [...path, ns, n]);
    seen.set(n, ns);
  };
  const axisNames = new Set<string>();
  for (const [n, s] of Object.entries(spec.inputs)) {
    claim(n, "inputs");
    names.set(n, s);
    for (const d of s) if (typeof d === "string") axisNames.add(d);
  }
  for (const [n, a] of Object.entries(spec.arrays ?? {})) { claim(n, "arrays"); names.set(n, a.shape); }
  const nodes = spec.nodes ?? {};
  for (const n of Object.keys(nodes)) claim(n, "nodes");

  const env: ArrayEnv = { names, axisNames, nets };
  const building = new Set<string>();
  const visit = (n: string): Shape => {
    const done = names.get(n);
    if (done) return done;
    const e = nodes[n];
    if (e === undefined) throw new SpecError(`unknown name "${n}"`, [...path, "nodes"]);
    if (building.has(n)) throw new SpecError(`node "${n}" depends on itself (${[...building, n].join(" -> ")})`, [...path, "nodes", n]);
    building.add(n);
    // resolve dependencies first so a node may be defined after its users
    for (const dep of exprNames(e)) if (!names.has(dep) && dep in nodes) visit(dep);
    const s = inferExpr(e, env, [...path, "nodes", n]);
    building.delete(n);
    names.set(n, s);
    return s;
  };
  const nodeShapes: Record<string, Shape> = {};
  for (const [n, a] of Object.entries(spec.arrays ?? {})) nodeShapes[n] = a.shape; // constants are internal arrays too
  for (const n of Object.keys(nodes)) nodeShapes[n] = visit(n);

  const outputs: Record<string, Shape> = {};
  for (const [n, declared] of Object.entries(spec.outputs)) {
    const p = [...path, "outputs", n];
    const s = names.get(n);
    if (!s) throw new SpecError(`output "${n}" is neither an input nor a node`, p);
    if (seen.get(n) === "arrays") throw new SpecError(`output "${n}" is a constant array`, p);
    if (declared.length !== s.length || declared.some((d, i) => !dimEq(d, s[i]!)))
      throw new SpecError(`output "${n}" is declared ${fmtShape(declared)} but has shape ${fmtShape(s)}`, p);
    outputs[n] = s;
  }
  return { inputs: { ...spec.inputs }, outputs, nodes: nodeShapes, batch: [] };
}

/** names referenced directly by an expression (not descending into called nets) */
export function exprNames(e: ArrayExpr, out = new Set<string>()): Set<string> {
  if (typeof e === "number") return out;
  if (typeof e === "string") { out.add(e); return out; }
  if (e.op === "arg") { out.add(e.name); return out; }
  if (e.op === "call") { for (const x of Object.values(e.inputs)) exprNames(x, out); return out; }
  const rec = e as unknown as Record<string, ArrayExpr | ArrayExpr[]>;
  for (const k of ["val", "vals", "min", "max", "cond", "indices"]) {
    const v = rec[k];
    if (v === undefined) continue;
    if (Array.isArray(v)) v.forEach((x) => exprNames(x, out));
    else exprNames(v, out);
  }
  return out;
}

/** bind arrays to inputs of a signature */
function bindSignature(sig: NetSignature, bind: Record<string, ArraySpec>, path: string[]): NetSignature {
  const shapes = Object.fromEntries(Object.entries(bind).map(([k, a]) => [k, arraySpecShape(a, [...path, k])]));
  const { batch, subst } = applyInputs(sig, shapes, path, "bind");
  const inputs = Object.fromEntries(Object.entries(sig.inputs).filter(([k]) => !(k in bind)));
  // a bound input becomes an internal array (still a valid `wrt`: the gradient at fixed weights)
  const bound = Object.fromEntries(Object.entries(sig.inputs).filter(([k]) => k in bind));
  return {
    inputs: mapShapes(inputs, (s) => applySubst(s, subst)),
    outputs: mapShapes(sig.outputs, (s) => applySubst(s, subst)),
    nodes: mapShapes({ ...sig.nodes, ...bound }, (s) => applySubst(s, subst)),
    batch,
  };
}

const resolve = (net: string | NetSpec, nets: NetResolver, path: string[]): NetSignature =>
  typeof net === "string" ? nets.net(net, path) : inferNet(net, nets, path);

function inferBind(spec: BoundNetSpec, nets: NetResolver, path: string[]): NetSignature {
  return bindSignature(resolve(spec.net, nets, [...path, "net"]), spec.bind, [...path, "bind"]);
}

function inferGrad(spec: GradNetSpec, nets: NetResolver, path: string[]): NetSignature {
  let sig = resolve(spec.net, nets, [...path, "net"]);
  if (spec.bind) sig = bindSignature(sig, spec.bind, [...path, "bind"]);
  const inputs: Record<string, Shape> = { ...sig.inputs };
  const outputs: Record<string, Shape> = {};
  const subst: Subst = new Map();
  const batches: Shape[] = [sig.batch];
  for (const [name, g] of Object.entries(spec.outputs)) {
    const p = [...path, "outputs", name];
    const of = sig.outputs[g.of];
    if (!of) throw new SpecError(`"${g.of}" is not an output of the net (outputs: ${Object.keys(sig.outputs).join(", ")})`, [...p, "of"]);
    const wrt = sig.inputs[g.wrt] ?? sig.nodes[g.wrt];
    if (!wrt) throw new SpecError(`"${g.wrt}" is neither an input nor an internal node of the net`, [...p, "wrt"]);
    if (g.seed === undefined) {
      if (of.length !== 0) throw new SpecError(`"${g.of}" has shape ${fmtShape(of)}; only a scalar output can be differentiated without a seed`, [...p, "of"]);
    } else if (typeof g.seed === "string") {
      const existing = inputs[g.seed];
      if (existing) {
        if (existing.length !== of.length || existing.some((d, i) => !dimEq(d, of[i]!)))
          throw new SpecError(`seed "${g.seed}" has shape ${fmtShape(existing)} but "${g.of}" has ${fmtShape(of)}`, [...p, "seed"]);
      } else if (g.seed in sig.nodes || g.seed in sig.outputs) {
        throw new SpecError(`seed "${g.seed}" names an internal array; a seed must be an input (existing or new)`, [...p, "seed"]);
      } else {
        inputs[g.seed] = of; // a new input of the gradient net
      }
    } else {
      const { batch, example } = splitBatch(of, arraySpecShape(g.seed, [...p, "seed"]), [...p, "seed"], `seed for "${g.of}"`);
      unifyShape(of, example, subst, [...p, "seed"], `seed for "${g.of}"`);
      batches.push(batch);
    }
    if (name in outputs) throw new SpecError(`gradient output "${name}" is defined twice`, p);
    outputs[name] = wrt;
  }
  for (const k of spec.keep ?? []) {
    const s = sig.outputs[k];
    if (!s) throw new SpecError(`keep: "${k}" is not an output of the net`, [...path, "keep"]);
    if (k in outputs) throw new SpecError(`"${k}" is both a kept output and a gradient output`, [...path, "keep"]);
    outputs[k] = s;
  }
  // the forward outputs are internal arrays of the gradient net
  return {
    inputs,
    outputs,
    nodes: { ...sig.nodes, ...sig.outputs },
    batch: broadcastShapes(batches, [...path, "outputs"], () => "seed batch axes"),
  };
}

/*******************************************************/
/* net-backed field data */

/**
 * Check a net-backed field: every remaining input of the net is given (by an
 * expression of the coordinates or by the frame), shapes agree, no batch axes
 * sneak in, and the chosen output has the field's shape. Returns the output's
 * per-example shape.
 */
export function inferNetField(
  spec: NetScalarFieldDataSpec | NetVectorFieldDataSpec,
  dimCount: number,
  nets: NetResolver = noNetResolver,
  path: string[] = ["data"],
): Shape {
  const sig = resolve(spec.net, nets, [...path, "net"]);
  if (sig.batch.length) throw new SpecError(`the net's bound arrays add batch axes ${fmtShape(sig.batch)}; a field needs one value per point — reduce over a declared axis inside the net`, [...path, "net"]);
  const arrays = new Map<string, Shape>();
  for (const [n, a] of Object.entries(spec.arrays ?? {})) arrays.set(n, arraySpecShape(a, [...path, "arrays", n]));
  const env: ArrayEnv = { names: arrays, axisNames: new Set(), coordDims: dimCount, nets };
  const given: Record<string, Shape> = {};
  for (const [n, e] of Object.entries(spec.inputs ?? {})) given[n] = inferExpr(e, env, [...path, "inputs", n]);
  if (spec.inputs === undefined) {
    // default: a net with one remaining input is a map of the point itself
    const names = Object.keys(sig.inputs);
    if (names.length !== 1)
      throw new SpecError(`no \`inputs\`: only a net with exactly one remaining input can take the point directly, this one has ${names.length ? names.map((n) => `"${n}"`).join(", ") : "none"}`, path);
    given[names[0]!] = [dimCount];
  }
  const { batch, subst } = applyInputs(sig, given, path, "input");
  if (batch.length) throw new SpecError(`inputs add batch axes ${fmtShape(batch)}; a field needs one value per point`, path);
  const missing = Object.keys(sig.inputs).filter((k) => !(k in given));
  if (missing.length) throw new SpecError(`net inputs ${missing.map((m) => `"${m}"`).join(", ")} are not given`, path);
  const out = sig.outputs[spec.output];
  if (!out) throw new SpecError(`net has no output "${spec.output}" (outputs: ${Object.keys(sig.outputs).join(", ")})`, [...path, "output"]);
  const shape = applySubst(out, subst);
  const want: Shape = spec.type === "net" ? [] : [dimCount];
  if (shape.length !== want.length || shape.some((d, i) => !dimEq(d, want[i]!)))
    throw new SpecError(`output "${spec.output}" has shape ${fmtShape(shape)}, a ${spec.type === "net" ? "scalar" : "vector"} field needs ${fmtShape(want)}`, [...path, "output"]);
  return shape;
}

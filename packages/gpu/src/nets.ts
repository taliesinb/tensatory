// Net-backed fields -> WGSL.
//
// One thread evaluates the whole net for ONE grid point, so inside the emitted
// function there is no batch: every array has exactly its declared per-example
// shape (symbolic sizes resolved from the bound arrays) and lives in a
// function-scope `var name: array<f32, N>`; constants (bound / baked arrays)
// live in the shared storage buffer. Ops become nested loops over the output
// shape with an inner accumulation for contractions and reductions. `call`
// inlines the callee. The result is an ordinary field function
// `fn <name>(p: vecD, pos: i32) -> f32 | vecD`, so every kernel (raster, exact
// isolines, streamlines, glyphs, marching tetrahedra) evaluates the net
// directly and net fields need no CPU fallback. Derivatives are net fields
// too (core's autodiff rewrites the program), so nothing special is needed.
//
// Sizes are bounded (NET_MAX_FLOATS): a net whose per-example intermediates
// do not fit is left to the CPU fallback (still correct, just `costly`).

import {
  NetScalarFieldData,
  NetVectorFieldData,
  PulledBackScalarFieldData,
  PulledBackVectorFieldData,
  SymbolicScalarFieldData,
  SymbolicVectorFieldData,
  anfProgram,
  compileNet,
  exprNames,
  inferExpr,
  pointInput,
  type FieldData,
  type NetField,
  type Program,
  type ProgramResolver,
  type Shape,
} from "@tensatory/core";
import type { ArrayExpr, ArrayReduceFn } from "@tensatory/schema";
import { f32, vecType } from "./wgsl";

/**
 * Most function-scope floats a transpiled net may declare; beyond this the CPU fallback samples the field. Safari's
 * WGSL compiler rejects a function whose variables exceed 8192 bytes (2048 f32); Chrome has no such limit, but a
 * bundle must render in both, so the tighter one rules (with room for scalars and loop counters).
 */
export const NET_MAX_FLOATS = 2000;

/**
 * Most WORK (multiply-adds, roughly) one thread may do per point. A net field is evaluated by ONE lane per grid point
 * as a serial chain of storage reads, latency-bound at ~10⁷ MAC/s regardless of how many points run alongside; the
 * MNIST MLP (256 examples × 269k weights = 7·10⁷ per point) takes ~10 s per dispatch at ANY grid size, which would
 * trip the GPU watchdog. Beyond this budget the field is left to the CPU path (`costly`); the right home for such
 * nets is a cooperative kernel (a workgroup per point), see notes/nets.md.
 */
export const NET_MAX_WORK = 4_000_000;
let WORK_LIMIT = NET_MAX_WORK;
/** override the work budget (diagnostics / benchmarks; Infinity lets any net through) */
export function setNetMaxWork(n: number): void { WORK_LIMIT = n; sizeCache = new WeakMap(); }

/**
 * Whether loop bounds are emitted through an opaque function (`nb_`) so the backend shader compiler cannot unroll
 * the loop nests. WGSL has no roll / unroll attribute (gpuweb#4110) and Tint's MSL printer emits plain loops, so the
 * runtime bound is the only lever; measured with apps/viewer/public/nettiming.html. Default on; a switch so the
 * probe can compare.
 */
let OPAQUE_BOUNDS = true;
export function setOpaqueLoopBounds(on: boolean): void { OPAQUE_BOUNDS = on; }

/** default streaming mode (NetEmitContext.stream overrides): auto = stream when the batched emission does not fit */
let STREAM: "auto" | "always" | "never" = "auto";
export function setEmitStream(mode: "auto" | "always" | "never"): void { STREAM = mode; sizeCache = new WeakMap(); }

/** what the emitter needs from the program builder */
export interface NetEmitContext {
  readonly D: number;
  /** override of NET_MAX_FLOATS (diagnostics) */
  readonly maxFloats?: number;
  /** override of NET_MAX_WORK (diagnostics) */
  readonly maxWork?: number;
  /** stream the dataset axis (see NetEmitter.streamed): auto = when the batched emission exceeds the limit (default) */
  readonly stream?: "auto" | "always" | "never";
  /** pack an array into the shared `data` buffer; returns its element offset */
  upload(data: ArrayLike<number>): number;
}

const UNARY: Record<string, (x: string) => string> = {
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
const BINARY: Record<string, (a: string, b: string) => string> = {
  sub: (a, b) => `(${a} - ${b})`, div: (a, b) => `(${a} / ${b})`, pow: (a, b) => `pow_(${a}, ${b})`,
  logBase: (a, b) => `logbase_(${a}, ${b})`, atan2: (a, b) => `atan2(${a}, ${b})`, mod: (a, b) => `mod_(${a}, ${b})`,
};
const COMPARE: Record<string, string> = { lt: "<", le: "<=", gt: ">", ge: ">=", eq: "==", ne: "!=" };
const NARY = new Set(["add", "mul", "min", "max", "mean", "rms"]);

/** an array inside the emitted function */
interface Arr {
  readonly shape: readonly number[];
  /** priv: function-scope array `base[i]`; data: storage `data[base + i]`; lit: a scalar expression (uniform over
   *  the shape); point: the point `p` ([D]); view: one example of a batched priv / data array (streaming);
   *  expr: a LAZY array — element `idx` is the expression `fn(idx)`, recomputed at every read, so nothing is stored
   *  (a displaced 784×256 weight `W + Σ tₖ Dₖ` reads straight from the storage buffer inside the matmul) */
  readonly kind: "priv" | "data" | "lit" | "point" | "view" | "expr";
  readonly base: string;
  /** expr: the element expression */
  readonly fn?: (idx: readonly string[]) => string;
  /** view: the full array's shape, the streamed axis and the loop variable indexing it, where the array lives */
  readonly full?: readonly number[];
  readonly sAxis?: number;
  readonly sVar?: string;
  readonly store?: "priv" | "data";
}

/** a program the streaming emitter cannot stream (the caller falls back to the batched emission) */
export class StreamError extends Error {}

const size = (shape: readonly number[]) => shape.reduce((a, b) => a * b, 1);

/**
 * Arrays at least this large whose operands are all STABLE (storage data, literals, the point, other lazy arrays —
 * never a function-scope array, whose buffer liveness reuses, nor a streamed view, whose loop variable is local) are
 * emitted lazily (Arr kind "expr") instead of being materialized. Below it, materializing is cheaper: a contraction
 * re-reads each element once per example, and recomputing a small array costs more than holding it.
 */
export const LAZY_MIN_FLOATS = 512;
const stable = (a: Arr): boolean => a.kind === "data" || a.kind === "lit" || a.kind === "point" || a.kind === "expr";
/** most summed elements an einsum unrolls into a lazy element expression (a displacement sums K ≤ 8 directions) */
const LAZY_MAX_SUMMED = 8;
const strides = (shape: readonly number[]): number[] => { const s = new Array<number>(shape.length); for (let d = shape.length - 1, acc = 1; d >= 0; d--) { s[d] = acc; acc *= shape[d]!; } return s; };
const dims = (shape: Shape, what: string): number[] => shape.map((d) => { if (typeof d !== "number") throw new Error(`${what}: unresolved symbolic size "${d}"`); return d; });

class NetEmitter {
  readonly lines: string[] = [];
  floats = 0;
  /** estimated per-point work: elements written by fills plus multiply-adds of contractions, times the streamed N */
  work = 0;
  private scale = 1;
  private n = 0;

  constructor(private readonly ctx: NetEmitContext, private readonly nets: ProgramResolver) {}

  private fresh(prefix: string): string { return `${prefix}${this.n++}`; }

  /** a loop bound the shader compiler cannot see through (so it does not unroll the loop nests: compile time) */
  private bound(n: number): string { return !OPAQUE_BOUNDS || n <= 2 ? String(n) : `nb_(${n})`; }

  /** function-scope arrays whose values are dead, by element count: reused before anything new is declared */
  private readonly free = new Map<number, string[]>();
  /** arrays allocated while emitting the current node (freed at its end unless they became the node's value) */
  private scratch: string[] = [];
  /** live node names per array base (aliases such as reshape share a base) */
  private readonly owners = new Map<string, Set<string>>();

  /** a function-scope array, reusing a dead one of the same size when there is one (private memory is the GPU's
   *  scarce resource here: every float a thread holds costs occupancy) */
  private readonly sizes = new Map<string, number>();
  private alloc(shape: readonly number[]): Arr {
    const n = Math.max(1, size(shape));
    // best fit: the smallest dead array that holds n floats (a larger one wastes nothing — it is already declared)
    let name: string | undefined;
    let bestSize = Infinity;
    for (const [sz, pool] of this.free) if (sz >= n && sz < bestSize && pool.length) { bestSize = sz; }
    if (bestSize < Infinity) name = this.free.get(bestSize)!.pop();
    if (name === undefined) {
      name = this.fresh("a");
      this.floats += n;
      this.sizes.set(name, n);
      this.lines.push(`  var ${name}: array<f32, ${n}>;`);
    }
    this.scratch.push(name);
    return { shape, kind: "priv", base: name };
  }

  private release(base: string): void {
    const n = this.sizes.get(base);
    if (n === undefined) return;
    const pool = this.free.get(n) ?? this.free.set(n, []).get(n)!;
    if (!pool.includes(base)) pool.push(base);
  }

  /** flat index expression of `idx` (one string per axis) in `shape` */
  private flat(shape: readonly number[], idx: readonly string[]): string {
    const st = strides(shape);
    const paren = (i: string) => (/^[A-Za-z_][A-Za-z_0-9]*$|^\d+$/.test(i) ? i : `(${i})`);
    const terms = idx.map((i, d) => (shape[d] === 1 ? null : st[d] === 1 ? paren(i) : `${paren(i)} * ${st[d]}`)).filter((t): t is string => t !== null);
    return terms.length ? terms.join(" + ") : "0";
  }

  /** read element `idx` of `a` (idx has a.shape.length entries) */
  read(a: Arr, idx: readonly string[]): string {
    switch (a.kind) {
      case "lit": return a.base;
      case "point": return this.ctx.D === 1 ? "p" : `p[${idx[0]}]`;
      case "priv": return `${a.base}[${this.flat(a.shape, idx)}]`;
      case "data": return `data[${a.base} + ${this.flat(a.shape, idx)}]`;
      case "expr": return a.fn!(idx);
      case "view": {
        const fullIdx = [...idx]; fullIdx.splice(a.sAxis!, 0, a.sVar!);
        const f = this.flat(a.full!, fullIdx);
        return a.store === "data" ? `data[${a.base} + ${f}]` : `${a.base}[${f}]`;
      }
    }
  }

  /** read `a` broadcast against an output of rank `outIdx.length` (numpy right alignment; size-1 axes index 0) */
  private readB(a: Arr, outIdx: readonly string[]): string {
    const off = outIdx.length - a.shape.length;
    return this.read(a, a.shape.map((s, d) => (s === 1 ? "0" : outIdx[off + d]!)));
  }

  /** nested loops over `shape`; `body` receives the index variables */
  private loops(shape: readonly number[], body: (idx: string[]) => void): void {
    const idx: string[] = [];
    const open = (d: number) => {
      if (d === shape.length) { body(idx); return; }
      const v = this.fresh("ix");
      this.lines.push(`  for (var ${v}: i32 = 0; ${v} < ${this.bound(shape[d]!)}; ${v}++) {`);
      idx.push(v);
      open(d + 1);
      idx.pop();
      this.lines.push(`  }`);
    };
    open(0);
  }

  /** write `expr(idx)` into every element of a new array of `shape` */
  private fill(shape: readonly number[], expr: (idx: string[]) => string): Arr {
    this.work += this.scale * size(shape);
    const out = this.alloc(shape);
    this.loops(shape, (idx) => this.lines.push(`  ${this.read(out, idx)} = ${expr(idx)};`));
    return out;
  }

  /** a contiguous copy of `a` in a function-scope array (for aliasing ops on literals / the point) */
  private materialize(a: Arr): Arr {
    if (a.kind === "priv" || a.kind === "data") return a;
    return this.fill(a.shape, (idx) => this.read(a, idx));
  }

  /*******************************************************/

  /** emit a whole program with its inputs bound to arrays; returns the outputs */
  program(prog: Program, inputs: ReadonlyMap<string, Arr>, path: string): Record<string, Arr> {
    const { env, sizeOf } = this.bindEnv(prog, inputs, path);
    const outputs = new Set(Object.values(prog.outputs));
    this.emitNodes(prog.nodes, env, sizeOf, path, outputs, !this.inCall);
    return Object.fromEntries(Object.entries(prog.outputs).map(([o, n]) => [o, env.get(n)!]));
  }

  /** bind a program's inputs and constants: the starting environment and the concrete symbolic sizes */
  private bindEnv(prog: Program, inputs: ReadonlyMap<string, Arr>, path: string): { env: Map<string, Arr>; sizeOf: (d: number | string) => number; sizes: Map<string, number> } {
    const env = new Map<string, Arr>();
    const sizes = new Map<string, number>();
    const bind = (name: string, actual: readonly number[], declared: Shape) => {
      if (actual.length !== declared.length) throw new Error(`${path}: "${name}" has rank ${actual.length}, declared ${declared.length} (a batch inside a transpiled net?)`);
      declared.forEach((d, i) => {
        if (typeof d === "number") { if (d !== actual[i]) throw new Error(`${path}: "${name}" axis ${i} is ${actual[i]}, declared ${d}`); }
        else { const prev = sizes.get(d); if (prev === undefined) sizes.set(d, actual[i]!); else if (prev !== actual[i]) throw new Error(`${path}: axis "${d}" is ${prev} and ${actual[i]}`); }
      });
    };
    for (const [n, shape] of Object.entries(prog.inputs)) {
      const a = inputs.get(n);
      if (!a) throw new Error(`${path}: input "${n}" not given`);
      bind(n, a.shape, shape);
      env.set(n, a);
    }
    for (const [n, c] of Object.entries(prog.consts)) {
      bind(n, c.arr.shape, c.shape);
      env.set(n, { shape: [...c.arr.shape], kind: "data", base: String(this.ctx.upload(c.arr.data)) });
    }
    const sizeOf = (d: number | string): number => {
      if (typeof d === "number") return d;
      const s = sizes.get(d);
      if (s === undefined) throw new Error(`${path}: symbolic size "${d}" is not bound`);
      return s;
    };
    return { env, sizeOf, sizes };
  }

  /**
   * Emit a list of nodes into `env`, with liveness-driven reuse of function-scope arrays when `manage` (not inside
   * an inlined call): names in `keep` are never released (outputs, values a later phase reads). `compute` overrides
   * how a node's value is produced (the streaming emitter's per-example / accumulating nodes).
   */
  private emitNodes(nodes: readonly { name: string; expr: ArrayExpr }[], env: Map<string, Arr>, sizeOf: (d: number | string) => number, path: string, keep: ReadonlySet<string>, manage: boolean, compute?: (node: { name: string; expr: ArrayExpr }, env: Map<string, Arr>) => Arr): void {
    // liveness: the last node that reads each name; outputs and everything a nested (inlined) call produces live on
    const lastUse = new Map<string, number>();
    nodes.forEach((node, i) => { for (const dep of exprNames(node.expr)) lastUse.set(dep, i); });
    const top = manage;
    const releaseDying = (node: { name: string; expr: ArrayExpr }, i: number) => {
      for (const dep of exprNames(node.expr)) {
        if (lastUse.get(dep) !== i || keep.has(dep)) continue;
        const a = env.get(dep);
        if (!a || a.kind !== "priv") continue;
        const own = this.owners.get(a.base);
        if (!own) continue;
        own.delete(dep);
        if (own.size === 0) { this.owners.delete(a.base); this.release(a.base); }
      }
    };
    nodes.forEach((node, i) => {
      this.scratch = [];
      // an elementwise node writes element i from the operands' element i (or a broadcast of a smaller operand): an
      // operand that dies here can be its output buffer — release it first so alloc picks it up (in-place update)
      if (top && isElementwise(node.expr)) releaseDying(node, i);
      const val = compute ? compute(node, env) : this.expr(node.expr, env, sizeOf, `${path}.${node.name}`);
      env.set(node.name, val);
      if (!top) return;
      // a folded node may ALIAS an operand (`add(x, 0)` is x): if that operand was just released, take it back
      if (val.kind === "priv") { const pool = this.free.get(this.sizes.get(val.base) ?? -1); const k = pool?.indexOf(val.base) ?? -1; if (k >= 0) pool!.splice(k, 1); }
      if (val.kind === "priv") (this.owners.get(val.base) ?? this.owners.set(val.base, new Set()).get(val.base)!).add(node.name);
      // scratch arrays that did not become the node's value are dead now
      for (const b of this.scratch) if (b !== val.base && !this.owners.has(b)) this.release(b);
      this.scratch = [];
      // operands whose last reader this was
      releaseDying(node, i);
    });
  }
  /*******************************************************/
  /* streaming: the dataset axis in time instead of space */

  /**
   * Emit `prog` with its symbolic axis S (the dataset axis, "N") STREAMED: instead of materializing every `[N, …]`
   * intermediate — the [N, 16] hidden layer alone is 1920 floats at N = 120, past Safari's 8 KB of function
   * variables — the nodes that carry S are computed one example at a time inside `for n < N`, and the nodes that
   * consume S (a reduce over it, an einsum contracting it: the loss's mean, the weight gradients' `ij,ik->kj`)
   * ACCUMULATE across the loop. Same batched program, same loops per op; the N loop merely moves outside the whole
   * per-example chain, so the footprint is that of one example. Semantically nothing changes: a streamed program
   * is legal exactly when S only ever ends in a reduction (no op couples examples) — anything else throws
   * StreamError and the caller keeps the batched emission.
   *
   * Nodes are classified by DEPENDENCE, not shape: a node whose value does not depend on a batched name — autodiff's
   * `1/N` seed `add(mul(nll, 0), 1)` has shape [N, 1] but reads nothing — is uniform and emitted once, outside
   * (`elementwise` folds it to one scalar). Levels: a batched node depending (through a non-batched node) on an
   * accumulated result belongs to a later loop, which recomputes the per-example chain it needs (the gradient of a
   * mean needs no second pass thanks to the fold above). Only the nodes `wanted` outputs need are emitted.
   */
  streamed(prog: Program, inputs: ReadonlyMap<string, Arr>, path: string, wanted: readonly string[]): Record<string, Arr> {
    const { env, sizeOf } = this.bindEnv(prog, inputs, path);
    const fail = (why: string): never => { throw new StreamError(`${path}: cannot stream: ${why}`); };
    // the streamed axis: the symbolic size of the largest concrete extent
    const symbolic = new Map<string, number>();
    for (const sh of [...Object.values(prog.inputs), ...Object.values(prog.consts).map((c) => c.shape)]) for (const d of sh) if (typeof d === "string") symbolic.set(d, sizeOf(d));
    if (!symbolic.size) fail("no symbolic axis");
    const S = [...symbolic.entries()].sort((a, b) => b[1] - a[1])[0]![0], n = symbolic.get(S)!;
    // full (symbolic) shapes of every name
    const names = new Map<string, Shape>();
    for (const [k, sh] of Object.entries(prog.inputs)) names.set(k, sh);
    for (const [k, c] of Object.entries(prog.consts)) names.set(k, c.shape);
    const shapeEnv = { names, axisNames: new Set(symbolic.keys()), nets: this.nets };
    for (const node of prog.nodes) names.set(node.name, inferExpr(node.expr, shapeEnv, [path, node.name]));
    const sPos = (sh: Shape): number | undefined => { const i = sh.indexOf(S); if (i >= 0 && sh.indexOf(S, i + 1) >= 0) fail(`"${S}" appears twice in a shape`); return i < 0 ? undefined : i; };
    const sPosOfExpr = (x: ArrayExpr): number | undefined => (typeof x === "number" ? undefined : typeof x === "string" ? sPos(names.get(x) ?? fail(`unknown name ${x}`)) : x.op === "arg" ? sPos(names.get(x.name)!) : sPos(inferExpr(x, shapeEnv, [path])));
    const concrete = (sh: Shape) => sh.map((d) => sizeOf(d));
    const rest = (sh: Shape) => { const p = sPos(sh); return concrete(p === undefined ? sh : sh.filter((_, i) => i !== p)); };
    // only what the wanted outputs need
    const byName = new Map(prog.nodes.map((nd) => [nd.name, nd]));
    const needed = new Set<string>();
    const visit = (nm: string) => { const nd = byName.get(nm); if (!nd || needed.has(nm)) return; needed.add(nm); for (const d of exprNames(nd.expr)) visit(d); };
    const wantedNames = wanted.map((o) => prog.outputs[o] ?? fail(`no output "${o}"`));
    wantedNames.forEach(visit);
    const nodes = prog.nodes.filter((nd) => needed.has(nd.name));
    // classification and scheduling (see above)
    const batched = new Set<string>(), lvl = new Map<string, number>(), avail = new Map<string, number>(), kind = new Map<string, "batched" | "boundary" | "plain">();
    for (const [k, sh] of names) if (!byName.has(k)) { if (sPos(sh) !== undefined) { batched.add(k); lvl.set(k, 0); } else avail.set(k, -1); }
    for (const nd of nodes) {
      const deps = [...effectiveNames(nd.expr)];
      const depBatched = deps.some((d) => batched.has(d));
      const hasS = sPos(names.get(nd.name)!) !== undefined;
      const need = Math.max(0, ...deps.map((d) => (batched.has(d) ? lvl.get(d)! : (avail.get(d) ?? -1) + 1)));
      if (!depBatched) { kind.set(nd.name, "plain"); avail.set(nd.name, Math.max(-1, ...deps.map((d) => avail.get(d) ?? -1))); }
      else if (hasS) { kind.set(nd.name, "batched"); batched.add(nd.name); lvl.set(nd.name, need); }
      else { kind.set(nd.name, "boundary"); lvl.set(nd.name, need); avail.set(nd.name, need); }
    }
    for (const w of wantedNames) if (batched.has(w)) fail(`output "${w}" carries the streamed axis`);
    const loops = Math.max(0, ...nodes.filter((nd) => kind.get(nd.name) !== "plain").map((nd) => lvl.get(nd.name)! + 1));
    // names read outside the phase that computes them stay allocated
    const group = (nm: string) => (kind.get(nm) === "batched" ? `b${lvl.get(nm)}` : `p${avail.get(nm)}`);
    const keep = new Set(wantedNames);
    for (const nd of nodes) for (const d of exprNames(nd.expr)) if (byName.has(d) && needed.has(d) && group(d) !== group(nd.name)) keep.add(d);
    const hints = (inLoop: boolean) => { const h = new Map<string, readonly number[]>(); for (const [k, sh] of names) h.set(k, inLoop ? rest(sh) : concrete(sh)); return h; };
    const plainAt = (k: number) => nodes.filter((nd) => kind.get(nd.name) === "plain" && avail.get(nd.name) === k);

    this.shapeHints = hints(false);
    this.emitNodes(plainAt(-1), env, sizeOf, path, keep, !this.inCall);
    for (const nd of plainAt(-1)) if (sPos(names.get(nd.name)!) !== undefined && env.get(nd.name)!.kind !== "lit") fail(`"${nd.name}" has the streamed axis but depends on no batched value and is not uniform`);

    for (let L = 0; L < loops; L++) {
      const boundaries = nodes.filter((nd) => kind.get(nd.name) === "boundary" && lvl.get(nd.name) === L);
      // the per-example chain this loop needs: batched nodes reachable from its boundaries through batched names
      const body = new Set<string>();
      const grow = (nm: string) => { for (const d of exprNames(byName.get(nm)!.expr)) if (batched.has(d) && byName.has(d) && !body.has(d)) { body.add(d); grow(d); } };
      boundaries.forEach((b) => grow(b.name));
      // accumulators
      const accs = new Map<string, { acc: Arr; fn: string }>();
      for (const b of boundaries) {
        const { fn } = this.perExample(b.expr, sPosOfExpr, S, n, path);
        if (!fn) fail(`"${b.name}" consumes the streamed axis with "${typeof b.expr === "object" ? b.expr.op : b.expr}", not a reduction`);
        this.scratch = [];
        const acc = this.alloc(concrete(names.get(b.name)!));
        this.scratch = [];
        (this.owners.get(acc.base) ?? this.owners.set(acc.base, new Set()).get(acc.base)!).add(b.name);
        const init = fn === "prod" ? "1.0" : fn === "max" || fn === "logsumexp" ? "-3.4e38" : fn === "min" ? "3.4e38" : "0.0";
        this.loops(acc.shape, (i) => this.lines.push(`  ${this.read(acc, i)} = ${init};`));
        accs.set(b.name, { acc, fn: fn! });
      }
      // the loop over examples: every array carrying S is read through a one-example view
      const sVar = this.fresh("n");
      this.lines.push(`  for (var ${sVar}: i32 = 0; ${sVar} < ${this.bound(n)}; ${sVar}++) {`);
      const outerScale = this.scale; this.scale *= n;
      const declared = new Set(this.sizes.keys());
      const loopEnv = new Map<string, Arr>();
      for (const [k, a] of env) { const p = sPos(names.get(k) ?? []); loopEnv.set(k, p === undefined ? a : this.view(a, p, sVar)); }
      this.shapeHints = hints(true);
      const order = nodes.filter((nd) => body.has(nd.name) || accs.has(nd.name));
      // nothing defined outside the loop may be released (or written in place) inside it: the next example reads it
      const outer = new Set(loopEnv.keys());
      this.emitNodes(order, loopEnv, sizeOf, `${path}.n`, outer, !this.inCall, (nd, e) => {
        const { expr, fn } = this.perExample(nd.expr, sPosOfExpr, S, n, path);
        const v = this.expr(expr, e, sizeOf, `${path}.${nd.name}`);
        const a = accs.get(nd.name);
        if (!a) return this.reconcile(v, rest(names.get(nd.name)!), nd.name);
        // accumulate this example's contribution
        const src = this.reconcile(v, a.acc.shape, nd.name);
        this.loops(a.acc.shape, (i) => {
          const x = this.read(a.acc, i), y = this.read(src, i);
          const upd = fn === "sum" || fn === "mean" ? `${x} + ${y}` : fn === "prod" ? `${x} * ${y}` : fn === "max" ? `max(${x}, ${y})` : fn === "min" ? `min(${x}, ${y})` : `lse2_(${x}, ${y})`;
          this.lines.push(`  ${x} = ${upd};`);
        });
        return a.acc;
      });
      this.lines.push(`  }`);
      this.scale = outerScale;
      // loop-scoped arrays are gone
      for (const [sz, pool] of this.free) this.free.set(sz, pool.filter((b) => declared.has(b)));
      for (const base of [...this.owners.keys()]) if (!declared.has(base)) this.owners.delete(base);
      for (const nm of [...this.sizes.keys()]) if (!declared.has(nm)) this.sizes.delete(nm);
      this.shapeHints = hints(false);
      for (const [nm, { acc, fn }] of accs) {
        if (fn === "mean") this.loops(acc.shape, (i) => this.lines.push(`  ${this.read(acc, i)} = ${this.read(acc, i)} / ${f32(n)};`));
        env.set(nm, acc);
      }
      this.emitNodes(plainAt(L), env, sizeOf, path, keep, !this.inCall);
    }
    this.shapeHints = new Map();
    return Object.fromEntries(wanted.map((o) => [o, env.get(prog.outputs[o]!)!]));
  }

  /** one example of a batched array (streaming): the loop variable indexes the streamed axis */
  private view(a: Arr, sAxis: number, sVar: string): Arr {
    const shape = a.shape.filter((_, i) => i !== sAxis);
    if (a.kind === "lit") return { ...a, shape };
    if (a.kind === "priv" || a.kind === "data") return { kind: "view", base: a.base, shape, full: a.shape, sAxis, sVar, store: a.kind };
    if (a.kind === "expr") return { kind: "expr", base: a.base, shape, fn: (idx) => { const full = [...idx]; full.splice(sAxis, 0, sVar); return a.fn!(full); } };
    throw new StreamError(`cannot view a ${a.kind} array per example`);
  }

  /** `v` as `shape` (an alias when contiguous and of the same size) */
  private reconcile(v: Arr, shape: readonly number[], what: string): Arr {
    if (v.shape.length === shape.length && v.shape.every((d, i) => d === shape[i])) return v;
    if (size(v.shape) !== size(shape)) throw new StreamError(`"${what}": per-example shape [${v.shape}] is not [${shape}]`);
    if (v.kind === "lit") return { ...v, shape };
    return { ...this.materialize(v), shape };
  }

  /**
   * Rewrite a node's expression for ONE example of its batched operands (their streamed axis dropped): axis
   * arguments shift past it, einsum letters lose it. `fn` set means the node consumes the axis — the returned
   * expression is this example's contribution, to be accumulated with `fn` across the loop.
   */
  private perExample(e: ArrayExpr, sPosOf: (x: ArrayExpr) => number | undefined, S: string, n: number, path: string): { expr: ArrayExpr; fn?: ArrayReduceFn } {
    const fail = (why: string): never => { throw new StreamError(`${path}: cannot stream: ${why}`); };
    if (typeof e !== "object") return { expr: e };
    const rankOf = (x: ArrayExpr): number => (typeof x === "number" ? 0 : typeof x === "string" || x.op === "arg" ? (this.shapeHints.get(typeof x === "string" ? x : (x as { name: string }).name)?.length ?? 0) + (sPosOf(x) === undefined ? 0 : 1) : fail("rank of a nested expression"));
    const norm = (axis: number, rank: number) => (axis < 0 ? rank + axis : axis);
    const shift = (axis: number, s: number) => (axis < s ? axis : axis - 1);
    if (isElementwise(e)) {
      // broadcasting commutes with dropping S only when S sits at the same right-aligned position in every operand
      const walk = (x: ArrayExpr, acc: Set<number>): void => {
        if (typeof x === "number") return;
        if (typeof x === "string" || x.op === "arg" || !isElementwise(x)) { const p = sPosOf(x); if (p !== undefined) acc.add(rankOf(x) - p); return; }
        for (const k of ["val", "vals", "min", "max", "cond"] as const) { const v = (x as unknown as Record<string, ArrayExpr | ArrayExpr[]>)[k]; if (v === undefined) continue; if (Array.isArray(v)) v.forEach((y) => walk(y, acc)); else walk(v, acc); }
      };
      const pos = new Set<number>(); walk(e, pos);
      if (pos.size > 1) fail(`elementwise operands hold "${S}" at different right-aligned positions`);
      return { expr: e };
    }
    const op = e.op;
    switch (op) {
      case "arg": case "stopGradient": case "oneHot": return { expr: e };
      case "matmul": {
        const [a, b] = e.vals;
        const ra = rankOf(a), rb = rankOf(b);
        if (ra === 1 && rb === 1) return this.perExample({ op: "einsum", subscripts: "k,k->", vals: [a, b] }, sPosOf, S, n, path);
        const L = Math.max(ra, rb) - 2;
        const lead = Array.from({ length: Math.max(L, 0) }, (_, i) => String.fromCharCode(0x41 + i));
        const la = lead.slice(lead.length - Math.max(ra - 2, 0)).join(""), lb = lead.slice(lead.length - Math.max(rb - 2, 0)).join("");
        const sub = rb === 1 ? `${la}ik,k->${lead.join("")}i` : ra === 1 ? `k,${lb}kj->${lead.join("")}j` : `${la}ik,${lb}kj->${lead.join("")}ij`;
        return this.perExample({ op: "einsum", subscripts: sub, vals: [a, b] }, sPosOf, S, n, path);
      }
      case "einsum": {
        const text = e.subscripts.replace(/\s+/g, "");
        const [lhs, rhs] = text.split("->");
        const terms = lhs!.split(",");
        let out = rhs;
        if (out === undefined) { const count = new Map<string, number>(); for (const l of lhs!.replace(/,/g, "")) count.set(l, (count.get(l) ?? 0) + 1); out = [...count.entries()].filter(([, c]) => c === 1).map(([l]) => l).sort().join(""); }
        let letter: string | undefined;
        const newTerms = terms.map((t, k) => {
          const p = sPosOf(e.vals[k]!);
          if (p === undefined) return t;
          const l = t[p]!;
          if (letter !== undefined && letter !== l) fail(`einsum: "${S}" is "${letter}" in one operand and "${l}" in another`);
          letter = l;
          return t.slice(0, p) + t.slice(p + 1);
        });
        if (letter === undefined) return { expr: e };
        terms.forEach((t, k) => { if (sPosOf(e.vals[k]!) === undefined && t.includes(letter!)) fail(`einsum: letter "${letter}" of the streamed axis also indexes an unbatched operand`); });
        const summed = !out.includes(letter);
        const newOut = out.replace(letter, "");
        return { expr: { ...e, subscripts: `${newTerms.join(",")}->${newOut}` }, ...(summed ? { fn: "sum" as const } : {}) };
      }
      case "reduce": {
        const p = sPosOf(e.val);
        if (p === undefined) return { expr: e };
        const r = rankOf(e.val);
        const axes = (e.axes === undefined ? Array.from({ length: r }, (_, i) => i) : e.axes.map((a) => norm(a, r)));
        if (!axes.includes(p)) return { expr: { ...e, axes: axes.map((a) => shift(a, p)) } };
        const others = axes.filter((a) => a !== p).map((a) => shift(a, p));
        if (e.fn === "logsumexp" && others.length) fail("logsumexp over the streamed axis together with other axes");
        // this example's contribution: the reduction over the remaining axes (a mean of equal-count means is the mean)
        const expr: ArrayExpr = others.length ? { ...e, axes: others, keepDims: false } : e.val;
        return { expr, fn: e.fn };
      }
      case "argmax": case "argmin": case "softmax": case "logSoftmax": {
        const p = sPosOf(e.val);
        if (p === undefined) return { expr: e };
        const ax = norm(e.axis ?? -1, rankOf(e.val));
        if (ax === p) fail(`${op} along the streamed axis`);
        return { expr: { ...e, axis: shift(ax, p) } };
      }
      case "reshape": {
        const p = sPosOf(e.val);
        if (p === undefined) return { expr: e };
        const j = e.shape.indexOf(S);
        if (j < 0 || e.shape.indexOf(S, j + 1) >= 0) fail(`reshape of a batched value must name "${S}" once in its shape`);
        return { expr: { ...e, shape: e.shape.filter((_, i) => i !== j) } };
      }
      case "transpose": {
        const p = sPosOf(e.val);
        if (p === undefined) return { expr: e };
        const r = rankOf(e.val);
        const perm = e.perm ?? Array.from({ length: r }, (_, i) => r - 1 - i);
        const j = perm.indexOf(p);
        return { expr: { ...e, perm: perm.filter((_, i) => i !== j).map((q) => shift(q, p)) } };
      }
      case "concat": {
        const ps = e.vals.map(sPosOf);
        if (ps.every((p) => p === undefined)) return { expr: e };
        if (ps.some((p) => p === undefined) || new Set(ps).size !== 1) fail("concat of batched and unbatched values");
        const ax = norm(e.axis, rankOf(e.vals[0]!));
        if (ax === ps[0]) fail("concat along the streamed axis");
        return { expr: { ...e, axis: shift(ax, ps[0]!) } };
      }
      case "slice": {
        const p = sPosOf(e.val);
        if (p === undefined) return { expr: e };
        const ax = norm(e.axis, rankOf(e.val));
        if (ax === p) fail("slice along the streamed axis");
        return { expr: { ...e, axis: shift(ax, p) } };
      }
      case "takeAlong": {
        const p = sPosOf(e.val), q = sPosOf(e.indices);
        if (p === undefined && q === undefined) return { expr: e };
        const r = rankOf(e.val);
        if (p === undefined || (q !== undefined && rankOf(e.indices) - q !== r - p)) fail("takeAlong: indices and values hold the streamed axis differently");
        const ax = norm(e.axis, r);
        if (ax === p) fail("takeAlong along the streamed axis");
        return { expr: { ...e, axis: shift(ax, p!) } };
      }
      case "call": return fail("a call with batched inputs");
      default: return { expr: e };
    }
  }

  private inCall = false;
  /** shapes of names that are not in the environment (streaming: a batched name inside a folded `mul(x, 0)`) */
  shapeHints: ReadonlyMap<string, readonly number[]> = new Map();

  /** shape of an expression whose operands are in `env` (symbolic sizes resolved) */
  private shapeOf(e: ArrayExpr, env: ReadonlyMap<string, Arr>, sizeOf: (d: number | string) => number): number[] {
    const names = new Map<string, Shape>();
    for (const [k, sh] of this.shapeHints) names.set(k, [...sh]);
    for (const [k, a] of env) names.set(k, a.shape);
    return dims(inferExpr(concretize(e, sizeOf), { names, axisNames: new Set(), nets: this.nets }, ["net"]), "net");
  }

  /** emit an expression */
  expr(e: ArrayExpr, env: ReadonlyMap<string, Arr>, sizeOf: (d: number | string) => number, path: string): Arr {
    if (typeof e === "number") return { shape: [], kind: "lit", base: f32(e) };
    if (typeof e === "string") return lookup(e, env, path);
    const sub = (x: ArrayExpr) => this.expr(x, env, sizeOf, path);
    const norm = (axis: number, rank: number) => (axis < 0 ? rank + axis : axis);
    const outShape = () => this.shapeOf(e, env, sizeOf);
    const op = e.op;
    switch (op) {
      case "arg": return lookup(e.name, env, path);
      case "coord": case "coordv": throw new Error(`${path}: coordinate leaves are folded into the program by core (fieldProgram)`);
      case "clamp": case "where": return this.elementwise(e, env, sizeOf, path);
      case "matmul": {
        const a = sub(e.vals[0]), b = sub(e.vals[1]);
        const ra = a.shape.length, rb = b.shape.length;
        if (ra === 1 && rb === 1) return this.einsum(["k", "k"], "", [a, b], outShape());
        const L = Math.max(ra, rb) - 2;
        const lead = Array.from({ length: Math.max(L, 0) }, (_, i) => String.fromCharCode(0x41 + i));
        const la = lead.slice(lead.length - Math.max(ra - 2, 0)).join(""), lb = lead.slice(lead.length - Math.max(rb - 2, 0)).join("");
        if (rb === 1) return this.einsum([la + "ik", "k"], lead.join("") + "i", [a, b], outShape());
        if (ra === 1) return this.einsum(["k", lb + "kj"], lead.join("") + "j", [a, b], outShape());
        return this.einsum([la + "ik", lb + "kj"], lead.join("") + "ij", [a, b], outShape());
      }
      case "einsum": {
        const text = e.subscripts.replace(/\s+/g, "");
        const [lhs, rhs] = text.split("->");
        const terms = lhs!.split(",");
        const vals = e.vals.map(sub);
        let out = rhs;
        if (out === undefined) {
          const count = new Map<string, number>();
          for (const l of lhs!.replace(/,/g, "")) count.set(l, (count.get(l) ?? 0) + 1);
          out = [...count.entries()].filter(([, c]) => c === 1).map(([l]) => l).sort().join("");
        }
        return this.einsum(terms, out, vals, outShape());
      }
      case "reduce": {
        const v = sub(e.val);
        const r = v.shape.length;
        const axes = new Set(e.axes === undefined ? v.shape.map((_, i) => i) : e.axes.map((a) => norm(a, r)));
        return this.reduce(v, e.fn, axes, e.keepDims ?? false, outShape());
      }
      case "argmax": case "argmin": {
        const v = sub(e.val);
        const ax = norm(e.axis ?? -1, v.shape.length);
        const out = this.alloc(outShape());
        this.loops(out.shape, (oi) => {
          const best = this.fresh("b"), arg = this.fresh("j"), k = this.fresh("k");
          const idx = [...oi]; idx.splice(ax, 0, k);
          const first = [...oi]; first.splice(ax, 0, "0");
          this.lines.push(`  var ${best}: f32 = ${this.read(v, first)}; var ${arg}: i32 = 0;`);
          this.lines.push(`  for (var ${k}: i32 = 1; ${k} < ${this.bound(v.shape[ax]!)}; ${k}++) { let x = ${this.read(v, idx)}; if (x ${op === "argmax" ? ">" : "<"} ${best}) { ${best} = x; ${arg} = ${k}; } }`);
          this.lines.push(`  ${this.read(out, oi)} = f32(${arg});`);
        });
        return out;
      }
      case "softmax": case "logSoftmax": {
        const v = this.materialize(sub(e.val));
        const ax = norm(e.axis ?? -1, v.shape.length);
        const keep = v.shape.map((s, d) => (d === ax ? 1 : s));
        const m = this.reduce(v, "max", new Set([ax]), true, keep);
        const ex = this.fill(v.shape, (i) => `exp(${this.read(v, i)} - ${this.readB(m, i)})`);
        const s = this.reduce(ex, "sum", new Set([ax]), true, keep);
        return op === "softmax"
          ? this.fill(v.shape, (i) => `${this.read(ex, i)} / ${this.readB(s, i)}`)
          : this.fill(v.shape, (i) => `(${this.read(v, i)} - ${this.readB(m, i)}) - log(${this.readB(s, i)})`);
      }
      case "reshape": {
        const v0 = sub(e.val);
        if (v0.kind === "lit") return { ...v0, shape: outShape() }; // uniform: any shape
        const v = this.materialize(v0);
        return { ...v, shape: outShape() }; // contiguous: an alias
      }
      case "transpose": {
        const v = sub(e.val);
        if (v.kind === "lit") return { ...v, shape: outShape() };
        const r = v.shape.length;
        const perm = e.perm ?? Array.from({ length: r }, (_, i) => r - 1 - i);
        return this.fill(outShape(), (oi) => { const idx = new Array<string>(r); perm.forEach((p, d) => { idx[p] = oi[d]!; }); return this.read(v, idx); });
      }
      case "concat": {
        const vals = e.vals.map(sub);
        const shape = outShape();
        const ax = norm(e.axis, shape.length);
        const out = this.alloc(shape);
        let at = 0;
        for (const v of vals) {
          const part = shape.map((s, d) => (d === ax ? v.shape[ax]! : s));
          this.loops(part, (i) => { const oi = i.map((x, d) => (d === ax ? `${x} + ${at}` : x)); this.lines.push(`  ${this.read(out, oi)} = ${this.readB(v, i)};`); });
          at += v.shape[ax]!;
        }
        return out;
      }
      case "slice": {
        const v = sub(e.val);
        const shape = outShape();
        if (v.kind === "lit") return { ...v, shape };
        const ax = norm(e.axis, v.shape.length);
        const n = v.shape[ax]!, step = e.step ?? 1;
        const clampIdx = (i: number | undefined, dflt: number, lo: number, hi: number) => (i === undefined ? dflt : Math.min(hi, Math.max(lo, i < 0 ? n + i : i)));
        const start = step > 0 ? clampIdx(e.start, 0, 0, n) : clampIdx(e.start, n - 1, -1, n - 1);
        return this.fill(shape, (i) => this.read(v, i.map((x, d) => (d === ax ? `${start} + ${x} * ${step}` : x))));
      }
      case "oneHot": {
        const v = sub(e.val);
        const shape = outShape();
        return this.fill(shape, (i) => `select(0.0, 1.0, i32(round_(${this.read(v, i.slice(0, -1))})) == ${i[i.length - 1]})`);
      }
      case "takeAlong": {
        const v = sub(e.val), ix = sub(e.indices);
        const shape = outShape();
        const ax = norm(e.axis, v.shape.length);
        const n = v.shape[ax]!;
        return this.fill(shape, (i) => {
          const j = `clamp(i32(round_(${this.readB(ix, i)})), 0, ${n - 1})`;
          const off = shape.length - v.shape.length;
          const idx = v.shape.map((s, d) => (d === ax ? j : s === 1 ? "0" : i[off + d]!));
          return this.read(v, idx);
        });
      }
      case "stopGradient": return sub(e.val);
      case "call": {
        const callee = typeof e.net === "string" ? this.nets.program(e.net, [path]) : compileNet(e.net, this.nets, [path]);
        const args = new Map(Object.entries(e.inputs).map(([k, v]) => [k, this.materialize(sub(v))]));
        const was = this.inCall; this.inCall = true;
        const saved = this.scratch; this.scratch = [];
        const outs = this.program(callee, args, `${path}.call`);
        this.scratch = [...saved, ...this.scratch]; this.inCall = was;
        const out = outs[e.output];
        if (!out) throw new Error(`${path}: called net has no output "${e.output}"`);
        return out;
      }
      default: {
        if (isElementwise(e)) return this.elementwise(e, env, sizeOf, path);
        throw new Error(`${path}: unknown array op "${String(op)}"`);
      }
    }
  }

  /**
   * An elementwise TREE (nested unary / nary / binary / compare / clamp / where over names, literals and
   * non-elementwise subexpressions) as ONE loop: each element is a single scalar expression, so a chain like
   * `relu(add(matmul, b))` or an autodiff mask `where(gt(y, 0), g, 0)` materializes nothing but its result.
   * Non-elementwise operands are emitted first (materialized); the tree's shape is inferred once.
   */
  private elementwise(e: ArrayExpr, env: ReadonlyMap<string, Arr>, sizeOf: (d: number | string) => number, path: string): Arr {
    const shape = this.shapeOf(e, env, sizeOf);
    // materialize non-elementwise / non-leaf operands once, keyed by object identity
    const leaves = new Map<object, Arr>();
    const prepare = (x: ArrayExpr): void => {
      if (typeof x !== "object") return;
      if (isElementwise(x)) { for (const k of ["val", "vals", "min", "max", "cond"] as const) { const v = (x as unknown as Record<string, ArrayExpr | ArrayExpr[]>)[k]; if (v === undefined) continue; if (Array.isArray(v)) v.forEach(prepare); else prepare(v); } return; }
      leaves.set(x, this.expr(x, env, sizeOf, path));
    };
    prepare(e);
    // names are resolved now (a lazy result may be read after `env` has grown)
    const names = new Map<string, Arr>();
    const nameOf = (x: string): Arr => { let a = names.get(x); if (!a) { a = lookup(x, env, path); names.set(x, a); } return a; };
    // (a name absent from env sits inside a folded `mul(x, 0)` — streaming's shapeHints — and is never read)
    const collect = (x: ArrayExpr): void => {
      if (typeof x === "string") { if (env.has(x)) nameOf(x); return; }
      if (typeof x !== "object") return;
      if (x.op === "arg") { if (env.has(x.name)) nameOf(x.name); return; }
      if (!isElementwise(x)) return;
      for (const k of ["val", "vals", "min", "max", "cond"] as const) { const v = (x as unknown as Record<string, ArrayExpr | ArrayExpr[]>)[k]; if (v === undefined) continue; if (Array.isArray(v)) v.forEach(collect); else collect(v); }
    };
    collect(e);
    // fold literal idioms of autodiff (mul by 0, add / mul of a unit)
    const scalar = (x: ArrayExpr, idx: readonly string[]): string => {
      if (typeof x === "number") return f32(x);
      if (typeof x === "string") return this.readB(nameOf(x), idx);
      if (x.op === "arg") return this.readB(nameOf(x.name), idx);
      if (!isElementwise(x)) return this.readB(leaves.get(x)!, idx);
      const op = x.op;
      if (op === "clamp") return `clamp(${scalar(x.val, idx)}, ${scalar(x.min, idx)}, ${scalar(x.max, idx)})`;
      if (op === "where") return `select(${scalar(x.vals[1], idx)}, ${scalar(x.vals[0], idx)}, ${scalar(x.cond, idx)} != 0.0)`;
      if (op in UNARY) return UNARY[op]!(scalar((x as { val: ArrayExpr }).val, idx));
      if (op in BINARY) { const [a, b] = (x as { vals: [ArrayExpr, ArrayExpr] }).vals; return BINARY[op]!(scalar(a, idx), scalar(b, idx)); }
      if (op in COMPARE) { const [a, b] = (x as { vals: [ArrayExpr, ArrayExpr] }).vals; return `select(0.0, 1.0, ${scalar(a, idx)} ${COMPARE[op]} ${scalar(b, idx)})`; }
      const vals = (x as { vals: ArrayExpr[] }).vals;
      if (op === "mul" && vals.some((v) => v === 0)) return "0.0";
      if (op === "add" || op === "mul") {
        const unit = op === "add" ? 0 : 1;
        const rest = vals.filter((v) => v !== unit);
        if (rest.length === 0) return f32(unit);
        if (rest.length === 1) return scalar(rest[0]!, idx);
        return `(${rest.map((v) => scalar(v, idx)).join(op === "add" ? " + " : " * ")})`;
      }
      const xs = vals.map((v) => scalar(v, idx));
      switch (op) {
        case "min": return xs.reduce((a, b) => `min(${a}, ${b})`);
        case "max": return xs.reduce((a, b) => `max(${a}, ${b})`);
        case "mean": return `((${xs.join(" + ")}) / ${f32(xs.length)})`;
        default: return `sqrt((${xs.map((v) => `${v} * ${v}`).join(" + ")}) / ${f32(xs.length)})`; // rms
      }
    };
    // a tree that folds to a literal or to one same-shaped operand needs no array
    const probe = scalar(e, shape.map((_, d) => `__i${d}`));
    if (/^[-+0-9.e()\s]+$/.test(probe) && !/__i/.test(probe)) return { shape, kind: "lit", base: probe };
    // a tree whose value does not depend on the index (every operand a literal or size-1 broadcast — autodiff's
    // `1/N` seeds): one scalar, held uniform over the shape (no array; the shape may include a streamed axis)
    if (!/__i/.test(probe)) { const u = this.fresh("u"); this.lines.push(`  let ${u}: f32 = ${probe};`); return { shape, kind: "lit", base: u }; }
    if (typeof e === "object" && (e.op === "add" || e.op === "mul")) {
      const unit = e.op === "add" ? 0 : 1;
      const rest = (e as { vals: ArrayExpr[] }).vals.filter((v) => v !== unit);
      if (rest.length === 1 && typeof rest[0] === "string") { const r = lookup(rest[0], env, path); if (r.shape.length === shape.length && r.shape.every((d, i) => d === shape[i])) return r; }
    }
    // a large tree over stable operands stays an expression: consumers read `scalar(e, idx)` in place
    if (size(shape) >= LAZY_MIN_FLOATS && [...names.values(), ...leaves.values()].every(stable))
      return { shape, kind: "expr", base: this.fresh("lz"), fn: (idx) => scalar(e, idx) };
    return this.fill(shape, (idx) => scalar(e, idx));
  }

  /** generalized einsum over declared axes (no batch here); size-1 letters broadcast */
  private einsum(terms: string[], out: string, vals: Arr[], shape: number[]): Arr {
    const sizes = new Map<string, number>();
    terms.forEach((t, k) => [...t].forEach((l, d) => { const s = vals[k]!.shape[d]!; const prev = sizes.get(l); if (prev === undefined || prev === 1) sizes.set(l, s); }));
    const summed = [...new Set(terms.join(""))].filter((l) => !out.includes(l));
    const outLetters = [...out];
    const distinct = [...new Set(outLetters)];
    const diagonal = distinct.length !== outLetters.length;
    // a large result over stable operands with a short contraction (a displacement's Σ_k t[k] D[k, …]) stays an
    // expression: the sum is unrolled into the element expression, nothing is stored
    const summedSize = summed.reduce((n, l) => n * sizes.get(l)!, 1);
    if (!diagonal && size(shape) >= LAZY_MIN_FLOATS && summedSize <= LAZY_MAX_SUMMED && vals.every(stable)) {
      const fn = (oi: readonly string[]): string => {
        const pos = new Map<string, string>(outLetters.map((l, d) => [l, oi[d]!]));
        const products: string[] = [];
        const inner = (d: number) => {
          if (d === summed.length) { products.push(vals.map((v, k) => this.read(v, [...terms[k]!].map((l, i) => (v.shape[i] === 1 ? "0" : pos.get(l)!)))).join(" * ")); return; }
          for (let i = 0; i < sizes.get(summed[d]!)!; i++) { pos.set(summed[d]!, String(i)); inner(d + 1); }
        };
        inner(0);
        return `(${products.join(" + ")})`;
      };
      return { shape, kind: "expr", base: this.fresh("lz"), fn };
    }
    this.work += this.scale * size(shape) * summedSize;
    const res = this.alloc(shape);
    // a letter repeated in the output writes the diagonal only: zero the array, then iterate each letter once
    if (diagonal) this.loops(shape, (oi) => this.lines.push(`  ${this.read(res, oi)} = 0.0;`));
    this.loops(distinct.map((l) => sizes.get(l)!), (di) => {
      const pos = new Map<string, string>(distinct.map((l, d) => [l, di[d]!]));
      const oi = outLetters.map((l) => pos.get(l)!);
      const acc = this.fresh("acc");
      this.lines.push(`  var ${acc}: f32 = 0.0;`);
      const inner = (d: number) => {
        if (d === summed.length) {
          const prod = vals.map((v, k) => this.read(v, [...terms[k]!].map((l, i) => (v.shape[i] === 1 ? "0" : pos.get(l)!)))).join(" * ");
          this.lines.push(`  ${acc} = ${acc} + ${prod};`);
          return;
        }
        const l = summed[d]!, v = this.fresh("s");
        this.lines.push(`  for (var ${v}: i32 = 0; ${v} < ${this.bound(sizes.get(l)!)}; ${v}++) {`);
        pos.set(l, v);
        inner(d + 1);
        this.lines.push(`  }`);
      };
      inner(0);
      this.lines.push(`  ${this.read(res, oi)} = ${acc};`);
    });
    return res;
  }

  private reduce(v: Arr, fn: ArrayReduceFn, axes: ReadonlySet<number>, keepDims: boolean, shape: number[]): Arr {
    const out = this.alloc(shape);
    const kept = v.shape.map((_, d) => d).filter((d) => !axes.has(d));
    const red = [...axes].sort((a, b) => a - b);
    const count = red.reduce((n, d) => n * v.shape[d]!, 1);
    // iterate over the kept axes; the output index drops (or keeps as 0) the reduced ones
    this.loops(kept.map((d) => v.shape[d]!), (ki) => {
      const outIdx = keepDims ? v.shape.map((_, d) => (axes.has(d) ? "0" : ki[kept.indexOf(d)]!)) : ki;
      const acc = this.fresh("acc");
      const init = fn === "prod" ? "1.0" : fn === "max" || fn === "logsumexp" ? "-3.4e38" : fn === "min" ? "3.4e38" : "0.0";
      this.lines.push(`  var ${acc}: f32 = ${init};`);
      const pass = (body: (x: string) => string) => {
        const rv: string[] = [];
        for (const d of red) { const s = this.fresh("r"); rv.push(s); this.lines.push(`  for (var ${s}: i32 = 0; ${s} < ${this.bound(v.shape[d]!)}; ${s}++) {`); }
        const idx = v.shape.map((_, d) => (axes.has(d) ? rv[red.indexOf(d)]! : ki[kept.indexOf(d)]!));
        this.lines.push(`  ${body(this.read(v, idx))}`);
        for (const _ of red) this.lines.push(`  }`);
      };
      switch (fn) {
        case "sum": case "mean": pass((x) => `${acc} = ${acc} + ${x};`); break;
        case "prod": pass((x) => `${acc} = ${acc} * ${x};`); break;
        case "max": pass((x) => `${acc} = max(${acc}, ${x});`); break;
        case "min": pass((x) => `${acc} = min(${acc}, ${x});`); break;
        case "logsumexp": {
          pass((x) => `${acc} = max(${acc}, ${x});`);
          const s = this.fresh("acc");
          this.lines.push(`  var ${s}: f32 = 0.0;`);
          pass((x) => `${s} = ${s} + exp(${x} - ${acc});`);
          this.lines.push(`  ${acc} = ${acc} + log(${s});`);
          break;
        }
      }
      if (fn === "mean") this.lines.push(`  ${acc} = ${acc} / ${f32(count)};`);
      this.lines.push(`  ${this.read(out, outIdx)} = ${acc};`);
    });
    return out;
  }
}

/**
 * Names an expression's VALUE depends on: `exprNames` minus the operands of a `mul` with a literal 0 among its
 * operands — the emitter folds those to 0 without reading them (autodiff's `add(mul(x, 0), 1)` seeds), so a node
 * built only from such references is uniform, whatever its shape says.
 */
function effectiveNames(e: ArrayExpr, out = new Set<string>()): Set<string> {
  if (typeof e === "number") return out;
  if (typeof e === "string") { out.add(e); return out; }
  if (e.op === "arg") { out.add(e.name); return out; }
  if (e.op === "call") { for (const x of Object.values(e.inputs)) effectiveNames(x, out); return out; }
  if (e.op === "mul" && e.vals.some((v) => v === 0)) return out;
  const rec = e as unknown as Record<string, ArrayExpr | ArrayExpr[]>;
  for (const k of ["val", "vals", "min", "max", "cond", "indices"]) { const v = rec[k]; if (v === undefined) continue; if (Array.isArray(v)) v.forEach((x) => effectiveNames(x, out)); else effectiveNames(v, out); }
  return out;
}

/** occurrences of each name in the program's node expressions (with multiplicity) */
function useCounts(prog: Program): Map<string, number> {
  const counts = new Map<string, number>();
  const walk = (e: ArrayExpr): void => {
    if (typeof e === "number") return;
    if (typeof e === "string") { counts.set(e, (counts.get(e) ?? 0) + 1); return; }
    if (e.op === "arg") { counts.set(e.name, (counts.get(e.name) ?? 0) + 1); return; }
    if (e.op === "call") { Object.values(e.inputs).forEach(walk); return; }
    for (const k of ["val", "vals", "min", "max", "cond", "indices"]) { const v = (e as unknown as Record<string, ArrayExpr | ArrayExpr[]>)[k]; if (v === undefined) continue; if (Array.isArray(v)) v.forEach(walk); else walk(v); }
  };
  for (const n of prog.nodes) walk(n.expr);
  for (const o of Object.values(prog.outputs)) counts.set(o, (counts.get(o) ?? 0) + 1);
  return counts;
}

/**
 * Substitute every elementwise node that is read exactly once, by an elementwise node, into that consumer. The
 * consumer's expression becomes a tree the emitter turns into one loop (`elementwise`), so the intermediate is
 * never materialized. Outputs and multi-use nodes stay.
 */
export function fuseElementwise(prog: Program): Program {
  const counts = useCounts(prog);
  const outputs = new Set(Object.values(prog.outputs));
  const consumer = new Map<string, number>(); // name -> index of its only reading node
  prog.nodes.forEach((node, i) => { for (const dep of exprNames(node.expr)) consumer.set(dep, i); });
  const fused = new Map<string, ArrayExpr>();
  const subst = (e: ArrayExpr): ArrayExpr => {
    if (typeof e === "number") return e;
    if (typeof e === "string") return fused.get(e) ?? e;
    if (e.op === "arg") return fused.get(e.name) ?? e;
    if (e.op === "call") return { ...e, inputs: Object.fromEntries(Object.entries(e.inputs).map(([k, v]) => [k, subst(v)])) };
    const out = { ...e } as unknown as Record<string, unknown>;
    for (const k of ["val", "vals", "min", "max", "cond", "indices"]) { const v = out[k]; if (v === undefined) continue; out[k] = Array.isArray(v) ? (v as ArrayExpr[]).map(subst) : subst(v as ArrayExpr); }
    return out as unknown as ArrayExpr;
  };
  const nodes: Program["nodes"] = [];
  prog.nodes.forEach((node) => {
    const expr = subst(node.expr);
    const c = consumer.get(node.name);
    if (isElementwise(expr) && !outputs.has(node.name) && counts.get(node.name) === 1 && c !== undefined && isElementwise(prog.nodes[c]!.expr)) {
      fused.set(node.name, expr);
      return;
    }
    nodes.push({ name: node.name, expr });
  });
  return { ...prog, nodes };
}

/** ops whose output element depends only on the operands' element at the same (broadcast) index */
function isElementwise(e: ArrayExpr): boolean {
  if (typeof e !== "object") return false;
  const op = e.op;
  return op in UNARY || op in BINARY || op in COMPARE || NARY.has(op) || op === "clamp" || op === "where";
}

function lookup(name: string, env: ReadonlyMap<string, Arr>, path: string): Arr {
  const a = env.get(name);
  if (!a) throw new Error(`${path}: unknown name "${name}"`);
  return a;
}

/** resolve symbolic sizes in reshape / oneHot (at any depth) so shape inference sees numbers */
function concretize(e: ArrayExpr, sizeOf: (d: number | string) => number): ArrayExpr {
  if (typeof e !== "object") return e;
  if (e.op === "call") return { ...e, inputs: Object.fromEntries(Object.entries(e.inputs).map(([k, v]) => [k, concretize(v, sizeOf)])) };
  const out = { ...e } as unknown as Record<string, unknown>;
  for (const k of ["val", "vals", "min", "max", "cond", "indices"]) {
    const v = out[k];
    if (v === undefined) continue;
    out[k] = Array.isArray(v) ? (v as ArrayExpr[]).map((x) => concretize(x, sizeOf)) : concretize(v as ArrayExpr, sizeOf);
  }
  if (e.op === "reshape") out.shape = e.shape.map((d) => (d === -1 ? -1 : sizeOf(d)));
  if (e.op === "oneHot") out.size = sizeOf(e.size);
  return out as unknown as ArrayExpr;
}

/*******************************************************/

export interface NetFunction {
  /** the WGSL function text */
  code: string;
  /** function-scope floats used */
  floats: number;
  /** estimated work per point (see NET_MAX_WORK) */
  work: number;
  /** the dataset axis was streamed */
  streamed: boolean;
}

/**
 * Emit `fn <name>(p, pos) -> f32 | vecD` evaluating a net field (a program
 * whose sole input is the point) at `p`. Returns undefined when the net's
 * intermediates exceed NET_MAX_FLOATS (the caller falls back to CPU sampling).
 */
export function emitNetField(name: string, field: NetField, ctx: NetEmitContext, outputs: string[] = [field.output]): NetFunction | undefined {
  const D = ctx.D;
  const max = ctx.maxFloats ?? NET_MAX_FLOATS, mode = ctx.stream ?? STREAM;
  // A-normal form (every intermediate is a node, so liveness — array reuse, in-place updates — sees all of them),
  // then single-use elementwise producers fused back into their elementwise consumers (one loop, no intermediate)
  const prog = fuseElementwise(anfProgram(field.program, field.nets));
  // two emissions may run (batched, then streamed): the constants go into `data` once
  const offsets = new Map<ArrayLike<number>, number>();
  const upload = (d: ArrayLike<number>) => { let o = offsets.get(d); if (o === undefined) { o = ctx.upload(d); offsets.set(d, o); } return o; };
  const attempt = (streamed: boolean) => {
    const em = new NetEmitter({ ...ctx, upload }, field.nets);
    const inputs = new Map<string, Arr>([[pointInput(field.program), { shape: [D], kind: "point", base: "p" }]]);
    const outs = streamed ? em.streamed(prog, inputs, "net", outputs) : em.program(prog, inputs, "net");
    return { em, outs, streamed };
  };
  let r = mode === "always" ? undefined : attempt(false);
  if (mode !== "never" && (r === undefined || r.em.floats > max)) {
    try { const s = attempt(true); if (r === undefined || s.em.floats < r.em.floats) r = s; } catch (e) { if (!(e instanceof StreamError)) throw e; }
  }
  if (r === undefined || r.em.floats > max || r.em.work > (ctx.maxWork ?? WORK_LIMIT)) return undefined;
  const { em, outs } = r;
  // the outputs' elements concatenated: a scalar, a vecD, or (value + gradient) a vec(D+1)
  const comps: string[] = [];
  for (const o of outputs) {
    const out = outs[o];
    if (!out) throw new Error(`net field has no output "${o}"`);
    const n = out.shape.reduce((a, b) => a * b, 1);
    for (let i = 0; i < n; i++) comps.push(em.read(out, out.shape.length === 0 ? [] : [String(i)]));
  }
  if (comps.length > 4) throw new Error(`net field function would return ${comps.length} components (max 4)`);
  const type = comps.length === 1 ? "f32" : vecType(comps.length);
  const ret = comps.length === 1 ? comps[0]! : `${type}(${comps.join(", ")})`;
  return { code: `fn ${name}(p: ${vecType(D)}, pos: i32) -> ${type} {\n${em.lines.join("\n")}\n  return ${ret};\n}`, floats: em.floats, work: em.work, streamed: r.streamed };
}

/** function-scope floats and per-point work a net field needs (a dry emission), cached per program */
let sizeCache = new WeakMap<Program, { floats: number; work: number }>();
function netFieldSize(fd: NetScalarFieldData | NetVectorFieldData): { floats: number; work: number } {
  let n = sizeCache.get(fd.field.program);
  if (n === undefined) {
    try {
      const r = emitNetField("dry", fd.field, { D: fd.dimCount, upload: () => 0 });
      n = r ? { floats: r.floats, work: r.work } : { floats: Infinity, work: Infinity };
    } catch { n = { floats: Infinity, work: Infinity }; }
    sizeCache.set(fd.field.program, n);
  }
  return n;
}
export const netFieldFloats = (fd: NetScalarFieldData | NetVectorFieldData): number => netFieldSize(fd).floats;
export const netFieldWork = (fd: NetScalarFieldData | NetVectorFieldData): number => netFieldSize(fd).work;

/**
 * Whether the GPU program can evaluate `fd` without sampling it on the CPU:
 * cheap data always; net fields (and whatever is derived from them) when the
 * net fits in function-scope memory AND one lane can evaluate it in bounded
 * time (NET_MAX_WORK).
 */
export function gpuTranspilable(fd: FieldData): boolean {
  if (!fd.costly) return true;
  if (fd instanceof NetScalarFieldData || fd instanceof NetVectorFieldData) { const s = netFieldSize(fd); return s.floats <= NET_MAX_FLOATS && s.work <= WORK_LIMIT; }
  if (fd instanceof SymbolicScalarFieldData || fd instanceof SymbolicVectorFieldData)
    return [...Object.values(fd.args.scalars), ...Object.values(fd.args.vectors)].every(gpuTranspilable);
  if (fd instanceof PulledBackScalarFieldData || fd instanceof PulledBackVectorFieldData) return gpuTranspilable(fd.inner);
  return false;
}

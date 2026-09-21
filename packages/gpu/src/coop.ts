// The COOPERATIVE net kernel: one workgroup per grid point (notes/nets.md "Step 0, measured").
//
// The per-thread emitter (nets.ts) evaluates a whole net in one lane: optimal for small nets, usable at any point
// (isoline vertices, streamline steps), but a serial chain of storage reads — the MNIST MLP took 10 s per dispatch at
// any grid size. Here a WORKGROUP of WG threads evaluates the net for ONE point, and the same program is emitted with
// a different loop-to-thread mapping:
//
//   * arrays live in WORKGROUP memory (`var<workgroup>`, module scope; 16 KB by default, 32 KB on Apple GPUs) with the
//     same liveness-driven reuse as function-scope arrays in the per-thread emitter;
//   * every element loop is STRIDED across the workgroup (`for (f = lid; f < n; f += WG)`) and followed by a
//     `workgroupBarrier()`, so a node's readers see its writers; barriers are only ever emitted in uniform control
//     flow (loop bounds are literals or `nb_`, which reads read-only storage);
//   * the dataset axis is streamed in TILES of E examples (not one at a time): the arrays that carry it keep the axis
//     at size E, read through a tile view `sVar + e`, and the boundary reductions accumulate per tile. A tile lets a
//     contraction read its weight element ONCE for E examples — the per-lane kernel was memory-bound on the weight
//     stream, and the step-0 kernel gained 5× from E = 1 to 8;
//   * contractions have three mappings: OUTPUT-PARALLEL with a REGISTER TILE (thread ↔ output unit j, E accumulators,
//     the weight `W(t)[i, j]` read once per i: the matvecs), plain output-parallel (small outputs: the logits, the
//     per-example cross-entropy), and CONTRACTION-PARALLEL with a workgroup tree reduction (a small output over a
//     large contraction: the loss's mean, the gradient's ⟨g, h·Dₖ⟩ with output [K]).
//
// The kernel writes ONE value (or a small vector) per point into a resident values grid; the fused kernels then read
// that grid the way `costly` fields are already consumed (`residentReader`). Nets this kernel cannot take
// (StreamError: the dataset axis is consumed by something other than a reduction) stay on the CPU path.

import { anfProgram, inferExpr, pointInput, type NetField, type Program, type Shape } from "@tensatory/core";
import type { ArrayExpr, ArrayReduceFn } from "@tensatory/schema";
import { NetEmitter, StreamError, fuseElementwise, size, strides, type Arr, type NetEmitContext } from "./nets";
import { f32, gridWgsl, vecType, type GridRef } from "./wgsl";
import type { NetScalarFieldData, NetVectorFieldData } from "@tensatory/core";

/** threads per cooperative workgroup (WebGPU's default maxComputeInvocationsPerWorkgroup) */
export const COOP_WG = 256;
/** floats of workgroup memory the emitter may use at the default limit (16 KB), minus the reduction scratch */
export const COOP_DEFAULT_FLOATS = 16384 / 4 - COOP_WG - 64;
/** tile sizes tried, largest first (each must divide the dataset axis) */
const TILES = [8, 4, 2, 1];
/** most output elements a contraction-parallel mapping accumulates per thread */
const SMALL_OUT = 32;
/** most tile elements a register tile holds per thread */
const MAX_REGISTER_TILE = 16;

/** defaults of the two code-shape knobs (see CoopEmitContext); set from measurements in cooptiming.html */
let UNROLL_TILE = true;
let EXAMPLE_BOUND: "opaque" | "literal" = "opaque";
let FORCE_TILE: number | undefined;
export function setCoopCodeShape(o: { unroll?: boolean; exampleBound?: "opaque" | "literal"; tile?: number | null }): void {
  if (o.unroll !== undefined) UNROLL_TILE = o.unroll;
  if (o.exampleBound !== undefined) EXAMPLE_BOUND = o.exampleBound;
  if (o.tile !== undefined) FORCE_TILE = o.tile ?? undefined;
  capableCache = new WeakMap();
}

let WG_FLOATS = COOP_DEFAULT_FLOATS;
/** set from the device's maxComputeWorkgroupStorageSize (GpuBackend.create) */
export function setCoopWorkgroupBytes(bytes: number): void { WG_FLOATS = Math.floor(bytes / 4) - COOP_WG - 64; capableCache = new WeakMap(); }
export const coopWorkgroupFloats = (): number => WG_FLOATS;

export interface CoopEmitContext extends NetEmitContext {
  /** floats of workgroup memory available to arrays (device limit / 4 minus the reduction scratch) */
  readonly workgroupFloats?: number;
  /** force a tile size (tests) */
  readonly tile?: number;
  /** emit the small per-thread loops over the register tile UNROLLED (constant indices into the accumulators) instead
   *  of as `for` loops; measured with apps/viewer/public/cooptiming.html */
  readonly unroll?: boolean;
  /** the example loop's bound: opaque (`nb_`, the shader compiler cannot unroll the huge body) or literal */
  readonly exampleBound?: "opaque" | "literal";
}

const REDUCE_INIT = (fn: string) => (fn === "prod" ? "1.0" : fn === "max" || fn === "logsumexp" ? "-3.4e38" : fn === "min" ? "3.4e38" : "0.0");
const combine = (fn: string, x: string, y: string) => (fn === "sum" || fn === "mean" ? `${x} + ${y}` : fn === "prod" ? `${x} * ${y}` : fn === "max" ? `max(${x}, ${y})` : fn === "min" ? `min(${x}, ${y})` : `lse2_(${x}, ${y})`);

export class CoopEmitter extends NetEmitter {
  /** module-scope declarations (workgroup arrays) */
  readonly decls: string[] = [];
  /** the current tile size (examples per pass of the streamed loop) */
  E = 1;
  /** number of workgroup barriers emitted (diagnostics) */
  barriers = 0;
  private get unroll(): boolean { return (this.ctx as CoopEmitContext).unroll ?? UNROLL_TILE; }
  private get exampleBound(): "opaque" | "literal" { return (this.ctx as CoopEmitContext).exampleBound ?? EXAMPLE_BOUND; }

  protected override declare(name: string, n: number): void { this.decls.push(`var<workgroup> ${name}: array<f32, ${n}>;`); }

  /** literal bounds: the step-0 kernel ran 2–3× faster with the contraction loops unrolled; the tile loop (a huge
   *  body) gets `nb_` explicitly */
  protected override bound(n: number): string { return String(n); }

  /** workgroup arrays outlive the loop that allocated them: hand the loop's dead ones back to the pool */
  protected override endLoopScope(declared: ReadonlySet<string>): void {
    for (const base of [...this.owners.keys()]) if (!declared.has(base)) { this.owners.delete(base); this.release(base); }
  }

  protected sync(): void { this.lines.push(`  workgroupBarrier();`); this.barriers++; }

  /** every element loop is strided over the workgroup, then a barrier */
  protected override loops(shape: readonly number[], body: (idx: string[]) => void): void {
    const n = size(shape);
    if (n === 0) return;
    const f = this.fresh("fl");
    this.lines.push(`  for (var ${f}: i32 = lid; ${f} < ${n}; ${f} += WG) {`);
    const st = strides(shape);
    const idx = shape.map((s, d) => {
      if (s === 1) return "0";
      const v = this.fresh("ix");
      this.lines.push(`  let ${v}: i32 = ${st[d] === 1 ? f : `(${f} / ${st[d]})`}${d === 0 ? "" : ` % ${s}`};`);
      return v;
    });
    body(idx);
    this.lines.push(`  }`);
    this.sync();
  }

  /** plain per-thread nested loops (inside a strided loop, or for a small output every thread walks); `unrolled`
   *  emits the body once per index combination with literal indices */
  private serialLoops(shape: readonly number[], body: (idx: string[]) => void, unrolled = false): void {
    const idx: string[] = [];
    if (unrolled) {
      const open = (d: number) => {
        if (d === shape.length) { body(idx); return; }
        for (let i = 0; i < shape[d]!; i++) { idx.push(String(i)); open(d + 1); idx.pop(); }
      };
      open(0);
      return;
    }
    const open = (d: number) => {
      if (d === shape.length) { body(idx); return; }
      if (shape[d] === 1) { idx.push("0"); open(d + 1); idx.pop(); return; }
      const v = this.fresh("j");
      this.lines.push(`  for (var ${v}: i32 = 0; ${v} < ${shape[d]}; ${v}++) {`);
      idx.push(v);
      open(d + 1);
      idx.pop();
      this.lines.push(`  }`);
    };
    open(0);
  }

  /** reduce `expr` (one value per thread) across the workgroup with `fn`; the result is in `red_[0]` after this */
  private treeReduce(expr: string, fn: string): void {
    this.lines.push(`  red_[lid] = ${expr};`);
    this.sync();
    const s = this.fresh("hs");
    this.lines.push(`  for (var ${s}: i32 = WG / 2; ${s} > 0; ${s} = ${s} / 2) {`);
    this.lines.push(`    if (lid < ${s}) { red_[lid] = ${combine(fn, "red_[lid]", `red_[lid + ${s}]`)}; }`);
    this.lines.push(`    workgroupBarrier();`);
    this.barriers++;
    this.lines.push(`  }`);
  }

  /** tile view: the streamed axis stays, at size E, indexed `sVar + e` */
  protected override view(a: Arr, sAxis: number, sVar: string): Arr {
    const shape = a.shape.map((s, i) => (i === sAxis ? this.E : s));
    if (a.kind === "lit") return { ...a, shape };
    if (a.kind === "priv" || a.kind === "data") return { kind: "view", base: a.base, shape, full: a.shape, sAxis, sVar, store: a.kind };
    if (a.kind === "expr") return { kind: "expr", base: a.base, shape, fn: (idx) => { const full = [...idx]; full[sAxis] = `(${sVar} + ${idx[sAxis]})`; return a.fn!(full); } };
    throw new StreamError(`cannot view a ${a.kind} array per tile`);
  }

  override read(a: Arr, idx: readonly string[]): string {
    if (a.kind === "view") {
      const fullIdx = [...idx];
      fullIdx[a.sAxis!] = `(${a.sVar} + ${idx[a.sAxis!]})`;
      const f = this.flat(a.full!, fullIdx);
      return a.store === "data" ? `data[${a.base} + ${f}]` : `${a.base}[${f}]`;
    }
    return super.read(a, idx);
  }

  /*******************************************************/
  /* contractions and reductions: the three mappings */

  protected override einsum(terms: string[], out: string, vals: Arr[], shape: number[]): Arr {
    const sizes = new Map<string, number>();
    terms.forEach((t, k) => [...t].forEach((l, d) => { const s = vals[k]!.shape[d]!; const prev = sizes.get(l); if (prev === undefined || prev === 1) sizes.set(l, s); }));
    const summed = [...new Set(terms.join(""))].filter((l) => !out.includes(l));
    const outLetters = [...out];
    const diagonal = new Set(outLetters).size !== outLetters.length;
    const summedSize = summed.reduce((n, l) => n * sizes.get(l)!, 1);
    const outSize = size(shape);
    // the diagonal write: the base emitter's code, over strided loops
    if (diagonal) return super.einsum(terms, out, vals, shape);
    // LAZY (never materialized), as in the base emitter — but a tile VIEW of storage data counts as stable here: the
    // node is only ever read inside the tile loop, where its loop variable is in scope. This keeps the displacement
    // term `einsum(t, XD)` of a hoisted layer an expression the following elementwise fill reads in place.
    const stable = (a: Arr) => a.kind === "data" || a.kind === "lit" || a.kind === "point" || a.kind === "expr" || (a.kind === "view" && a.store === "data");
    if (outSize >= 512 && summedSize <= 8 && vals.every(stable)) {
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
    this.work += this.scale * outSize * summedSize;
    const res = this.alloc(shape);
    const readAt = (k: number, pos: ReadonlyMap<string, string>) => this.read(vals[k]!, [...terms[k]!].map((l, i) => (vals[k]!.shape[i] === 1 ? "0" : pos.get(l)!)));
    const product = (pos: ReadonlyMap<string, string>, skip = -1) => vals.map((_, k) => (k === skip ? null : readAt(k, pos))).filter((x): x is string => x !== null).join(" * ");
    const summedLoops = (pos: Map<string, string>, body: () => void) => {
      const open = (d: number) => {
        if (d === summed.length) { body(); return; }
        const l = summed[d]!, v = this.fresh("s");
        this.lines.push(`  for (var ${v}: i32 = 0; ${v} < ${sizes.get(l)}; ${v}++) {`);
        pos.set(l, v);
        open(d + 1);
        this.lines.push(`  }`);
      };
      open(0);
    };

    // CONTRACTION-PARALLEL: a small output over a large contraction — every thread sums a strided slice of the summed
    // index space for every output element, then the workgroup reduces
    if (outSize <= SMALL_OUT && summedSize >= 2 * COOP_WG && summed.length) {
      const acc = this.fresh("acc");
      this.lines.push(`  var ${acc}: array<f32, ${outSize}>;`);
      this.lines.push(`  for (var q: i32 = 0; q < ${outSize}; q++) { ${acc}[q] = 0.0; }`);
      const f = this.fresh("fl");
      this.lines.push(`  for (var ${f}: i32 = lid; ${f} < ${summedSize}; ${f} += WG) {`);
      const st = strides(summed.map((l) => sizes.get(l)!));
      const pos = new Map<string, string>();
      summed.forEach((l, d) => { const v = this.fresh("ix"); this.lines.push(`  let ${v}: i32 = ${st[d] === 1 ? f : `(${f} / ${st[d]})`}${d === 0 ? "" : ` % ${sizes.get(l)}`};`); pos.set(l, v); });
      this.serialLoops(shape, (oi) => {
        outLetters.forEach((l, d) => pos.set(l, oi[d]!));
        this.lines.push(`  ${acc}[${this.flat(shape, oi)}] = ${acc}[${this.flat(shape, oi)}] + ${product(pos)};`);
      });
      this.lines.push(`  }`);
      for (let o = 0; o < outSize; o++) {
        this.treeReduce(`${acc}[${o}]`, "sum");
        this.lines.push(`  if (lid == 0) { ${res.base}[${o}] = red_[0]; }`);
        this.sync();
      }
      return res;
    }

    // OUTPUT-PARALLEL. With a REGISTER TILE when a storage-resident operand (the weight) lacks some output letters (the
    // tile axis e): thread ↔ the letters it has, E accumulators, the weight read once per summed index
    const heavy = vals.map((_, k) => k).filter((k) => vals[k]!.kind === "data" || vals[k]!.kind === "expr" || (vals[k]!.kind === "view" && vals[k]!.store === "data")).sort((a, b) => size(vals[b]!.shape) - size(vals[a]!.shape))[0];
    const tileLetters = heavy === undefined ? [] : outLetters.filter((l) => !terms[heavy]!.includes(l) && sizes.get(l)! > 1);
    const tileSize = tileLetters.reduce((n, l) => n * sizes.get(l)!, 1);
    const parLetters = outLetters.filter((l) => !tileLetters.includes(l));
    const parSize = parLetters.reduce((n, l) => n * sizes.get(l)!, 1);
    if (heavy !== undefined && vals.length >= 2 && tileLetters.length && tileSize <= MAX_REGISTER_TILE && parSize >= COOP_WG / 2) {
      this.loops(parLetters.map((l) => sizes.get(l)!), (pi) => {
        const pos = new Map<string, string>(parLetters.map((l, d) => [l, pi[d]!]));
        const acc = this.fresh("acc");
        const tileShape = tileLetters.map((l) => sizes.get(l)!);
        // unrolled: one scalar per tile element (WebKit's compiler kept an indexed private array in memory)
        const slot = (ti: readonly string[]) => (this.unroll ? `${acc}_${this.flat(tileShape, ti)}` : `${acc}[${this.flat(tileShape, ti)}]`);
        if (this.unroll) this.lines.push(`  ${Array.from({ length: tileSize }, (_, q) => `var ${acc}_${q}: f32 = 0.0;`).join(" ")}`);
        else { this.lines.push(`  var ${acc}: array<f32, ${tileSize}>;`); this.lines.push(`  for (var q: i32 = 0; q < ${tileSize}; q++) { ${acc}[q] = 0.0; }`); }
        summedLoops(pos, () => {
          const w = this.fresh("w");
          this.lines.push(`  let ${w}: f32 = ${readAt(heavy, pos)};`);
          this.serialLoops(tileShape, (ti) => {
            tileLetters.forEach((l, d) => pos.set(l, ti[d]!));
            this.lines.push(`  ${slot(ti)} = ${slot(ti)} + ${product(pos, heavy)} * ${w};`);
          }, this.unroll);
        });
        this.serialLoops(tileShape, (ti) => {
          tileLetters.forEach((l, d) => pos.set(l, ti[d]!));
          this.lines.push(`  ${this.read(res, outLetters.map((l) => pos.get(l)!))} = ${slot(ti)};`);
        }, this.unroll);
      });
      return res;
    }
    // plain: thread ↔ output element, serial contraction
    this.loops(shape, (oi) => {
      const pos = new Map<string, string>(outLetters.map((l, d) => [l, oi[d]!]));
      const acc = this.fresh("acc");
      this.lines.push(`  var ${acc}: f32 = 0.0;`);
      summedLoops(pos, () => this.lines.push(`  ${acc} = ${acc} + ${product(pos)};`));
      this.lines.push(`  ${this.read(res, oi)} = ${acc};`);
    });
    return res;
  }

  protected override reduce(v: Arr, fn: ArrayReduceFn, axes: ReadonlySet<number>, keepDims: boolean, shape: number[]): Arr {
    const kept = v.shape.map((_, d) => d).filter((d) => !axes.has(d));
    const red = [...axes].sort((a, b) => a - b);
    const count = red.reduce((n, d) => n * v.shape[d]!, 1);
    const outSize = size(shape);
    // CONTRACTION-PARALLEL: a small output over a long reduction (the loss's mean over the tile and the classes)
    if (outSize <= SMALL_OUT && count >= 2 * COOP_WG && fn !== "logsumexp") {
      const out = this.alloc(shape);
      const acc = this.fresh("acc");
      this.lines.push(`  var ${acc}: array<f32, ${outSize}>;`);
      this.lines.push(`  for (var q: i32 = 0; q < ${outSize}; q++) { ${acc}[q] = ${REDUCE_INIT(fn)}; }`);
      const f = this.fresh("fl");
      this.lines.push(`  for (var ${f}: i32 = lid; ${f} < ${count}; ${f} += WG) {`);
      const st = strides(red.map((d) => v.shape[d]!));
      const rv = red.map((d, i) => { const x = this.fresh("ix"); this.lines.push(`  let ${x}: i32 = ${st[i] === 1 ? f : `(${f} / ${st[i]})`}${i === 0 ? "" : ` % ${v.shape[d]}`};`); return x; });
      const keptShape = kept.map((d) => v.shape[d]!);
      this.serialLoops(keptShape, (ki) => {
        const idx = v.shape.map((_, d) => (axes.has(d) ? rv[red.indexOf(d)]! : ki[kept.indexOf(d)]!));
        const a = `${acc}[${this.flat(keptShape, ki)}]`;
        this.lines.push(`  ${a} = ${combine(fn, a, this.read(v, idx))};`);
      });
      this.lines.push(`  }`);
      const outIdxOf = (k: number): string[] => { const st2 = strides(keptShape); const ki = keptShape.map((s, d) => String(Math.floor(k / st2[d]!) % s)); return keepDims ? v.shape.map((_, d) => (axes.has(d) ? "0" : ki[kept.indexOf(d)]!)) : ki; };
      for (let o = 0; o < outSize; o++) {
        this.treeReduce(`${acc}[${o}]`, fn);
        this.lines.push(`  if (lid == 0) { ${this.read(out, outIdxOf(o))} = red_[0]${fn === "mean" ? ` / ${f32(count)}` : ""}; }`);
        this.sync();
      }
      return out;
    }
    return super.reduce(v, fn, axes, keepDims, shape);
  }

  /*******************************************************/
  /* the tiled stream */

  /**
   * `streamed` with the streamed axis kept at size E per pass: the loop steps by E, arrays carrying the axis are
   * tile views, and the boundary nodes (reductions consuming the axis) accumulate the tile's contribution. The
   * per-example legality rules are the same (`perExample` is consulted for them and for the accumulation function);
   * a `mean` over the axis is summed per tile and divided by the full count at the end.
   */
  tiled(prog: Program, inputs: ReadonlyMap<string, Arr>, path: string, wanted: readonly string[], E: number): Record<string, Arr> {
    const P = this.streamPlan(prog, inputs, path, wanted);
    const { env, sizeOf, fail, S, n, names, sPos, sPosOfExpr, concrete, nodes, keep, plainAt } = P;
    if (n % E !== 0) fail(`tile ${E} does not divide ${n}`);
    this.E = E;
    const tileSizeOf = (d: number | string): number => (d === S ? E : sizeOf(d));
    const tileShape = (sh: Shape) => sh.map(tileSizeOf);
    const hints = (inLoop: boolean) => { const h = new Map<string, readonly number[]>(); for (const [k, sh] of names) h.set(k, inLoop ? tileShape(sh) : concrete(sh)); return h; };
    const nameShape = new Map(names);
    const fullShape = (x: ArrayExpr): Shape => (typeof x === "string" ? names.get(x)! : typeof x === "object" && x.op === "arg" ? names.get(x.name)! : inferExpr(x, { names: nameShape, axisNames: new Set([S]), nets: this.nets }, [path]));

    this.shapeHints = hints(false);
    this.emitNodes(plainAt(-1), env, sizeOf, path, keep, true);
    for (const nd of plainAt(-1)) if (sPos(names.get(nd.name)!) !== undefined && env.get(nd.name)!.kind !== "lit") fail(`"${nd.name}" has the streamed axis but depends on no batched value and is not uniform`);

    for (let L = 0; L < P.loops; L++) {
      const { boundaries, body } = P.bodyOf(L);
      const accs = new Map<string, { acc: Arr; fn: string; divisor: number }>();
      for (const b of boundaries) {
        const fn = this.perExample(b.expr, sPosOfExpr, S, n, path).fn;
        if (fn === undefined) return fail(`"${b.name}" consumes the streamed axis with "${typeof b.expr === "object" ? b.expr.op : b.expr}", not a reduction`);
        // a mean over the streamed axis (and maybe others): summed per tile, divided by the full count at the end
        let divisor = 1;
        if (fn === "mean" && typeof b.expr === "object" && b.expr.op === "reduce") {
          const sh = fullShape(b.expr.val);
          const r = sh.length;
          const axes = b.expr.axes === undefined ? sh.map((_, i) => i) : b.expr.axes.map((a) => (a < 0 ? r + a : a));
          divisor = axes.reduce((p, a) => p * sizeOf(sh[a]!), 1);
        }
        this.scratch = [];
        const acc = this.alloc(concrete(names.get(b.name)!));
        this.scratch = [];
        (this.owners.get(acc.base) ?? this.owners.set(acc.base, new Set()).get(acc.base)!).add(b.name);
        this.loops(acc.shape, (i) => this.lines.push(`  ${this.read(acc, i)} = ${REDUCE_INIT(fn)};`));
        accs.set(b.name, { acc, fn, divisor });
      }
      const sVar = this.fresh("n");
      this.lines.push(`  for (var ${sVar}: i32 = 0; ${sVar} < ${this.exampleBound === "literal" ? n : `nb_(${n})`}; ${sVar} += ${E}) {`);
      const outerScale = this.scale; this.scale *= n / E;
      const declared = new Set(this.sizes.keys());
      const loopEnv = new Map<string, Arr>();
      for (const [k, a] of env) { const p = sPos(names.get(k) ?? []); loopEnv.set(k, p === undefined ? a : this.view(a, p, sVar)); }
      this.shapeHints = hints(true);
      const order = nodes.filter((nd) => body.has(nd.name) || accs.has(nd.name));
      const outer = new Set(loopEnv.keys());
      this.emitNodes(order, loopEnv, tileSizeOf, `${path}.n`, outer, true, (nd, e) => {
        const { fn } = this.perExample(nd.expr, sPosOfExpr, S, n, path); // legality of this op on a tile
        const a = accs.get(nd.name);
        let expr = nd.expr;
        if (a && fn === "mean" && typeof expr === "object" && expr.op === "reduce") expr = { ...expr, fn: "sum" };
        const v = this.expr(expr, e, tileSizeOf, `${path}.${nd.name}`);
        if (!a) return this.reconcile(v, tileShape(names.get(nd.name)!), nd.name);
        const src = this.reconcile(v, a.acc.shape, nd.name);
        this.loops(a.acc.shape, (i) => { const x = this.read(a.acc, i); this.lines.push(`  ${x} = ${combine(a.fn, x, this.read(src, i))};`); });
        return a.acc;
      });
      this.lines.push(`  }`);
      this.scale = outerScale;
      this.endLoopScope(declared);
      this.shapeHints = hints(false);
      for (const [nm, { acc, fn, divisor }] of accs) {
        if (fn === "mean") this.loops(acc.shape, (i) => this.lines.push(`  ${this.read(acc, i)} = ${this.read(acc, i)} / ${f32(divisor)};`));
        env.set(nm, acc);
      }
      this.emitNodes(plainAt(L), env, sizeOf, path, keep, true);
    }
    this.shapeHints = new Map();
    return Object.fromEntries(wanted.map((o) => [o, env.get(prog.outputs[o]!)!]));
  }
}

/*******************************************************/

export interface CoopKernel {
  /** the WGSL entry point + workgroup declarations (to be appended to a ProgramBuilder library) */
  code: string;
  /** workgroup floats used */
  floats: number;
  /** estimated multiply-adds per point (all threads together) */
  work: number;
  /** the tile size chosen */
  tile: number;
  /** output components per point */
  channels: number;
  barriers: number;
}

/**
 * Emit the cooperative kernel `main` for a net field: one workgroup per grid point, the point from the dispatch
 * grid header (`data`, as the sampling kernels do) offset by `params[0]` (the chunk), the wanted outputs' elements
 * written to `out[point * channels + c]` by thread 0. Returns undefined when the net cannot be streamed or does not
 * fit workgroup memory at any tile size.
 */
export function emitCoopKernel(field: NetField, ctx: CoopEmitContext, outputs: string[], G: GridRef): CoopKernel | undefined {
  const D = ctx.D;
  const budget = ctx.workgroupFloats ?? COOP_DEFAULT_FLOATS;
  const prog = fuseElementwise(anfProgram(field.program, field.nets));
  const offsets = new Map<ArrayLike<number>, number>();
  const upload = (d: ArrayLike<number>) => { let o = offsets.get(d); if (o === undefined) { o = ctx.upload(d); offsets.set(d, o); } return o; };
  const tiles = ctx.tile !== undefined ? [ctx.tile] : FORCE_TILE !== undefined ? [FORCE_TILE] : TILES;
  let best: { em: CoopEmitter; outs: Record<string, Arr>; E: number } | undefined;
  for (const E of tiles) {
    const em = new CoopEmitter({ ...ctx, upload }, field.nets);
    const inputs = new Map<string, Arr>([[pointInput(field.program), { shape: [D], kind: "point", base: "p" }]]);
    let outs: Record<string, Arr>;
    try { outs = em.tiled(prog, inputs, "net", outputs, E); }
    catch (e) { if (e instanceof StreamError) { if (/does not divide/.test(e.message)) continue; return undefined; } throw e; }
    if (em.floats <= budget) { best = { em, outs, E }; break; }
  }
  if (!best) return undefined;
  const { em, outs, E } = best;
  const comps: string[] = [];
  for (const o of outputs) {
    const out = outs[o];
    if (!out) throw new Error(`net field has no output "${o}"`);
    const n = out.shape.reduce((a, b) => a * b, 1);
    for (let i = 0; i < n; i++) comps.push(em.read(out, out.shape.length === 0 ? [] : [String(i)]));
  }
  // grid position -> point, row-major, from the header
  const idx: string[] = ["  var rem: i32 = point;"];
  const pc: string[] = [];
  for (let d = 0; d < D; d++) {
    idx.push(`  let gs${d}: i32 = ${G.s(String(d))}; let gg${d}: i32 = rem / gs${d}; rem = rem - gg${d} * gs${d};`);
    pc.push(`${G.a(String(d))} + f32(gg${d}) * ${G.h(String(d))}`);
  }
  const p = D === 1 ? pc[0]! : `${vecType(D)}(${pc.join(", ")})`;
  const code = `@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<u32>;
const WG: i32 = ${COOP_WG};
var<workgroup> red_: array<f32, ${COOP_WG}>;
${em.decls.join("\n")}
@compute @workgroup_size(${COOP_WG}) fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid3: vec3<u32>) {
  let point: i32 = i32(wid.x + params[0]);
  if (point >= ${G.count}) { return; }
  let lid: i32 = i32(lid3.x);
${idx.join("\n")}
  let p = ${p};
  let pos: i32 = point;
${em.lines.join("\n")}
  if (lid == 0) {
${comps.map((c, i) => `    out[point * ${comps.length} + ${i}] = ${c};`).join("\n")}
  }
}`;
  return { code, floats: em.floats, work: em.work, tile: E, channels: comps.length, barriers: em.barriers };
}

/** dry emissions per program and budget: whether the cooperative kernel can take a net field */
let capableCache = new WeakMap<Program, Map<string, boolean>>();
export function clearCoopCache(): void { capableCache = new WeakMap(); }

/**
 * Whether a cooperative kernel can evaluate `fd` on the current device (its dataset axis streams and its arrays fit
 * workgroup memory at some tile size). A dry emission, cached per program and output.
 */
export function coopCapable(fd: NetScalarFieldData | NetVectorFieldData): boolean {
  const key = `${fd.field.output}|${WG_FLOATS}`;
  let m = capableCache.get(fd.field.program);
  if (!m) capableCache.set(fd.field.program, (m = new Map()));
  let ok = m.get(key);
  if (ok === undefined) {
    try { ok = emitCoopKernel(fd.field, { D: fd.dimCount, upload: () => 0 }, [fd.field.output], gridWgsl("dg_", "data", 0).ref) !== undefined; }
    catch { ok = false; }
    m.set(key, ok);
  }
  return ok;
}

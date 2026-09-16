// Streamlines of a vector field: RK4 on the *unit* field so points are spaced
// uniformly (one step = `step` world units); integrated both ways from each
// seed. Three seeding strategies ("modes") deal with oversampling:
//
//  * stratified — one jittered seed per cell of a grid filling the box, every
//    line integrated to the step cap. Cheap and parallel, but lines pile up
//    where the field converges (valleys of a gradient descent) and leave
//    starved patches elsewhere.
//  * evenly-spaced — Jobard–Lefer: new seeds are placed d_sep beside existing
//    lines and a line stops when it comes within d_test of another one.
//    Uniform density everywhere. Sequential (CPU); the result is a seed list
//    with per-seed step budgets that any integrator (the GPU kernels included)
//    re-integrates to the same lines.
//  * coverage — stratified, then a histogram of vertices on the seed grid
//    adds seeds in starved cells (one or two rounds). Fixes the dark patches,
//    the extra lines still drain into the valleys.

import type { VectorFieldData } from "../fields/fieldData";
import type { Box } from "../geometry/box";

export type StreamlineMode = "stratified" | "evenly-spaced" | "coverage";
export const STREAMLINE_MODES: readonly StreamlineMode[] = ["stratified", "evenly-spaced", "coverage"];

export interface StreamlineOptions {
  /**
   * stratified / coverage: number of seed cells (one jittered seed per cell; the actual count is the nearest grid).
   * evenly-spaced: target density — the separation d_sep is the side of a cell of that grid, so the density is
   * comparable to a stratified set of `count` lines without the pile-up.
   */
  count: number;
  /** max integration steps in each direction */
  maxSteps: number;
  /** step length in world units */
  step: number;
  /** +1 follows the field (ascent for a gradient), -1 flows against it (descent) */
  sign?: 1 | -1;
  /**
   * integrate against the direction as well, so a line passes through its seed (default true). The viewer
   * passes false: a line then STARTS at its seed and runs in the chosen direction only, so line starts stay
   * as distributed as the seeds and lines only concentrate where the flow converges (their ends) — with both
   * ways, the backward halves of descending lines are ascending lines and pile up at the field's sources.
   */
  bidirectional?: boolean;
  /** RNG seed for the jitter */
  seed?: number;
  /** restrict seeds / integration to this box (defaults to the field's box) */
  box?: Box;
  /** seeding strategy, default "stratified" */
  mode?: StreamlineMode;
  /** evenly-spaced: d_test / d_sep, the fraction of the separation at which a line stops near another (default 0.5) */
  testRatio?: number;
  /** coverage: number of fill rounds (default 2) */
  rounds?: number;
}

export interface Streamline {
  /** flat [x0, y0, x1, y1, ...] in world coordinates (2D); D-dimensional in general */
  points: Float64Array;
  /** total arc length (steps * step) */
  length: number;
  /** per-line random phase in [0, 1) for particle animation */
  phase: number;
}

/** deterministic LCG in [0, 1); exact 32-bit arithmetic (Math.imul) so other implementations (WGSL) reproduce it bit for bit */
export function lcg(seed: number): () => number {
  let s = seed >>> 0;
  return () => ((s = (Math.imul(s, 1103515245) + 12345) & 0x7fffffff) / 0x7fffffff);
}

export interface StreamlineSeeds {
  /** flat [x0, y0, x1, y1, ...] seed points */
  points: Float64Array;
  /** per-line phase in [0, 1) */
  phases: Float64Array;
  /**
   * optional per-seed step budgets, flat [back0, fwd0, back1, fwd1, ...] (each ≤ maxSteps); absent = maxSteps
   * both ways. Evenly-spaced planning stops lines where they approach others and records the step counts here,
   * so re-integrating the seeds (CPU or GPU, f64 or f32) reproduces the planned lines without the spatial test.
   */
  budgets?: Uint32Array;
}

/** seeds together with the lines they were planned from (already integrated) */
export interface StreamlinePlan {
  seeds: StreamlineSeeds;
  lines: Streamline[];
  /** the separation lines were planned at (evenly-spaced), or the seed-cell side (others) */
  separation: number;
}

/** side of a seed cell: ≈`count` cells as close to cubes as possible filling the box */
export function seedCellSide(box: Box, count: number): number {
  const D = box.dimCount;
  const vol = box.size.reduce((a, b) => a * (b || 1), 1);
  return Math.pow(vol / Math.max(1, count), 1 / D);
}

/** the seed grid of `streamlineSeeds`: cells per axis */
function seedCells(box: Box, count: number): number[] {
  const side = seedCellSide(box, count);
  return box.size.map((sz) => Math.max(1, Math.round((sz || side) / side)));
}

/** jittered seeds: ≈`count` cells as close to cubes as possible filling the box, one random point per cell */
export function streamlineSeeds(box: Box, count: number, seed = 12345): StreamlineSeeds {
  const D = box.dimCount;
  const rand = lcg(seed);
  const size = box.size;
  const cnt = seedCells(box, count);
  const total = cnt.reduce((a, b) => a * b, 1);
  const points = new Float64Array(total * D), phases = new Float64Array(total);
  const idx = new Array<number>(D).fill(0);
  for (let c = 0; c < total; c++) {
    let rem = c;
    for (let d = D - 1; d >= 0; d--) { idx[d] = rem % cnt[d]!; rem = Math.floor(rem / cnt[d]!); }
    for (let d = 0; d < D; d++) points[c * D + d] = box.a[d]! + ((idx[d]! + rand()) / cnt[d]!) * size[d]!;
    phases[c] = rand();
  }
  return { points, phases };
}

/** integrate with the strategy of `opts.mode` (default stratified) */
export function integrateStreamlines(field: VectorFieldData, opts: StreamlineOptions): Streamline[] {
  return planStreamlines(field, opts).lines;
}

/** seeds (with budgets where the mode needs them) and lines for any mode */
export function planStreamlines(field: VectorFieldData, opts: StreamlineOptions): StreamlinePlan {
  const mode = opts.mode ?? "stratified";
  if (mode === "evenly-spaced") return evenlySpacedStreamlines(field, opts);
  if (mode === "coverage") return coverageStreamlines(field, opts);
  const box = opts.box ?? field.box;
  const seeds = streamlineSeeds(box, opts.count, opts.seed);
  return { seeds, lines: integrateFromSeeds(field, seeds, opts), separation: seedCellSide(box, opts.count) };
}

type IntegrateOpts = Pick<StreamlineOptions, "maxSteps" | "step" | "sign" | "box" | "bidirectional">;

/** one-direction RK4 integrator on the unit field; `stop(q, n)` (n = steps taken so far) ends a line early */
function makeIntegrator(field: VectorFieldData, opts: IntegrateOpts) {
  const D = field.dimCount;
  const box = opts.box ?? field.box;
  const sgn = opts.sign ?? 1;
  const h = opts.step;
  const tmp = new Float64Array(D), q = new Float64Array(D);

  /** unit direction at q (into out); false where the field vanishes / is undefined / outside */
  const dir = (at: Float64Array, out: Float64Array): boolean => {
    if (!box.contains(at, 1e-12)) return false;
    field.fn(at, -1, out);
    let l = 0;
    for (let d = 0; d < D; d++) l += out[d]! * out[d]!;
    if (!(l > 1e-24) || !Number.isFinite(l)) return false;
    const k = sgn / Math.sqrt(l);
    for (let d = 0; d < D; d++) out[d]! *= k;
    return true;
  };
  const k1 = new Float64Array(D), k2 = new Float64Array(D), k3 = new Float64Array(D), k4 = new Float64Array(D);
  return (seed: ArrayLike<number>, s: 1 | -1, steps: number, stop?: (q: Float64Array, n: number) => boolean): number[] => {
    const pts: number[] = [];
    for (let d = 0; d < D; d++) q[d] = seed[d]!;
    for (let n = 0; n < steps; n++) {
      if (!dir(q, k1)) break;
      for (let d = 0; d < D; d++) tmp[d] = q[d]! + s * 0.5 * h * k1[d]!;
      if (!dir(tmp, k2)) break;
      for (let d = 0; d < D; d++) tmp[d] = q[d]! + s * 0.5 * h * k2[d]!;
      if (!dir(tmp, k3)) break;
      for (let d = 0; d < D; d++) tmp[d] = q[d]! + s * h * k3[d]!;
      if (!dir(tmp, k4)) break;
      for (let d = 0; d < D; d++) q[d]! += (s * h / 6) * (k1[d]! + 2 * k2[d]! + 2 * k3[d]! + k4[d]!);
      if (!box.contains(q, 1e-12)) break;
      if (stop && stop(q, n + 1)) break;
      for (let d = 0; d < D; d++) pts.push(q[d]!);
    }
    return pts;
  };
}

/** assemble back (nearest-first) + seed + fwd into one line; undefined when it has fewer than 2 points */
function assemble(D: number, seed: ArrayLike<number>, back: number[], fwd: number[], h: number, phase: number): Streamline | undefined {
  const n = back.length / D + 1 + fwd.length / D;
  if (n < 2) return undefined;
  const pts = new Float64Array(n * D);
  let o = 0;
  for (let i = back.length / D - 1; i >= 0; i--) for (let d = 0; d < D; d++) pts[o++] = back[i * D + d]!;
  for (let d = 0; d < D; d++) pts[o++] = seed[d]!;
  for (let i = 0; i < fwd.length; i++) pts[o++] = fwd[i]!;
  return { points: pts, length: (n - 1) * h, phase };
}

/**
 * Integrate one streamline per seed (both directions unless `bidirectional` is false, `seeds.budgets`
 * capping the steps per direction below `opts.maxSteps` where present). Seeds whose line has fewer than
 * 2 points are skipped, so the result may be shorter than the seed list.
 */
export function integrateFromSeeds(field: VectorFieldData, seeds: StreamlineSeeds, opts: IntegrateOpts): Streamline[] {
  const D = field.dimCount;
  const integrate = makeIntegrator(field, opts);
  const total = seeds.phases.length;
  const lines: Streamline[] = [];
  const seed = new Float64Array(D);
  const M = opts.maxSteps, MB = opts.bidirectional === false ? 0 : M, budgets = seeds.budgets;
  for (let c = 0; c < total; c++) {
    for (let d = 0; d < D; d++) seed[d] = seeds.points[c * D + d]!;
    const nb = budgets ? Math.min(MB, budgets[2 * c]!) : MB, nf = budgets ? Math.min(M, budgets[2 * c + 1]!) : M;
    const line = assemble(D, seed, integrate(seed, -1, nb), integrate(seed, 1, nf), opts.step, seeds.phases[c]!);
    if (line) lines.push(line);
  }
  return lines;
}

/*******************************************************/
/* evenly-spaced (Jobard–Lefer) */

/** growable point set hashed on a grid of `cell` (2D or 3D); each point remembers its line and ordinal along it */
class PointHash {
  private readonly cells = new Map<number, number[]>();
  private readonly D: number;
  private readonly pts: number[] = [];
  private line: number[] = [];
  private ord: number[] = [];
  /** `a`: hash origin; `n`: cells per axis (for the key) */
  constructor(private readonly a: readonly number[], private readonly cell: number, private readonly n: readonly number[]) { this.D = a.length; }
  get count(): number { return this.line.length; }
  private key(c0: number, c1: number, c2: number): number { return c0 + this.n[0]! * (c1 + this.n[1]! * c2); }
  private cellOf(p: ArrayLike<number>, d: number): number { return Math.floor((p[d]! - this.a[d]!) / this.cell); }
  add(p: ArrayLike<number>, line: number, ord: number): void {
    const i = this.line.length;
    for (let d = 0; d < this.D; d++) this.pts.push(p[d]!);
    this.line.push(line); this.ord.push(ord);
    const k = this.key(this.cellOf(p, 0), this.cellOf(p, 1), this.D > 2 ? this.cellOf(p, 2) : 0);
    const list = this.cells.get(k);
    if (list) list.push(i); else this.cells.set(k, [i]);
  }
  /**
   * is any point within `r` (≤ cell) of p? Points of line `self` closer than `minOrd` ordinals to
   * ordinal `ord` are ignored (a line must not stop on its own recent past).
   */
  near(p: ArrayLike<number>, r: number, self = -1, ord = 0, minOrd = 0): boolean {
    const r2 = r * r, D = this.D;
    const c0 = this.cellOf(p, 0), c1 = this.cellOf(p, 1), c2 = D > 2 ? this.cellOf(p, 2) : 0;
    const k2lo = D > 2 ? c2 - 1 : 0, k2hi = D > 2 ? c2 + 1 : 0;
    for (let k = k2lo; k <= k2hi; k++) for (let j = c1 - 1; j <= c1 + 1; j++) for (let i = c0 - 1; i <= c0 + 1; i++) {
      const list = this.cells.get(this.key(i, j, k));
      if (!list) continue;
      for (const q of list) {
        if (this.line[q] === self && Math.abs(this.ord[q]! - ord) < minOrd) continue;
        let d2 = 0;
        for (let d = 0; d < D; d++) { const dd = this.pts[q * D + d]! - p[d]!; d2 += dd * dd; }
        if (d2 < r2) return true;
      }
    }
    return false;
  }
}

/** unit vectors perpendicular to the unit tangent t: one in 2D (left normal), two in 3D (an orthonormal pair) */
function perpendiculars(t: ArrayLike<number>, D: number): number[][] {
  if (D === 2) return [[-t[1]!, t[0]!]];
  // 3D: n1 = t × (the axis least aligned with t), n2 = t × n1
  const ax = Math.abs(t[0]!), ay = Math.abs(t[1]!), az = Math.abs(t[2]!);
  const u = ax <= ay && ax <= az ? [1, 0, 0] : ay <= az ? [0, 1, 0] : [0, 0, 1];
  const c = (a: ArrayLike<number>, b: ArrayLike<number>) => [a[1]! * b[2]! - a[2]! * b[1]!, a[2]! * b[0]! - a[0]! * b[2]!, a[0]! * b[1]! - a[1]! * b[0]!];
  const n1 = c(t, u), l1 = Math.hypot(...n1);
  const n1u = n1.map((x) => x / l1);
  return [n1u, c(t, n1u)];
}

/**
 * Evenly-spaced streamlines (Jobard & Lefer 1997), 2D and 3D. Separation d_sep = seed-cell side for
 * `count`; a line stops within d_test = testRatio·d_sep of another line (or of its own distant past);
 * candidate seeds lie d_sep to either side of every vertex of accepted lines (in 3D: four candidates,
 * ± two perpendiculars of the tangent), FIFO, and when that front is exhausted the jittered stratified
 * seeds farther than d_sep from every line restart it (disconnected regions, regions the flow never
 * reaches). Deterministic for a given `seed`.
 */
export function evenlySpacedStreamlines(field: VectorFieldData, opts: StreamlineOptions): StreamlinePlan {
  const D = field.dimCount;
  if (D !== 2 && D !== 3) throw new Error("evenly-spaced streamlines are 2D or 3D only");
  const box = opts.box ?? field.box;
  const h = opts.step, M = opts.maxSteps, MB = opts.bidirectional === false ? 0 : M;
  const dSep = seedCellSide(box, opts.count);
  const dTest = (opts.testRatio ?? 0.5) * dSep;
  const minOrd = Math.max(2, Math.ceil((2 * dSep) / h)); // own points closer along the arc than this do not stop a line
  const hash = new PointHash(box.a, dSep, box.size.map((sz) => Math.ceil(sz / dSep) + 3));
  const integrate = makeIntegrator(field, opts);
  const rand = lcg((opts.seed ?? 12345) ^ 0x5bd1e995);
  const fallback = streamlineSeeds(box, opts.count, opts.seed);

  const lines: Streamline[] = [];
  const seedPts: number[] = [], phases: number[] = [], budgets: number[] = [];
  const queue: number[] = []; // indices into `lines` whose sides still have to be inspected
  // generous cap: the front can only be this long if every line is a single step (degenerate fields)
  const vol = box.size.reduce((a, b) => a * b, 1);
  const maxLines = Math.max(64, Math.ceil((8 * vol) / (Math.pow(dSep, D - 1) * h)));

  const tryLine = (sp: number[]): boolean => {
    if (!box.contains(sp, 1e-12) || hash.near(sp, dSep)) return false;
    const id = lines.length;
    // points enter the hash as they are produced (ordinal = signed step), so the forward half also avoids the backward half
    const stopFor = (s: 1 | -1) => (q: Float64Array, n: number): boolean => {
      if (hash.near(q, dTest, id, s * n, minOrd)) return true;
      hash.add(q, id, s * n);
      return false;
    };
    hash.add(sp, id, 0);
    const back = integrate(sp, -1, MB, stopFor(-1));
    const fwd = integrate(sp, 1, M, stopFor(1));
    const line = assemble(D, sp, back, fwd, h, rand());
    if (!line) return false; // its seed stays in the hash: nothing else can start there either
    lines.push(line); queue.push(id);
    seedPts.push(...sp); phases.push(line.phase); budgets.push(back.length / D, fwd.length / D);
    return true;
  };

  let nextFallback = 0;
  const restart = (): boolean => {
    while (nextFallback < fallback.phases.length) {
      const i = nextFallback++;
      if (tryLine(Array.from(fallback.points.subarray(D * i, D * i + D)))) return true;
    }
    return false;
  };
  restart();
  const t = new Array<number>(D).fill(0);
  while (queue.length && lines.length < maxLines) {
    const l = lines[queue.shift()!]!.points;
    const n = l.length / D;
    for (let k = 0; k < n && lines.length < maxLines; k++) {
      const i0 = Math.max(0, k - 1), i1 = Math.min(n - 1, k + 1);
      let tl = 0;
      for (let d = 0; d < D; d++) { t[d] = l[D * i1 + d]! - l[D * i0 + d]!; tl += t[d]! * t[d]!; }
      tl = Math.sqrt(tl);
      if (!(tl > 0)) continue;
      for (let d = 0; d < D; d++) t[d]! /= tl;
      for (const nrm of perpendiculars(t, D)) {
        tryLine(nrm.map((x, d) => l[D * k + d]! + x * dSep));
        tryLine(nrm.map((x, d) => l[D * k + d]! - x * dSep));
      }
    }
    if (!queue.length) restart();
  }
  return {
    seeds: { points: Float64Array.from(seedPts), phases: Float64Array.from(phases), budgets: Uint32Array.from(budgets) },
    lines,
    separation: dSep,
  };
}

/*******************************************************/
/* coverage fill */

/**
 * Stratified seeds, then `rounds` (default 2) of coverage fill: vertices are histogrammed on the seed
 * grid and every starved cell (fewer than ¼ of the median hit count of the occupied cells, at least
 * one vertex) gets one more jittered seed. Any dimension.
 */
export function coverageStreamlines(field: VectorFieldData, opts: StreamlineOptions): StreamlinePlan {
  const D = field.dimCount;
  const box = opts.box ?? field.box;
  const cnt = seedCells(box, opts.count);
  const total = cnt.reduce((a, b) => a * b, 1);
  const rand = lcg((opts.seed ?? 12345) ^ 0x27d4eb2f);
  const base = streamlineSeeds(box, opts.count, opts.seed);
  const points = [...base.points], phases = [...base.phases];
  const lines = integrateFromSeeds(field, base, opts);
  const hits = new Uint32Array(total);
  const cellOf = (pts: Float64Array, i: number): number => {
    let c = 0;
    for (let d = 0; d < D; d++) {
      const t = (pts[i * D + d]! - box.a[d]!) / (box.size[d]! || 1);
      c = c * cnt[d]! + Math.min(cnt[d]! - 1, Math.max(0, Math.floor(t * cnt[d]!)));
    }
    return c;
  };
  const count = (ls: Streamline[]) => { for (const l of ls) for (let i = 0; i < l.points.length / D; i++) hits[cellOf(l.points, i)]!++; };
  count(lines);
  for (let round = 0; round < (opts.rounds ?? 2); round++) {
    const occupied = Array.from(hits).filter((v) => v > 0).sort((a, b) => a - b);
    if (!occupied.length) break;
    const threshold = Math.max(1, 0.25 * occupied[occupied.length >> 1]!);
    const add: number[] = [], addPhase: number[] = [];
    const idx = new Array<number>(D).fill(0);
    for (let c = 0; c < total; c++) {
      if (hits[c]! >= threshold) continue;
      let rem = c;
      for (let d = D - 1; d >= 0; d--) { idx[d] = rem % cnt[d]!; rem = Math.floor(rem / cnt[d]!); }
      for (let d = 0; d < D; d++) add.push(box.a[d]! + ((idx[d]! + rand()) / cnt[d]!) * box.size[d]!);
      addPhase.push(rand());
    }
    if (!addPhase.length) break;
    const extra = integrateFromSeeds(field, { points: Float64Array.from(add), phases: Float64Array.from(addPhase) }, opts);
    count(extra);
    lines.push(...extra); points.push(...add); phases.push(...addPhase);
  }
  return { seeds: { points: Float64Array.from(points), phases: Float64Array.from(phases) }, lines, separation: seedCellSide(box, opts.count) };
}

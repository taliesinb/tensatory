// Runtime curves: γ: [t0, t1] → ℝ^D (schema/curves.ts, notes/curves.md).
//
// Mirrors the field data design: `kind` symbolic (evaluable at any t) or
// sampled (`sampleTimes`, interpolated between), `point(t)`, `velocity(t)`
// (exact for symbolic data, the interpolant's derivative for sampled data),
// `sampleOn(ts)` and `polyline(t0, t1, n)` for drawing. A flow curve integrates
// its vector field once (RK4, fixed step) into a sampled curve and evaluates
// through it. Pushforwards move the points (curves are covariant; fields pull
// back).

import { EvalError, SpecError } from "../errors";
import type { VExpr } from "../symbolic/ast";
import { compileVector, pureContext, type VectorFn } from "../symbolic/compile";
import { diffVector } from "../symbolic/diff";
import type { VectorFieldData } from "../fields/fieldData";

export type CurveKind = "symbolic" | "sampled";
export type CurveInterp = "linear" | "cubic" | "step";

export abstract class CurveData {
  abstract readonly kind: CurveKind;
  abstract readonly dimCount: number;
  /** [t0, t1] */
  abstract readonly interval: readonly [number, number];
  /** the parameter values of the samples; defined iff kind === "sampled" */
  abstract readonly sampleTimes: Float64Array | undefined;
  /** γ(t) written into `out` (t clamped to the interval is the caller's business: outside it is undefined) */
  abstract pointInto(t: number, out: Float64Array): boolean;
  /** γ'(t) */
  abstract velocityInto(t: number, out: Float64Array): boolean;

  get t0(): number { return this.interval[0]; }
  get t1(): number { return this.interval[1]; }
  contains(t: number, eps = 1e-12): boolean { return t >= this.t0 - eps && t <= this.t1 + eps; }

  point(t: number): number[] | undefined {
    const out = new Float64Array(this.dimCount);
    return this.pointInto(t, out) ? Array.from(out) : undefined;
  }
  velocity(t: number): number[] | undefined {
    const out = new Float64Array(this.dimCount);
    return this.velocityInto(t, out) ? Array.from(out) : undefined;
  }

  /** γ at every t of `ts`, flat [n, D] (NaN outside the interval) */
  sampleOn(ts: ArrayLike<number>): Float64Array {
    const D = this.dimCount;
    const out = new Float64Array(ts.length * D);
    const p = new Float64Array(D);
    for (let i = 0; i < ts.length; i++) {
      if (this.pointInto(ts[i]!, p)) out.set(p, i * D); else out.fill(NaN, i * D, (i + 1) * D);
    }
    return out;
  }

  /**
   * The points to draw the curve over [a, b] ⊂ interval as a polyline, flat [n, D]: sampled data with linear /
   * step interpolation gives its own samples inside (a, b) plus the exact end points; anything else is sampled
   * uniformly at `n` points (cubic data also keeps its samples as vertices so the markers lie on the line).
   */
  polyline(a = this.t0, b = this.t1, n = 256): Float64Array {
    const lo = Math.max(this.t0, Math.min(a, b)), hi = Math.min(this.t1, Math.max(a, b));
    if (!(hi >= lo)) return new Float64Array(0);
    const ts = this.polylineTimes(lo, hi, n);
    return this.sampleOn(ts);
  }

  /** the parameter values `polyline` evaluates at (overridden by sampled data to include its own samples) */
  protected polylineTimes(lo: number, hi: number, n: number): Float64Array {
    if (hi === lo) return Float64Array.of(lo);
    return Float64Array.from({ length: n }, (_, i) => lo + ((hi - lo) * i) / (n - 1));
  }

  /** arc length from t0 to t, by the polyline (n chords over the interval) */
  arcLength(t = this.t1, n = 1024): number {
    const pts = this.polyline(this.t0, Math.min(t, this.t1), n);
    const D = this.dimCount;
    let s = 0;
    for (let i = D; i < pts.length; i += D) { let d2 = 0; for (let k = 0; k < D; k++) { const d = pts[i + k]! - pts[i - D + k]!; d2 += d * d; } s += Math.sqrt(d2); }
    return s;
  }
}

/*******************************************************/
/* symbolic: γ(t) as an expression of t (dimCount = 1 scalar language; coord 0 is t) */

/**
 * The expression is normalized in the MANIFOLD's dimension (its vectors are D-vectors) but only `coord 0` — t — may
 * appear among the coordinate leaves: the other coordinates, `coordv` and `grad` have no meaning on a curve.
 */
function checkCurveExpr(e: unknown, path: string[]): void {
  if (typeof e !== "object" || e === null) return;
  if (Array.isArray(e)) { for (const x of e) checkCurveExpr(x, path); return; }
  const n = e as { k?: string; index?: number };
  if (n.k === "coord" && n.index !== 0) throw new SpecError(`a curve's expression may only use coordinate 0 (the parameter t), not coordinate ${n.index}`, path);
  if (n.k === "coordv" || n.k === "grad") throw new SpecError(`"${n.k}" has no meaning in a curve's expression (only t = coordinate 0)`, path);
  for (const v of Object.values(e)) checkCurveExpr(v, path);
}

export class SymbolicCurveData extends CurveData {
  readonly kind = "symbolic" as const;
  readonly sampleTimes = undefined;
  private readonly fn: VectorFn;
  private dfn: VectorFn | undefined;
  private readonly tv: Float64Array;

  /** @param ast normalized in `dimCount` dimensions (see checkCurveExpr) */
  constructor(readonly ast: VExpr, readonly dimCount: number, readonly interval: readonly [number, number], path: string[] = []) {
    super();
    if (!(interval[1] > interval[0])) throw new SpecError(`interval [${interval}] must have t0 < t1`, path);
    checkCurveExpr(ast, [...path, "expr"]);
    this.tv = new Float64Array(dimCount); // only [0] is ever read
    this.fn = compileVector(ast, pureContext(dimCount));
  }
  pointInto(t: number, out: Float64Array): boolean {
    if (!this.contains(t)) return false;
    this.tv[0] = t;
    const r = this.fn(this.tv, -1, out);
    if (r !== out) out.set(r);
    return true;
  }
  velocityInto(t: number, out: Float64Array): boolean {
    if (!this.contains(t)) return false;
    this.dfn ??= compileVector(diffVector(this.ast, 0, this.dimCount), pureContext(this.dimCount));
    this.tv[0] = t;
    const r = this.dfn(this.tv, -1, out);
    if (r !== out) out.set(r);
    return true;
  }
}

/*******************************************************/
/* sampled: points at parameter values */

export class SampledCurveData extends CurveData {
  readonly kind = "sampled" as const;
  readonly interval: readonly [number, number];
  readonly sampleTimes: Float64Array;
  /** flat [N, D] */
  readonly points: Float64Array;
  /** flat [N, D] tangents at the samples: given (`velocities`), or estimated for the cubic interpolant */
  private readonly tangents: Float64Array | undefined;
  readonly count: number;

  constructor(
    readonly dimCount: number,
    points: Float64Array,
    times: Float64Array | undefined,
    readonly interp: CurveInterp = "linear",
    velocities?: Float64Array,
    readonly closed = false,
    readonly labels?: readonly string[],
    path: string[] = [],
  ) {
    super();
    const D = dimCount;
    if (points.length % D !== 0) throw new SpecError(`points hold ${points.length} numbers, not a multiple of ${D}`, path);
    const N = points.length / D;
    if (N < 1) throw new SpecError("a sampled curve needs at least one point", path);
    if (times !== undefined && times.length !== N) throw new SpecError(`times has ${times.length} entries for ${N} points`, path);
    if (velocities !== undefined && velocities.length !== points.length) throw new SpecError(`velocities hold ${velocities.length} numbers, points ${points.length}`, path);
    if (labels && labels.length !== N) throw new SpecError(`labels has ${labels.length} entries for ${N} points`, path);
    this.count = N;
    this.points = points;
    this.sampleTimes = times ?? Float64Array.from({ length: N }, (_, i) => i);
    for (let i = 1; i < N; i++) if (!(this.sampleTimes[i]! > this.sampleTimes[i - 1]!)) throw new SpecError(`times must be strictly increasing (t[${i}] = ${this.sampleTimes[i]} after ${this.sampleTimes[i - 1]})`, path);
    if (closed && N >= 2) {
      // the curve returns to its first point: one extra segment back to it, over one more time step
      const dt = N > 1 ? (this.sampleTimes[N - 1]! - this.sampleTimes[0]!) / (N - 1) : 1;
      this.interval = [this.sampleTimes[0]!, this.sampleTimes[N - 1]! + dt];
    } else this.interval = [this.sampleTimes[0]!, this.sampleTimes[N - 1]!];
    if (interp === "cubic") this.tangents = velocities ?? this.catmullRom();
    else if (velocities) this.tangents = velocities;
  }

  /** the time of sample i, with the closing segment's end being the first sample again */
  private timeAt(i: number): number { return i < this.count ? this.sampleTimes[i]! : this.interval[1]; }
  private pointAt(i: number, k: number): number { return this.points[(i % this.count) * this.dimCount + k]!; }
  private tangentAt(i: number, k: number): number { return this.tangents![(i % this.count) * this.dimCount + k]!; }
  private get segments(): number { return this.closed && this.count >= 2 ? this.count : this.count - 1; }

  /** Catmull–Rom tangents (finite differences of the neighbours; one-sided at open ends) */
  private catmullRom(): Float64Array {
    const N = this.count, D = this.dimCount, out = new Float64Array(N * D);
    if (N < 2) return out;
    for (let i = 0; i < N; i++) {
      const prev = i > 0 ? i - 1 : this.closed ? N - 1 : i, next = i < N - 1 ? i + 1 : this.closed ? 0 : i;
      let dt: number;
      if (this.closed) { const step = this.interval[1] - this.interval[0]; const tp = i > 0 ? this.sampleTimes[prev]! : this.sampleTimes[0]! - (this.interval[1] - this.sampleTimes[N - 1]!), tn = i < N - 1 ? this.sampleTimes[next]! : this.interval[1]; dt = tn - tp || step; }
      else dt = this.sampleTimes[next]! - this.sampleTimes[prev]!;
      for (let k = 0; k < D; k++) out[i * D + k] = (this.pointAt(next, k) - this.pointAt(prev, k)) / dt;
    }
    return out;
  }

  /** the segment index i with time(i) <= t <= time(i+1), and the local parameter u ∈ [0, 1] */
  private locate(t: number): { i: number; u: number } | undefined {
    if (!this.contains(t)) return undefined;
    const segs = this.segments;
    if (segs === 0) return { i: 0, u: 0 };
    const tt = Math.max(this.interval[0], Math.min(this.interval[1], t));
    // binary search over the sample times (the closing segment is the last)
    let lo = 0, hi = segs - 1;
    while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (this.timeAt(mid) <= tt) lo = mid; else hi = mid - 1; }
    const a = this.timeAt(lo), b = this.timeAt(lo + 1);
    return { i: lo, u: b > a ? (tt - a) / (b - a) : 0 };
  }

  pointInto(t: number, out: Float64Array): boolean {
    const loc = this.locate(t);
    if (!loc) return false;
    const { i, u } = loc, D = this.dimCount;
    if (this.segments === 0) { for (let k = 0; k < D; k++) out[k] = this.pointAt(0, k); return true; }
    switch (this.interp) {
      case "step": for (let k = 0; k < D; k++) out[k] = this.pointAt(u < 1 ? i : i + 1, k); return true;
      case "linear": for (let k = 0; k < D; k++) out[k] = (1 - u) * this.pointAt(i, k) + u * this.pointAt(i + 1, k); return true;
      case "cubic": {
        const h = this.timeAt(i + 1) - this.timeAt(i);
        const h00 = 2 * u ** 3 - 3 * u ** 2 + 1, h10 = u ** 3 - 2 * u ** 2 + u, h01 = -2 * u ** 3 + 3 * u ** 2, h11 = u ** 3 - u ** 2;
        for (let k = 0; k < D; k++) out[k] = h00 * this.pointAt(i, k) + h10 * h * this.tangentAt(i, k) + h01 * this.pointAt(i + 1, k) + h11 * h * this.tangentAt(i + 1, k);
        return true;
      }
    }
  }

  velocityInto(t: number, out: Float64Array): boolean {
    const loc = this.locate(t);
    if (!loc) return false;
    const { i, u } = loc, D = this.dimCount;
    if (this.segments === 0) { out.fill(0); return true; }
    const h = this.timeAt(i + 1) - this.timeAt(i);
    switch (this.interp) {
      case "step": out.fill(0); return true;
      case "linear":
        if (this.tangents) { for (let k = 0; k < D; k++) out[k] = (1 - u) * this.tangentAt(i, k) + u * this.tangentAt(i + 1, k); return true; }
        for (let k = 0; k < D; k++) out[k] = (this.pointAt(i + 1, k) - this.pointAt(i, k)) / h;
        return true;
      case "cubic": {
        const d00 = 6 * u ** 2 - 6 * u, d10 = 3 * u ** 2 - 4 * u + 1, d01 = -6 * u ** 2 + 6 * u, d11 = 3 * u ** 2 - 2 * u;
        for (let k = 0; k < D; k++) out[k] = (d00 * this.pointAt(i, k) + d10 * h * this.tangentAt(i, k) + d01 * this.pointAt(i + 1, k) + d11 * h * this.tangentAt(i + 1, k)) / h;
        return true;
      }
    }
  }

  protected override polylineTimes(lo: number, hi: number, n: number): Float64Array {
    // the samples inside (lo, hi) are vertices; linear / step need nothing else, cubic gets `n` uniform points too
    const ts: number[] = [lo];
    const segs = this.segments;
    for (let i = 1; i <= segs; i++) { const t = this.timeAt(i); if (t > lo && t < hi) ts.push(t); }
    if (this.interp === "step") { // a staircase: both ends of every jump
      const out: number[] = [];
      for (let j = 0; j < ts.length; j++) { out.push(ts[j]!); if (j > 0) out.splice(out.length - 1, 0, ts[j]! - 1e-9 * Math.max(1, Math.abs(ts[j]!))); }
      out.push(hi);
      return Float64Array.from(out);
    }
    if (this.interp === "cubic" && hi > lo) for (let i = 1; i < n - 1; i++) ts.push(lo + ((hi - lo) * i) / (n - 1));
    if (hi > lo) ts.push(hi);
    return Float64Array.from(new Set(ts)).sort((a, b) => a - b);
  }
}

/*******************************************************/
/* flow: the integral curve of a vector field from a start point */

export interface FlowMethod { integrator: "rk4" | "euler"; step: number }

/**
 * ẋ = F(x) from `start` at t = 0, evaluated over [t0, t1] (t0 < 0 runs backwards). Integrated once, lazily, into a
 * finely sampled curve (fixed step); `point` / `velocity` interpolate it linearly / read F.
 */
export class FlowCurveData extends CurveData {
  readonly kind = "symbolic" as const;
  readonly sampleTimes = undefined;
  private inner: SampledCurveData | undefined;
  private readonly tmp: Float64Array;

  constructor(readonly field: VectorFieldData, readonly start: readonly number[], readonly interval: readonly [number, number], readonly method: FlowMethod, path: string[] = []) {
    super();
    if (start.length !== field.dimCount) throw new SpecError(`start has ${start.length} components, the field ${field.dimCount}`, path);
    if (!(interval[1] > interval[0])) throw new SpecError(`interval [${interval}] must have t0 < t1`, path);
    if (!(method.step > 0)) throw new SpecError(`step must be positive, got ${method.step}`, path);
    if (Math.abs(interval[1]) / method.step + Math.abs(interval[0]) / method.step > 5e6) throw new SpecError(`flow would take more than 5·10⁶ steps (interval [${interval}], step ${method.step})`, path);
    this.tmp = new Float64Array(field.dimCount);
  }
  get dimCount(): number { return this.field.dimCount; }

  private F(p: Float64Array, out: Float64Array): void {
    const r = this.field.fn(p, -1, out);
    if (r !== out) out.set(r);
  }
  private stepFrom(p: Float64Array, h: number, out: Float64Array): void {
    const D = this.dimCount;
    const k1 = new Float64Array(D), k2 = new Float64Array(D), k3 = new Float64Array(D), k4 = new Float64Array(D), q = new Float64Array(D);
    this.F(p, k1);
    if (this.method.integrator === "euler") { for (let k = 0; k < D; k++) out[k] = p[k]! + h * k1[k]!; return; }
    for (let k = 0; k < D; k++) q[k] = p[k]! + 0.5 * h * k1[k]!; this.F(q, k2);
    for (let k = 0; k < D; k++) q[k] = p[k]! + 0.5 * h * k2[k]!; this.F(q, k3);
    for (let k = 0; k < D; k++) q[k] = p[k]! + h * k3[k]!; this.F(q, k4);
    for (let k = 0; k < D; k++) out[k] = p[k]! + (h / 6) * (k1[k]! + 2 * k2[k]! + 2 * k3[k]! + k4[k]!);
  }

  /** integrate from t = 0 to `to` in steps of ±step, recording the samples with |t| >= |from| (same sign) */
  private run(from: number, to: number): { ts: number[]; pts: number[] } {
    const D = this.dimCount, h = Math.sign(to) * this.method.step;
    const ts: number[] = [], pts: number[] = [];
    let p = Float64Array.from(this.start), t = 0;
    const n = Math.ceil(Math.abs(to) / this.method.step - 1e-9);
    const record = () => { ts.push(t); pts.push(...p); };
    if (Math.abs(from) < 1e-12) record();
    for (let i = 0; i < n; i++) {
      const next = new Float64Array(D);
      const hh = i === n - 1 ? to - t : h; // land exactly on `to`
      this.stepFrom(p, hh, next);
      p = next; t += hh;
      if (!p.every(Number.isFinite)) throw new EvalError(`flow left the finite numbers at t = ${t}`);
      if (Math.abs(t) >= Math.abs(from) - 1e-9) record();
    }
    return { ts, pts };
  }

  /** the integrated curve, computed on first use */
  get sampled(): SampledCurveData {
    if (this.inner) return this.inner;
    const [t0, t1] = this.interval, D = this.dimCount;
    let ts: number[] = [], pts: number[] = [];
    if (t0 < 0) { const b = this.run(t1 < 0 ? t1 : 0, t0); ts = b.ts.reverse(); pts = []; for (let i = b.ts.length - 1; i >= 0; i--) pts.push(...b.pts.slice(i * D, (i + 1) * D)); }
    if (t1 > 0) {
      const f = this.run(t0 > 0 ? t0 : 0, t1);
      // t = 0 appears in both halves: keep one
      const skip = ts.length && f.ts.length && Math.abs(f.ts[0]! - ts[ts.length - 1]!) < 1e-12 ? 1 : 0;
      ts.push(...f.ts.slice(skip)); pts.push(...f.pts.slice(skip * D));
    }
    // times must be strictly increasing; the interval is exactly what was asked
    const T = Float64Array.from(ts), P = Float64Array.from(pts);
    this.inner = new SampledCurveData(D, P, T, "linear");
    return this.inner;
  }

  pointInto(t: number, out: Float64Array): boolean { return this.contains(t) && this.sampled.pointInto(Math.max(this.t0, Math.min(this.t1, t)), out); }
  velocityInto(t: number, out: Float64Array): boolean {
    if (!this.pointInto(t, this.tmp)) return false;
    this.F(this.tmp, out);
    return true;
  }
  protected override polylineTimes(lo: number, hi: number, n: number): Float64Array {
    // the integration steps are the vertices (like a sampled curve), never coarser than the caller asked
    const s = this.sampled;
    const ts: number[] = [lo];
    for (const t of s.sampleTimes) if (t > lo && t < hi) ts.push(t);
    if (hi > lo) ts.push(hi);
    if (ts.length < n && hi > lo) for (let i = 1; i < n - 1; i++) ts.push(lo + ((hi - lo) * i) / (n - 1));
    return Float64Array.from(new Set(ts)).sort((a, b) => a - b);
  }
}

/*******************************************************/
/* pushforwards */

/** γ'(t) = origin + scale ⊙ (γ(t) − origin) + vec: translate and scale in one affine map */
export class AffineCurveData extends CurveData {
  readonly kind: CurveKind;
  readonly sampleTimes: Float64Array | undefined;
  constructor(readonly inner: CurveData, private readonly origin: readonly number[], private readonly scale: readonly number[], private readonly vec: readonly number[]) {
    super();
    this.kind = inner.kind;
    this.sampleTimes = inner.sampleTimes;
  }
  get dimCount(): number { return this.inner.dimCount; }
  get interval(): readonly [number, number] { return this.inner.interval; }
  pointInto(t: number, out: Float64Array): boolean {
    if (!this.inner.pointInto(t, out)) return false;
    for (let k = 0; k < out.length; k++) out[k] = this.origin[k]! + this.scale[k]! * (out[k]! - this.origin[k]!) + this.vec[k]!;
    return true;
  }
  velocityInto(t: number, out: Float64Array): boolean {
    if (!this.inner.velocityInto(t, out)) return false;
    for (let k = 0; k < out.length; k++) out[k] = this.scale[k]! * out[k]!;
    return true;
  }
  protected override polylineTimes(lo: number, hi: number, n: number): Float64Array {
    return (this.inner as unknown as { polylineTimes(lo: number, hi: number, n: number): Float64Array }).polylineTimes(lo, hi, n);
  }
}

export const translateCurve = (inner: CurveData, vec: readonly number[]): CurveData =>
  new AffineCurveData(inner, new Array<number>(inner.dimCount).fill(0), new Array<number>(inner.dimCount).fill(1), vec);
export const scaleCurve = (inner: CurveData, origin: readonly number[] | undefined, scale: number | readonly number[]): CurveData =>
  new AffineCurveData(inner, origin ?? new Array<number>(inner.dimCount).fill(0), typeof scale === "number" ? new Array<number>(inner.dimCount).fill(scale) : scale, new Array<number>(inner.dimCount).fill(0));

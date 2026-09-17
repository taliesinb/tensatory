// Adaptive resolution: the grid the fields are sampled on (isolines / isosurfaces, raster, streamlines) is
// not a control but a feedback loop over a ladder of resolutions, with two tiers:
//
//   * `moving`  — used while the levels change every frame (▶ animation, dragging value / split): bounded by
//                 the frame budget (30 fps), the memory cap and the ladder;
//   * `settled` — used once the levels stop (after the 2D isolines' settle delay): bounded by the latency of
//                 ONE recomputation (a few hundred ms is fine when idle), the memory cap and the ladder.
//
// Both start low and ramp one ladder step per measured computation. Frame time is the interval between the
// rendered frame's rAF and the next one (which includes the GPU stall) or the JS time of the frame, whichever
// is larger. Frame times are quantized by vsync, so a 60 fps window does not say how much headroom there is:
// the moving tier then PROBES one step up and steps back (remembering the failure for a while) if the budget
// is exceeded; when the measured time is above one vsync it is extrapolated by (n′/n)ᴰ instead. Render-only
// frames (orbiting, particles) that miss the budget step the displayed tier down too: triangle count is a
// render cost. Memory is extrapolated from the caller's split into ∝ nᴰ (grids) and ∝ nᴰ⁻¹ (meshes, lines)
// parts; a step whose prediction exceeds the cap is not taken, and a frame whose working set alone exceeds the
// cap steps down. Everything bounded per context (field, levels, options) so a failed step in one picture does
// not haunt another; `?res=` / `?res3=` pin a resolution and disable the loop.

export type Tier = "moving" | "settled";

export interface FrameReport {
  /** frame time in ms (rAF interval after the rendered frame, or its JS time if larger) */
  ms: number;
  /** which tier's resolution the frame used */
  tier: Tier;
  /** the frame recomputed geometry (grid / levels changed); false = it only rendered what was resident */
  recomputed: boolean;
  /** the frame built a pipeline / kernel (shader compile): its time is not representative ... */
  compiled: boolean;
  /** ... unless its JS time alone already blows the budget: a compile stalls the GPU, not the main thread, so a long
   *  JS time is real CPU work (a costly field sampled on the dispatch grid) and is judged like any slow sample */
  jsMs?: number;
  /** memory after the frame: `total` = everything held (device + JS arrays; the part outside the working set is
   *  trimmable), `volume` / `surface` = the frame's working set, ∝ nᴰ and ∝ nᴰ⁻¹ */
  bytes: { total: number; volume: number; surface: number };
  /** the frame's working set alone exceeds the cap even after trimming caches */
  overCap: boolean;
  /** what the resolution is spent on (field ids, levels, flags): failures are remembered per context */
  ctx: string;
}

export interface AutoResState { moving: number; settled: number; measured?: boolean }

/** the moving-tier frame budget (30 fps) and the target left below it before stepping up */
const FRAME_BUDGET_MS = 33.4, FRAME_TARGET_MS = 26;
/** one recomputation while idle may take this long */
const SETTLED_BUDGET_MS = 200;
/** a frame time at or under this is vsync-bound (60 Hz): no information about headroom */
const VSYNC_MS = 17.5;
/** frame-time window (median) for down decisions and probe validation */
const WINDOW = 8;
/** samples discarded after a resolution change (allocations, first dispatches) */
const COOLDOWN = 3;
/** how long a failed step is remembered per context */
const FAIL_TTL_MS = 60_000;
/** headroom kept under the memory cap when predicting a step */
const MEM_HEADROOM = 0.85;

/** ×√2 per step from `lo` to `hi`, rounded to friendly values */
export function ladder(lo: number, hi: number): number[] {
  const out: number[] = [];
  for (let n = lo; n <= hi; n *= 2) { out.push(n); if (Math.round(n * 1.5) < hi) out.push(Math.round(n * 1.5)); }
  return out.filter((n) => n <= hi).sort((a, b) => a - b);
}

export class AutoRes {
  /** ladder indices per tier; moving ≤ settled */
  moving: number;
  settled: number;
  /** a fixed resolution (URL pin): the loop is off */
  pin: number | undefined;
  /** memory cap in bytes */
  capBytes = 1024 * 2 ** 20;
  /** called whenever a tier changed (the caller marks the scene dirty and persists); `dir` +1 = up, −1 = down */
  onChange: ((tier: Tier, dir: 1 | -1) => void) | undefined;
  /** called when a settled recomputation could not be timed (it compiled shaders): the caller recomputes the same
   *  geometry once more so the next sample is clean */
  onRemeasure: (() => void) | undefined;
  /** the last decision, for the status row — terse: it shares one line with the memory figure.
   *  `mov` / `set` = the tier, ↑ / ↓ a step, `@n` the resolution held, `→n` the next step's predicted cost */
  note = "";

  private readonly failed = new Map<string, number>(); // `${ctx}|${tier}|${idx}` -> expiry
  private cooldown = 0;
  private movingTimes: number[] = [];
  private renderTimes: number[] = [];
  private lastCtx = "";
  /** the moving tier has had a full window of its own samples (before that it follows the settled tier two steps below) */
  private movingMeasured = false;
  /** a settled hold was confirmed by a second, clean sample (the first after a context change may carry setup work) */
  private confirmed = false;
  /** the previous settled sample compiled: the driver may finish the compile in the NEXT frame, so that one is discarded too */
  private afterCompile = false;
  /** a slow settled sample is confirmed by a second one before it fails a step (main-thread stalls — a bundle
   *  loading, devtools — inflate rAF intervals too) */
  private slowSeen = false;

  /**
   * @param steps  the resolution ladder (grid points along the longest box side)
   * @param dims   2 or 3: cost extrapolation exponent (cells ∝ nᴰ)
   * @param start  initial resolution (snapped down onto the ladder)
   */
  constructor(readonly steps: number[], readonly dims: number, start: number) {
    this.moving = this.settled = this.indexOf(start);
  }

  /** ladder index of the largest step ≤ n (0 when n is below the ladder) */
  indexOf(n: number): number { let i = 0; for (let k = 0; k < this.steps.length; k++) if (this.steps[k]! <= n) i = k; return i; }

  /** the resolution to use this frame */
  resolution(tier: Tier): number { return this.pin ?? this.steps[tier === "moving" ? this.moving : this.settled]!; }

  state(): AutoResState { return { moving: this.moving, settled: this.settled, measured: this.movingMeasured }; }
  restore(s: AutoResState | undefined): void {
    if (!s) return;
    const clamp = (i: number) => Math.max(0, Math.min(this.steps.length - 1, Math.round(i)));
    this.settled = clamp(s.settled); this.moving = Math.min(clamp(s.moving), this.settled);
    this.movingMeasured = !!s.measured;
    this.reset();
  }

  private reset(): void { this.movingTimes = []; this.renderTimes = []; this.cooldown = COOLDOWN; this.confirmed = false; this.afterCompile = false; this.slowSeen = false; }

  private isFailed(ctx: string, tier: Tier, idx: number): boolean {
    const k = `${ctx}|${tier}|${idx}`, t = this.failed.get(k);
    if (t === undefined) return false;
    if (t < performance.now()) { this.failed.delete(k); return false; }
    return true;
  }
  private fail(ctx: string, tier: Tier, idx: number): void { this.failed.set(`${ctx}|${tier}|${idx}`, performance.now() + FAIL_TTL_MS); }

  private set(tier: Tier, idx: number, why: string): void {
    idx = Math.max(0, Math.min(this.steps.length - 1, idx));
    const dir: 1 | -1 = idx > (tier === "settled" ? this.settled : this.moving) ? 1 : -1;
    if (tier === "settled") {
      if (idx === this.settled) return;
      this.settled = idx; this.moving = Math.min(this.moving, idx);
      // one recomputation may take ~6× a frame: about two ladder steps (×2.8 cells each) — the moving guess until it is measured
      if (!this.movingMeasured) this.moving = Math.max(this.moving, idx - 2);
    } else { idx = Math.min(idx, this.settled); if (idx === this.moving) return; this.moving = idx; }
    this.note = `${tier === "moving" ? "mov" : "set"}${dir > 0 ? "↑" : "↓"}${this.steps[idx]} ${why}`;
    this.reset();
    this.onChange?.(tier, dir);
  }

  /** predicted working set (bytes) if the resolution moved from index `from` to `to`; what is outside the working set is trimmable */
  private predictBytes(r: FrameReport, from: number, to: number): number {
    const ratio = this.steps[to]! / this.steps[from]!;
    return r.bytes.volume * ratio ** this.dims + r.bytes.surface * ratio ** (this.dims - 1);
  }
  private memoryAllows(r: FrameReport, from: number, to: number): boolean { return this.predictBytes(r, from, to) <= this.capBytes * MEM_HEADROOM; }

  /** feed one rendered frame; may change a tier (see `onChange`) */
  report(r: FrameReport): void {
    if (this.pin !== undefined) return;
    if (r.ctx !== this.lastCtx) { this.lastCtx = r.ctx; this.reset(); }
    const idx = r.tier === "moving" ? this.moving : this.settled;
    const top = this.steps.length - 1;
    if (r.overCap) { this.fail(r.ctx, r.tier, idx); this.set(r.tier, idx - 1, "mem cap"); return; }
    // the 75th percentile: up to a quarter of the window may hiccup (frame times are vsync-quantized: 17 / 33 / 50 ms)
    const median = (xs: number[]) => { const s = [...xs].sort((a, b) => a - b); return s[Math.min(s.length - 1, Math.floor(s.length * 0.75))]!; };

    if (r.tier === "settled" && r.recomputed) {
      // one recomputation while idle: a single latency sample decides (no cooldown: the sample after a step IS the probe)
      const cpuSlow = (r.jsMs ?? 0) > SETTLED_BUDGET_MS * 1.5;
      if ((r.compiled || this.afterCompile) && !cpuSlow) { this.afterCompile = r.compiled; this.onRemeasure?.(); return; }
      this.afterCompile = false;
      if (r.ms > SETTLED_BUDGET_MS * 1.5) {
        // (a remeasure would hit the caches, so CPU-slow samples are believed at once)
        if (!this.slowSeen && !cpuSlow) { this.slowSeen = true; this.note = `set@${this.steps[idx]} ${r.ms.toFixed(0)}ms, remeasuring`; this.onRemeasure?.(); return; }
        if (idx > 0) { this.fail(r.ctx, "settled", idx); this.set("settled", idx - 1, `${r.ms.toFixed(0)}ms`); }
        return;
      }
      this.slowSeen = false;
      if (idx < top && !this.isFailed(r.ctx, "settled", idx + 1) && this.memoryAllows(r, idx, idx + 1)) {
        const predicted = Math.max(r.ms, VSYNC_MS) * (this.steps[idx + 1]! / this.steps[idx]!) ** this.dims;
        if (predicted <= SETTLED_BUDGET_MS) { this.set("settled", idx + 1, `~${predicted.toFixed(0)}ms`); }
        else {
          this.note = `set@${this.steps[idx]} →${this.steps[idx + 1]} ~${predicted.toFixed(0)}ms`;
          // one clean re-measurement before believing a hold: the sample may have carried a mode / space switch
          if (!this.confirmed) { this.confirmed = true; this.onRemeasure?.(); }
        }
      } else if (idx < top && !this.isFailed(r.ctx, "settled", idx + 1)) this.note = `set@${this.steps[idx]} →${this.steps[idx + 1]} ~${(this.predictBytes(r, idx, idx + 1) / 2 ** 20).toFixed(0)}MB`;
      return;
    }
    if (r.compiled || this.cooldown > 0) { this.cooldown = Math.max(0, this.cooldown - 1); return; }

    if (!r.recomputed) {
      // render-only frame: resident geometry drawn again (orbit, particles); slow means too many triangles / segments
      this.renderTimes.push(r.ms); if (this.renderTimes.length > WINDOW) this.renderTimes.shift();
      if (this.renderTimes.length === WINDOW && median(this.renderTimes) > FRAME_BUDGET_MS && idx > 0) {
        this.fail(r.ctx, r.tier, idx); this.set(r.tier, idx - 1, `draw ${median(this.renderTimes).toFixed(0)}ms`);
      }
      return;
    }

    if (r.tier === "settled") return; // a settled frame that only rendered was handled above
    // moving tier: frame-time window
    this.movingTimes.push(r.ms); if (this.movingTimes.length > WINDOW) this.movingTimes.shift();
    if (this.movingTimes.length < WINDOW) return;
    this.movingMeasured = true;
    const med = median(this.movingTimes);
    if (med > FRAME_BUDGET_MS) {
      if (idx > 0) { this.fail(r.ctx, "moving", idx); this.set("moving", idx - 1, `${med.toFixed(0)}ms`); }
     
      return;
    }
   
    if (idx >= top || this.isFailed(r.ctx, "moving", idx + 1) || !this.memoryAllows(r, idx, idx + 1)) return;
    // a frame budget met at n means one recomputation at the next step is fine too: while the animation runs the
    // settled tier gets no samples of its own, so a moving step up carries it along
    const up = (why: string) => { if (this.settled <= idx) this.settled = idx + 1; this.set("moving", idx + 1, why); };
    if (med <= VSYNC_MS) { up("probe"); return; }
    const predicted = med * (this.steps[idx + 1]! / this.steps[idx]!) ** this.dims;
    if (predicted <= FRAME_TARGET_MS) up(`~${predicted.toFixed(0)}ms`);
  }
}

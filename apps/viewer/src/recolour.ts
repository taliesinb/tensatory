// Progressive exact recolouring of resident-coloured geometry (gpu/src/recolour.ts) — the viewer side.
//
// Render sites that coloured a set from a resident grid (a costly colour field) register the set here each
// frame; after the frame, `step` spends a fixed record budget over the registered sets, round-robin, writing exact
// colours into their buffers in place. Progress lives on the set (`RecolourProgress`, reset by the caller when the
// set is re-dispatched); when everything registered is done the frame loop stops re-rendering.

import type { ScalarFieldData } from "@tensatory/core";
import { gpuTranspilable, recolourStep, recolourer, type GpuBackend, type RecolourProgress, type Recolourer, type RecordLayout } from "@tensatory/gpu";
import { Cache } from "./cache";

/**
 * Scheduling. One batch per frame with up to IN_FLIGHT outstanding (`onSubmittedWorkDone` takes ~4 frames to come
 * back, so gating on it would idle the GPU three frames in four; an unbounded queue would pile seconds of work
 * behind a slow frame). With the pile bounded, the rAF interval IS the GPU-load signal: when the GPU cannot keep
 * up, presenting blocks and frames stretch to ≈ batch cost. The budget follows that interval — grows while frames
 * stay under TARGET_MS, shrinks when they run long — so the recolouring takes what the frame has to spare. Per-batch
 * GPU timing was tried and is useless here: `onSubmittedWorkDone` latency is quantized to frame boundaries.
 */
export const RECOLOUR_BUDGET_MIN = 16384, RECOLOUR_BUDGET_MAX = 1 << 19, RECOLOUR_TARGET_MS = 24, IN_FLIGHT = 2;

interface Job { r: Recolourer; buffer: GPUBuffer; indirect: GPUBuffer; total: number; progress: RecolourProgress }

export class Recolour {
  private readonly kernels = new Cache<Recolourer>(8, () => {});
  private jobs: Job[] = [];
  private budget = RECOLOUR_BUDGET_MIN * 2;
  private outstanding = 0;
  private dispatchedLastFrame = false;
  constructor(private readonly gpu: GpuBackend) {}

  /** forget the kernels (a bundle change or revision: the field behind a key is not the same field any more) */
  clear(): void { this.kernels.clear(); this.jobs = []; }

  /** a render begins: it re-registers the sets it draws (the list persists between renders so batches can run
   *  every frame while the image is refreshed only every few — re-rendering four translucent million-triangle
   *  shells is ~40 ms of GPU, far more than a batch) */
  beginFrame(): void { this.jobs = []; }

  /** register a set for this frame: `total` = its record count if known, else its capacity */
  add(field: ScalarFieldData, fieldKey: string, layout: RecordLayout, buffer: GPUBuffer, indirect: GPUBuffer, total: number, progress: RecolourProgress): void {
    if (progress.done) return;
    if (!gpuTranspilable(field)) { progress.done = true; return; } // the GPU cannot evaluate it exactly: the resident colour stays
    const r = this.kernels.getOr(`${fieldKey}|${layout.D}|${layout.floats}|${layout.points.map((p) => `${p.pos}:${p.col}`).join(",")}`, () => recolourer(this.gpu, field, layout));
    this.jobs.push({ r, buffer, indirect, total, progress });
  }

  /**
   * Dispatch this frame's batch; `frameMs` is the interval of the frame that just ended (it carried the previous
   * batch). Returns whether more frames are needed (sets pending or batches in flight).
   */
  step(frameMs: number): boolean {
    const jobs = this.jobs.filter((j) => !j.progress.done);
    if (!jobs.length) { this.budget = RECOLOUR_BUDGET_MIN * 2; this.dispatchedLastFrame = false; return this.outstanding > 0; }
    if (!this.fixed) {
      // a long frame means the GPU is behind (whether or not this frame dispatched): shrink; grow only when frames
      // are short AND the pipeline is not full (a full pipeline with short frames is a GPU still catching up)
      if (frameMs > RECOLOUR_TARGET_MS * 1.5) this.budget = Math.max(RECOLOUR_BUDGET_MIN, Math.round(this.budget * 0.6));
      else if (frameMs < RECOLOUR_TARGET_MS && this.dispatchedLastFrame && this.outstanding < IN_FLIGHT) this.budget = Math.min(RECOLOUR_BUDGET_MAX, Math.round(this.budget * 1.2));
    }
    this.dispatchedLastFrame = false;
    // a compile in flight (the recolour kernel's own, say) would DEFER the dispatch: nothing submitted this frame
    // and the whole pile landing at once when the compile finishes
    if (this.gpu.compiling > 0 || this.outstanding >= IN_FLIGHT) return true;
    if (this.fixed) this.budget = this.fixed;
    const now = performance.now();
    if (now - this.stats.lastAt > 500) { this.stats.firstAt = now; if (!this.fixed) this.budget = RECOLOUR_BUDGET_MIN * 2; } // a new run ramps from the bottom
    recolourStep(jobs, this.budget);
    this.stats.batches++; this.stats.records += this.budget; this.stats.lastAt = now;
    this.outstanding++; this.dispatchedLastFrame = true;
    // a promise that never settles (a lost device, a browser quirk) must not wedge the pipeline: count it done after a while
    let settled = false;
    const release = () => { if (!settled) { settled = true; this.outstanding--; } };
    void this.gpu.device.queue.onSubmittedWorkDone().then(release, release);
    setTimeout(release, 2000);
    return true;
  }

  /** diagnostics: batches dispatched, records (incl. empty slots), first / last dispatch time of the current run */
  readonly stats = { batches: 0, records: 0, firstAt: 0, lastAt: 0 };
  /** diagnostics: pin the budget (0 = adaptive) */
  fixed = 0;
  /** sets registered by the last render that are not exact yet */
  get pending(): number { return this.jobs.filter((j) => !j.progress.done).length; }
  /** the current per-batch record budget (diagnostics) */
  get currentBudget(): number { return this.budget; }
}

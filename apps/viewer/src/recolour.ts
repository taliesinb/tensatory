// Progressive exact recolouring of resident-coloured geometry (gpu/src/recolour.ts) — the viewer side.
//
// Render sites that coloured a set from a resident grid (a costly colour field) register the set here each
// frame; after the frame, `step` spends a fixed record budget over the registered sets, round-robin, writing exact
// colours into their buffers in place. Progress lives on the set (`RecolourProgress`, reset by the caller when the
// set is re-dispatched); when everything registered is done the frame loop stops re-rendering.

import type { ScalarFieldData } from "@tensatory/core";
import { gpuTranspilable, recolourStep, recolourer, type GpuBackend, type RecolourProgress, type Recolourer, type RecordLayout } from "@tensatory/gpu";
import { Cache } from "./cache";

/** records recoloured per frame in total — ~9 ms of GPU for the iris net (≈ 37 k evaluations per 10 ms) */
export const RECOLOUR_BUDGET = 32768;

interface Job { r: Recolourer; buffer: GPUBuffer; indirect: GPUBuffer; total: number; progress: RecolourProgress }

export class Recolour {
  private readonly kernels = new Cache<Recolourer>(8, () => {});
  private jobs: Job[] = [];
  constructor(private readonly gpu: GpuBackend) {}

  /** register a set for this frame: `total` = its record count if known, else its capacity */
  add(field: ScalarFieldData, fieldKey: string, layout: RecordLayout, buffer: GPUBuffer, indirect: GPUBuffer, total: number, progress: RecolourProgress): void {
    if (progress.done) return;
    if (!gpuTranspilable(field)) { progress.done = true; return; } // the GPU cannot evaluate it exactly: the resident colour stays
    const r = this.kernels.getOr(`${fieldKey}|${layout.D}|${layout.floats}|${layout.points.map((p) => `${p.pos}:${p.col}`).join(",")}`, () => recolourer(this.gpu, field, layout));
    this.jobs.push({ r, buffer, indirect, total, progress });
  }

  /** dispatch this frame's batches; returns whether more frames are needed */
  step(budget = RECOLOUR_BUDGET): boolean {
    const jobs = this.jobs;
    this.jobs = [];
    return recolourStep(jobs, budget);
  }

  get pending(): number { return this.jobs.length; }
}

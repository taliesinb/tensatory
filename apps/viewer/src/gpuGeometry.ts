// Asynchronous GPU geometry for the viewer: exact isolines and streamline
// integration. Same pattern as the sampler: cached results, `undefined` while
// a computation is in flight, `onReady` when it lands.

import type { ContourResult, DenseGrid, ScalarFieldData, Streamline, StreamlineSeeds, VectorFieldData } from "@tensatory/core";
import { gpuExactIsoContours, gpuIntegrateFromSeeds, gpuProjector, type GpuBackend, type GpuProjector, type GpuStreamlineOptions } from "@tensatory/gpu";

export class GpuGeometry {
  private readonly projectors = new Map<string, GpuProjector>();
  private readonly contourCache = new Map<string, ContourResult>();
  private readonly streamCache = new Map<string, Streamline[]>();
  private readonly pending = new Set<string>();

  constructor(readonly gpu: GpuBackend, private readonly onReady: () => void) {}

  get busy(): boolean { return this.pending.size > 0; }

  clear(): void { this.projectors.clear(); this.contourCache.clear(); this.streamCache.clear(); this.pending.clear(); }

  /** exact isolines of a symbolic field at `level`; undefined while computing */
  contours(key: string, fieldKey: string, field: ScalarFieldData, grid: DenseGrid, values: ArrayLike<number>, level: number, tolerance: number): ContourResult | undefined {
    const have = this.contourCache.get(key);
    if (have) return have;
    if (this.pending.has(key)) return undefined;
    this.pending.add(key);
    let proj = this.projectors.get(fieldKey);
    if (!proj) this.projectors.set(fieldKey, (proj = gpuProjector(this.gpu, field, grid)));
    const t0 = performance.now();
    gpuExactIsoContours(this.gpu, proj, field, grid, values, level, { tolerance })
      .then((r) => {
        if (this.contourCache.size > 64) this.contourCache.delete(this.contourCache.keys().next().value!);
        this.contourCache.set(key, r);
        console.log(`isoline (GPU) level ${level.toPrecision(3)}: ${r.vertexCount} vertices, |f − c| ≤ ${r.maxResidual.toExponential(1)}, ${(performance.now() - t0).toFixed(1)} ms`);
      })
      .catch((e) => console.error(`GPU isolines failed (${key}):`, e))
      .finally(() => { this.pending.delete(key); this.onReady(); });
    return undefined;
  }

  /** streamlines from `seeds`; undefined while computing */
  streamlines(key: string, field: VectorFieldData, seeds: StreamlineSeeds, opts: GpuStreamlineOptions): Streamline[] | undefined {
    const have = this.streamCache.get(key);
    if (have) return have;
    if (this.pending.has(key)) return undefined;
    this.pending.add(key);
    const t0 = performance.now();
    gpuIntegrateFromSeeds(this.gpu, field, seeds, opts)
      .then((lines) => {
        if (this.streamCache.size > 16) this.streamCache.delete(this.streamCache.keys().next().value!);
        this.streamCache.set(key, lines);
        console.log(`streamlines (GPU): ${lines.length} lines in ${(performance.now() - t0).toFixed(1)} ms`);
      })
      .catch((e) => console.error(`GPU streamlines failed (${key}):`, e))
      .finally(() => { this.pending.delete(key); this.onReady(); });
    return undefined;
  }
}

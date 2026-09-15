// Fully GPU-resident geometry for the fused (GPU compute + GPU render) path.
// Everything here is synchronous from the CPU's point of view: kernels are
// enqueued and the GPU queue orders them before the render pass, so a frame
// needs no readback and no await. Results are cached by key; a level change
// re-dispatches into the same segment set.

import type { DenseGrid, ScalarFieldData, StreamlineSeeds, VectorFieldData } from "@tensatory/core";
import {
  allocSegments,
  blurResidentSync,
  fusedIsolines,
  fusedStreamlines,
  resetSegments,
  sampleResidentSync,
  smoothedIsolines,
  uploadGrid,
  uploadSegments,
  type FusedIsolines,
  type FusedStreamlineOptions,
  type FusedStreamlines,
  type GpuBackend,
  type GpuGrid,
  type GpuSegments,
  type SmoothedIsolines,
} from "@tensatory/gpu";

function lru<V>(map: Map<string, V>, max: number, dispose: (v: V) => void): void {
  while (map.size > max) { const k = map.keys().next().value!; dispose(map.get(k)!); map.delete(k); }
}

export class FusedGeometry {
  private readonly grids = new Map<string, GpuGrid>();
  private readonly isoKernels = new Map<string, FusedIsolines>();
  private readonly smoothKernels = new Map<string, SmoothedIsolines>();
  private readonly blurred = new Map<string, GpuGrid>();
  private readonly isoSets = new Map<string, { segs: GpuSegments; stamp: string }>();
  private readonly streamKernels = new Map<string, { kernel: FusedStreamlines; segs: GpuSegments }>();
  private readonly uploaded = new Map<string, GpuSegments>();

  constructor(readonly gpu: GpuBackend) {}

  clear(): void {
    for (const g of this.grids.values()) g.destroy();
    for (const s of this.isoSets.values()) s.segs.destroy();
    for (const s of this.streamKernels.values()) s.segs.destroy();
    for (const s of this.uploaded.values()) s.destroy();
    for (const k of this.smoothKernels.values()) k.destroy();
    for (const g of this.blurred.values()) g.destroy();
    this.grids.clear(); this.isoKernels.clear(); this.isoSets.clear(); this.streamKernels.clear(); this.uploaded.clear(); this.smoothKernels.clear(); this.blurred.clear();
  }

  /** resident samples of `field` on `grid` (enqueued on first use) */
  grid(key: string, field: ScalarFieldData | VectorFieldData, grid: DenseGrid): GpuGrid {
    let g = this.grids.get(key);
    if (!g) { this.grids.set(key, (g = sampleResidentSync(this.gpu, field, grid))); lru(this.grids, 24, (v) => v.destroy()); }
    return g;
  }

  /** CPU values as a resident grid (CPU compute + GPU render) */
  uploadGrid(key: string, grid: DenseGrid, values: ArrayLike<number>, channels: number): GpuGrid {
    let g = this.grids.get(key);
    if (!g) { this.grids.set(key, (g = uploadGrid(this.gpu, grid, values, channels))); lru(this.grids, 24, (v) => v.destroy()); }
    return g;
  }

  /** box-blurred copy of a resident grid (core's `boxBlur` semantics) */
  blur(key: string, values: GpuGrid, radius: number): GpuGrid {
    if (radius <= 0) return values;
    const k = `${key}|blur${radius}`;
    let g = this.blurred.get(k);
    if (!g) { this.blurred.set(k, (g = blurResidentSync(this.gpu, values, radius))); lru(this.blurred, 8, (v) => v.destroy()); }
    return g;
  }

  /** marching-squares isolines of a resident grid with Taubin smoothing on the edge graph (non-exact path with `line` > 0) */
  smoothedIsolines(kernelKey: string, setKey: string, values: GpuGrid, colour: ScalarFieldData | undefined, level: number, iterations: number): GpuSegments {
    let kernel = this.smoothKernels.get(kernelKey);
    if (!kernel) { this.smoothKernels.set(kernelKey, (kernel = smoothedIsolines(this.gpu, values, colour))); lru(this.smoothKernels, 8, (v) => v.destroy()); }
    let set = this.isoSets.get(setKey);
    if (set && set.segs.capacity !== kernel.capacity) { set.segs.destroy(); this.isoSets.delete(setKey); set = undefined; }
    if (!set) { this.isoSets.set(setKey, (set = { segs: allocSegments(this.gpu, kernel.capacity, false), stamp: "" })); lru(this.isoSets, 32, (v) => v.segs.destroy()); }
    const stamp = `${kernelKey}|${level}|it${iterations}`;
    if (set.stamp !== stamp) { resetSegments(this.gpu, set.segs); kernel.dispatch(set.segs, level, iterations); set.stamp = stamp; }
    return set.segs;
  }

  /**
   * Segments of the isoline of `field` at `level`. `kernelKey` identifies (field, grid, colour field);
   * `setKey` identifies the slot (kernel + level index) whose segment set is reused across level changes.
   */
  isolines(kernelKey: string, setKey: string, field: ScalarFieldData, values: GpuGrid, colour: ScalarFieldData | undefined, level: number, tol: number, exact?: boolean): GpuSegments {
    let kernel = this.isoKernels.get(kernelKey);
    if (!kernel) { this.isoKernels.set(kernelKey, (kernel = fusedIsolines(this.gpu, field, values, colour, { exact }))); lru(this.isoKernels, 16, () => {}); }
    const cells = (values.grid.size[0]! - 1) * (values.grid.size[1]! - 1);
    const capacity = Math.min(kernel.capacity, Math.max(cells * 8, 65536));
    let set = this.isoSets.get(setKey);
    if (set && set.segs.capacity !== capacity) { set.segs.destroy(); this.isoSets.delete(setKey); set = undefined; }
    if (!set) { this.isoSets.set(setKey, (set = { segs: allocSegments(this.gpu, capacity, false), stamp: "" })); lru(this.isoSets, 32, (v) => v.segs.destroy()); }
    const stamp = `${kernelKey}|${level}|${tol.toExponential(3)}`;
    if (set.stamp !== stamp) {
      resetSegments(this.gpu, set.segs);
      kernel.dispatch(set.segs, level, tol);
      set.stamp = stamp;
    }
    return set.segs;
  }

  /** segments of the streamlines through resident `vectors` from `seeds` (key covers everything that affects them) */
  streamlines(key: string, vectors: GpuGrid, seeds: StreamlineSeeds, opts: FusedStreamlineOptions, colour: ScalarFieldData | undefined): GpuSegments {
    let e = this.streamKernels.get(key);
    if (!e) {
      const kernel = fusedStreamlines(this.gpu, vectors, seeds, opts, colour);
      const segs = allocSegments(this.gpu, kernel.capacity, true);
      kernel.dispatch(segs);
      this.streamKernels.set(key, (e = { kernel, segs }));
      lru(this.streamKernels, 8, (v) => v.segs.destroy());
    }
    return e.segs;
  }

  /** CPU-computed lines packed as Seg records (GPU render of CPU geometry) */
  uploadedSegments(key: string, pack: () => Float32Array, particles: boolean): GpuSegments {
    let s = this.uploaded.get(key);
    if (!s) { this.uploaded.set(key, (s = uploadSegments(this.gpu, pack(), particles))); lru(this.uploaded, 32, (v) => v.destroy()); }
    return s;
  }
}

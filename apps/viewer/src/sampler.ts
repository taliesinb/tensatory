// The sampling service: field values on grids, from the GPU when available,
// else from core on the CPU. Results are cached by key. GPU results arrive
// asynchronously: `request` returns undefined while a computation is in
// flight and calls `onReady` when it lands (the caller re-renders); CPU
// sampling stays synchronous so a CPU-only session behaves exactly as before.

import type { DenseGrid, ScalarFieldData, VectorFieldData } from "@tensatory/core";
import { GpuBackend, gpuSampleOn } from "@tensatory/gpu";

export type Backend = "gpu" | "cpu";
export type Values = Float64Array | Float32Array;

export class Sampler {
  private readonly cache = new Map<string, Values>();
  private readonly pending = new Set<string>();
  private readonly checked = new Set<string>();
  gpu: GpuBackend | undefined;
  backend: Backend = "cpu";
  /** compute every GPU sample on the CPU too and report the deviation (see the log) */
  check = false;
  /** human-readable description of the compute backend */
  label = "CPU";

  constructor(private readonly onReady: () => void) {}

  /** initialize; `prefer` "cpu" skips WebGPU entirely */
  async init(prefer: "auto" | "cpu" | "gpu"): Promise<void> {
    if (prefer === "cpu") return;
    try {
      this.gpu = await GpuBackend.create();
    } catch (e) {
      console.warn("WebGPU unavailable:", e);
    }
    if (this.gpu) { this.backend = "gpu"; this.label = `GPU (${this.gpu.adapterInfo || "WebGPU"})`; }
    else if (prefer === "gpu") console.warn("backend=gpu requested but no WebGPU adapter; using the CPU");
  }

  clear(): void { this.cache.clear(); this.pending.clear(); this.checked.clear(); }

  /**
   * Values of `field` on `grid` (row-major; D per point for vectors), NaN where
   * the grid leaves the field's box. Undefined while the GPU is computing.
   */
  request(key: string, field: ScalarFieldData | VectorFieldData, grid: DenseGrid): Values | undefined {
    const have = this.cache.get(key);
    if (have) return have;
    if (this.backend === "cpu" || !this.gpu) {
      const v = this.cpu(field, grid);
      this.store(key, v);
      return v;
    }
    if (this.pending.has(key)) return undefined;
    this.pending.add(key);
    const t0 = performance.now();
    gpuSampleOn(this.gpu, field, grid)
      .then((v) => {
        this.maskOutside(v, field, grid);
        this.store(key, v);
        if (this.check && !this.checked.has(key)) { this.checked.add(key); this.compare(key, v, field, grid, performance.now() - t0); }
      })
      .catch((e) => {
        console.error(`GPU sampling failed for ${key}; falling back to the CPU:`, e);
        this.store(key, this.cpu(field, grid));
      })
      .finally(() => { this.pending.delete(key); this.onReady(); });
    return undefined;
  }

  private store(key: string, v: Values): void {
    if (this.cache.size > 48) this.cache.delete(this.cache.keys().next().value!);
    this.cache.set(key, v);
  }

  /** core's sampling, with NaN outside the field's own box */
  cpu(field: ScalarFieldData | VectorFieldData, grid: DenseGrid): Float64Array {
    const inside = field.box.contains(grid.box.a, 1e-12) && field.box.contains(grid.box.b, 1e-12);
    const v = field.sampleOn(grid);
    if (!inside) this.maskOutside(v, field, grid);
    return v;
  }

  private maskOutside(v: Values, field: ScalarFieldData | VectorFieldData, grid: DenseGrid): void {
    if (field.box.contains(grid.box.a, 1e-12) && field.box.contains(grid.box.b, 1e-12)) return;
    const D = grid.dimCount, ch = field.rank === "scalar" ? 1 : D;
    const p = new Float64Array(D);
    for (let i = 0; i < grid.sampleCount; i++) {
      grid.pointInto(i, p);
      if (!field.box.contains(p, 1e-12)) for (let c = 0; c < ch; c++) v[i * ch + c] = NaN;
    }
  }

  private compare(key: string, gpu: Values, field: ScalarFieldData | VectorFieldData, grid: DenseGrid, gpuMs: number): void {
    const t0 = performance.now();
    const cpu = this.cpu(field, grid);
    const cpuMs = performance.now() - t0;
    let scale = 0;
    for (let i = 0; i < cpu.length; i++) if (Number.isFinite(cpu[i]!)) scale = Math.max(scale, Math.abs(cpu[i]!));
    let worst = 0, nanMismatch = 0;
    for (let i = 0; i < cpu.length; i++) {
      const a = cpu[i]!, b = gpu[i]!;
      if (Number.isNaN(a) || Number.isNaN(b)) { if (Number.isNaN(a) !== Number.isNaN(b)) nanMismatch++; continue; }
      worst = Math.max(worst, Math.abs(a - b) / (1e-5 * (scale || 1) + 2e-4 * Math.abs(a)));
    }
    const msg = `agreement ${key}: worst ${worst.toFixed(2)}× tolerance, ${nanMismatch} NaN mismatches; gpu ${gpuMs.toFixed(1)} ms, cpu ${cpuMs.toFixed(1)} ms`;
    if (worst > 1 || nanMismatch) console.warn(msg); else console.log(msg);
  }
}

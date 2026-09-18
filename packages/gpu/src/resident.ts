// GPU-resident grids: sampled field values that stay on the device for the
// raster pass and the fused geometry kernels. Read back only on demand.

import type { DenseGrid, ScalarFieldData, VectorFieldData } from "@tensatory/core";
import { RESIDENT_USAGE, type GpuBackend } from "./device";
import { buildSampleProgram } from "./program";

export interface GpuGrid {
  grid: DenseGrid;
  channels: number;
  /** f32 values, row-major, `channels` per point */
  buffer: GPUBuffer;
  destroy(): void;
}

/** sample `field` on `grid` into a resident buffer (no readback) */
export async function sampleResident(backend: GpuBackend, field: ScalarFieldData | VectorFieldData, grid: DenseGrid): Promise<GpuGrid> {
  const program = buildSampleProgram(field, grid);
  const n = program.sampleCount * program.channels;
  const { kept: [buffer] } = await backend.runKernel({
    code: program.code,
    invocations: program.sampleCount,
    buffers: [{ role: "rw", size: n * 4, keep: true }, { role: "r", data: program.data }],
  });
  return { grid, channels: program.channels, buffer: buffer!, destroy: () => buffer!.destroy() };
}

/** enqueue sampling into a resident buffer without waiting (queue-ordered before later passes) */
export function sampleResidentSync(backend: GpuBackend, field: ScalarFieldData | VectorFieldData, grid: DenseGrid): GpuGrid {
  const program = buildSampleProgram(field, grid);
  const n = program.sampleCount * program.channels;
  const [buffer] = backend.dispatch({
    code: program.code,
    invocations: program.sampleCount,
    buffers: [{ role: "rw", size: n * 4, keep: true }, { role: "r", data: program.data }],
  });
  return { grid, channels: program.channels, buffer: buffer!, destroy: () => buffer!.destroy() };
}

/** upload CPU values as a resident grid (for CPU-compute / GPU-render) */
export function uploadGrid(backend: GpuBackend, grid: DenseGrid, values: ArrayLike<number>, channels: number): GpuGrid {
  const data = values instanceof Float32Array ? values : Float32Array.from(values as ArrayLike<number>);
  const buffer = backend.createBuffer({ size: Math.max(16, data.byteLength), usage: RESIDENT_USAGE });
  backend.device.queue.writeBuffer(buffer, 0, data as unknown as BufferSource);
  return { grid, channels, buffer, destroy: () => buffer.destroy() };
}

/** read a resident grid back to the CPU */
export async function readGrid(backend: GpuBackend, g: GpuGrid): Promise<Float32Array> {
  await backend.whenIdle(); // deferred dispatches (async compiles) land first
  const dev = backend.device;
  const size = g.grid.sampleCount * g.channels * 4;
  const read = dev.createBuffer({ size: Math.max(16, size), usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const enc = dev.createCommandEncoder();
  enc.copyBufferToBuffer(g.buffer, 0, read, 0, Math.max(16, size));
  dev.queue.submit([enc.finish()]);
  await read.mapAsync(GPUMapMode.READ);
  const out = new Float32Array(read.getMappedRange().slice(0, size));
  read.unmap(); read.destroy();
  return out;
}

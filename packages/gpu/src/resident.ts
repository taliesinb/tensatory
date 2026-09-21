// GPU-resident grids: sampled field values that stay on the device for the
// raster pass and the fused geometry kernels. Read back only on demand.

import type { DenseGrid, ScalarFieldData, VectorFieldData } from "@tensatory/core";
import { RESIDENT_USAGE, programKernels, type GpuBackend } from "./device";
import { buildSampleProgram, type GpuProgram, type ResidentProvider } from "./program";

export interface GpuGrid {
  grid: DenseGrid;
  channels: number;
  /** f32 values, row-major, `channels` per point */
  buffer: GPUBuffer;
  destroy(): void;
}

/** the resident output buffer and the uploaded data of a program, and its dispatches over them */
function prepare(backend: GpuBackend, program: GpuProgram): { buffer: GPUBuffer; data: GPUBuffer; kernels: ReturnType<typeof programKernels> } {
  const dev = backend.device;
  const n = program.sampleCount * program.channels;
  const buffer = backend.createBuffer({ size: Math.max(16, n * 4), usage: RESIDENT_USAGE });
  const data = dev.createBuffer({ size: Math.max(16, program.data.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  dev.queue.writeBuffer(data, 0, program.data as unknown as BufferSource);
  return { buffer, data, kernels: programKernels(program, { role: "rw", buffer }, { role: "r", buffer: data }) };
}

/** sample `field` on `grid` into a resident buffer (no readback); `resident` serves nets that are arguments */
export async function sampleResident(backend: GpuBackend, field: ScalarFieldData | VectorFieldData, grid: DenseGrid, resident?: ResidentProvider): Promise<GpuGrid> {
  const program = buildSampleProgram(field, grid, resident);
  const { buffer, data, kernels } = prepare(backend, program);
  try { for (const k of kernels) await backend.runKernel(k); } finally { data.destroy(); }
  return { grid, channels: program.channels, buffer, destroy: () => buffer.destroy() };
}

/**
 * Enqueue sampling into a resident buffer without waiting (queue-ordered before later passes). A cooperative program
 * is several chunked dispatches; the data buffer is released once every dispatch has been submitted.
 */
export function sampleResidentSync(backend: GpuBackend, field: ScalarFieldData | VectorFieldData, grid: DenseGrid, resident?: ResidentProvider): GpuGrid {
  const program = buildSampleProgram(field, grid, resident);
  const { buffer, data, kernels } = prepare(backend, program);
  for (const k of kernels) backend.dispatch(k);
  void backend.whenIdle().then(() => data.destroy());
  return { grid, channels: program.channels, buffer, destroy: () => buffer.destroy() };
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

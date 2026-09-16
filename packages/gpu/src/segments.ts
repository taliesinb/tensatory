// The one line-geometry record every source produces and the renderer draws:
//   Seg { a, b: vec2  endpoints (world) | ca, cb: colour values at a and b
//         arc: arc length at a | len: total line length (0 = no particles) | phase }
// Fused kernels append segments through an atomic counter in an indirect-draw
// buffer; CPU polylines are packed into the same layout and uploaded.

import type { Polyline, Streamline } from "@tensatory/core";
import { RESIDENT_USAGE, type GpuBackend } from "./device";

export const SEG_FLOATS = 10; // 40 bytes, vec2-aligned
export const SEG_WGSL = `
struct Seg { a: vec2<f32>, b: vec2<f32>, ca: f32, cb: f32, arc: f32, len: f32, phase: f32, pad: f32 }
struct Indirect { vertexCount: u32, instanceCount: atomic<u32>, firstVertex: u32, firstInstance: u32 }
`;
/** WGSL: append `s` to `segs` (bound as array<Seg>) using `ind` (Indirect); CAP is the capacity constant */
export const SEG_APPEND_WGSL = `
fn appendSeg(s: Seg) {
  let i = atomicAdd(&ind.instanceCount, 1u);
  if (i < CAP) { segs[i] = s; }
}`;

export interface GpuSegments {
  buffer: GPUBuffer;
  /** [6, instanceCount, 0, 0] for drawIndirect; instanceCount may exceed capacity (extra instances draw nothing) */
  indirect: GPUBuffer;
  capacity: number;
  /** whether the segments carry particle data (len > 0) */
  particles: boolean;
  destroy(): void;
}

/** allocate an empty, resident segment set with an indirect buffer initialized to 6 vertices / 0 instances */
export function allocSegments(backend: GpuBackend, capacity: number, particles: boolean): GpuSegments {
  const dev = backend.device;
  const buffer = backend.createBuffer({ size: Math.max(16, capacity * SEG_FLOATS * 4), usage: RESIDENT_USAGE });
  const indirect = backend.createBuffer({ size: 16, usage: RESIDENT_USAGE });
  dev.queue.writeBuffer(indirect, 0, new Uint32Array([6, 0, 0, 0]));
  return { buffer, indirect, capacity, particles, destroy: () => { buffer.destroy(); indirect.destroy(); } };
}

/** reset the instance counter of a segment set (before a kernel appends into it again) */
export function resetSegments(backend: GpuBackend, s: GpuSegments): void {
  backend.device.queue.writeBuffer(s.indirect, 0, new Uint32Array([6, 0, 0, 0]));
}

/** pack polylines (optionally with per-vertex colour values) into Seg records */
export function packPolylines(lines: Polyline[], values?: (ArrayLike<number> | undefined)[]): Float32Array {
  let count = 0;
  for (const l of lines) count += Math.max(0, l.length / 2 - 1);
  const out = new Float32Array(count * SEG_FLOATS);
  let o = 0;
  lines.forEach((l, li) => {
    const v = values?.[li];
    for (let i = 0; i + 3 < l.length; i += 2) {
      out[o] = l[i]!; out[o + 1] = l[i + 1]!; out[o + 2] = l[i + 2]!; out[o + 3] = l[i + 3]!;
      out[o + 4] = v ? v[i / 2]! : 0; out[o + 5] = v ? v[i / 2 + 1]! : 0;
      // arc, len, phase = 0: no particles
      o += SEG_FLOATS;
    }
  });
  return out;
}

/** pack streamlines with arc / length / phase for particle animation */
export function packStreamlines(lines: Streamline[], step: number, values?: (ArrayLike<number> | undefined)[]): Float32Array {
  let count = 0;
  for (const l of lines) count += Math.max(0, l.points.length / 2 - 1);
  const out = new Float32Array(count * SEG_FLOATS);
  let o = 0;
  lines.forEach((l, li) => {
    const v = values?.[li], p = l.points;
    for (let i = 0; i + 3 < p.length; i += 2) {
      out[o] = p[i]!; out[o + 1] = p[i + 1]!; out[o + 2] = p[i + 2]!; out[o + 3] = p[i + 3]!;
      out[o + 4] = v ? v[i / 2]! : 0; out[o + 5] = v ? v[i / 2 + 1]! : 0;
      out[o + 6] = (i / 2) * step; out[o + 7] = l.length; out[o + 8] = l.phase;
      o += SEG_FLOATS;
    }
  });
  return out;
}

/** upload packed records as a resident segment set */
export function uploadSegments(backend: GpuBackend, data: Float32Array, particles: boolean): GpuSegments {
  const count = data.length / SEG_FLOATS;
  const s = allocSegments(backend, Math.max(1, count), particles);
  if (count) backend.device.queue.writeBuffer(s.buffer, 0, data as unknown as BufferSource);
  backend.device.queue.writeBuffer(s.indirect, 0, new Uint32Array([6, count, 0, 0]));
  return s;
}

/** read a segment set back (tests / debugging): the valid records, as Seg floats */
export async function readSegments(backend: GpuBackend, s: GpuSegments): Promise<Float32Array> {
  const dev = backend.device;
  const readInd = dev.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const readSeg = dev.createBuffer({ size: s.buffer.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const enc = dev.createCommandEncoder();
  enc.copyBufferToBuffer(s.indirect, 0, readInd, 0, 16);
  enc.copyBufferToBuffer(s.buffer, 0, readSeg, 0, s.buffer.size);
  dev.queue.submit([enc.finish()]);
  await readInd.mapAsync(GPUMapMode.READ);
  const n = Math.min(s.capacity, new Uint32Array(readInd.getMappedRange())[1]!);
  readInd.unmap(); readInd.destroy();
  await readSeg.mapAsync(GPUMapMode.READ);
  const out = new Float32Array(readSeg.getMappedRange().slice(0, n * SEG_FLOATS * 4));
  readSeg.unmap(); readSeg.destroy();
  return out;
}

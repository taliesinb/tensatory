// Line geometry in 3D: Seg3 records (the 3D counterpart of segments.ts' Seg),
// packed from CPU polylines / streamlines or appended by kernels, drawn by
// GpuRenderer3D as screen-space thick quads with depth.
//
//   Seg3 { a: vec3, ca | b: vec3, cb | arc, len, phase, pad }   (12 floats, 48 bytes)
//
// The renderer also draws 2D Seg sets (segments.ts) embedded on a box face
// (axis + depth), so the fused 2D isoline kernels serve the face isolines.

import type { Streamline } from "@tensatory/core";
import { RESIDENT_USAGE, type GpuBackend } from "./device";

export const SEG3_FLOATS = 12;
export const SEG3_WGSL = `
struct Seg3 { a: vec3<f32>, ca: f32, b: vec3<f32>, cb: f32, arc: f32, len: f32, phase: f32, pad: f32 }
struct Indirect3 { vertexCount: u32, instanceCount: atomic<u32>, firstVertex: u32, firstInstance: u32 }
`;
export const SEG3_APPEND_WGSL = `
fn appendSeg3(s: Seg3) {
  let i = atomicAdd(&ind.instanceCount, 1u);
  if (i < CAP) { segs[i] = s; }
}`;

export interface GpuSegments3 {
  buffer: GPUBuffer;
  /** [6, instanceCount, 0, 0] */
  indirect: GPUBuffer;
  capacity: number;
  particles: boolean;
  destroy(): void;
}

export function allocSegments3(backend: GpuBackend, capacity: number, particles: boolean): GpuSegments3 {
  const dev = backend.device;
  const buffer = backend.createBuffer({ size: Math.max(48, capacity * SEG3_FLOATS * 4), usage: RESIDENT_USAGE });
  const indirect = backend.createBuffer({ size: 16, usage: RESIDENT_USAGE });
  dev.queue.writeBuffer(indirect, 0, new Uint32Array([6, 0, 0, 0]));
  return { buffer, indirect, capacity, particles, destroy: () => { buffer.destroy(); indirect.destroy(); } };
}
export function resetSegments3(backend: GpuBackend, s: GpuSegments3): void {
  backend.device.queue.writeBuffer(s.indirect, 0, new Uint32Array([6, 0, 0, 0]));
}
export function uploadSegments3(backend: GpuBackend, data: Float32Array, particles: boolean): GpuSegments3 {
  const count = data.length / SEG3_FLOATS;
  const s = allocSegments3(backend, Math.max(1, count), particles);
  if (count) backend.device.queue.writeBuffer(s.buffer, 0, data as unknown as BufferSource);
  backend.device.queue.writeBuffer(s.indirect, 0, new Uint32Array([6, count, 0, 0]));
  return s;
}

/** pack 3D polylines (xyz triples) with optional per-vertex colour values */
export function packPolylines3(lines: ArrayLike<number>[], values?: (ArrayLike<number> | undefined)[]): Float32Array {
  let count = 0;
  for (const l of lines) count += Math.max(0, l.length / 3 - 1);
  const out = new Float32Array(count * SEG3_FLOATS);
  let o = 0;
  lines.forEach((l, li) => {
    const v = values?.[li];
    for (let i = 0; i + 5 < l.length; i += 3) {
      out[o] = l[i]!; out[o + 1] = l[i + 1]!; out[o + 2] = l[i + 2]!; out[o + 3] = v ? v[i / 3]! : 0;
      out[o + 4] = l[i + 3]!; out[o + 5] = l[i + 4]!; out[o + 6] = l[i + 5]!; out[o + 7] = v ? v[i / 3 + 1]! : 0;
      o += SEG3_FLOATS;
    }
  });
  return out;
}

/** pack 3D streamlines with arc / length / phase for particles */
export function packStreamlines3(lines: Streamline[], step: number, values?: (ArrayLike<number> | undefined)[]): Float32Array {
  let count = 0;
  for (const l of lines) count += Math.max(0, l.points.length / 3 - 1);
  const out = new Float32Array(count * SEG3_FLOATS);
  let o = 0;
  lines.forEach((l, li) => {
    const v = values?.[li], p = l.points;
    for (let i = 0; i + 5 < p.length; i += 3) {
      out[o] = p[i]!; out[o + 1] = p[i + 1]!; out[o + 2] = p[i + 2]!; out[o + 3] = v ? v[i / 3]! : 0;
      out[o + 4] = p[i + 3]!; out[o + 5] = p[i + 4]!; out[o + 6] = p[i + 5]!; out[o + 7] = v ? v[i / 3 + 1]! : 0;
      out[o + 8] = (i / 3) * step; out[o + 9] = l.length; out[o + 10] = l.phase;
      o += SEG3_FLOATS;
    }
  });
  return out;
}

/** pack filled 3D triangles (flat [baseLeft, baseRight, apex], 9 floats each) as Seg3 records: a, b = the base, (arc, len, phase) = the apex */
export function packTriangles3(tris: ArrayLike<number>, values?: ArrayLike<number>): Float32Array {
  const count = Math.floor(tris.length / 9);
  const out = new Float32Array(count * SEG3_FLOATS);
  for (let i = 0; i < count; i++) {
    const o = i * SEG3_FLOATS, t = i * 9, c = values ? values[i]! : 0;
    out[o] = tris[t]!; out[o + 1] = tris[t + 1]!; out[o + 2] = tris[t + 2]!; out[o + 3] = c;
    out[o + 4] = tris[t + 3]!; out[o + 5] = tris[t + 4]!; out[o + 6] = tris[t + 5]!; out[o + 7] = c;
    out[o + 8] = tris[t + 6]!; out[o + 9] = tris[t + 7]!; out[o + 10] = tris[t + 8]!;
  }
  return out;
}

/** the 12 edges of a box as one polyline set */
export function boxEdges(a: ArrayLike<number>, b: ArrayLike<number>): Float32Array {
  const corner = (m: number) => [m & 1 ? b[0]! : a[0]!, m & 2 ? b[1]! : a[1]!, m & 4 ? b[2]! : a[2]!];
  const lines: number[][] = [];
  for (let m = 0; m < 8; m++) for (const bit of [1, 2, 4]) if (!(m & bit)) lines.push([...corner(m), ...corner(m | bit)]);
  return packPolylines3(lines);
}

/** read a Seg3 set back (tests / debugging): the valid records as floats */
export async function readSegments3(backend: GpuBackend, s: GpuSegments3): Promise<Float32Array> {
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
  const out = new Float32Array(readSeg.getMappedRange().slice(0, n * SEG3_FLOATS * 4));
  readSeg.unmap(); readSeg.destroy();
  return out;
}

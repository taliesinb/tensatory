// Fused 3D streamlines: the vec3 twin of fusedStreamlines (fused.ts). One
// thread per seed integrates RK4 both ways (within the seed's step budgets)
// through a resident 3-channel vector grid (trilinear) into scratch slots, then
// appends Seg3 records with arc / length / phase and the colour program's value
// at both ends. Seeds and budgets come from core's planners (stratified, JL,
// coverage), exactly as in 2D.

import { DenseGrid, type StreamlineSeeds } from "@tensatory/core";
import { colourCode, type ColourSource } from "./colour";
import type { GpuBackend } from "./device";
import type { FusedStreamlineOptions } from "./fused";
import { SEG3_APPEND_WGSL, SEG3_WGSL, type GpuSegments3 } from "./lines3d";
import { ProgramBuilder } from "./program";
import type { GpuGrid } from "./resident";
import { f32 } from "./wgsl";

export const SEED3_FLOATS = 8; // x, y, z, phase, back budget, forward budget, base slot, pad

/** pack 3D seeds with their budgets and prefix-summed point slots (see flow.ts packSeeds for 2D) */
export function packSeeds3(seeds: StreamlineSeeds, maxSteps: number, bidirectional = true): { data: Float32Array; lines: number; points: number; segments: number } {
  const lines = seeds.phases.length, b = seeds.budgets, maxBack = bidirectional ? maxSteps : 0;
  const data = new Float32Array(lines * SEED3_FLOATS);
  let points = 0, segments = 0;
  for (let i = 0; i < lines; i++) {
    const nb = b ? Math.min(maxBack, b[2 * i]!) : maxBack, nf = b ? Math.min(maxSteps, b[2 * i + 1]!) : maxSteps;
    const o = i * SEED3_FLOATS;
    data[o] = seeds.points[i * 3]!; data[o + 1] = seeds.points[i * 3 + 1]!; data[o + 2] = seeds.points[i * 3 + 2]!; data[o + 3] = seeds.phases[i]!;
    data[o + 4] = nb; data[o + 5] = nf; data[o + 6] = points + nb;
    points += nb + 1 + nf; segments += nb + nf;
  }
  return { data, lines, points, segments };
}

export interface FusedStreamlines3 {
  run(segs: GpuSegments3): Promise<void>;
  dispatch(segs: GpuSegments3): void;
  /** exact segment count for this seed set */
  readonly capacity: number;
  destroy(): void;
}

export function fusedStreamlines3(backend: GpuBackend, vectors: GpuGrid, seeds: StreamlineSeeds, opts: FusedStreamlineOptions, colour?: ColourSource): FusedStreamlines3 {
  const grid = vectors.grid;
  if (grid.dimCount !== 3 || vectors.channels !== 3) throw new Error("fusedStreamlines3 needs a resident 3D vector grid");
  const box = opts.box ?? grid.box;
  const sgn = opts.sign ?? 1, h = opts.step;
  const packed = packSeeds3(seeds, opts.maxSteps, opts.bidirectional);
  const lines = packed.lines;
  const b = new ProgramBuilder(new DenseGrid([2, 2, 2], grid.box));
  const cc = colourCode(b, colour, 3, "0.0", 6);
  const lib = b.library();
  const [nx, ny, nz] = grid.size as [number, number, number];
  const [sx, sy, sz] = grid.strides as [number, number, number];
  const eps = 1e-6 * Math.max(...box.size);
  const capacity = Math.max(1, packed.segments);
  const code = `${lib.code}
${SEG3_WGSL}
@group(0) @binding(0) var<storage, read_write> segs: array<Seg3>;
@group(0) @binding(2) var<storage, read> vec: array<f32>;
@group(0) @binding(3) var<storage, read_write> ind: Indirect3;
@group(0) @binding(4) var<storage, read> seeds: array<f32>;
@group(0) @binding(5) var<storage, read_write> scratch: array<f32>;
const CAP: u32 = ${capacity}u;
${SEG3_APPEND_WGSL}
const A: vec3<f32> = vec3<f32>(${f32(box.a[0]!)}, ${f32(box.a[1]!)}, ${f32(box.a[2]!)});
const B: vec3<f32> = vec3<f32>(${f32(box.b[0]!)}, ${f32(box.b[1]!)}, ${f32(box.b[2]!)});
const EPS: f32 = ${f32(eps)};
const H: f32 = ${f32(h)}; const SGN: f32 = ${f32(sgn)};
fn inBox(q: vec3<f32>) -> bool { return all(q >= A - EPS) && all(q <= B + EPS); }
fn at(i: i32, j: i32, k: i32) -> vec3<f32> { let o = (i * ${sx} + j * ${sy} + k * ${sz}) * 3; return vec3<f32>(vec[o], vec[o + 1], vec[o + 2]); }
// trilinear read of the resident vector grid
fn field(p: vec3<f32>) -> vec3<f32> {
  let g = clamp((p - vec3<f32>(${f32(grid.box.a[0]!)}, ${f32(grid.box.a[1]!)}, ${f32(grid.box.a[2]!)})) / vec3<f32>(${f32(grid.spacing[0]!)}, ${f32(grid.spacing[1]!)}, ${f32(grid.spacing[2]!)}), vec3<f32>(0.0), vec3<f32>(${f32(nx - 1)}, ${f32(ny - 1)}, ${f32(nz - 1)}));
  var i0 = vec3<i32>(floor(g));
  i0 = min(i0, vec3<i32>(${Math.max(0, nx - 2)}, ${Math.max(0, ny - 2)}, ${Math.max(0, nz - 2)}));
  let f = g - vec3<f32>(i0);
  let i1 = min(i0 + 1, vec3<i32>(${nx - 1}, ${ny - 1}, ${nz - 1}));
  let c00 = mix(at(i0.x, i0.y, i0.z), at(i1.x, i0.y, i0.z), f.x);
  let c10 = mix(at(i0.x, i1.y, i0.z), at(i1.x, i1.y, i0.z), f.x);
  let c01 = mix(at(i0.x, i0.y, i1.z), at(i1.x, i0.y, i1.z), f.x);
  let c11 = mix(at(i0.x, i1.y, i1.z), at(i1.x, i1.y, i1.z), f.x);
  return mix(mix(c00, c10, f.y), mix(c01, c11, f.y), f.z);
}
fn dir(q: vec3<f32>) -> vec4<f32> {
  if (!inBox(q)) { return vec4<f32>(0.0); }
  let v = field(q);
  let l = dot(v, v);
  if (!isfinite_(l) || !(l > 1e-24)) { return vec4<f32>(0.0); }
  return vec4<f32>(v * (SGN / sqrt(l)), 1.0);
}
fn integrate(seed: vec3<f32>, s: f32, base: i32, stepDir: i32, steps: i32) -> i32 {
  var q = seed; var n: i32 = 0;
  for (var k = 0; k < steps; k++) {
    let k1 = dir(q); if (k1.w == 0.0) { break; }
    let k2 = dir(q + s * 0.5 * H * k1.xyz); if (k2.w == 0.0) { break; }
    let k3 = dir(q + s * 0.5 * H * k2.xyz); if (k3.w == 0.0) { break; }
    let k4 = dir(q + s * H * k3.xyz); if (k4.w == 0.0) { break; }
    q = q + (s * H / 6.0) * (k1.xyz + 2.0 * k2.xyz + 2.0 * k3.xyz + k4.xyz);
    if (!inBox(q)) { break; }
    let o = (base + stepDir * (k + 1)) * 3;
    scratch[o] = q.x; scratch[o + 1] = q.y; scratch[o + 2] = q.z; n++;
  }
  return n;
}
${cc.code}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = i32(id.x);
  if (i >= ${lines}) { return; }
  let s = i * ${SEED3_FLOATS};
  let seed = vec3<f32>(seeds[s], seeds[s + 1], seeds[s + 2]);
  let phase = seeds[s + 3];
  let base = i32(seeds[s + 6]);
  scratch[base * 3] = seed.x; scratch[base * 3 + 1] = seed.y; scratch[base * 3 + 2] = seed.z;
  let nb = integrate(seed, -1.0, base, -1, i32(seeds[s + 4]));
  let nf = integrate(seed, 1.0, base, 1, i32(seeds[s + 5]));
  let n = nb + 1 + nf;
  if (n < 2) { return; }
  let len = f32(n - 1) * H;
  let start = base - nb;
  var prev = vec3<f32>(scratch[start * 3], scratch[start * 3 + 1], scratch[start * 3 + 2]);
  var cPrev = colour_(prev);
  for (var k = 1; k < n; k++) {
    let o = (start + k) * 3;
    let cur = vec3<f32>(scratch[o], scratch[o + 1], scratch[o + 2]);
    let cCur = colour_(cur);
    var sg: Seg3; sg.a = prev; sg.b = cur; sg.ca = cPrev; sg.cb = cCur; sg.arc = f32(k - 1) * H; sg.len = len; sg.phase = phase; sg.pad = 0.0;
    appendSeg3(sg);
    prev = cur; cPrev = cCur;
  }
}`;
  const kernel = (segs: GpuSegments3) => ({
    code,
    invocations: Math.max(1, lines),
    buffers: [
      { role: "rw" as const, buffer: segs.buffer },
      { role: "r" as const, data: lib.data },
      { role: "r" as const, buffer: vectors.buffer },
      { role: "rw" as const, buffer: segs.indirect },
      { role: "r" as const, data: packed.data.length ? packed.data : new Float32Array(SEED3_FLOATS) },
      { role: "rw" as const, size: Math.max(16, packed.points * 3 * 4) },
      ...cc.buffers,
    ],
  });
  return {
    capacity,
    async run(segs) { await backend.runKernel(kernel(segs)); },
    dispatch(segs) { backend.dispatch(kernel(segs)); },
    destroy() { /* nothing resident of its own */ },
  };
}

// Streamline integration on the GPU: one invocation per seed, RK4 on the unit
// field in both directions, through a vector field (dense grids are read with
// bilinear interpolation, symbolic fields are transpiled). Mirrors core's
// `integrateFromSeeds` in f32.

import { DenseGrid, type Box, type Streamline, type StreamlineSeeds, type VectorFieldData } from "@tensatory/core";
import type { GpuBackend } from "./device";
import { ProgramBuilder } from "./program";
import { f32 } from "./wgsl";

export interface GpuStreamlineOptions {
  maxSteps: number;
  step: number;
  sign?: 1 | -1;
  box?: Box;
}

export async function gpuIntegrateFromSeeds(backend: GpuBackend, field: VectorFieldData, seeds: StreamlineSeeds, opts: GpuStreamlineOptions): Promise<Streamline[]> {
  const box = opts.box ?? field.box;
  const sgn = opts.sign ?? 1;
  const h = opts.step, M = opts.maxSteps;
  const lines = seeds.phases.length;
  // the builder needs some dispatch grid for direct reads; none apply here (pos = -1)
  const b = new ProgramBuilder(field.samplePoints ?? new DenseGrid([2, 2], field.box));
  const vf = b.vector(field);
  const lib = b.library();
  const eps = 1e-6 * Math.max(box.size[0]!, box.size[1]!);
  const stride = 2 * M + 1; // points per line: back(M) + seed + fwd(M)
  const code = `${lib.code}
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<storage, read> seeds: array<f32>;
@group(0) @binding(3) var<storage, read_write> counts: array<u32>;
const A: vec2<f32> = vec2<f32>(${f32(box.a[0]!)}, ${f32(box.a[1]!)});
const B: vec2<f32> = vec2<f32>(${f32(box.b[0]!)}, ${f32(box.b[1]!)});
const EPS: f32 = ${f32(eps)};
const H: f32 = ${f32(h)}; const SGN: f32 = ${f32(sgn)}; const M: i32 = ${M}; const STRIDE: i32 = ${stride};
fn inBox(q: vec2<f32>) -> bool { return q.x >= A.x - EPS && q.x <= B.x + EPS && q.y >= A.y - EPS && q.y <= B.y + EPS; }
// unit direction, or (0,0) with ok = false where the field vanishes / is undefined / outside
fn dir(q: vec2<f32>) -> vec3<f32> {
  if (!inBox(q)) { return vec3<f32>(0.0, 0.0, 0.0); }
  let v = ${vf}(q, -1);
  let l = dot(v, v);
  if (!isfinite_(l) || !(l > 1e-24)) { return vec3<f32>(0.0, 0.0, 0.0); }
  return vec3<f32>(v * (SGN / sqrt(l)), 1.0);
}
// integrate from the seed in direction s (±1); writes points and returns their count
fn integrate(seed: vec2<f32>, s: f32, base: i32, stepDir: i32) -> u32 {
  var q = seed; var n: u32 = 0u;
  for (var k = 0; k < M; k++) {
    let k1 = dir(q); if (k1.z == 0.0) { break; }
    let k2 = dir(q + s * 0.5 * H * k1.xy); if (k2.z == 0.0) { break; }
    let k3 = dir(q + s * 0.5 * H * k2.xy); if (k3.z == 0.0) { break; }
    let k4 = dir(q + s * H * k3.xy); if (k4.z == 0.0) { break; }
    q = q + (s * H / 6.0) * (k1.xy + 2.0 * k2.xy + 2.0 * k3.xy + k4.xy);
    if (!inBox(q)) { break; }
    let o = (base + stepDir * (k + 1)) * 2;
    out[o] = q.x; out[o + 1] = q.y; n++;
  }
  return n;
}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = i32(id.x);
  if (i >= ${lines}) { return; }
  let seed = vec2<f32>(seeds[i * 2], seeds[i * 2 + 1]);
  let base = i * STRIDE + M; // the seed's slot
  out[base * 2] = seed.x; out[base * 2 + 1] = seed.y;
  counts[i * 2] = integrate(seed, -1.0, base, -1);
  counts[i * 2 + 1] = integrate(seed, 1.0, base, 1);
}`;
  const { read: [ptsBuf, cntBuf] } = await backend.runKernel({
    code,
    invocations: lines,
    buffers: [
      { role: "rw", size: lines * stride * 2 * 4, readback: true },
      { role: "r", data: lib.data },
      { role: "r", data: Float32Array.from(seeds.points) },
      { role: "rw", size: lines * 2 * 4, readback: true },
    ],
  });
  const pts = new Float32Array(ptsBuf!), counts = new Uint32Array(cntBuf!);
  const out: Streamline[] = [];
  for (let i = 0; i < lines; i++) {
    const nb = counts[i * 2]!, nf = counts[i * 2 + 1]!;
    const n = nb + 1 + nf;
    if (n < 2) continue;
    const start = i * stride + M - nb;
    out.push({ points: Float64Array.from(pts.subarray(start * 2, (start + n) * 2)), length: (n - 1) * h, phase: seeds.phases[i]! });
  }
  return out;
}

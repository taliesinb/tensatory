// Streamline integration on the GPU: one invocation per seed, RK4 on the unit
// field in both directions, through a vector field (dense grids are read with
// bilinear interpolation, symbolic fields are transpiled). Mirrors core's
// `integrateFromSeeds` in f32, including per-seed step budgets.

import { DenseGrid, type Box, type Streamline, type StreamlineSeeds, type VectorFieldData } from "@tensatory/core";
import type { GpuBackend } from "./device";
import { ProgramBuilder } from "./program";
import { f32 } from "./wgsl";

export interface GpuStreamlineOptions {
  maxSteps: number;
  step: number;
  sign?: 1 | -1;
  box?: Box;
  /** also integrate against the direction from each seed (default true); false = lines start at their seeds */
  bidirectional?: boolean;
}

/** floats per seed in the packed seed buffer: x, y, phase, back budget, forward budget, slot of the seed point */
export const SEED_FLOATS = 6;

export interface PackedSeeds {
  /** [x, y, phase, nb, nf, base] per seed; base = index of the seed's point slot (back points before it, forward after) */
  data: Float32Array;
  lines: number;
  /** total point slots = Σ (nb + 1 + nf) */
  points: number;
  /** worst-case segment count = Σ (nb + nf) */
  segments: number;
}

/** pack seeds with their budgets (capped at `maxSteps`, defaulting to it; the backward budget is 0 when not `bidirectional`) and prefix-summed point slots */
export function packSeeds(seeds: StreamlineSeeds, maxSteps: number, bidirectional = true): PackedSeeds {
  const lines = seeds.phases.length, b = seeds.budgets, maxBack = bidirectional ? maxSteps : 0;
  const data = new Float32Array(lines * SEED_FLOATS);
  let points = 0, segments = 0;
  for (let i = 0; i < lines; i++) {
    const nb = b ? Math.min(maxBack, b[2 * i]!) : maxBack, nf = b ? Math.min(maxSteps, b[2 * i + 1]!) : maxSteps;
    const o = i * SEED_FLOATS;
    data[o] = seeds.points[i * 2]!; data[o + 1] = seeds.points[i * 2 + 1]!; data[o + 2] = seeds.phases[i]!;
    data[o + 3] = nb; data[o + 4] = nf; data[o + 5] = points + nb;
    points += nb + 1 + nf; segments += nb + nf;
  }
  return { data, lines, points, segments };
}

/** WGSL: the RK4 loop shared by the readback and the fused kernel; `dir(q) -> vec3` (unit direction, z = ok) must be defined */
export function integrateWgsl(pointsBuffer: string): string {
  return `
// integrate from the seed in direction s (±1) for at most 'steps' steps; writes points into slots base ± k and returns their count
fn integrate(seed: vec2<f32>, s: f32, base: i32, stepDir: i32, steps: i32) -> i32 {
  var q = seed; var n: i32 = 0;
  for (var k = 0; k < steps; k++) {
    let k1 = dir(q); if (k1.z == 0.0) { break; }
    let k2 = dir(q + s * 0.5 * H * k1.xy); if (k2.z == 0.0) { break; }
    let k3 = dir(q + s * 0.5 * H * k2.xy); if (k3.z == 0.0) { break; }
    let k4 = dir(q + s * H * k3.xy); if (k4.z == 0.0) { break; }
    q = q + (s * H / 6.0) * (k1.xy + 2.0 * k2.xy + 2.0 * k3.xy + k4.xy);
    if (!inBox(q)) { break; }
    let o = (base + stepDir * (k + 1)) * 2;
    ${pointsBuffer}[o] = q.x; ${pointsBuffer}[o + 1] = q.y; n++;
  }
  return n;
}`;
}

export async function gpuIntegrateFromSeeds(backend: GpuBackend, field: VectorFieldData, seeds: StreamlineSeeds, opts: GpuStreamlineOptions): Promise<Streamline[]> {
  const box = opts.box ?? field.box;
  const sgn = opts.sign ?? 1;
  const h = opts.step;
  const packed = packSeeds(seeds, opts.maxSteps, opts.bidirectional);
  const lines = packed.lines;
  if (!lines) return [];
  // the builder needs some dispatch grid for direct reads; none apply here (pos = -1)
  const b = new ProgramBuilder(field.samplePoints ?? new DenseGrid([2, 2], field.box));
  const vf = b.vector(field);
  const lib = b.library();
  const eps = 1e-6 * Math.max(box.size[0]!, box.size[1]!);
  const code = `${lib.code}
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<storage, read> seeds: array<f32>;
@group(0) @binding(3) var<storage, read_write> counts: array<u32>;
const A: vec2<f32> = vec2<f32>(${f32(box.a[0]!)}, ${f32(box.a[1]!)});
const B: vec2<f32> = vec2<f32>(${f32(box.b[0]!)}, ${f32(box.b[1]!)});
const EPS: f32 = ${f32(eps)};
const H: f32 = ${f32(h)}; const SGN: f32 = ${f32(sgn)};
fn inBox(q: vec2<f32>) -> bool { return q.x >= A.x - EPS && q.x <= B.x + EPS && q.y >= A.y - EPS && q.y <= B.y + EPS; }
// unit direction, or (0,0) with ok = false where the field vanishes / is undefined / outside
fn dir(q: vec2<f32>) -> vec3<f32> {
  if (!inBox(q)) { return vec3<f32>(0.0, 0.0, 0.0); }
  let v = ${vf}(q, -1);
  let l = dot(v, v);
  if (!isfinite_(l) || !(l > 1e-24)) { return vec3<f32>(0.0, 0.0, 0.0); }
  return vec3<f32>(v * (SGN / sqrt(l)), 1.0);
}
${integrateWgsl("out")}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = i32(id.x);
  if (i >= ${lines}) { return; }
  let s = i * ${SEED_FLOATS};
  let seed = vec2<f32>(seeds[s], seeds[s + 1]);
  let base = i32(seeds[s + 5]); // the seed's slot
  out[base * 2] = seed.x; out[base * 2 + 1] = seed.y;
  counts[i * 2] = u32(integrate(seed, -1.0, base, -1, i32(seeds[s + 3])));
  counts[i * 2 + 1] = u32(integrate(seed, 1.0, base, 1, i32(seeds[s + 4])));
}`;
  const { read: [ptsBuf, cntBuf] } = await backend.runKernel({
    code,
    invocations: lines,
    buffers: [
      { role: "rw", size: Math.max(4, packed.points * 2 * 4), readback: true },
      { role: "r", data: lib.data },
      { role: "r", data: packed.data },
      { role: "rw", size: lines * 2 * 4, readback: true },
    ],
  });
  const pts = new Float32Array(ptsBuf!), counts = new Uint32Array(cntBuf!);
  const out: Streamline[] = [];
  for (let i = 0; i < lines; i++) {
    const nb = counts[i * 2]!, nf = counts[i * 2 + 1]!;
    const n = nb + 1 + nf;
    if (n < 2) continue;
    const start = packed.data[i * SEED_FLOATS + 5]! - nb;
    out.push({ points: Float64Array.from(pts.subarray(start * 2, (start + n) * 2)), length: (n - 1) * h, phase: seeds.phases[i]! });
  }
  return out;
}

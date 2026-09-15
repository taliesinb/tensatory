// Isolines on the GPU: marching squares as a per-cell kernel, and Newton
// projection of vertices onto the level set as a per-vertex kernel. The
// adaptive chord refinement of core's `exactIsoContours` is driven from the
// CPU as a handful of projection rounds (one dispatch per round).

import {
  DenseGrid,
  joinSegments,
  type ContourResult,
  type Polyline,
  type ScalarFieldData,
} from "@tensatory/core";
import type { GpuBackend } from "./device";
import { ProgramBuilder } from "./program";
import { f32 } from "./wgsl";

/*******************************************************/
/* marching squares */

function marchingSquaresCode(grid: DenseGrid): string {
  const [nx, ny] = grid.size as [number, number];
  const sx = grid.strides[0]!, sy = grid.strides[1]!;
  return `
@group(0) @binding(0) var<storage, read_write> segs: array<f32>;
@group(0) @binding(1) var<storage, read_write> counts: array<u32>;
@group(0) @binding(2) var<storage, read> vals: array<f32>;
@group(0) @binding(3) var<storage, read> lvl: array<f32>;
const NX: i32 = ${nx}; const NY: i32 = ${ny};
fn t_(v0: f32, v1: f32, level: f32) -> f32 { return select((level - v0) / (v1 - v0), 0.5, v0 == v1); }
fn isnan_(v: f32) -> bool { let b = bitcast<u32>(v); return (b & 0x7f800000u) == 0x7f800000u && (b & 0x007fffffu) != 0u; }
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = i32(id.x);
  if (c >= (NX - 1) * (NY - 1)) { return; }
  let i = c / (NY - 1); let j = c % (NY - 1);
  let level = lvl[0];
  let v00 = vals[i * ${sx} + j * ${sy}]; let v10 = vals[(i + 1) * ${sx} + j * ${sy}];
  let v11 = vals[(i + 1) * ${sx} + (j + 1) * ${sy}]; let v01 = vals[i * ${sx} + (j + 1) * ${sy}];
  counts[c] = 0u;
  if (isnan_(v00) || isnan_(v10) || isnan_(v11) || isnan_(v01)) { return; }
  var code: u32 = 0u;
  if (v00 >= level) { code |= 1u; } if (v10 >= level) { code |= 2u; } if (v11 >= level) { code |= 4u; } if (v01 >= level) { code |= 8u; }
  if (code == 0u || code == 15u) { return; }
  let x0 = ${f32(grid.box.a[0]!)} + f32(i) * ${f32(grid.spacing[0]!)};
  let y0 = ${f32(grid.box.a[1]!)} + f32(j) * ${f32(grid.spacing[1]!)};
  let hx = ${f32(grid.spacing[0]!)}; let hy = ${f32(grid.spacing[1]!)};
  let B = vec2<f32>(x0 + t_(v00, v10, level) * hx, y0);
  let R = vec2<f32>(x0 + hx, y0 + t_(v10, v11, level) * hy);
  let T = vec2<f32>(x0 + t_(v01, v11, level) * hx, y0 + hy);
  let L = vec2<f32>(x0, y0 + t_(v00, v01, level) * hy);
  var a0: vec2<f32>; var b0: vec2<f32>; var a1: vec2<f32>; var b1: vec2<f32>; var n: u32 = 1u;
  switch (code) {
    case 1u, 14u: { a0 = L; b0 = B; }
    case 2u, 13u: { a0 = B; b0 = R; }
    case 3u, 12u: { a0 = L; b0 = R; }
    case 4u, 11u: { a0 = R; b0 = T; }
    case 6u, 9u:  { a0 = B; b0 = T; }
    case 7u, 8u:  { a0 = L; b0 = T; }
    default: {
      let centreHigh = 0.25 * (v00 + v10 + v11 + v01) >= level;
      n = 2u;
      if ((code == 5u) == centreHigh) { a0 = L; b0 = T; a1 = B; b1 = R; } else { a0 = L; b0 = B; a1 = R; b1 = T; }
    }
  }
  let o = c * 8;
  segs[o] = a0.x; segs[o + 1] = a0.y; segs[o + 2] = b0.x; segs[o + 3] = b0.y;
  if (n == 2u) { segs[o + 4] = a1.x; segs[o + 5] = a1.y; segs[o + 6] = b1.x; segs[o + 7] = b1.y; }
  counts[c] = n;
}`;
}

/** GPU counterpart of core's `marchingSquaresSegments`: segments in the same cell order */
export async function gpuMarchingSquaresSegments(backend: GpuBackend, grid: DenseGrid, values: ArrayLike<number>, level: number): Promise<Float32Array> {
  const cells = (grid.size[0]! - 1) * (grid.size[1]! - 1);
  const [segBuf, cntBuf] = await backend.runKernel({
    code: marchingSquaresCode(grid),
    invocations: cells,
    buffers: [
      { role: "rw", size: cells * 8 * 4, readback: true },
      { role: "rw", size: cells * 4, readback: true },
      { role: "r", data: values instanceof Float32Array ? values : Float32Array.from(values as ArrayLike<number>) },
      { role: "r", data: Float32Array.of(level) },
    ],
  });
  const segs = new Float32Array(segBuf!), counts = new Uint32Array(cntBuf!);
  let total = 0;
  for (let c = 0; c < cells; c++) total += counts[c]!;
  const out = new Float32Array(total * 4);
  let o = 0;
  for (let c = 0; c < cells; c++) for (let k = 0; k < counts[c]!; k++) { out.set(segs.subarray(c * 8 + k * 4, c * 8 + k * 4 + 4), o); o += 4; }
  return out;
}

/*******************************************************/
/* projection onto the level set */

export interface GpuProjector {
  /** project points ([x, y, maxDist] per vertex) onto {f = level}; returns [x, y, ok] per vertex */
  project(vertices: Float32Array, level: number): Promise<Float32Array>;
}

/**
 * Builds the projection kernel for a symbolic field: `f` and its two partials
 * are transpiled once, and every dispatch projects a batch of vertices with
 * the same damped Newton + bisection fallback as core's `projectToLevel`
 * (f32 tolerances).
 */
export function gpuProjector(backend: GpuBackend, field: ScalarFieldData, grid: DenseGrid): GpuProjector {
  const b = new ProgramBuilder(grid);
  const fn = b.scalar(field), dx = b.scalar(field, [0]), dy = b.scalar(field, [1]);
  const lib = b.library();
  const { a, b: bb } = field.box;
  const eps = 1e-6 * Math.max(field.box.size[0]!, field.box.size[1]!);
  const code = `${lib.code}
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<storage, read> verts: array<f32>;
@group(0) @binding(3) var<storage, read> lvl: array<f32>;
const A: vec2<f32> = vec2<f32>(${f32(a[0]!)}, ${f32(a[1]!)});
const B: vec2<f32> = vec2<f32>(${f32(bb[0]!)}, ${f32(bb[1]!)});
const EPS: f32 = ${f32(eps)};
fn inBox(q: vec2<f32>) -> bool { return q.x >= A.x - EPS && q.x <= B.x + EPS && q.y >= A.y - EPS && q.y <= B.y + EPS; }
fn resid(q: vec2<f32>, level: f32) -> f32 { return ${fn}(q, -1) - level; }
fn grad(q: vec2<f32>, lock: vec2<f32>) -> vec2<f32> { return vec2<f32>(${dx}(q, -1), ${dy}(q, -1)) * lock; }
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = i32(id.x);
  if (i >= i32(arrayLength(&verts)) / 3) { return; }
  let p = vec2<f32>(verts[i * 3], verts[i * 3 + 1]);
  let maxDist = verts[i * 3 + 2];
  let level = lvl[0];
  let scale = max(1.0, abs(level));
  let tolF = 1e-6 * scale;
  // coordinates on box faces stay locked there
  let lock = vec2<f32>(select(1.0, 0.0, abs(p.x - A.x) < EPS || abs(p.x - B.x) < EPS), select(1.0, 0.0, abs(p.y - A.y) < EPS || abs(p.y - B.y) < EPS));
  var q = p;
  var r = resid(q, level);
  var ok = 0.0;
  let g0 = grad(q, lock);
  let g02 = dot(g0, g0);
  if (!isfinite_(r) || !(g02 > 1e-24)) { out[i * 3] = p.x; out[i * 3 + 1] = p.y; out[i * 3 + 2] = 0.0; return; }
  let r0 = r;
  // 1. damped Newton
  for (var it = 0; it < 12; it++) {
    if (abs(r) < tolF) { break; }
    let g = grad(q, lock);
    let g2 = dot(g, g);
    if (!(g2 > 1e-24)) { break; }
    var k = r / g2;
    var accepted = false;
    for (var damp = 0; damp < 5; damp++) {
      let trial = q - k * g;
      let d = trial - p;
      if (inBox(trial) && dot(d, d) <= maxDist * maxDist) {
        let rt = resid(trial, level);
        if (isfinite_(rt) && abs(rt) < abs(r)) { q = trial; r = rt; accepted = true; break; }
      }
      k = k * 0.5;
    }
    if (!accepted) { break; }
  }
  if (abs(r) < tolF || abs(r) < 1e-5 * scale) { ok = 1.0; }
  else {
    // 2. bracket a sign change along the initial gradient line, then bisect
    let dir = -sign(r0) * g0 / sqrt(g02);
    var lo = 0.0; var hi = maxDist / 64.0; var found = false;
    for (var n = 0; n < 12; n++) {
      let t = p + hi * dir;
      if (!inBox(t)) { break; }
      let rh = resid(t, level);
      if (!isfinite_(rh)) { break; }
      if (sign(rh) != sign(r0)) { found = true; break; }
      lo = hi; hi = hi * 2.0;
      if (hi > maxDist) { break; }
    }
    if (found) {
      for (var it = 0; it < 40; it++) {
        let mid = 0.5 * (lo + hi);
        let rm = resid(p + mid * dir, level);
        if (abs(rm) < tolF) { lo = mid; hi = mid; break; }
        if (sign(rm) == sign(r0)) { lo = mid; } else { hi = mid; }
      }
      q = p + 0.5 * (lo + hi) * dir; ok = 1.0;
    }
  }
  // snap locked coordinates exactly onto the boundary
  if (lock.x == 0.0) { q.x = select(B.x, A.x, abs(q.x - A.x) < abs(q.x - B.x)); }
  if (lock.y == 0.0) { q.y = select(B.y, A.y, abs(q.y - A.y) < abs(q.y - B.y)); }
  out[i * 3] = q.x; out[i * 3 + 1] = q.y; out[i * 3 + 2] = ok;
}`;
  return {
    async project(vertices, level) {
      const n = vertices.length / 3;
      if (n === 0) return new Float32Array(0);
      const [out] = await backend.runKernel({
        code,
        invocations: n,
        buffers: [
          { role: "rw", size: n * 3 * 4, readback: true },
          { role: "r", data: lib.data },
          { role: "r", data: vertices },
          { role: "r", data: Float32Array.of(level) },
        ],
      });
      return new Float32Array(out!, 0, n * 3);
    },
  };
}

/*******************************************************/
/* exact isolines: seed on the GPU, project on the GPU, refine in rounds */

function chordDistance(ax: number, ay: number, bx: number, by: number, mx: number, my: number): number {
  const dx = bx - ax, dy = by - ay, l2 = dx * dx + dy * dy;
  if (l2 === 0) return Math.hypot(mx - ax, my - ay);
  const t = Math.min(1, Math.max(0, ((mx - ax) * dx + (my - ay) * dy) / l2));
  return Math.hypot(mx - (ax + t * dx), my - (ay + t * dy));
}

export async function gpuExactIsoContours(
  backend: GpuBackend,
  projector: GpuProjector,
  field: ScalarFieldData,
  grid: DenseGrid,
  values: ArrayLike<number>,
  level: number,
  opts: { tolerance: number; maxRounds?: number },
): Promise<ContourResult> {
  const tol = opts.tolerance, maxRounds = opts.maxRounds ?? 12;
  const cell = Math.max(grid.spacing[0]!, grid.spacing[1]!);
  const segs = await gpuMarchingSquaresSegments(backend, grid, values, level);
  // f32 endpoints of a shared vertex computed in two cells differ by ~1e-7 relative: join with an f32-scale tolerance
  const seeds = joinSegments(Float64Array.from(segs), Math.min(grid.spacing[0]!, grid.spacing[1]!) * 1e-4 || 1e-7);

  // 1. project every seed vertex (closed loops: the repeated last vertex is dropped, re-added after)
  const closed = seeds.map((s) => s.length / 2 > 2 && s[0] === s[s.length - 2] && s[1] === s[s.length - 1]);
  const batch: number[] = [];
  seeds.forEach((s, li) => { const n = s.length / 2 - (closed[li] ? 1 : 0); for (let i = 0; i < n; i++) batch.push(s[2 * i]!, s[2 * i + 1]!, 1.5 * cell); });
  const proj = await projector.project(Float32Array.from(batch), level);
  // polylines as arrays of [x, y]
  const lines: number[][][] = [];
  let o = 0;
  seeds.forEach((s, li) => {
    const n = s.length / 2 - (closed[li] ? 1 : 0);
    const v: number[][] = [];
    for (let i = 0; i < n; i++, o++) v.push(proj[o * 3 + 2]! > 0 ? [proj[o * 3]!, proj[o * 3 + 1]!] : [s[2 * i]!, s[2 * i + 1]!]);
    if (closed[li]) v.push(v[0]!);
    lines.push(v);
  });

  // 2. refinement rounds: project the midpoints of all chords still marked, split where they deviate
  let marks = lines.map((v) => new Array<boolean>(Math.max(0, v.length - 1)).fill(true));
  for (let round = 0; round < maxRounds; round++) {
    const req: number[] = [], where: [number, number][] = [];
    lines.forEach((v, li) => {
      for (let i = 0; i + 1 < v.length; i++) {
        if (!marks[li]![i]) continue;
        const [ax, ay] = v[i] as [number, number], [bx, by] = v[i + 1] as [number, number];
        const len = Math.hypot(bx - ax, by - ay);
        if (len <= 2 * tol) continue;
        req.push(0.5 * (ax + bx), 0.5 * (ay + by), len); where.push([li, i]);
      }
    });
    if (!req.length) break;
    const res = await projector.project(Float32Array.from(req), level);
    // insert from the back so indices stay valid
    const inserts = new Map<number, [number, number, number][]>(); // line -> [(index, x, y)]
    where.forEach(([li, i], k) => {
      if (!(res[k * 3 + 2]! > 0)) return;
      const mx = res[k * 3]!, my = res[k * 3 + 1]!;
      const [ax, ay] = lines[li]![i] as [number, number], [bx, by] = lines[li]![i + 1] as [number, number];
      if (chordDistance(ax, ay, bx, by, mx, my) > tol) { let l = inserts.get(li); if (!l) inserts.set(li, (l = [])); l.push([i, mx, my]); }
    });
    if (!inserts.size) break;
    const nextMarks = lines.map((v) => new Array<boolean>(Math.max(0, v.length - 1)).fill(false));
    for (const [li, list] of inserts) {
      const v = lines[li]!;
      const m = nextMarks[li]!;
      // rebuild the line with insertions; new chords marked for the next round
      const out: number[][] = [], outMarks: boolean[] = [];
      const byIndex = new Map(list.map(([i, x, y]) => [i, [x, y]] as const));
      for (let i = 0; i < v.length; i++) {
        out.push(v[i]!);
        if (i + 1 < v.length) {
          const ins = byIndex.get(i);
          if (ins) { out.push([ins[0], ins[1]]); outMarks.push(true, true); } else outMarks.push(false);
        }
      }
      lines[li] = out; nextMarks[li] = outMarks; void m;
    }
    marks = nextMarks;
  }

  const polylines: Polyline[] = lines.map((v) => Float64Array.from(v.flat()));
  let vertexCount = 0, maxResidual = 0;
  for (const l of polylines) {
    vertexCount += l.length / 2;
    for (let i = 0; i < l.length; i += 2) { const r = Math.abs(field.fn([l[i]!, l[i + 1]!], -1) - level); if (r > maxResidual) maxResidual = r; }
  }
  return { lines: polylines, method: "exact", vertexCount, maxResidual };
}

// Grid passes on resident data: statistics (reduction), separable box blur,
// and Taubin-smoothed marching-squares isolines on the edge graph.

import { DenseGrid, type ScalarFieldData } from "@tensatory/core";
import { RESIDENT_USAGE, type GpuBackend } from "./device";
import { levelParams, marchingSquaresWgsl } from "./isolines";
import { ProgramBuilder } from "./program";
import type { GpuGrid } from "./resident";
import { SEG_WGSL, type GpuSegments } from "./segments";
import { GRID_FLOATS, gridWgsl, packGrid } from "./wgsl";

/** every pass here takes its grid(s) from a header in its params buffer, so a pass compiles once per field / shape and
 *  serves every grid (an adaptive resolution or a moving crop compiles nothing) */
const P_GRID = gridWgsl("pg_", "params", 4); // params: [p0, p1, p2, p3, grid header]
const P_GRID2 = gridWgsl("qg_", "params", 4 + GRID_FLOATS); // a second grid after the first
function gridParams(p: number[], grid: DenseGrid, grid2?: DenseGrid): Float32Array {
  const f = new Float32Array(4 + GRID_FLOATS * (grid2 ? 2 : 1));
  p.forEach((v, i) => { f[i] = v; });
  packGrid(grid, f, 4);
  if (grid2) packGrid(grid2, f, 4 + GRID_FLOATS);
  return f;
}

const ISNAN = `fn isnan_(v: f32) -> bool { let b = bitcast<u32>(v); return (b & 0x7f800000u) == 0x7f800000u && (b & 0x007fffffu) != 0u; }
fn isfinite_(v: f32) -> bool { return (bitcast<u32>(v) & 0x7f800000u) != 0x7f800000u; }`;

/*******************************************************/
/* statistics */

export interface GpuStats { min: number; max: number; posMin: number; mean: number; finite: number }

const STATS_WG = 256;
function statsCode(channels: number, channel: number): string {
  return `
${ISNAN}
@group(0) @binding(0) var<storage, read_write> partials: array<f32>;
@group(0) @binding(1) var<storage, read> vals: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<f32>;
var<workgroup> wmin: array<f32, 64>; var<workgroup> wmax: array<f32, 64>; var<workgroup> wpos: array<f32, 64>; var<workgroup> wsum: array<f32, 64>; var<workgroup> wcnt: array<f32, 64>;
@compute @workgroup_size(64) fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
  var mn = 3.4e38; var mx = -3.4e38; var pm = 3.4e38; var sum = 0.0; var cnt = 0.0;
  let stride = ${STATS_WG}u * 64u;
  let n = bitcast<u32>(params[0]);
  for (var i = wid.x * 64u + lid.x; i < n; i += stride) {
    let v = vals[i * ${channels}u + ${channel}u];
    if (isfinite_(v)) { mn = min(mn, v); mx = max(mx, v); if (v > 0.0) { pm = min(pm, v); } sum += v; cnt += 1.0; }
  }
  wmin[lid.x] = mn; wmax[lid.x] = mx; wpos[lid.x] = pm; wsum[lid.x] = sum; wcnt[lid.x] = cnt;
  workgroupBarrier();
  for (var s = 32u; s > 0u; s = s >> 1u) {
    if (lid.x < s) {
      wmin[lid.x] = min(wmin[lid.x], wmin[lid.x + s]); wmax[lid.x] = max(wmax[lid.x], wmax[lid.x + s]);
      wpos[lid.x] = min(wpos[lid.x], wpos[lid.x + s]); wsum[lid.x] += wsum[lid.x + s]; wcnt[lid.x] += wcnt[lid.x + s];
    }
    workgroupBarrier();
  }
  if (lid.x == 0u) { let o = wid.x * 5u; partials[o] = wmin[0]; partials[o + 1] = wmax[0]; partials[o + 2] = wpos[0]; partials[o + 3] = wsum[0]; partials[o + 4] = wcnt[0]; }
}`;
}

/** min / max / positive min / mean over the finite values of one channel of a resident grid */
export async function gpuStats(backend: GpuBackend, g: GpuGrid, channel = 0): Promise<GpuStats> {
  const n = g.grid.sampleCount;
  const { read: [buf] } = await backend.runKernel({
    code: statsCode(g.channels, channel),
    invocations: STATS_WG * 64,
    buffers: [{ role: "rw", size: STATS_WG * 5 * 4, readback: true }, { role: "r", buffer: g.buffer }, { role: "r", data: (() => { const f = new Float32Array(4); new Uint32Array(f.buffer)[0] = n; return f; })() }],
  });
  const p = new Float32Array(buf!);
  let min = Infinity, max = -Infinity, posMin = Infinity, sum = 0, cnt = 0;
  for (let w = 0; w < STATS_WG; w++) {
    if (p[w * 5 + 4]! > 0) { min = Math.min(min, p[w * 5]!); max = Math.max(max, p[w * 5 + 1]!); posMin = Math.min(posMin, p[w * 5 + 2]!); sum += p[w * 5 + 3]!; cnt += p[w * 5 + 4]!; }
  }
  return { min: cnt ? min : NaN, max: cnt ? max : NaN, posMin: posMin < 3e38 ? posMin : NaN, mean: cnt ? sum / cnt : NaN, finite: cnt };
}

/*******************************************************/
/* separable box blur */

function blurAxisCode(axis: number, radius: number): string {
  return `
@group(0) @binding(0) var<storage, read_write> dst: array<f32>;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<f32>;
${P_GRID.code}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let v = i32(id.x);
  if (v >= ${P_GRID.ref.count}) { return; }
  let D = ${P_GRID.ref.n(String(axis))}; let S = ${P_GRID.ref.s(String(axis))};
  let c = (v / S) % D;
  var sum = 0.0; var cnt = 0.0;
  for (var k = -${radius}; k <= ${radius}; k++) {
    let cc = c + k;
    if (cc < 0 || cc >= D) { continue; }
    sum += src[v + k * S]; cnt += 1.0;
  }
  dst[v] = sum / cnt;
}`;
}

/** core's `boxBlur` on a resident scalar grid: a new resident grid (enqueued, no readback) */
export function blurResidentSync(backend: GpuBackend, src: GpuGrid, radius: number): GpuGrid {
  const grid = src.grid, n = grid.sampleCount;
  const r = Math.floor(radius);
  if (r <= 0 || src.channels !== 1) return src;
  let cur = src.buffer;
  const temps: GPUBuffer[] = [];
  for (let axis = 0; axis < grid.dimCount; axis++) {
    if (grid.size[axis]! < 2) continue;
    const [out] = backend.dispatch({
      code: blurAxisCode(axis, r),
      invocations: n,
      buffers: [{ role: "rw", size: n * 4, keep: true }, { role: "r", buffer: cur }, { role: "r", data: gridParams([], grid) }],
    });
    if (cur !== src.buffer) temps.push(cur);
    cur = out!;
  }
  for (const t of temps) t.destroy(); // accounted buffers: destroyed after any pending compile's deferred dispatch
  if (cur === src.buffer) return src;
  const buffer = cur;
  return { grid, channels: 1, buffer, destroy: () => buffer.destroy() };
}

/*******************************************************/
/* plane samples of a 3D field / resident grid, built once, dispatched per depth */

export interface PlanePass {
  /** fill `into` (grid2.sampleCount f32) with the values on the plane `axis = depth`; enqueued */
  dispatch(depth: number, into: GPUBuffer): void;
  destroy(): void;
}

/** evaluate a 3D field exactly on the plane `axis = depth` at the points of `grid2`: one program per (field, axis, grid2) */
export function planeSampler(backend: GpuBackend, field: ScalarFieldData, axis: number, grid2: DenseGrid): PlanePass {
  const keep = [0, 1, 2].filter((d) => d !== axis) as [number, number];
  const b = new ProgramBuilder(new DenseGrid([2, 2, 2], field.box));
  const fn = b.scalar(field);
  const lib = b.library();
  const n = grid2.sampleCount;
  const c = ["", "", ""]; c[axis] = "params[0]"; c[keep[0]] = "w0"; c[keep[1]] = "w1";
  const G = P_GRID.ref;
  const code = `${lib.code}
@group(0) @binding(0) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<f32>;
${P_GRID.code}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let v = i32(id.x);
  if (v >= ${G.count}) { return; }
  let s0 = ${G.s("0")};
  let u0 = v / s0; let u1 = v - u0 * s0;
  let w0 = ${G.a("0")} + f32(u0) * ${G.h("0")};
  let w1 = ${G.a("1")} + f32(u1) * ${G.h("1")};
  dst[v] = ${fn}(vec3<f32>(${c.join(", ")}), -1);
}`;
  return {
    dispatch(depth, into) { backend.dispatch({ code, invocations: n, buffers: [{ role: "rw", buffer: into }, { role: "r", data: lib.data }, { role: "r", data: gridParams([depth], grid2) }] }); },
    destroy() { /* nothing resident */ },
  };
}

/** trilinear samples of a resident 3D grid on the plane `axis = depth` at the points of `grid2`: one program per (grid, axis, grid2) */
export function planeSlicer(backend: GpuBackend, src: GpuGrid, axis: number, grid2: DenseGrid): PlanePass {
  const code = sliceCode(src, axis);
  const n = grid2.sampleCount;
  return {
    dispatch(depth, into) { backend.dispatch({ code, invocations: n, buffers: [{ role: "rw", buffer: into }, { role: "r", buffer: src.buffer }, { role: "r", data: gridParams([depth], src.grid, grid2) }] }); },
    destroy() { /* nothing resident */ },
  };
}

/*******************************************************/
/* plane slice of a resident 3D grid */

/**
 * Sample a resident 3D scalar grid on the plane `axis = depth` at the points of the 2D grid
 * `grid2` (trilinear, like core's interpolation of dense data): the face values of a sampled or
 * blurred volume, so the face outlines match the mesh boundary by construction.
 */
function sliceCode(src: GpuGrid, axis: number): string {
  const g = src.grid;
  if (g.dimCount !== 3 || src.channels !== 1) throw new Error("plane slice needs a resident 3D scalar grid");
  const keep = [0, 1, 2].filter((d) => d !== axis) as [number, number];
  const V = P_GRID.ref, Q = P_GRID2.ref; // the volume grid, the plane grid
  // grid position (fractional) of a world coordinate along each axis, clamped like the reader in program.ts
  const coord = (d: number, expr: string) => `select(0.0, clamp((${expr} - ${V.a(String(d))}) / ${V.h(String(d))}, 0.0, f32(${V.n(String(d))} - 1)), ${V.n(String(d))} > 1)`;
  return `
@group(0) @binding(0) var<storage, read_write> dst: array<f32>;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<f32>;
${P_GRID.code}
${P_GRID2.code}
fn at(i: i32, j: i32, k: i32) -> f32 { return src[min(i, ${V.n("0")} - 1) * ${V.s("0")} + min(j, ${V.n("1")} - 1) * ${V.s("1")} + min(k, ${V.n("2")} - 1) * ${V.s("2")}]; }
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let v = i32(id.x);
  if (v >= ${Q.count}) { return; }
  let qs0 = ${Q.s("0")};
  let u0 = v / qs0; let u1 = v - u0 * qs0;
  let w0 = ${Q.a("0")} + f32(u0) * ${Q.h("0")};
  let w1 = ${Q.a("1")} + f32(u1) * ${Q.h("1")};
  var gp: vec3<f32>;
  gp[${axis}] = ${coord(axis, "params[0]")};
  gp[${keep[0]}] = ${coord(keep[0], "w0")};
  gp[${keep[1]}] = ${coord(keep[1], "w1")};
  let i0 = vec3<i32>(floor(gp)); let f = gp - vec3<f32>(i0);
  let c00 = mix(at(i0.x, i0.y, i0.z), at(i0.x + 1, i0.y, i0.z), f.x);
  let c10 = mix(at(i0.x, i0.y + 1, i0.z), at(i0.x + 1, i0.y + 1, i0.z), f.x);
  let c01 = mix(at(i0.x, i0.y, i0.z + 1), at(i0.x + 1, i0.y, i0.z + 1), f.x);
  let c11 = mix(at(i0.x, i0.y + 1, i0.z + 1), at(i0.x + 1, i0.y + 1, i0.z + 1), f.x);
  dst[v] = mix(mix(c00, c10, f.y), mix(c01, c11, f.y), f.z);
}`;
}

/**
 * Sample a resident 3D scalar grid on the plane `axis = depth` at the points of the 2D grid
 * `grid2` (trilinear, like core's interpolation of dense data): the face values of a sampled or
 * blurred volume, so the face outlines match the mesh boundary by construction.
 */
export function sliceResidentSync(backend: GpuBackend, src: GpuGrid, axis: number, depth: number, grid2: DenseGrid): GpuGrid {
  const n = grid2.sampleCount;
  const [out] = backend.dispatch({ code: sliceCode(src, axis), invocations: n, buffers: [{ role: "rw", size: Math.max(16, n * 4), keep: true }, { role: "r", buffer: src.buffer }, { role: "r", data: gridParams([depth], src.grid, grid2) }] });
  const buffer = out!;
  return { grid: grid2, channels: 1, buffer, destroy: () => buffer.destroy() };
}

/*******************************************************/
/* Taubin-smoothed marching squares on the edge graph */

export interface SmoothedIsolines {
  /** marching squares of the resident grid at `level`, smoothed with `iterations` Taubin λ|μ rounds, appended into `segs` */
  dispatch(segs: GpuSegments, level: number, iterations: number): void;
  readonly capacity: number;
  destroy(): void;
}

/**
 * Non-exact isolines (sampled or blurred fields) with Taubin smoothing, fully on
 * the GPU. Every marching-squares vertex lies on a grid edge shared by two
 * cells, and each cell holds at most two segments, so a vertex's two polyline
 * neighbours are found from the adjacent cells' segment lists: the smoothing
 * runs on the edge graph without ever joining polylines. Vertices with fewer
 * than two neighbours (open ends at the box) stay fixed, like core.
 */
export function smoothedIsolines(backend: GpuBackend, values: GpuGrid, colour?: ScalarFieldData | "level"): SmoothedIsolines {
  const grid = values.grid;
  const [nx, ny] = grid.size as [number, number];
  const cells = (nx - 1) * (ny - 1);
  const hEdges = (nx - 1) * ny, vEdges = nx * (ny - 1), edges = hEdges + vEdges;
  const capacity = cells * 2;
  const b = new ProgramBuilder(grid);
  const col = colour && colour !== "level" ? b.scalar(colour) : undefined;
  const lib = b.library();
  // persistent scratch: edge positions (ping-pong), edge used flags, cell segments as edge ids
  const posA = backend.createBuffer({ size: Math.max(16, edges * 8), usage: RESIDENT_USAGE });
  const posB = backend.createBuffer({ size: Math.max(16, edges * 8), usage: RESIDENT_USAGE });
  const cellSegs = backend.createBuffer({ size: Math.max(16, cells * 4 * 4), usage: RESIDENT_USAGE }); // e0a, e0b, e1a, e1b (i32; -1 = none)
  // the grid comes from the params header (params[4..]) in every pass: NX / NY / edges are runtime values
  const G = P_GRID.ref;
  const DIMS = `${P_GRID.code}
fn NX_() -> i32 { return ${G.n("0")}; }
fn NY_() -> i32 { return ${G.n("1")}; }
fn HEDGES_() -> i32 { return (NX_() - 1) * NY_(); }
fn EDGES_() -> i32 { return HEDGES_() + NX_() * (NY_() - 1); }`;
  const EDGES = `
fn hEdge(i: i32, j: i32) -> i32 { return i * NY_() + j; }                   // bottom edge of cell (i, j): (i,j)-(i+1,j)
fn vEdge(i: i32, j: i32) -> i32 { return HEDGES_() + i * (NY_() - 1) + j; }    // left edge of cell (i, j): (i,j)-(i,j+1)
`;
  const seedCode = `
${ISNAN}
@group(0) @binding(0) var<storage, read_write> pos: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> cseg: array<i32>;
@group(0) @binding(2) var<storage, read> vals: array<f32>;
@group(0) @binding(3) var<storage, read> params: array<f32>;
${DIMS}
${marchingSquaresWgsl(G)}
${EDGES}
// which edge a marching-squares endpoint lies on: the cell side it is nearest to (normalized distances, so a
// last-bit difference between this expression and cellSegments' — fast-math may contract y0 + hy differently
// here and there — cannot misfile a vertex)
fn edgeOf(p: vec2<f32>, i: i32, j: i32, x0: f32, y0: f32) -> i32 {
  let hx = msH().x; let hy = msH().y;
  let dB = abs(p.y - y0) / hy; let dT = abs(p.y - (y0 + hy)) / hy; let dL = abs(p.x - x0) / hx; let dR = abs(p.x - (x0 + hx)) / hx;
  let m = min(min(dB, dT), min(dL, dR));
  if (m == dB) { return hEdge(i, j); }
  if (m == dT) { return hEdge(i, j + 1); }
  if (m == dL) { return vEdge(i, j); }
  return vEdge(i + 1, j);
}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = i32(id.x);
  if (c >= msCells()) { return; }
  let r = cellSegments(c, params[0]);
  let NY = NY_();
  let i = c / (NY - 1); let j = c % (NY - 1);
  let x0 = msA().x + f32(i) * msH().x;
  let y0 = msA().y + f32(j) * msH().y;
  var e = vec4<i32>(-1, -1, -1, -1);
  if (r.n >= 1u) { e.x = edgeOf(r.a0, i, j, x0, y0); e.y = edgeOf(r.b0, i, j, x0, y0); pos[e.x] = r.a0; pos[e.y] = r.b0; }
  if (r.n == 2u) { e.z = edgeOf(r.a1, i, j, x0, y0); e.w = edgeOf(r.b1, i, j, x0, y0); pos[e.z] = r.a1; pos[e.w] = r.b1; }
  cseg[c * 4] = e.x; cseg[c * 4 + 1] = e.y; cseg[c * 4 + 2] = e.z; cseg[c * 4 + 3] = e.w;
}`;
  const passCode = `
@group(0) @binding(0) var<storage, read_write> dst: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> src: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> cseg: array<i32>;
@group(0) @binding(3) var<storage, read> params: array<f32>;
${DIMS}
${EDGES}
// the other endpoint of the segment of cell c that uses edge e (-1 if none)
fn partner(c: i32, e: i32) -> i32 {
  if (c < 0 || c >= (NX_() - 1) * (NY_() - 1)) { return -1; }
  let o = c * 4;
  if (cseg[o] == e) { return cseg[o + 1]; } if (cseg[o + 1] == e) { return cseg[o]; }
  if (cseg[o + 2] == e) { return cseg[o + 3]; } if (cseg[o + 3] == e) { return cseg[o + 2]; }
  return -1;
}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let e = i32(id.x);
  if (e >= EDGES_()) { return; }
  let NX = NX_(); let NY = NY_(); let HEDGES = HEDGES_();
  var c0: i32; var c1: i32;
  if (e < HEDGES) { let i = e / NY; let j = e % NY; c0 = select(-1, i * (NY - 1) + (j - 1), j > 0); c1 = select(-1, i * (NY - 1) + j, j < NY - 1); }
  else { let k = e - HEDGES; let i = k / (NY - 1); let j = k % (NY - 1); c0 = select(-1, (i - 1) * (NY - 1) + j, i > 0); c1 = select(-1, i * (NY - 1) + j, i < NX - 1); }
  let n0 = partner(c0, e); let n1 = partner(c1, e);
  let p = src[e];
  if (n0 < 0 || n1 < 0) { dst[e] = p; return; } // unused edge or open end: fixed
  let lap = 0.5 * (src[n0] + src[n1]) - p;
  dst[e] = p + params[0] * lap;
}`;
  const emitCode = `${lib.code}
${SEG_WGSL}
@group(0) @binding(0) var<storage, read_write> segs: array<Seg>;
@group(0) @binding(2) var<storage, read> pos: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> ind: Indirect;
@group(0) @binding(4) var<storage, read> cseg: array<i32>;
@group(0) @binding(5) var<storage, read> params: array<f32>;
${DIMS}
// the set's real capacity (bits) in params[0]; the atomic counts every segment, so the indirect counter is the true size
fn appendSeg(s: Seg) {
  let i = atomicAdd(&ind.instanceCount, 1u);
  if (i < bitcast<u32>(params[0])) { segs[i] = s; }
}
fn colour_(p: vec2<f32>) -> f32 { return ${colour === "level" ? "params[1]" : col ? `${col}(p, -1)` : "0.0"}; }
fn emit(ea: i32, eb: i32) {
  let a = pos[ea]; let b = pos[eb];
  var s: Seg; s.a = a; s.b = b; s.ca = colour_(a); s.cb = colour_(b); s.arc = 0.0; s.len = 0.0; s.phase = 0.0; s.pad = 0.0;
  appendSeg(s);
}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = i32(id.x);
  if (c >= (NX_() - 1) * (NY_() - 1)) { return; }
  let o = c * 4;
  if (cseg[o] >= 0) { emit(cseg[o], cseg[o + 1]); }
  if (cseg[o + 2] >= 0) { emit(cseg[o + 2], cseg[o + 3]); }
}`;
  return {
    capacity,
    dispatch(segs, level, iterations) {
      backend.dispatch({ code: seedCode, invocations: cells, buffers: [{ role: "rw", buffer: posA }, { role: "rw", buffer: cellSegs }, { role: "r", buffer: values.buffer }, { role: "r", data: levelParams(level, grid) }] });
      let cur = posA, other = posB;
      for (let it = 0; it < iterations; it++) {
        for (const k of [0.5, -0.53]) {
          backend.dispatch({ code: passCode, invocations: edges, buffers: [{ role: "rw", buffer: other }, { role: "r", buffer: cur }, { role: "r", buffer: cellSegs }, { role: "r", data: gridParams([k], grid) }] });
          [cur, other] = [other, cur];
        }
      }
      const cap = gridParams([0, level], grid); new Uint32Array(cap.buffer)[0] = Math.min(0xffffffff, segs.capacity); // params[1] = level (colour "level")
      backend.dispatch({ code: emitCode, invocations: cells, buffers: [{ role: "rw", buffer: segs.buffer }, { role: "r", data: lib.data }, { role: "r", buffer: cur }, { role: "rw", buffer: segs.indirect }, { role: "r", buffer: cellSegs }, { role: "r", data: cap }] });
    },
    destroy() { posA.destroy(); posB.destroy(); cellSegs.destroy(); },
  };
}

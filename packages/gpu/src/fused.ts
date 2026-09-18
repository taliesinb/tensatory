// Fused geometry kernels: everything stays on the device. They read a resident
// grid (GpuGrid) and append Seg records to a resident segment set through an
// atomic counter, ready for drawIndirect. No readback, no CPU join.
//
//  * fusedIsolines: per cell — marching squares, Newton projection of the two
//    endpoints (symbolic fields), in-thread adaptive midpoint refinement into
//    up to MAXP points, colour-field evaluation per vertex, append.
//  * fusedStreamlines: per seed — RK4 both ways (within the seed's step
//    budgets) into scratch slots, then append segments with arc / length /
//    phase and colour.

import { DenseGrid, type ScalarFieldData, type StreamlineSeeds } from "@tensatory/core";
import type { GpuBackend } from "./device";
import { colourCode, type ColourSource } from "./colour";
import { levelParams, marchingSquaresWgsl, projectionWgsl } from "./isolines";
import { ProgramBuilder } from "./program";
import type { GpuGrid } from "./resident";
import { SEG_APPEND_WGSL, SEG_WGSL, type GpuSegments } from "./segments";
import { SEED_FLOATS, integrateWgsl, packSeeds } from "./flow";
import { f32, gridWgsl } from "./wgsl";

/** the isoline kernels' grid header: params[4..] */
const ISO_GRID = gridWgsl("pg_", "params", 4);

export const ISO_MAXP = 17; // points per refined seed segment (16 pieces)

export interface FusedIsolineOptions {
  /** project the vertices onto the level set (default: the field is symbolic) */
  exact?: boolean;
  /** the values are a 3D field sampled on the plane `axis = depth` (depth passed per dispatch): exact projection uses the 3D field restricted to it */
  slice?: { field: ScalarFieldData; axis: number };
}
const sliceBox = (s: { field: ScalarFieldData; axis: number }) => {
  const keep = [0, 1, 2].filter((d) => d !== s.axis);
  const a = keep.map((d) => s.field.box.a[d]!), b = keep.map((d) => s.field.box.b[d]!);
  return { a, b, size: a.map((x, i) => b[i]! - x) };
};

export interface FusedIsolines {
  /** append the isolines at `level` (world tolerance `tol`) into `segs`; the caller resets the set first when reusing it. `depth`: the plane of a slice kernel */
  run(segs: GpuSegments, level: number, tol: number, depth?: number): Promise<void>;
  /** same, enqueued without waiting (queue-ordered before later passes) */
  dispatch(segs: GpuSegments, level: number, tol: number, depth?: number): void;
  /** worst-case segment count for one level */
  readonly capacity: number;
  destroy(): void;
}

/**
 * Build the fused isoline kernel for `field` sampled as `values` (resident).
 * `colour` (optional) is evaluated at every vertex into ca / cb; symbolic
 * fields are projected exactly, sampled fields keep the marching-squares
 * segments.
 */
export function fusedIsolines(backend: GpuBackend, field: ScalarFieldData | undefined, values: GpuGrid, colour?: ColourSource, opts: FusedIsolineOptions = {}): FusedIsolines {
  const grid = values.grid;
  const slice = opts.slice;
  const exact = opts.exact ?? (slice ? slice.field.kind === "symbolic" : field?.kind === "symbolic");
  if (exact && !field && !slice) throw new Error("fusedIsolines: exact projection needs the field");
  // a slice of a 3D field: the programs are 3D, wrapped as 2D functions on the plane
  const b = new ProgramBuilder(slice ? new DenseGrid([2, 2, 2], slice.field.box) : grid);
  let fn = "", vg = "";
  const extra: string[] = [];
  if (exact && slice) {
    const keep = [0, 1, 2].filter((d) => d !== slice.axis) as [number, number];
    const f3 = b.scalar(slice.field), vg3 = b.valueGradient(slice.field);
    const lift = (q: string) => { const c = ["", "", ""]; c[slice.axis] = "params[2]"; c[keep[0]] = `${q}.x`; c[keep[1]] = `${q}.y`; return `vec3<f32>(${c.join(", ")})`; };
    fn = "sl_f"; vg = "sl_vg";
    extra.push(`fn sl_f(q: vec2<f32>, pos: i32) -> f32 { return ${f3}(${lift("q")}, -1); }`);
    extra.push(`fn sl_vg(q: vec2<f32>, pos: i32) -> vec3<f32> { let v = ${vg3}(${lift("q")}, -1); return vec3<f32>(v.x, v[${keep[0] + 1}], v[${keep[1] + 1}]); }`);
  } else if (exact) { fn = b.scalar(field!); vg = b.valueGradient(field!); }
  const cc = colourCode(b, colour, 2, "params[0]", 5); // see colour.ts: field / resident grid / level
  const lib = b.library();
  lib.code += `\n${extra.join("\n")}`;
  const cells = (grid.size[0]! - 1) * (grid.size[1]! - 1);
  const capacity = cells * 2 * (ISO_MAXP - 1);
  const code = `${lib.code}
${SEG_WGSL}
@group(0) @binding(0) var<storage, read_write> segs: array<Seg>;
@group(0) @binding(2) var<storage, read> vals: array<f32>;
@group(0) @binding(3) var<storage, read_write> ind: Indirect;
@group(0) @binding(4) var<storage, read> params: array<f32>;
const MAXP: i32 = ${ISO_MAXP};
// params: [level, tol, depth, capacity (bits), grid header]. The set's real capacity comes with the params (sets
// are sized from measured counts, smaller than the worst case); the atomic counts every segment regardless, so
// the indirect counter is the true size even when the set is full. The grid header keeps the code the same for
// every grid of a field: no compiles when the resolution or the crop changes.
fn appendSeg(s: Seg) {
  let i = atomicAdd(&ind.instanceCount, 1u);
  if (i < bitcast<u32>(params[3])) { segs[i] = s; }
}
${ISO_GRID.code}
${marchingSquaresWgsl(ISO_GRID.ref)}
${exact ? projectionWgsl(fn, vg, slice ? sliceBox(slice) : field!.box) : ""}
${cc.code}
fn chordDist_(a: vec2<f32>, b: vec2<f32>, m: vec2<f32>) -> f32 {
  let d = b - a; let l2 = dot(d, d);
  if (l2 == 0.0) { return distance(m, a); }
  let t = clamp(dot(m - a, d) / l2, 0.0, 1.0);
  return distance(m, a + t * d);
}
// refine one seed segment a-b into a polyline of up to MAXP points and append it
fn emit(a0: vec2<f32>, b0: vec2<f32>, level: f32, tol: f32) {
  var pts: array<vec2<f32>, ${ISO_MAXP}>;
  var n: i32 = 2;
  pts[0] = a0; pts[1] = b0;
${exact ? `
  let cellMax = 1.5 * max(msH().x, msH().y);
  let tolW = 0.25 * tol; // vertices within a quarter of the chord tolerance (a sixteenth of a pixel) of the curve
  let pa = project_(a0, cellMax, level, tolW); if (pa.z > 0.0) { pts[0] = pa.xy; }
  let pb = project_(b0, cellMax, level, tolW); if (pb.z > 0.0) { pts[1] = pb.xy; }
  // adaptive midpoint refinement, in passes (each pass splits the chords still too coarse)
  for (var round = 0; round < 4; round++) {
    var inserted = false;
    var i: i32 = 0;
    loop {
      if (i + 1 >= n || n >= MAXP) { break; }
      let A = pts[i]; let B = pts[i + 1];
      let len = distance(A, B);
      if (len > 2.0 * tol) {
        let m = project_(0.5 * (A + B), len, level, tolW);
        if (m.z > 0.0 && chordDist_(A, B, m.xy) > tol) {
          for (var k = n; k > i + 1; k--) { pts[k] = pts[k - 1]; }
          pts[i + 1] = m.xy; n++; inserted = true;
          i += 2; continue;
        }
      }
      i++;
    }
    if (!inserted) { break; }
  }` : ""}
  var cPrev = colour_(pts[0]);
  for (var j = 0; j + 1 < n; j++) {
    let cNext = colour_(pts[j + 1]);
    var s: Seg; s.a = pts[j]; s.b = pts[j + 1]; s.ca = cPrev; s.cb = cNext; s.arc = 0.0; s.len = 0.0; s.phase = 0.0; s.pad = 0.0;
    appendSeg(s);
    cPrev = cNext;
  }
}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = i32(id.x);
  if (c >= msCells()) { return; }
  let level = params[0]; let tol = params[1];
  let r = cellSegments(c, level);
  if (r.n == 0u) { return; }
  emit(r.a0, r.b0, level, tol);
  if (r.n == 2u) { emit(r.a1, r.b1, level, tol); }
}`;
  const kernel = (segs: GpuSegments, level: number, tol: number, depth = 0) => ({
    code,
    invocations: cells,
    buffers: [
      { role: "rw" as const, buffer: segs.buffer },
      { role: "r" as const, data: lib.data },
      { role: "r" as const, buffer: values.buffer },
      { role: "rw" as const, buffer: segs.indirect },
      { role: "r" as const, data: (() => { const f = levelParams(level, grid, [tol, depth]); new Uint32Array(f.buffer)[3] = Math.min(0xffffffff, segs.capacity); return f; })() },
      ...cc.buffers,
    ],
  });
  return {
    capacity,
    async run(segs, level, tol, depth) { await backend.runKernel(kernel(segs, level, tol, depth)); },
    dispatch(segs, level, tol, depth) { backend.dispatch(kernel(segs, level, tol, depth)); },
    destroy() { /* nothing resident of its own */ },
  };
}

export interface FusedStreamlines {
  /** integrate from the seeds and append the segments into `segs` */
  run(segs: GpuSegments): Promise<void>;
  dispatch(segs: GpuSegments): void;
  /** worst-case segment count: Σ over seeds of the step budgets both ways (2·maxSteps without budgets) */
  readonly capacity: number;
  destroy(): void;
}

export interface FusedStreamlineOptions {
  maxSteps: number;
  step: number;
  sign?: 1 | -1;
  /** integration box (defaults to the vector grid's box) */
  box?: { a: readonly number[]; b: readonly number[]; size: number[] };
  /** also integrate against the direction from each seed (default true); false = lines start at their seeds */
  bidirectional?: boolean;
}

/**
 * Build the fused streamline kernel over a resident vector grid (bilinear
 * interpolation), with optional colour field evaluated at every vertex.
 * Seeds may carry per-seed step budgets (evenly-spaced planning): the kernel
 * re-integrates exactly that many steps each way, and its scratch and segment
 * capacity are sized from the budgets.
 */
export function fusedStreamlines(backend: GpuBackend, vectors: GpuGrid, seeds: StreamlineSeeds, opts: FusedStreamlineOptions, colour?: ColourSource): FusedStreamlines {
  const grid = vectors.grid;
  const box = opts.box ?? grid.box;
  const sgn = opts.sign ?? 1, h = opts.step;
  const packed = packSeeds(seeds, opts.maxSteps, opts.bidirectional);
  const lines = packed.lines;
  // the vector grid is read through a DenseVectorFieldData reader emitted by the builder; bind its data to the resident buffer
  // (the builder would upload a copy: instead we emit the reader against binding 2 by giving it a placeholder and rebinding)
  const b = new ProgramBuilder(new DenseGrid([2, 2], grid.box));
  const cc = colourCode(b, colour, 2, "0.0", 6);
  const lib = b.library();
  const [nx, ny] = grid.size as [number, number];
  const sx = grid.strides[0]!, sy = grid.strides[1]!;
  const eps = 1e-6 * Math.max(box.size[0]!, box.size[1]!);
  const capacity = Math.max(1, packed.segments);
  const code = `${lib.code}
${SEG_WGSL}
@group(0) @binding(0) var<storage, read_write> segs: array<Seg>;
@group(0) @binding(2) var<storage, read> vec: array<f32>;
@group(0) @binding(3) var<storage, read_write> ind: Indirect;
@group(0) @binding(4) var<storage, read> seeds: array<f32>;
@group(0) @binding(5) var<storage, read_write> scratch: array<f32>;
const CAP: u32 = ${capacity}u;
${SEG_APPEND_WGSL}
const A: vec2<f32> = vec2<f32>(${f32(box.a[0]!)}, ${f32(box.a[1]!)});
const B: vec2<f32> = vec2<f32>(${f32(box.b[0]!)}, ${f32(box.b[1]!)});
const EPS: f32 = ${f32(eps)};
const H: f32 = ${f32(h)}; const SGN: f32 = ${f32(sgn)};
fn inBox(q: vec2<f32>) -> bool { return q.x >= A.x - EPS && q.x <= B.x + EPS && q.y >= A.y - EPS && q.y <= B.y + EPS; }
// bilinear read of the resident vector grid
fn field(p: vec2<f32>) -> vec2<f32> {
  let g0 = clamp((p.x - ${f32(grid.box.a[0]!)}) / ${f32(grid.spacing[0]!)}, 0.0, ${f32(nx - 1)});
  let g1 = clamp((p.y - ${f32(grid.box.a[1]!)}) / ${f32(grid.spacing[1]!)}, 0.0, ${f32(ny - 1)});
  var i0 = i32(floor(g0)); if (i0 >= ${nx - 1}) { i0 = ${Math.max(0, nx - 2)}; }
  var i1 = i32(floor(g1)); if (i1 >= ${ny - 1}) { i1 = ${Math.max(0, ny - 2)}; }
  let f0 = g0 - f32(i0); let f1 = g1 - f32(i1);
  let j0 = min(i0 + 1, ${nx - 1}); let j1 = min(i1 + 1, ${ny - 1});
  let o00 = (i0 * ${sx} + i1 * ${sy}) * 2; let o10 = (j0 * ${sx} + i1 * ${sy}) * 2; let o01 = (i0 * ${sx} + j1 * ${sy}) * 2; let o11 = (j0 * ${sx} + j1 * ${sy}) * 2;
  let v00 = vec2<f32>(vec[o00], vec[o00 + 1]); let v10 = vec2<f32>(vec[o10], vec[o10 + 1]);
  let v01 = vec2<f32>(vec[o01], vec[o01 + 1]); let v11 = vec2<f32>(vec[o11], vec[o11 + 1]);
  return (1.0 - f0) * (1.0 - f1) * v00 + f0 * (1.0 - f1) * v10 + (1.0 - f0) * f1 * v01 + f0 * f1 * v11;
}
fn dir(q: vec2<f32>) -> vec3<f32> {
  if (!inBox(q)) { return vec3<f32>(0.0, 0.0, 0.0); }
  let v = field(q);
  let l = dot(v, v);
  if (!isfinite_(l) || !(l > 1e-24)) { return vec3<f32>(0.0, 0.0, 0.0); }
  return vec3<f32>(v * (SGN / sqrt(l)), 1.0);
}
${integrateWgsl("scratch")}
${cc.code}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = i32(id.x);
  if (i >= ${lines}) { return; }
  let s = i * ${SEED_FLOATS};
  let seed = vec2<f32>(seeds[s], seeds[s + 1]);
  let phase = seeds[s + 2];
  let base = i32(seeds[s + 5]);
  scratch[base * 2] = seed.x; scratch[base * 2 + 1] = seed.y;
  let nb = integrate(seed, -1.0, base, -1, i32(seeds[s + 3]));
  let nf = integrate(seed, 1.0, base, 1, i32(seeds[s + 4]));
  let n = nb + 1 + nf;
  if (n < 2) { return; }
  let len = f32(n - 1) * H;
  let start = base - nb;
  var prev = vec2<f32>(scratch[start * 2], scratch[start * 2 + 1]);
  var cPrev = colour_(prev);
  for (var k = 1; k < n; k++) {
    let o = (start + k) * 2;
    let cur = vec2<f32>(scratch[o], scratch[o + 1]);
    let cCur = colour_(cur);
    var s: Seg; s.a = prev; s.b = cur; s.ca = cPrev; s.cb = cCur; s.arc = f32(k - 1) * H; s.len = len; s.phase = phase; s.pad = 0.0;
    appendSeg(s);
    prev = cur; cPrev = cCur;
  }
}`;
  const kernel = (segs: GpuSegments) => ({
    code,
    invocations: Math.max(1, lines),
    buffers: [
      { role: "rw" as const, buffer: segs.buffer },
      { role: "r" as const, data: lib.data },
      { role: "r" as const, buffer: vectors.buffer },
      { role: "rw" as const, buffer: segs.indirect },
      { role: "r" as const, data: packed.data.length ? packed.data : new Float32Array(SEED_FLOATS) },
      { role: "rw" as const, size: Math.max(8, packed.points * 2 * 4) },
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

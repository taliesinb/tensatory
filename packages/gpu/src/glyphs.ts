// Fused vector-field glyphs: core's arrow glyphs (flow/glyphs.ts) computed
// entirely on the device. Two dispatches, no readback:
//
//  1. measure — one thread per lattice point evaluates the vector field there
//     (transpiled / buffer-read through the program builder), stores the vector
//     and folds its norm into ONE atomic maximum. Positive f32 bit patterns
//     order like u32, so `atomicMax` on the bits is a max over the norms; NaN /
//     ∞ (exponent all ones) are skipped, exactly as core's `maxNorm`.
//  2. emit — one thread per point reads its vector and the maximum, builds the
//     centred arrow (shaft + two barbs, the same formulas as core) and appends
//     three Seg (2D) or Seg3 (3D) records with the colour field's value at the
//     point, for the line pipelines of the two renderers.
//
// The lattice travels in the params buffer as a list of packed grids (the
// cosets) and the set's capacity with it, so one compiled kernel per (field,
// colour) serves every spacing and view: a pan or zoom re-dispatches, compiles
// nothing and uploads nothing but the few floats of the lattice.

import { CHEVRON_SPREAD, DenseGrid, GLYPH_FILL, GLYPH_HEAD, GLYPH_STYLES, HEAD_SPREAD, TRIANGLE_HALF_WIDTH, type GlyphOptions, type GlyphStyle, type Lattice, type ScalarFieldData, type VectorFieldData } from "@tensatory/core";
import { RESIDENT_USAGE, type GpuBackend } from "./device";
import { SEG3_APPEND_WGSL, SEG3_WGSL, type GpuSegments3 } from "./lines3d";
import { ProgramBuilder } from "./program";
import { SEG_APPEND_WGSL, SEG_WGSL, type GpuSegments } from "./segments";
import { GRID_FLOATS, f32, packGrid, vecType } from "./wgsl";

/** params: [pointCount (bits), cosets (bits), fill, head, spacing, capacity (bits), style (bits: index into GLYPH_STYLES), minLength, then one packed grid per coset] */
const LATTICE_OFF = 8;

/** per-dispatch glyph parameters (the rest of GlyphOptions is fixed when the kernel is built) */
export interface GlyphDispatch { style?: GlyphStyle; minLength?: number }

export function packLattice(l: Lattice, fill: number, head: number, capacity: number, style: GlyphStyle = "arrow", minLength = 0): Float32Array {
  const f = new Float32Array(LATTICE_OFF + GRID_FLOATS * Math.max(1, l.cosets.length));
  const u = new Uint32Array(f.buffer);
  u[0] = l.pointCount; u[1] = l.cosets.length; u[5] = Math.min(0xffffffff, capacity); u[6] = Math.max(0, GLYPH_STYLES.indexOf(style));
  f[2] = fill; f[3] = head; f[4] = l.spacing; f[7] = minLength;
  l.cosets.forEach((g, c) => packGrid(g, f, LATTICE_OFF + GRID_FLOATS * c));
  return f;
}

/** WGSL: `latticePoint(i)` — the i'th point of the packed lattice in `params` (coset by coset, row-major within each) */
function latticeWgsl(D: number): string {
  const T = vecType(D);
  return `
fn lat_u(c: i32, k: i32) -> i32 { return bitcast<i32>(params[${LATTICE_OFF} + c * ${GRID_FLOATS} + k]); }
fn lat_f(c: i32, k: i32) -> f32 { return params[${LATTICE_OFF} + c * ${GRID_FLOATS} + k]; }
fn latticeCount() -> i32 { return bitcast<i32>(params[0]); }
fn latticePoint(i0: i32) -> ${T} {
  let nc = bitcast<i32>(params[1]);
  var c: i32 = 0; var rem: i32 = i0;
  loop { let n = lat_u(c, 0); if (rem < n || c + 1 >= nc) { break; } rem = rem - n; c = c + 1; }
  var p: ${T};
  for (var d: i32 = 0; d < ${D}; d++) {
    let s = lat_u(c, 5 + d); let g = rem / s; rem = rem - g * s;
    p[d] = lat_f(c, 8 + d) + f32(g) * lat_f(c, 11 + d);
  }
  return p;
}`;
}

export interface FusedGlyphs {
  /** evaluate the field on `lattice`, normalize and append the glyphs (style / cutoff from `d`, defaults from the
   *  options) into `segs` (allocated by the caller with at least `capacityFor(lattice)` records and reset; a fuller
   *  set is counted but not written, like the other kernels) */
  dispatch(segs: GpuSegments | GpuSegments3, lattice: Lattice, d?: GlyphDispatch): void;
  run(segs: GpuSegments | GpuSegments3, lattice: Lattice, d?: GlyphDispatch): Promise<void>;
  /** segments a lattice appends at most: three per point */
  capacityFor(lattice: Lattice): number;
  /** the normalizing norm of the last dispatch (the longest vector sampled), read back once the queue reaches it */
  readMaxNorm(): Promise<number>;
  destroy(): void;
}

/**
 * Build the glyph kernels for `field` (D = 2 or 3), coloured by `colour` at each point when given. The lattice is a
 * dispatch parameter. The vectors buffer (D f32 per point, grown as lattices need) and the maximum are resident and
 * owned by the returned object.
 */
export function fusedGlyphs(backend: GpuBackend, field: VectorFieldData, colour?: ScalarFieldData, opts: GlyphOptions = {}): FusedGlyphs {
  const D = field.dimCount;
  if (D !== 2 && D !== 3) throw new Error(`fusedGlyphs: ${D}D fields are not supported`);
  const T = vecType(D);
  const b = new ProgramBuilder(new DenseGrid(new Array<number>(D).fill(2), field.box));
  const vf = b.vector(field);
  const col = colour ? b.scalar(colour) : undefined;
  const lib = b.library();
  const fill = opts.fill ?? GLYPH_FILL, head = opts.head ?? GLYPH_HEAD, defaultStyle = opts.style ?? "arrow", defaultMin = opts.minLength ?? 0;
  let vecs = backend.createBuffer({ size: 16, usage: RESIDENT_USAGE });
  const mx = backend.createBuffer({ size: 16, usage: RESIDENT_USAGE });

  const measure = `${lib.code}
@group(0) @binding(0) var<storage, read_write> vecs: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<f32>;
@group(0) @binding(3) var<storage, read_write> mx: array<atomic<u32>>;
${latticeWgsl(D)}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = i32(id.x);
  if (i >= latticeCount()) { return; }
  let v = ${vf}(latticePoint(i), -1);
  for (var d: i32 = 0; d < ${D}; d++) { vecs[i * ${D} + d] = v[d]; }
  let l2 = dot(v, v);
  if (isfinite_(l2) && l2 > 0.0) { atomicMax(&mx[0], bitcast<u32>(sqrt(l2))); }
}`;

  // the glyph normal: 2D the left normal; 3D normal to u and the axis it is least aligned with (ties as in core)
  const normal = D === 2
    ? `fn glyphNormal(u: vec2<f32>) -> vec2<f32> { return vec2<f32>(-u.y, u.x); }`
    : `fn glyphNormal(u: vec3<f32>) -> vec3<f32> {
  let a = abs(u);
  var e = vec3<f32>(0.0, 0.0, 1.0);
  if (a.x <= a.y && a.x <= a.z) { e = vec3<f32>(1.0, 0.0, 0.0); } else if (a.y <= a.z) { e = vec3<f32>(0.0, 1.0, 0.0); }
  let c = cross(u, e);
  let l = length(c);
  return select(c / l, c, l == 0.0);
}`;
  const seg = D === 2
    ? { structs: SEG_WGSL, append: SEG_APPEND_WGSL, arr: "Seg", ind: "Indirect", call: "appendSeg", make: `
fn rec(a: vec2<f32>, b: vec2<f32>, c: f32) -> Seg { var s: Seg; s.a = a; s.b = b; s.ca = c; s.cb = c; s.arc = 0.0; s.len = 0.0; s.phase = 0.0; s.pad = 0.0; return s; }
// a filled triangle: base a-b, apex in (arc, len) — read by the renderer's triangle pipeline
fn tri(a: vec2<f32>, b: vec2<f32>, apex: vec2<f32>, c: f32) -> Seg { var s = rec(a, b, c); s.arc = apex.x; s.len = apex.y; return s; }` }
    : { structs: SEG3_WGSL, append: SEG3_APPEND_WGSL, arr: "Seg3", ind: "Indirect3", call: "appendSeg3", make: `
fn rec(a: vec3<f32>, b: vec3<f32>, c: f32) -> Seg3 { var s: Seg3; s.a = a; s.b = b; s.ca = c; s.cb = c; s.arc = 0.0; s.len = 0.0; s.phase = 0.0; s.pad = 0.0; return s; }
fn tri(a: vec3<f32>, b: vec3<f32>, apex: vec3<f32>, c: f32) -> Seg3 { var s = rec(a, b, c); s.arc = apex.x; s.len = apex.y; s.phase = apex.z; return s; }` };
  const emit = `${lib.code}
${seg.structs}
@group(0) @binding(0) var<storage, read_write> segs: array<${seg.arr}>;
@group(0) @binding(2) var<storage, read> vecs: array<f32>;
@group(0) @binding(3) var<storage, read_write> ind: ${seg.ind};
@group(0) @binding(4) var<storage, read> params: array<f32>;
@group(0) @binding(5) var<storage, read> mx: array<u32>;
const SPREAD: f32 = ${f32(HEAD_SPREAD)};
const CHEVRON: f32 = ${f32(CHEVRON_SPREAD)};
const TRI_W: f32 = ${f32(TRIANGLE_HALF_WIDTH)};
// the set's real capacity comes with the params (bits); the atomic counts every segment regardless
${seg.append.replace("CAP", "bitcast<u32>(params[5])")}
${latticeWgsl(D)}
${normal}
${seg.make}
fn colour_(p: ${T}) -> f32 { return ${col ? `${col}(p, -1)` : "0.0"}; }
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = i32(id.x);
  if (i >= latticeCount()) { return; }
  var v: ${T};
  for (var d: i32 = 0; d < ${D}; d++) { v[d] = vecs[i * ${D} + d]; }
  let l2 = dot(v, v);
  if (!isfinite_(l2) || !(l2 > 0.0)) { return; }
  let vmax = bitcast<f32>(mx[0]);
  if (!(vmax > 0.0)) { return; }
  let fill = params[2]; let head = params[3]; let spacing = params[4];
  let len = sqrt(l2);
  let L = fill * spacing * len / vmax;
  if (L < params[7]) { return; } // shorter than the cutoff: visual noise, not drawn
  let u = v / len;
  let nrm = glyphNormal(u);
  let p = latticePoint(i);
  let tip = p + 0.5 * L * u;
  let c = colour_(p);
  let style = bitcast<u32>(params[6]);
  if (style == 1u) {
    // head: a chevron of length L centred on p
    let back = tip - L * u;
    ${seg.call}(rec(back + CHEVRON * L * nrm, tip, c));
    ${seg.call}(rec(tip, back - CHEVRON * L * nrm, c));
  } else if (style == 2u) {
    // triangle: one filled record, base centred on p, apex at the arrow's tip
    let w = TRI_W * 0.5 * L;
    ${seg.call}(tri(p + w * nrm, p - w * nrm, tip, c));
  } else {
    let tail = p - 0.5 * L * u;
    let back = tip - head * L * u;
    ${seg.call}(rec(tail, tip, c));
    ${seg.call}(rec(tip, back + SPREAD * head * L * nrm, c));
    ${seg.call}(rec(tip, back - SPREAD * head * L * nrm, c));
  }
}`;

  const kernels = (segs: GpuSegments | GpuSegments3, lattice: Lattice, d: GlyphDispatch) => {
    if (lattice.dimCount !== D) throw new Error("fusedGlyphs: lattice and field dimensions differ");
    const n = Math.max(1, lattice.pointCount);
    const bytes = Math.max(16, n * D * 4);
    if (vecs.size < bytes) { vecs.destroy(); vecs = backend.createBuffer({ size: bytes, usage: RESIDENT_USAGE }); } // safe: earlier dispatches are already submitted
    const params = packLattice(lattice, fill, head, segs.capacity, d.style ?? defaultStyle, d.minLength ?? defaultMin);
    backend.write(mx, 0, zero);
    return [
      { code: measure, invocations: n, buffers: [{ role: "rw" as const, buffer: vecs }, { role: "r" as const, data: lib.data }, { role: "r" as const, data: params }, { role: "rw" as const, buffer: mx }] },
      { code: emit, invocations: n, buffers: [{ role: "rw" as const, buffer: segs.buffer }, { role: "r" as const, data: lib.data }, { role: "r" as const, buffer: vecs }, { role: "rw" as const, buffer: segs.indirect }, { role: "r" as const, data: params }, { role: "r" as const, buffer: mx }] },
    ];
  };
  const zero = new Uint32Array(4);
  return {
    capacityFor: (lattice) => 3 * Math.max(1, lattice.pointCount),
    dispatch(segs, lattice, d = {}) { for (const k of kernels(segs, lattice, d)) backend.dispatch(k); },
    async run(segs, lattice, d = {}) { for (const k of kernels(segs, lattice, d)) await backend.runKernel(k); },
    async readMaxNorm() {
      const bits = await backend.readCounter(mx, 0);
      return new Float32Array(Uint32Array.of(bits).buffer)[0]!;
    },
    destroy() { vecs.destroy(); mx.destroy(); },
  };
}

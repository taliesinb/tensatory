// Isosurface meshes on the device: resident triangle sets appended by a fused
// marching-tetrahedra kernel (or uploaded from a CPU IsoMesh), drawn with
// drawIndirect. Same shape as segments.ts for lines.
//
//   Vert { p: vec3 position | n: vec3 unit normal (towards increasing f) | c: colour value | pad }
//
// One thread per grid cell: eight values → mask → six tetrahedra → the
// TET_TRIANGLES table (shared with core) → vertices on the tet edges, moved
// onto the true level set along the exact gradient when the field is symbolic
// (the 3D twin of the 2D projection: damped Newton, bisection fallback, box
// faces locked), with a normal from the exact gradient (symbolic) or central
// differences of the resident values, a colour from the colour program,
// appended through an atomic vertex counter.

import { CUBE, TET_EDGES, TET_TRIANGLES, TETS, type IsoMesh, type ScalarFieldData } from "@tensatory/core";
import { RESIDENT_USAGE, type GpuBackend } from "./device";
import { ProgramBuilder } from "./program";
import type { GpuGrid } from "./resident";
import { f32 } from "./wgsl";

export const VERT_FLOATS = 8; // 32 bytes
export const VERT_WGSL = `
struct Vert { p: vec3<f32>, c: f32, n: vec3<f32>, pad: f32 }
struct MeshIndirect { vertexCount: atomic<u32>, instanceCount: u32, firstVertex: u32, firstInstance: u32 }
`;

export interface GpuMesh {
  /** Vert records */
  buffer: GPUBuffer;
  /** [vertexCount, 1, 0, 0] for drawIndirect; vertexCount may exceed 3 × capacity (extra vertices draw nothing) */
  indirect: GPUBuffer;
  /** triangles */
  capacity: number;
  destroy(): void;
}

export function allocMesh(backend: GpuBackend, capacity: number): GpuMesh {
  const dev = backend.device;
  const buffer = backend.createBuffer({ size: Math.max(32, capacity * 3 * VERT_FLOATS * 4), usage: RESIDENT_USAGE });
  const indirect = backend.createBuffer({ size: 16, usage: RESIDENT_USAGE });
  dev.queue.writeBuffer(indirect, 0, new Uint32Array([0, 1, 0, 0]));
  return { buffer, indirect, capacity, destroy: () => { buffer.destroy(); indirect.destroy(); } };
}

export function resetMesh(backend: GpuBackend, m: GpuMesh): void {
  backend.write(m.indirect, 0, new Uint32Array([0, 1, 0, 0]));
}

/** pack a CPU IsoMesh into Vert records */
export function packMesh(mesh: IsoMesh): Float32Array {
  const n = mesh.triangleCount * 3;
  const out = new Float32Array(n * VERT_FLOATS);
  for (let v = 0; v < n; v++) {
    const o = v * VERT_FLOATS;
    out[o] = mesh.positions[v * 3]!; out[o + 1] = mesh.positions[v * 3 + 1]!; out[o + 2] = mesh.positions[v * 3 + 2]!;
    out[o + 3] = mesh.values ? mesh.values[v]! : 0;
    out[o + 4] = mesh.normals[v * 3]!; out[o + 5] = mesh.normals[v * 3 + 1]!; out[o + 6] = mesh.normals[v * 3 + 2]!;
  }
  return out;
}

export function uploadMesh(backend: GpuBackend, data: Float32Array): GpuMesh {
  const verts = data.length / VERT_FLOATS;
  const m = allocMesh(backend, Math.max(1, verts / 3));
  if (verts) backend.write(m.buffer, 0, data as unknown as BufferSource);
  backend.write(m.indirect, 0, new Uint32Array([verts, 1, 0, 0]));
  return m;
}

/** read a mesh back (tests / debugging) as an IsoMesh */
export async function readMesh(backend: GpuBackend, m: GpuMesh): Promise<IsoMesh> {
  await backend.whenIdle(); // deferred dispatches (async compiles) land first
  const dev = backend.device;
  const readInd = dev.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const readV = dev.createBuffer({ size: m.buffer.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const enc = dev.createCommandEncoder();
  enc.copyBufferToBuffer(m.indirect, 0, readInd, 0, 16);
  enc.copyBufferToBuffer(m.buffer, 0, readV, 0, m.buffer.size);
  dev.queue.submit([enc.finish()]);
  await readInd.mapAsync(GPUMapMode.READ);
  const nv = Math.min(m.capacity * 3, new Uint32Array(readInd.getMappedRange())[0]!);
  readInd.unmap(); readInd.destroy();
  await readV.mapAsync(GPUMapMode.READ);
  const raw = new Float32Array(readV.getMappedRange().slice(0, nv * VERT_FLOATS * 4));
  readV.unmap(); readV.destroy();
  const positions = new Float32Array(nv * 3), normals = new Float32Array(nv * 3), values = new Float32Array(nv);
  for (let v = 0; v < nv; v++) {
    const o = v * VERT_FLOATS;
    positions[v * 3] = raw[o]!; positions[v * 3 + 1] = raw[o + 1]!; positions[v * 3 + 2] = raw[o + 2]!;
    values[v] = raw[o + 3]!;
    normals[v * 3] = raw[o + 4]!; normals[v * 3 + 1] = raw[o + 5]!; normals[v * 3 + 2] = raw[o + 6]!;
  }
  return { positions, normals, values, triangleCount: nv / 3 };
}

/** WGSL: `project3_(p, maxDist, level) -> vec4 (q, ok)` — Newton along the exact gradient of a symbolic 3D field, box faces locked */
export function projection3Wgsl(fn: string, valueGradient: string, box: { a: readonly number[]; b: readonly number[]; size: number[] }): string {
  const eps = 1e-6 * Math.max(...box.size);
  return `
const PA3: vec3<f32> = vec3<f32>(${f32(box.a[0]!)}, ${f32(box.a[1]!)}, ${f32(box.a[2]!)});
const PB3: vec3<f32> = vec3<f32>(${f32(box.b[0]!)}, ${f32(box.b[1]!)}, ${f32(box.b[2]!)});
const PEPS3: f32 = ${f32(eps)};
fn inBox3_(q: vec3<f32>) -> bool { return all(q >= PA3 - PEPS3) && all(q <= PB3 + PEPS3); }
fn resid3_(q: vec3<f32>, level: f32) -> f32 { return ${fn}(q, -1) - level; }
fn fg3_(q: vec3<f32>, level: f32, lock: vec3<f32>) -> vec4<f32> { let v = ${valueGradient}(q, -1); return vec4<f32>(v.x - level, v.yzw * lock); }
// see projectionWgsl (isolines.ts): one combined evaluation per Newton step, step-size stop, geometric acceptance
fn project3_(p: vec3<f32>, maxDist: f32, level: f32, tolW: f32) -> vec4<f32> {
  let scale = max(1.0, abs(level));
  let tolF = 1e-6 * scale;
  let onLo = abs(p - PA3) < vec3<f32>(PEPS3); let onHi = abs(p - PB3) < vec3<f32>(PEPS3);
  let lock = select(vec3<f32>(1.0), vec3<f32>(0.0), onLo | onHi);
  var q = p;
  var rg = fg3_(q, level, lock);
  var r = rg.x; var g = rg.yzw;
  var ok = 0.0;
  let g0 = g;
  let g02 = dot(g0, g0);
  if (!isfinite_(r) || !(g02 > 1e-24)) { return vec4<f32>(p, 0.0); }
  let r0 = r;
  var gn = sqrt(g02);
  for (var it = 0; it < 12; it++) {
    if (abs(r) < tolF) { break; }
    let g2 = dot(g, g);
    if (!(g2 > 1e-24)) { break; }
    gn = sqrt(g2);
    var k = r / g2;
    var accepted = false;
    for (var damp = 0; damp < 4; damp++) {
      let trial = q - k * g;
      let d = trial - p;
      if (inBox3_(trial) && dot(d, d) <= maxDist * maxDist) {
        let t = fg3_(trial, level, lock);
        if (isfinite_(t.x) && abs(t.x) < abs(r)) { q = trial; r = t.x; g = t.yzw; accepted = true; break; }
      }
      k = k * 0.5;
    }
    if (!accepted) { break; }
    if (abs(k) * gn < tolW) { break; }
  }
  if (abs(r) < tolF || abs(r) < 1e-5 * scale || abs(r) < tolW * gn) { ok = 1.0; }
  else {
    let dir = -sign(r0) * g0 / sqrt(g02);
    var lo = 0.0; var hi = maxDist / 64.0; var found = false;
    for (var n = 0; n < 12; n++) {
      let t = p + hi * dir;
      if (!inBox3_(t)) { break; }
      let rh = resid3_(t, level);
      if (!isfinite_(rh)) { break; }
      if (sign(rh) != sign(r0)) { found = true; break; }
      lo = hi; hi = hi * 2.0;
      if (hi > maxDist) { break; }
    }
    if (found) {
      for (var it = 0; it < 40; it++) {
        let mid = 0.5 * (lo + hi);
        let rm = resid3_(p + mid * dir, level);
        if (abs(rm) < tolF || hi - lo < tolW) { lo = mid; hi = mid; break; }
        if (sign(rm) == sign(r0)) { lo = mid; } else { hi = mid; }
      }
      q = p + 0.5 * (lo + hi) * dir; ok = 1.0;
    }
  }
  // locked coordinates snap back onto their face
  q = select(q, select(PB3, PA3, abs(q - PA3) < abs(q - PB3)), lock == vec3<f32>(0.0));
  return vec4<f32>(q, ok);
}`;
}

export interface FusedIsosurfaceOptions {
  /** the field `values` samples; symbolic data gives exact normals and (unless `exact: false`) projected vertices */
  field?: ScalarFieldData;
  /** project vertices onto the level set along the exact gradient (default: field is symbolic) */
  exact?: boolean;
  /** colour field evaluated at every vertex */
  colour?: ScalarFieldData;
}

export interface FusedIsosurface {
  /** append the isosurface at `level` into `mesh` (reset it first when reusing) */
  run(mesh: GpuMesh, level: number): Promise<void>;
  dispatch(mesh: GpuMesh, level: number): void;
  /** worst-case triangle count: 2 per tetrahedron */
  readonly capacity: number;
  destroy(): void;
}

/**
 * Build the fused marching-tetrahedra kernel for `values` (a resident 3D scalar
 * grid). With a symbolic `field` the vertices are projected onto the true level
 * set and the normals are the exact gradient there; otherwise normals come from
 * central differences of the values. `colour` is evaluated at every vertex.
 *
 * The grid (size, strides, origin, spacing) and the mesh capacity travel in the
 * params buffer, not in the WGSL, so the shader — and its compiled pipeline —
 * is the same for every resolution of a field: an adaptive resolution ramps
 * without shader compiles. The kernel counts every triangle through the atomic
 * even when the mesh is full, so the indirect counter is the true size.
 */
export function fusedIsosurface(backend: GpuBackend, values: GpuGrid, opts: FusedIsosurfaceOptions = {}): FusedIsosurface {
  const grid = values.grid;
  if (grid.dimCount !== 3 || values.channels !== 1) throw new Error("fusedIsosurface needs a resident 3D scalar grid");
  const [nx, ny, nz] = grid.size as [number, number, number];
  const [sx, sy, sz] = grid.strides as [number, number, number];
  const [hx, hy, hz] = grid.spacing as [number, number, number];
  const b = new ProgramBuilder(grid);
  const symbolic = opts.field?.kind === "symbolic" ? opts.field : undefined;
  const exact = opts.exact ?? !!symbolic;
  if (exact && !symbolic) throw new Error("fusedIsosurface: exact projection needs a symbolic field");
  const fn = symbolic && exact ? b.scalar(symbolic) : "", grad = symbolic ? b.gradient(symbolic) : "", vg = symbolic && exact ? b.valueGradient(symbolic) : ""; // normals need the gradient even without projection
  const colour = opts.colour;
  const col = colour ? b.scalar(colour) : undefined;
  const maxDist = Math.hypot(hx, hy, hz);
  const lib = b.library();
  const cells = (nx - 1) * (ny - 1) * (nz - 1);
  const capacity = cells * 12;
  // tables shared with core
  const triTable: number[] = [];
  for (let t = 0; t < 6; t++) for (let m = 0; m < 16; m++) { const tris = TET_TRIANGLES[t]![m]!; for (let q = 0; q < 6; q++) triTable.push(q < tris.length ? tris[q]! : -1); }
  const code = `${lib.code}
${VERT_WGSL}
@group(0) @binding(0) var<storage, read_write> verts: array<Vert>;
@group(0) @binding(2) var<storage, read> vals: array<f32>;
@group(0) @binding(3) var<storage, read_write> ind: MeshIndirect;
@group(0) @binding(4) var<storage, read> params: array<f32>;
// params: [level, capV(bits), nx, ny, nz, sx, sy, sz (bits), ax, ay, az, hx, hy, hz, maxDist]
fn pi_(i: i32) -> i32 { return bitcast<i32>(params[i]); }
const CUBE_: array<vec3<i32>, 8> = array<vec3<i32>, 8>(${CUBE.map(([a, b2, c]) => `vec3<i32>(${a}, ${b2}, ${c})`).join(", ")});
const TETS_: array<vec4<i32>, 6> = array<vec4<i32>, 6>(${TETS.map((T) => `vec4<i32>(${T.join(", ")})`).join(", ")});
const EDGE_A: array<i32, ${TET_EDGES.length}> = array<i32, ${TET_EDGES.length}>(${TET_EDGES.map(([a]) => a).join(", ")});
const EDGE_B: array<i32, ${TET_EDGES.length}> = array<i32, ${TET_EDGES.length}>(${TET_EDGES.map(([, b2]) => b2).join(", ")});
const TRI: array<i32, ${triTable.length}> = array<i32, ${triTable.length}>(${triTable.join(", ")});
fn val_(i: i32, j: i32, k: i32) -> f32 { return vals[i * pi_(5) + j * pi_(6) + k * pi_(7)]; }
// central differences (one-sided at the faces), world units
fn gridGrad_(i: i32, j: i32, k: i32) -> vec3<f32> {
  let H = vec3<f32>(params[11], params[12], params[13]);
  let i0 = max(i - 1, 0); let i1 = min(i + 1, pi_(2) - 1);
  let j0 = max(j - 1, 0); let j1 = min(j + 1, pi_(3) - 1);
  let k0 = max(k - 1, 0); let k1 = min(k + 1, pi_(4) - 1);
  return vec3<f32>(
    select((val_(i1, j, k) - val_(i0, j, k)) / (f32(i1 - i0) * H.x), 0.0, i1 == i0),
    select((val_(i, j1, k) - val_(i, j0, k)) / (f32(j1 - j0) * H.y), 0.0, j1 == j0),
    select((val_(i, j, k1) - val_(i, j, k0)) / (f32(k1 - k0) * H.z), 0.0, k1 == k0));
}
${exact ? projection3Wgsl(fn, vg, symbolic!.box) : ""}
fn colour_(p: vec3<f32>) -> f32 { return ${col ? `${col}(p, -1)` : "0.0"}; }
fn vertex_(c: vec3<i32>, v: array<f32, 8>, e: i32, level: f32) -> Vert {
  let A = vec3<f32>(params[8], params[9], params[10]);
  let H = vec3<f32>(params[11], params[12], params[13]);
  let a = EDGE_A[e]; let bb = EDGE_B[e];
  let va = v[a]; let vb = v[bb];
  var t: f32 = 0.0;
  if (va != vb) { t = clamp((level - va) / (vb - va), 0.0, 1.0); }
  let ca = CUBE_[a]; let cb = CUBE_[bb];
  let g = vec3<f32>(c) + vec3<f32>(ca) + (vec3<f32>(cb) - vec3<f32>(ca)) * t;
  var p = A + g * H;
  ${exact ? `let pr = project3_(p, params[14], level, 1e-3 * params[14]); if (pr.w > 0.5) { p = pr.xyz; }` : ""}
  var out: Vert;
  out.p = p;
  ${symbolic ? `let gr = ${grad}(p, -1);` : `let ga = gridGrad_(c.x + ca.x, c.y + ca.y, c.z + ca.z); let gb = gridGrad_(c.x + cb.x, c.y + cb.y, c.z + cb.z); let gr = ga + (gb - ga) * t;`}
  let l = length(gr);
  if (l > 0.0 && isfinite_(l)) { out.n = gr / l; } else { out.n = vec3<f32>(0.0, 0.0, 1.0); }
  out.c = colour_(p);
  out.pad = 0.0;
  return out;
}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let NY = pi_(3); let NZ = pi_(4);
  let cells = (pi_(2) - 1) * (NY - 1) * (NZ - 1);
  let cell = i32(id.x);
  if (cell >= cells) { return; }
  let capV = bitcast<u32>(params[1]);
  // cell index -> (i, j, k), k fastest like the grid
  let i = cell / ((NY - 1) * (NZ - 1));
  let r = cell - i * ((NY - 1) * (NZ - 1));
  let j = r / (NZ - 1);
  let k = r - j * (NZ - 1);
  let level = params[0];
  var v: array<f32, 8>;
  var mask: u32 = 0u;
  for (var c = 0; c < 8; c++) {
    let o = CUBE_[c];
    let f = val_(i + o.x, j + o.y, k + o.z);
    if (isnan_(f)) { return; }
    v[c] = f;
    if (f >= level) { mask = mask | (1u << u32(c)); }
  }
  if (mask == 0u || mask == 255u) { return; }
  let cidx = vec3<i32>(i, j, k);
  for (var t = 0; t < 6; t++) {
    let T = TETS_[t];
    let m = ((mask >> u32(T.x)) & 1u) | (((mask >> u32(T.y)) & 1u) << 1u) | (((mask >> u32(T.z)) & 1u) << 2u) | (((mask >> u32(T.w)) & 1u) << 3u);
    let base = (t * 16 + i32(m)) * 6;
    for (var q = 0; q < 6; q += 3) {
      let e0 = TRI[base + q];
      if (e0 < 0) { break; }
      let e1 = TRI[base + q + 1]; let e2 = TRI[base + q + 2];
      let at = atomicAdd(&ind.vertexCount, 3u);
      if (at + 3u <= capV) {
        verts[at] = vertex_(cidx, v, e0, level);
        verts[at + 1u] = vertex_(cidx, v, e1, level);
        verts[at + 2u] = vertex_(cidx, v, e2, level);
      }
    }
  }
}`;
  const params = (mesh: GpuMesh, level: number) => {
    const f = new Float32Array(16), u = new Uint32Array(f.buffer);
    f[0] = level; u[1] = Math.min(0xffffffff, mesh.capacity * 3); u[2] = nx; u[3] = ny; u[4] = nz; u[5] = sx; u[6] = sy; u[7] = sz;
    f[8] = grid.box.a[0]!; f[9] = grid.box.a[1]!; f[10] = grid.box.a[2]!; f[11] = hx; f[12] = hy; f[13] = hz; f[14] = maxDist;
    return f;
  };
  const kernel = (mesh: GpuMesh, level: number) => ({
    code,
    invocations: cells,
    buffers: [
      { role: "rw" as const, buffer: mesh.buffer },
      { role: "r" as const, data: lib.data },
      { role: "r" as const, buffer: values.buffer },
      { role: "rw" as const, buffer: mesh.indirect },
      { role: "r" as const, data: params(mesh, level) },
    ],
  });
  return {
    capacity,
    async run(mesh, level) { await backend.runKernel(kernel(mesh, level)); },
    dispatch(mesh, level) { backend.dispatch(kernel(mesh, level)); },
    destroy() { /* nothing resident besides the caller's buffers */ },
  };
}

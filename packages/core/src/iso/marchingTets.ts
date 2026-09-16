// Isosurfaces of a scalar field sampled on a dense 3D grid: marching tetrahedra.
//
// Every grid cell is split into the six tetrahedra around its (0,0,0)–(1,1,1)
// diagonal; each tetrahedron yields 0, 1 or 2 triangles whose vertices are the
// linear zero crossings of f − level along its edges. The output is a triangle
// soup (no welding): three vertices per triangle, each with a position, a unit
// normal and an optional colour value. Normals come from the gradient of the
// field — the exact gradient of a symbolic field evaluated at the vertex, or
// central differences of the grid values interpolated along the edge — and
// point towards INCREASING field values; triangles are wound counter-clockwise
// seen from that side, so the geometric and the shading normals agree.
//
// Cells are visited in grid order (i, j, k) and tetrahedra in table order, so
// the CPU output is deterministic; the GPU kernel (packages/gpu) emits the same
// triangles in an unspecified order — tests compare sorted.

import type { Point } from "@tensatory/schema";
import { DenseGrid } from "../geometry/grid";

/** the cube's corners, bit c of a cell mask ↔ CUBE[c] */
export const CUBE: readonly (readonly [number, number, number])[] = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]];
/** six tetrahedra around the 0–6 diagonal, as corner indices */
export const TETS: readonly (readonly [number, number, number, number])[] = [[0, 5, 1, 6], [0, 1, 2, 6], [0, 2, 3, 6], [0, 3, 7, 6], [0, 7, 4, 6], [0, 4, 5, 6]];

/** the 19 distinct tetrahedron edges as corner pairs (a < b) */
export const TET_EDGES: readonly (readonly [number, number])[] = (() => {
  const seen = new Set<number>(); const edges: [number, number][] = [];
  for (const T of TETS) for (let a = 0; a < 4; a++) for (let b = a + 1; b < 4; b++) {
    const i = Math.min(T[a]!, T[b]!), j = Math.max(T[a]!, T[b]!);
    if (!seen.has(i * 8 + j)) { seen.add(i * 8 + j); edges.push([i, j]); }
  }
  return edges;
})();
const EDGE_ID = new Int8Array(64).fill(-1);
TET_EDGES.forEach(([a, b], e) => { EDGE_ID[a * 8 + b] = e; });
const edgeId = (a: number, b: number): number => EDGE_ID[Math.min(a, b) * 8 + Math.max(a, b)]!;

/**
 * Per tetrahedron, per "high" mask (bit c set ⇔ value at corner T[c] ≥ level): the
 * triangles as edge ids, wound so that the geometric normal points towards the
 * high corners. Flattened as 3 ids per triangle; 0, 3 or 6 entries.
 */
export const TET_TRIANGLES: readonly (readonly (readonly number[])[])[] = TETS.map((T) => {
  const mid = (e: number): [number, number, number] => { const [a, b] = TET_EDGES[e]!; return [0, 1, 2].map((k) => (CUBE[a]![k]! + CUBE[b]![k]!) / 2) as [number, number, number]; };
  const orient = (tri: [number, number, number], high: number[]): number[] => {
    const p = tri.map(mid);
    const u = [0, 1, 2].map((k) => p[1]![k]! - p[0]![k]!), w = [0, 1, 2].map((k) => p[2]![k]! - p[0]![k]!);
    const n = [u[1]! * w[2]! - u[2]! * w[1]!, u[2]! * w[0]! - u[0]! * w[2]!, u[0]! * w[1]! - u[1]! * w[0]!];
    const cen = [0, 1, 2].map((k) => (p[0]![k]! + p[1]![k]! + p[2]![k]!) / 3);
    const hc = [0, 1, 2].map((k) => high.reduce((s, c) => s + CUBE[c]![k]!, 0) / high.length);
    const d = [0, 1, 2].reduce((s, k) => s + n[k]! * (hc[k]! - cen[k]!), 0);
    return d < 0 ? [tri[0], tri[2], tri[1]] : tri; // normal must point to the high side
  };
  const cases: number[][] = [];
  for (let mask = 0; mask < 16; mask++) {
    const H: number[] = [], L: number[] = [];
    for (let c = 0; c < 4; c++) (mask >> c & 1 ? H : L).push(T[c]!);
    let tris: number[] = [];
    if (H.length === 1) tris = orient([edgeId(H[0]!, L[0]!), edgeId(H[0]!, L[1]!), edgeId(H[0]!, L[2]!)], H);
    else if (H.length === 3) tris = orient([edgeId(L[0]!, H[0]!), edgeId(L[0]!, H[1]!), edgeId(L[0]!, H[2]!)], H);
    else if (H.length === 2) {
      const e00 = edgeId(H[0]!, L[0]!), e01 = edgeId(H[0]!, L[1]!), e11 = edgeId(H[1]!, L[1]!), e10 = edgeId(H[1]!, L[0]!);
      tris = [...orient([e00, e01, e11], H), ...orient([e00, e11, e10], H)];
    }
    cases.push(tris);
  }
  return cases;
});

/** triangle soup: 3 vertices per triangle */
export interface IsoMesh {
  /** xyz per vertex: 9 floats per triangle */
  positions: Float32Array;
  /** unit normal per vertex, towards increasing field values: 9 floats per triangle */
  normals: Float32Array;
  /** colour value per vertex (3 per triangle) when a colour source was given */
  values?: Float32Array;
  triangleCount: number;
}

export interface MarchingTetOptions {
  /** exact gradient at a point (symbolic fields); default: central differences of the grid values */
  gradient?: (p: Point) => ArrayLike<number> | undefined;
  /** colour values on the same grid, interpolated along the edge */
  colour?: ArrayLike<number>;
  /** colour evaluated at the vertex (symbolic colour fields); takes precedence over `colour` */
  colourAt?: (p: Point) => number;
}

/** isosurface of grid `values` at `level` (see the module comment) */
export function marchingTetrahedra(grid: DenseGrid, values: ArrayLike<number>, level: number, opts: MarchingTetOptions = {}): IsoMesh {
  if (grid.dimCount !== 3) throw new Error(`marchingTetrahedra needs a 3D grid, got ${grid.dimCount}D`);
  const [nx, ny, nz] = grid.size as [number, number, number];
  const [sx, sy, sz] = grid.strides as [number, number, number];
  const [hx, hy, hz] = grid.spacing as [number, number, number];
  const [ax, ay, az] = grid.box.a as [number, number, number];
  const OFF = CUBE.map(([a, b, c]) => a * sx + b * sy + c * sz);
  let pos = new Float32Array(9 * 4096), nrm = new Float32Array(9 * 4096), col = opts.colour || opts.colourAt ? new Float32Array(3 * 4096) : undefined;
  let nt = 0;
  const grow = () => {
    if ((nt + 2) * 9 > pos.length) {
      const p2 = new Float32Array(pos.length * 2); p2.set(pos); pos = p2;
      const n2 = new Float32Array(nrm.length * 2); n2.set(nrm); nrm = n2;
      if (col) { const c2 = new Float32Array(col.length * 2); c2.set(col); col = c2; }
    }
  };
  const v = new Float64Array(8);
  // central-difference gradient at a grid point (one-sided at the faces), in world units
  const gradAt = (i: number, j: number, k: number, out: Float64Array) => {
    const p = i * sx + j * sy + k * sz;
    const d = (s: number, n: number, idx: number, h: number) => {
      const lo = idx > 0 ? p - s : p, hi = idx < n - 1 ? p + s : p;
      return h > 0 ? (values[hi]! - values[lo]!) / (((hi - lo) / s) * h) : 0;
    };
    out[0] = d(sx, nx, i, hx); out[1] = d(sy, ny, j, hy); out[2] = d(sz, nz, k, hz);
  };
  const ga = new Float64Array(3), gb = new Float64Array(3);
  const pt: Point = [0, 0, 0];
  const emit = (i: number, j: number, k: number, e: number, out: number) => {
    const [a, b] = TET_EDGES[e]!;
    const va = v[a]!, vb = v[b]!;
    const t = va === vb ? 0 : Math.min(1, Math.max(0, (level - va) / (vb - va)));
    const ca = CUBE[a]!, cb = CUBE[b]!;
    const fi = i + ca[0] + (cb[0] - ca[0]) * t, fj = j + ca[1] + (cb[1] - ca[1]) * t, fk = k + ca[2] + (cb[2] - ca[2]) * t;
    pt[0] = ax + fi * hx; pt[1] = ay + fj * hy; pt[2] = az + fk * hz;
    pos[out] = pt[0]; pos[out + 1] = pt[1]; pos[out + 2] = pt[2];
    let gx: number, gy: number, gz: number;
    const g = opts.gradient?.(pt);
    if (g) { gx = g[0]!; gy = g[1]!; gz = g[2]!; }
    else {
      gradAt(i + ca[0], j + ca[1], k + ca[2], ga); gradAt(i + cb[0], j + cb[1], k + cb[2], gb);
      gx = ga[0]! + (gb[0]! - ga[0]!) * t; gy = ga[1]! + (gb[1]! - ga[1]!) * t; gz = ga[2]! + (gb[2]! - ga[2]!) * t;
    }
    const l = Math.hypot(gx, gy, gz);
    if (l > 0 && Number.isFinite(l)) { nrm[out] = gx / l; nrm[out + 1] = gy / l; nrm[out + 2] = gz / l; }
    else { nrm[out] = 0; nrm[out + 1] = 0; nrm[out + 2] = 1; }
    if (col) {
      const o3 = out / 3;
      if (opts.colourAt) col[o3] = opts.colourAt(pt);
      else { const c = opts.colour!, pa = (i + ca[0]) * sx + (j + ca[1]) * sy + (k + ca[2]) * sz, pb = (i + cb[0]) * sx + (j + cb[1]) * sy + (k + cb[2]) * sz; col[o3] = c[pa]! + (c[pb]! - c[pa]!) * t; }
    }
  };
  for (let i = 0; i + 1 < nx; i++) for (let j = 0; j + 1 < ny; j++) for (let k = 0; k + 1 < nz; k++) {
    const p0 = i * sx + j * sy + k * sz;
    let mask = 0, bad = false, lo = Infinity, hi = -Infinity;
    for (let c = 0; c < 8; c++) {
      const f = values[p0 + OFF[c]!]!; v[c] = f;
      if (Number.isNaN(f)) { bad = true; break; }
      if (f >= level) mask |= 1 << c;
      if (f < lo) lo = f; if (f > hi) hi = f;
    }
    if (bad || mask === 0 || mask === 255) continue;
    for (let t = 0; t < 6; t++) {
      const T = TETS[t]!;
      const m = (mask >> T[0] & 1) | ((mask >> T[1] & 1) << 1) | ((mask >> T[2] & 1) << 2) | ((mask >> T[3] & 1) << 3);
      const tris = TET_TRIANGLES[t]![m]!;
      if (!tris.length) continue;
      grow();
      for (let q = 0; q < tris.length; q += 3) {
        const out = nt * 9;
        emit(i, j, k, tris[q]!, out); emit(i, j, k, tris[q + 1]!, out + 3); emit(i, j, k, tris[q + 2]!, out + 6);
        nt++;
      }
    }
  }
  return { positions: pos.subarray(0, nt * 9), normals: nrm.subarray(0, nt * 9), values: col?.subarray(0, nt * 3), triangleCount: nt };
}

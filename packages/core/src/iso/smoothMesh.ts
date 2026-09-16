// Smoothing of isosurface soups: weld shared vertices (bit-identical positions
// from the same tetrahedron edge), Taubin λ|μ smoothing of the positions,
// area-weighted face normals, optional normal blurring, back to a soup. The
// prototype's `smooth` / `smoothNormals`, on the triangle soup marchingTets
// produces. For sampled data whose grid noise shows as facets; exact
// (projected) surfaces do not need it.

import type { IsoMesh } from "./marchingTets";

export interface IndexedMesh {
  positions: Float32Array;
  normals: Float32Array;
  values?: Float32Array;
  indices: Uint32Array;
  vertexCount: number;
}

/** weld a soup by exact position (vertices of a shared edge are computed identically in every tetrahedron) */
export function weldMesh(m: IsoMesh): IndexedMesh {
  const n = m.triangleCount * 3;
  const index = new Map<string, number>();
  const pos: number[] = [], nrm: number[] = [], val: number[] = [];
  const indices = new Uint32Array(n);
  for (let v = 0; v < n; v++) {
    const x = m.positions[v * 3]!, y = m.positions[v * 3 + 1]!, z = m.positions[v * 3 + 2]!;
    const key = `${x},${y},${z}`;
    let id = index.get(key);
    if (id === undefined) {
      id = pos.length / 3; index.set(key, id);
      pos.push(x, y, z); nrm.push(m.normals[v * 3]!, m.normals[v * 3 + 1]!, m.normals[v * 3 + 2]!);
      if (m.values) val.push(m.values[v]!);
    }
    indices[v] = id;
  }
  return { positions: Float32Array.from(pos), normals: Float32Array.from(nrm), values: m.values ? Float32Array.from(val) : undefined, indices, vertexCount: pos.length / 3 };
}

export function expandMesh(m: IndexedMesh): IsoMesh {
  const n = m.indices.length;
  const positions = new Float32Array(n * 3), normals = new Float32Array(n * 3), values = m.values ? new Float32Array(n) : undefined;
  for (let v = 0; v < n; v++) {
    const id = m.indices[v]!;
    positions.set(m.positions.subarray(id * 3, id * 3 + 3), v * 3);
    normals.set(m.normals.subarray(id * 3, id * 3 + 3), v * 3);
    if (values) values[v] = m.values![id]!;
  }
  return { positions, normals, values, triangleCount: n / 3 };
}

/** neighbour averaging of a per-vertex vec3 attribute, weight w (in place) */
function averagePass(m: IndexedMesh, attr: Float32Array, w: number, acc: Float32Array, deg: Float32Array): void {
  acc.fill(0); deg.fill(0);
  const I = m.indices;
  for (let i = 0; i < I.length; i += 3) for (let k = 0; k < 3; k++) {
    const a = I[i + k]!, b = I[i + ((k + 1) % 3)]!;
    for (let d = 0; d < 3; d++) { acc[a * 3 + d]! += attr[b * 3 + d]!; acc[b * 3 + d]! += attr[a * 3 + d]!; }
    deg[a]!++; deg[b]!++;
  }
  for (let v = 0; v < m.vertexCount; v++) {
    const dg = deg[v]!; if (!dg) continue;
    for (let d = 0; d < 3; d++) attr[v * 3 + d]! += w * (acc[v * 3 + d]! / dg - attr[v * 3 + d]!);
  }
}

/** area-weighted vertex normals from the faces, oriented like the existing normals (which point towards increasing f) */
export function faceNormals(m: IndexedMesh): void {
  const P = m.positions, I = m.indices, acc = new Float32Array(m.vertexCount * 3);
  for (let i = 0; i < I.length; i += 3) {
    const a = I[i]! * 3, b = I[i + 1]! * 3, c = I[i + 2]! * 3;
    const ux = P[b]! - P[a]!, uy = P[b + 1]! - P[a + 1]!, uz = P[b + 2]! - P[a + 2]!, wx = P[c]! - P[a]!, wy = P[c + 1]! - P[a + 1]!, wz = P[c + 2]! - P[a + 2]!;
    const nx = uy * wz - uz * wy, ny = uz * wx - ux * wz, nz = ux * wy - uy * wx;
    for (const vi of [I[i]!, I[i + 1]!, I[i + 2]!]) { acc[vi * 3]! += nx; acc[vi * 3 + 1]! += ny; acc[vi * 3 + 2]! += nz; }
  }
  for (let v = 0; v < m.vertexCount; v++) {
    const o = v * 3;
    let nx = acc[o]!, ny = acc[o + 1]!, nz = acc[o + 2]!;
    const l = Math.hypot(nx, ny, nz);
    if (!(l > 0)) continue;
    nx /= l; ny /= l; nz /= l;
    // keep the field orientation: the winding was chosen so face normals point to the high side already
    m.normals[o] = nx; m.normals[o + 1] = ny; m.normals[o + 2] = nz;
  }
}

/**
 * Taubin λ|μ smoothing of the welded positions (`iterations` rounds), then face normals; `normalIterations`
 * further averaging rounds of the normals. Vertices on the box faces move with their neighbours (as in the
 * prototype); with iterations = 0 only the normals are treated.
 */
export function taubinSmoothMesh(m: IndexedMesh, iterations: number, normalIterations = 0, lambda = 0.5, mu = -0.53): void {
  const acc = new Float32Array(m.vertexCount * 3), deg = new Float32Array(m.vertexCount);
  if (iterations > 0) {
    for (let it = 0; it < iterations; it++) { averagePass(m, m.positions, lambda, acc, deg); averagePass(m, m.positions, mu, acc, deg); }
    faceNormals(m);
  }
  for (let it = 0; it < normalIterations; it++) averagePass(m, m.normals, 1, acc, deg);
  if (normalIterations > 0) for (let v = 0; v < m.vertexCount; v++) {
    const o = v * 3, l = Math.hypot(m.normals[o]!, m.normals[o + 1]!, m.normals[o + 2]!);
    if (l > 0) { m.normals[o]! /= l; m.normals[o + 1]! /= l; m.normals[o + 2]! /= l; }
  }
}

/** convenience: weld → smooth → soup */
export function smoothIsoMesh(m: IsoMesh, iterations: number, normalIterations = 0): IsoMesh {
  if (iterations <= 0 && normalIterations <= 0) return m;
  const w = weldMesh(m);
  taubinSmoothMesh(w, iterations, normalIterations);
  return expandMesh(w);
}

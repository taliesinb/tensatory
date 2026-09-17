// Static vector-field glyphs ("hedgehog" arrows): a vector field sampled on a
// lattice and drawn as one arrow per sample, scaled so that the LONGEST vector
// actually sampled fills (most of) the lattice spacing — glyphs never overlap.
//
// The lattice is the densest packing of the dimension, so a given spacing packs
// the most samples: hexagonal in 2D (every point has six neighbours at the
// same distance), face-centred cubic in 3D (twelve). Both are unions of a few
// interleaved rectangular grids ("cosets": two for hex, four for FCC), so every
// existing sampling path — CPU, GPU read-back, GPU-resident — serves them
// unchanged, and a kernel walks a lattice as a list of DenseGrids.
//
// Glyph geometry lives here (CPU) and is mirrored bit for bit by the fused GPU
// kernel in @tensatory/gpu (glyphs.ts); the agreement test compares the two.

import { Box } from "../geometry/box";
import { DenseGrid } from "../geometry/grid";

export interface Lattice {
  readonly dimCount: number;
  /** nearest-neighbour distance (world units) */
  readonly spacing: number;
  /** the interleaved rectangular grids whose union is the lattice; points are numbered coset by coset, row-major within each */
  readonly cosets: DenseGrid[];
  readonly pointCount: number;
}

/**
 * Coset structure of the lattice with nearest-neighbour distance `s` in `D` dimensions: the period along each axis
 * (the same for every coset) and the cosets' offsets from the anchor.
 *  * 2D hexagonal: rows `s·√3/2` apart, alternate rows shifted by `s/2` → period (s, s√3), offsets (0, 0), (s/2, s√3/2).
 *  * 3D face-centred cubic: cubic cell `a = s·√2` with the four FCC sites → period (a, a, a), offsets 0, (½,½,0)a, (½,0,½)a, (0,½,½)a.
 *  * otherwise: the cubic lattice (one coset).
 */
export function latticeCosets(D: number, s: number): { period: number[]; offsets: number[][] } {
  if (D === 2) return { period: [s, s * Math.sqrt(3)], offsets: [[0, 0], [s / 2, (s * Math.sqrt(3)) / 2]] };
  if (D === 3) {
    const a = s * Math.SQRT2, h = a / 2;
    return { period: [a, a, a], offsets: [[0, 0, 0], [h, h, 0], [h, 0, h], [0, h, h]] };
  }
  return { period: new Array<number>(D).fill(s), offsets: [new Array<number>(D).fill(0)] };
}

/**
 * The lattice points inside `region` (closed), anchored so that `anchor` is a lattice point: anchoring at a fixed
 * world point (the field's box corner) keeps every glyph in place while the region (the view) moves. Empty cosets
 * are omitted; a region thinner than the period along an axis may hold a single row.
 */
export function latticeIn(region: Box, spacing: number, anchor: readonly number[] = region.a): Lattice {
  const D = region.dimCount;
  if (!(spacing > 0)) throw new RangeError(`lattice spacing must be positive, got ${spacing}`);
  const { period, offsets } = latticeCosets(D, spacing);
  const eps = 1e-9;
  const cosets: DenseGrid[] = [];
  let pointCount = 0;
  for (const off of offsets) {
    const size: number[] = [], a: number[] = [], b: number[] = [];
    let empty = false;
    for (let d = 0; d < D; d++) {
      const o = anchor[d]! + off[d]!, h = period[d]!;
      const i0 = Math.ceil((region.a[d]! - o) / h - eps), i1 = Math.floor((region.b[d]! - o) / h + eps);
      if (i1 < i0) { empty = true; break; }
      size.push(i1 - i0 + 1); a.push(o + i0 * h); b.push(o + i1 * h);
    }
    if (empty) continue;
    const g = new DenseGrid(size, new Box(a, b));
    cosets.push(g);
    pointCount += g.sampleCount;
  }
  return { dimCount: D, spacing, cosets, pointCount };
}

/** every lattice point, flat D-dimensional, in lattice order (coset by coset, row-major) */
export function latticePoints(l: Lattice): Float64Array {
  const D = l.dimCount, out = new Float64Array(l.pointCount * D), p = new Float64Array(D);
  let o = 0;
  for (const g of l.cosets) for (let i = 0; i < g.sampleCount; i++) { g.pointInto(i, p); out.set(p, o); o += D; }
  return out;
}

/**
 * Glyph shapes, all within the same length budget L (so neighbours never overlap in any style):
 *  * `arrow` — a shaft of length L centred on the point with two barbs at its tip (3 segments);
 *  * `head` — only the arrowhead: a chevron of length L centred on the point, its tip at `p + L/2·u` (2 segments);
 *  * `triangle` — a solid, narrow triangle with its base centred on the point and its apex where the arrow's tip
 *    would be, `p + L/2·u` (one filled triangle record, not lines).
 */
export type GlyphStyle = "arrow" | "head" | "triangle";
export const GLYPH_STYLES: readonly GlyphStyle[] = ["arrow", "head", "triangle"];

export interface GlyphOptions {
  /** length of the glyph at the maximum norm, as a fraction of the lattice spacing (default 0.9: neighbours never touch) */
  fill?: number;
  /** arrowhead length as a fraction of the shaft (default 0.3) */
  head?: number;
  /** the shape (default "arrow") */
  style?: GlyphStyle;
  /** glyphs shorter than this (world units; the viewer passes a few pixels' worth) are not drawn: visual noise cutoff */
  minLength?: number;
}
export const GLYPH_FILL = 0.9, GLYPH_HEAD = 0.3;
/** the barbs leave the tip at 30° from the shaft: back `head·L` along it, out `HEAD_SPREAD·head·L` sideways */
export const HEAD_SPREAD = 0.55;
/** the `head` chevron: back L from the tip, out `CHEVRON_SPREAD·L` sideways (a 44° opening — narrower than the arrowhead, so the direction reads) */
export const CHEVRON_SPREAD = 0.4;
/** the `triangle` base's half-width as a fraction of its length L/2 (base : length = 0.44) */
export const TRIANGLE_HALF_WIDTH = 0.22;
/** records (segments, or one filled triangle) a glyph of `style` appends at most */
export const glyphSegments = (style: GlyphStyle): number => (style === "head" ? 2 : style === "triangle" ? 1 : 3);

/** the largest finite norm among `vectors` (flat, D per point); 0 when there is none */
export function maxNorm(vectors: ArrayLike<number>, D: number): number {
  let best = 0;
  for (let i = 0; i + D <= vectors.length; i += D) {
    let l2 = 0;
    for (let d = 0; d < D; d++) { const v = vectors[i + d]!; l2 += v * v; }
    if (Number.isFinite(l2) && l2 > best) best = l2;
  }
  return Math.sqrt(best);
}

/** a unit vector perpendicular to the unit vector `u` (D = 2: its left normal; D = 3: normal to `u` and the axis it is least aligned with) */
export function glyphNormal(u: ArrayLike<number>, D: number): number[] {
  if (D === 2) return [-u[1]!, u[0]!];
  if (D === 3) {
    const ax = Math.abs(u[0]!), ay = Math.abs(u[1]!), az = Math.abs(u[2]!);
    const e = ax <= ay && ax <= az ? [1, 0, 0] : ay <= az ? [0, 1, 0] : [0, 0, 1];
    const n = [u[1]! * e[2]! - u[2]! * e[1]!, u[2]! * e[0]! - u[0]! * e[2]!, u[0]! * e[1]! - u[1]! * e[0]!];
    const l = Math.hypot(n[0]!, n[1]!, n[2]!) || 1;
    return [n[0]! / l, n[1]! / l, n[2]! / l];
  }
  const n = new Array<number>(D).fill(0); n[1 % D] = 1; return n;
}

export interface Glyphs {
  /** the polylines (arrow / head styles), flat D-dimensional: arrow [tail, tip, barb] + [tip, barb']; head [barb, tip, barb'] */
  lines: Float64Array[];
  /** the lattice point (index into `points`) each polyline belongs to */
  point: Int32Array;
  /** filled triangles (triangle style), flat `[baseLeft, baseRight, apex]` — 3·D floats per glyph */
  triangles: Float64Array;
  /** the lattice point of each triangle */
  triPoint: Int32Array;
  /** the normalizing norm used (the longest vector sampled), 0 when nothing was drawn */
  maxNorm: number;
}

/**
 * Glyphs for `vectors` sampled at `points` (both flat, D per entry): each is centred on its point and points along
 * its vector, with length `L = fill · spacing · |v| / maxNorm` (see `GlyphStyle` for the shapes). Points with a zero
 * or non-finite vector get no glyph. `maxNorm` defaults to the longest vector present (the "actually sampled"
 * normalization).
 */
export function arrowGlyphs(points: ArrayLike<number>, vectors: ArrayLike<number>, D: number, spacing: number, opts: GlyphOptions & { maxNorm?: number } = {}): Glyphs {
  const fill = opts.fill ?? GLYPH_FILL, head = opts.head ?? GLYPH_HEAD, style = opts.style ?? "arrow", minLength = opts.minLength ?? 0;
  const vmax = opts.maxNorm ?? maxNorm(vectors, D);
  const lines: Float64Array[] = [], point: number[] = [], tris: number[] = [], triPoint: number[] = [];
  const done = (): Glyphs => ({ lines, point: Int32Array.from(point), triangles: Float64Array.from(tris), triPoint: Int32Array.from(triPoint), maxNorm: vmax > 0 ? vmax : 0 });
  if (!(vmax > 0)) return done();
  const n = Math.floor(Math.min(points.length, vectors.length) / D);
  const u = new Array<number>(D);
  for (let i = 0; i < n; i++) {
    let l2 = 0;
    for (let d = 0; d < D; d++) { const v = vectors[i * D + d]!; l2 += v * v; }
    if (!Number.isFinite(l2) || !(l2 > 0)) continue;
    const len = Math.sqrt(l2), L = (fill * spacing * len) / vmax;
    if (L < minLength) continue;
    for (let d = 0; d < D; d++) u[d] = vectors[i * D + d]! / len;
    const nrm = glyphNormal(u, D);
    if (style === "arrow") {
      const shaft = new Float64Array(3 * D), barb = new Float64Array(2 * D);
      for (let d = 0; d < D; d++) {
        const c = points[i * D + d]!, ud = u[d]!, nd = nrm[d]!;
        const tail = c - 0.5 * L * ud, tip = c + 0.5 * L * ud;
        shaft[d] = tail; shaft[D + d] = tip; shaft[2 * D + d] = tip - head * L * ud + HEAD_SPREAD * head * L * nd;
        barb[d] = tip; barb[D + d] = tip - head * L * ud - HEAD_SPREAD * head * L * nd;
      }
      lines.push(shaft, barb); point.push(i, i);
    } else if (style === "head") {
      const chevron = new Float64Array(3 * D);
      for (let d = 0; d < D; d++) {
        const c = points[i * D + d]!, ud = u[d]!, nd = nrm[d]!;
        const tip = c + 0.5 * L * ud, back = tip - L * ud;
        chevron[d] = back + CHEVRON_SPREAD * L * nd; chevron[D + d] = tip; chevron[2 * D + d] = back - CHEVRON_SPREAD * L * nd;
      }
      lines.push(chevron); point.push(i);
    } else {
      const w = TRIANGLE_HALF_WIDTH * 0.5 * L, o = tris.length;
      tris.length = o + 3 * D;
      for (let d = 0; d < D; d++) {
        const c = points[i * D + d]!, ud = u[d]!, nd = nrm[d]!;
        tris[o + d] = c + w * nd; tris[o + D + d] = c - w * nd; tris[o + 2 * D + d] = c + 0.5 * L * ud;
      }
      triPoint.push(i);
    }
  }
  return done();
}

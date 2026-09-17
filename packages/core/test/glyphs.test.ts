import { describe, expect, it } from "vitest";
import { Box, CHEVRON_SPREAD, TRIANGLE_HALF_WIDTH, arrowGlyphs, latticeIn, latticePoints, maxNorm } from "../src";

/** nearest-neighbour distance of a flat point set (brute force; small sets only) */
function nearestNeighbour(pts: Float64Array, D: number): number {
  const n = pts.length / D;
  let best = Infinity;
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
    let d2 = 0;
    for (let d = 0; d < D; d++) { const dd = pts[i * D + d]! - pts[j * D + d]!; d2 += dd * dd; }
    best = Math.min(best, d2);
  }
  return Math.sqrt(best);
}
function neighbourCount(pts: Float64Array, D: number, i: number, dist: number): number {
  const n = pts.length / D;
  let c = 0;
  for (let j = 0; j < n; j++) {
    if (j === i) continue;
    let d2 = 0;
    for (let d = 0; d < D; d++) { const dd = pts[i * D + d]! - pts[j * D + d]!; d2 += dd * dd; }
    if (Math.abs(Math.sqrt(d2) - dist) < 1e-9) c++;
  }
  return c;
}

describe("lattices", () => {
  it("2D: hexagonal — six neighbours at the spacing, denser than a square grid, anchored", () => {
    const box = new Box([-1, -1], [1, 1]);
    const l = latticeIn(box, 0.2);
    expect(l.cosets.length).toBe(2);
    const pts = latticePoints(l);
    expect(pts.length / 2).toBe(l.pointCount);
    expect(nearestNeighbour(pts, 2)).toBeCloseTo(0.2, 9);
    // an interior point has six neighbours at exactly the spacing
    let interior = -1;
    for (let i = 0; i < l.pointCount; i++) if (Math.abs(pts[2 * i]!) < 0.3 && Math.abs(pts[2 * i + 1]!) < 0.3) { interior = i; break; }
    expect(neighbourCount(pts, 2, interior, 0.2)).toBe(6);
    // every point inside the box; the anchor (box corner) is a lattice point
    for (let i = 0; i < l.pointCount; i++) expect(box.contains([pts[2 * i]!, pts[2 * i + 1]!], 1e-9)).toBe(true);
    expect(Array.from(pts).some((_, k) => k % 2 === 0 && Math.abs(pts[k]! + 1) < 1e-12 && Math.abs(pts[k + 1]! + 1) < 1e-12)).toBe(true);
    // ~2/√3 ≈ 1.155 times the points of a square grid of the same spacing (area per point s²·√3/2 vs s²); a fine
    // lattice so the boundary rows do not dominate
    const fine = latticeIn(box, 0.02);
    expect(fine.pointCount / (101 * 101)).toBeGreaterThan(1.1);
    expect(fine.pointCount / (101 * 101)).toBeLessThan(1.2);
  });
  it("2D: a moved region keeps the same points (world anchoring)", () => {
    const a = latticeIn(new Box([0, 0], [1, 1]), 0.13, [0, 0]), b = latticeIn(new Box([0.3, 0.2], [1.3, 1.2]), 0.13, [0, 0]);
    const pa = latticePoints(a), pb = latticePoints(b);
    const key = (x: number, y: number) => `${x.toFixed(9)},${y.toFixed(9)}`;
    const setA = new Set<string>();
    for (let i = 0; i < a.pointCount; i++) setA.add(key(pa[2 * i]!, pa[2 * i + 1]!));
    let shared = 0;
    for (let i = 0; i < b.pointCount; i++) if (setA.has(key(pb[2 * i]!, pb[2 * i + 1]!))) shared++;
    expect(shared).toBeGreaterThan(b.pointCount * 0.4); // the overlap of the two regions
  });
  it("3D: face-centred cubic — twelve neighbours at the spacing", () => {
    const box = new Box([0, 0, 0], [1, 1, 1]);
    const l = latticeIn(box, 0.25);
    expect(l.cosets.length).toBe(4);
    const pts = latticePoints(l);
    expect(nearestNeighbour(pts, 3)).toBeCloseTo(0.25, 9);
    let interior = -1;
    for (let i = 0; i < l.pointCount; i++) if ([0, 1, 2].every((d) => pts[3 * i + d]! > 0.3 && pts[3 * i + d]! < 0.7)) { interior = i; break; }
    expect(interior).toBeGreaterThanOrEqual(0);
    expect(neighbourCount(pts, 3, interior, 0.25)).toBe(12);
    for (let i = 0; i < l.pointCount; i++) expect(box.contains([pts[3 * i]!, pts[3 * i + 1]!, pts[3 * i + 2]!], 1e-9)).toBe(true);
  });
  it("a thin region holds a single row; an empty coset is dropped", () => {
    const l = latticeIn(new Box([0, 0], [1, 0.05]), 0.1);
    expect(l.cosets.length).toBe(1);
    expect(l.cosets[0]!.size).toEqual([11, 1]);
  });
});

describe("arrow glyphs", () => {
  it("normalize to the longest vector, never exceed the fill, skip zero / NaN vectors, colour index per glyph", () => {
    const pts = Float64Array.from([0, 0, 1, 0, 0, 1, 1, 1]);
    const vec = Float64Array.from([2, 0, 0, 1, 0, 0, NaN, 1]);
    expect(maxNorm(vec, 2)).toBe(2);
    const g = arrowGlyphs(pts, vec, 2, 0.5);
    expect(g.maxNorm).toBe(2);
    expect(g.lines.length).toBe(4); // two glyphs (points 0 and 1), two polylines each
    expect(Array.from(g.point)).toEqual([0, 0, 1, 1]);
    // glyph 0: the longest vector, centred on (0, 0), along +x, length 0.9 · 0.5
    const s = g.lines[0]!;
    expect(s[0]).toBeCloseTo(-0.225); expect(s[1]).toBeCloseTo(0); expect(s[2]).toBeCloseTo(0.225); expect(s[3]).toBeCloseTo(0);
    // the barbs sit behind the tip, one either side
    expect(s[4]).toBeLessThan(s[2]!); expect(s[5]).toBeGreaterThan(0);
    const b = g.lines[1]!;
    expect(b[0]).toBeCloseTo(0.225); expect(b[3]).toBeLessThan(0);
    // glyph 1: half the norm → half the length, along +y
    const t = g.lines[2]!;
    expect(Math.hypot(t[2]! - t[0]!, t[3]! - t[1]!)).toBeCloseTo(0.225);
    expect(t[2]! - t[0]!).toBeCloseTo(0);
  });
  it("head: a chevron of length L centred on the point; triangle: base at the point, apex at the arrow's tip", () => {
    const pts = [1, 1], vec = [0, 3]; // L = 0.9 · spacing 1 = 0.9, along +y
    const h = arrowGlyphs(pts, vec, 2, 1, { style: "head" });
    expect(h.lines.length).toBe(1); expect(Array.from(h.point)).toEqual([0]);
    const c = h.lines[0]!; // [barb, tip, barb']
    expect(c[2]).toBeCloseTo(1); expect(c[3]).toBeCloseTo(1.45); // tip at p + L/2
    expect(c[1]).toBeCloseTo(0.55); expect(c[5]).toBeCloseTo(0.55); // barbs L back from the tip: centred on p
    expect(c[0]).toBeCloseTo(1 - CHEVRON_SPREAD * 0.9); expect(c[4]).toBeCloseTo(1 + CHEVRON_SPREAD * 0.9);
    const t = arrowGlyphs(pts, vec, 2, 1, { style: "triangle" });
    expect(t.lines.length).toBe(0); expect(t.triangles.length).toBe(6); expect(Array.from(t.triPoint)).toEqual([0]);
    const r = t.triangles; // [baseLeft, baseRight, apex]
    expect(r[1]).toBeCloseTo(1); expect(r[3]).toBeCloseTo(1); // the base is centred on the point
    expect(Math.abs(r[0]! - r[2]!)).toBeCloseTo(2 * TRIANGLE_HALF_WIDTH * 0.45); // narrow: its half-width is 0.22 of its length
    expect(r[4]).toBeCloseTo(1); expect(r[5]).toBeCloseTo(1.45); // apex where the arrow's tip is
  });
  it("3D barbs are perpendicular to the shaft", () => {
    const g = arrowGlyphs([0, 0, 0], [1, 2, 3], 3, 1);
    const s = g.lines[0]!;
    const u = [s[3]! - s[0]!, s[4]! - s[1]!, s[5]! - s[2]!];
    const barb = [s[6]! - s[3]!, s[7]! - s[4]!, s[8]! - s[5]!];
    // the barb's component across the shaft is non-zero and its along-shaft component points back
    const dot = u[0]! * barb[0]! + u[1]! * barb[1]! + u[2]! * barb[2]!;
    expect(dot).toBeLessThan(0);
    const cross = Math.hypot(u[1]! * barb[2]! - u[2]! * barb[1]!, u[2]! * barb[0]! - u[0]! * barb[2]!, u[0]! * barb[1]! - u[1]! * barb[0]!);
    expect(cross).toBeGreaterThan(0);
  });
});

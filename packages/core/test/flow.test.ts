import { describe, expect, it } from "vitest";
import { Box, DenseGrid, SymbolicVectorFieldData, boxBlur, buildScalarFieldData, buildVectorFieldData, integrateStreamlines, taubinSmooth } from "../src";

describe("boxBlur", () => {
  it("averages in-range neighbours per axis", () => {
    const g = new DenseGrid([3, 1], Box.fromSpec([[0, 1], [0, 0]]));
    expect(Array.from(boxBlur(g, [0, 3, 6], 1))).toEqual([1.5, 3, 4.5]);
    expect(Array.from(boxBlur(g, [0, 3, 6], 0))).toEqual([0, 3, 6]);
  });
});

describe("taubinSmooth", () => {
  it("keeps open endpoints fixed and closes loops", () => {
    const zig = Float64Array.from([0, 0, 1, 1, 2, 0, 3, 1, 4, 0]);
    const s = taubinSmooth(zig, 4);
    expect(s[0]).toBe(0); expect(s[1]).toBe(0); expect(s[8]).toBe(4); expect(s[9]).toBe(0);
    expect(Math.abs(s[3]! - 0.5)).toBeLessThan(0.5);
    const sq = Float64Array.from([0, 0, 1, 0, 1, 1, 0, 1, 0, 0]);
    const c = taubinSmooth(sq, 3);
    expect(c[0]).toBe(c[8]); expect(c[1]).toBe(c[9]);
  });
});

describe("streamlines", () => {
  it("follow a constant field at unit speed and stop at the box", () => {
    const v = buildVectorFieldData({ type: "symbolicv", box: [[0, 1], [0, 1]], expr: { op: "constv", value: [2, 0] } }, 2);
    const lines = integrateStreamlines(v, { count: 4, maxSteps: 1000, step: 0.05, seed: 1 });
    expect(lines.length).toBe(4);
    for (const l of lines) {
      const n = l.points.length / 2;
      // straight horizontal, spanning (almost) the whole box
      for (let i = 0; i < n; i++) expect(l.points[2 * i + 1]).toBeCloseTo(l.points[1]!, 12);
      expect(l.points[2 * (n - 1)]! - l.points[0]!).toBeGreaterThan(0.85);
      expect(l.length).toBeCloseTo((n - 1) * 0.05);
    }
  });
  it("descend a bowl's gradient towards the minimum", () => {
    const f = buildScalarFieldData({ type: "symbolic", box: [[-1, 1], [-1, 1]], expr: { op: "norm", vec: { op: "coordv" } } }, 2);
    const g = new SymbolicVectorFieldData({ k: "grad", s: { k: "arg", name: "f" } }, 2, { scalars: { f }, vectors: {} });
    const lines = integrateStreamlines(g, { count: 9, maxSteps: 200, step: 0.02, sign: -1, seed: 3 });
    for (const l of lines) {
      const n = l.points.length / 2;
      const rEnd = Math.hypot(l.points[2 * (n - 1)]!, l.points[2 * (n - 1) + 1]!);
      const rStart = Math.hypot(l.points[0]!, l.points[1]!);
      expect(rEnd).toBeLessThan(0.05); // the descent end reaches the centre
      expect(rStart).toBeGreaterThan(rEnd);
    }
  });
});

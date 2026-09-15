import { describe, expect, it } from "vitest";
import { Box, DenseGrid, buildScalarFieldData, contourField, isoContours, projectToLevel } from "../src";

const x = { op: "coord", index: 0 } as const;
const y = { op: "coord", index: 1 } as const;
const r2 = buildScalarFieldData({ type: "symbolic", box: [[-1, 1], [-1, 1]], expr: { op: "add", vals: [{ op: "square", val: x }, { op: "square", val: y }] } }, 2);

describe("exact isolines", () => {
  it("projects a point onto the level set along the gradient", () => {
    const q = projectToLevel(r2, [0.3, 0.1], 0.25, 1)!;
    expect(Math.hypot(q[0]!, q[1]!)).toBeCloseTo(0.5, 10);
    expect(q[1]! / q[0]!).toBeCloseTo(1 / 3, 6); // moved radially
    expect(projectToLevel(r2, [0, 0], 0.25, 1)).toBeUndefined(); // gradient vanishes
    expect(projectToLevel(r2, [0.3, 0.1], 0.25, 0.01)).toBeUndefined(); // too far
    // a point on the box boundary stays on it
    const b = projectToLevel(r2, [1, 0.3], 1.1, 1)!;
    expect(b[0]).toBe(1);
    expect(b[1]).toBeCloseTo(Math.sqrt(0.1), 10);
  });

  it("puts every vertex on the true circle and bounds the chord error", () => {
    const g = new DenseGrid([9, 9], r2.box); // very coarse seed
    const vals = r2.sampleOn(g);
    const tol = 1e-4;
    const res = contourField(r2, g, vals, 0.25, { tolerance: tol });
    expect(res.method).toBe("exact");
    expect(res.lines.length).toBe(1);
    expect(res.maxResidual).toBeLessThan(1e-10);
    const l = res.lines[0]!;
    // closed
    expect(l[0]).toBeCloseTo(l[l.length - 2]!, 12);
    expect(l[1]).toBeCloseTo(l[l.length - 1]!, 12);
    // chord sagitta of a circle of radius 0.5: s = R(1 - cos(θ/2)) <= tol
    for (let i = 0; i + 3 < l.length; i += 2) {
      expect(Math.hypot(l[i]!, l[i + 1]!)).toBeCloseTo(0.5, 9);
      const chord = Math.hypot(l[i + 2]! - l[i]!, l[i + 3]! - l[i + 1]!);
      const sagitta = 0.5 - Math.sqrt(Math.max(0, 0.25 - (chord / 2) ** 2));
      expect(sagitta).toBeLessThanOrEqual(tol * 1.01);
    }
    // far more vertices than the coarse seed, far fewer than needed by uniform sampling
    const seed = isoContours(g, vals, 0.25)[0]!;
    expect(l.length).toBeGreaterThan(seed.length * 3);
    expect(res.vertexCount).toBeLessThan(400);
  });

  it("handles a contour leaving the box and a level through a saddle", () => {
    const saddle = buildScalarFieldData({ type: "symbolic", box: [[-1, 1], [-1, 1]], expr: { op: "sub", vals: [{ op: "square", val: x }, { op: "square", val: y }] } }, 2);
    const g = new DenseGrid([16, 16], saddle.box);
    const vals = saddle.sampleOn(g);
    const res = contourField(saddle, g, vals, 0.3, { tolerance: 1e-5 });
    expect(res.maxResidual).toBeLessThan(1e-9);
    // both branches hit the box edge x = ±1 exactly
    for (const l of res.lines) {
      expect(Math.abs(l[0]!)).toBeCloseTo(1, 12);
      expect(Math.abs(l[l.length - 2]!)).toBeCloseTo(1, 12);
    }
    // through the saddle itself: must not throw; vertices at the origin keep their seed position
    const res0 = contourField(saddle, g, vals, 0, { tolerance: 1e-5 });
    expect(res0.lines.length).toBeGreaterThan(0);
    expect(Number.isFinite(res0.maxResidual)).toBe(true);
  });

  it("falls back to marching squares for sampled fields", () => {
    const dense = buildScalarFieldData({ type: "dense", box: [[-1, 1], [-1, 1]], samples: { type: "symbolic", shape: [9, 9], origin: [4, 4], expr: { op: "norm", vec: { op: "coordv" } } } }, 2);
    const res = contourField(dense, dense.samplePoints!, dense.sampleOn(dense.samplePoints!), 2, { tolerance: 1e-4 });
    expect(res.method).toBe("linear");
    expect(Number.isNaN(res.maxResidual)).toBe(true);
    expect(res.lines.length).toBe(1);
  });

  it("works for arbitrary boxes (non-unit spacing)", () => {
    const f = buildScalarFieldData({ type: "symbolic", box: [[0, 100], [-5, 5]], expr: { op: "add", vals: [{ op: "mul", vals: [0.01, x] }, { op: "sin", val: y }] } }, 2);
    const g = new DenseGrid([12, 12], f.box);
    const res = contourField(f, g, f.sampleOn(g), 0.7, { tolerance: 1e-3 });
    expect(res.maxResidual).toBeLessThan(1e-9);
    expect(res.lines.length).toBeGreaterThan(0);
    expect(new Box([0, -5], [100, 5]).contains(res.lines[0]!.subarray(0, 2), 1e-9)).toBe(true);
  });
});

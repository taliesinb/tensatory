import { describe, expect, it } from "vitest";
import { Box, DenseGrid, SymbolicVectorFieldData, buildScalarFieldData, contourField, marchingTetrahedra, projectToLevel, sliceScalarField } from "../src";

const sphereGrid = (n: number) => new DenseGrid([n, n, n], new Box([-2, -2, -2], [2, 2, 2]));
const sample = (g: DenseGrid, f: (x: number, y: number, z: number) => number) => {
  const v = new Float64Array(g.sampleCount);
  for (let p = 0; p < g.sampleCount; p++) { const [x, y, z] = g.point(p) as [number, number, number]; v[p] = f(x, y, z); }
  return v;
};

describe("marching tetrahedra", () => {
  const g = sphereGrid(33);
  const ball = sample(g, (x, y, z) => x * x + y * y + z * z);

  it("the level set of x²+y²+z² at 1 is the unit sphere: vertices on it, normals radial, closed and oriented", () => {
    const m = marchingTetrahedra(g, ball, 1);
    expect(m.triangleCount).toBeGreaterThan(500);
    let area = 0, worstR = 0, worstN = 1;
    for (let t = 0; t < m.triangleCount; t++) {
      const P = [0, 1, 2].map((v) => [0, 1, 2].map((k) => m.positions[t * 9 + v * 3 + k]!));
      const N = [0, 1, 2].map((v) => [0, 1, 2].map((k) => m.normals[t * 9 + v * 3 + k]!));
      for (let v = 0; v < 3; v++) {
        const r = Math.hypot(...P[v]!);
        worstR = Math.max(worstR, Math.abs(r - 1));
        const dot = (P[v]![0]! * N[v]![0]! + P[v]![1]! * N[v]![1]! + P[v]![2]! * N[v]![2]!) / r;
        worstN = Math.min(worstN, dot);
      }
      // geometric normal (winding) agrees with the shading normal
      const u = [0, 1, 2].map((k) => P[1]![k]! - P[0]![k]!), w = [0, 1, 2].map((k) => P[2]![k]! - P[0]![k]!);
      const gn = [u[1]! * w[2]! - u[2]! * w[1]!, u[2]! * w[0]! - u[0]! * w[2]!, u[0]! * w[1]! - u[1]! * w[0]!];
      const l = Math.hypot(...gn);
      area += l / 2;
      if (l > 1e-9) expect(gn[0]! * N[0]![0]! + gn[1]! * N[0]![1]! + gn[2]! * N[0]![2]!).toBeGreaterThan(0);
    }
    expect(worstR).toBeLessThan(0.01); // linear interpolation of a quadratic, h = 0.125
    expect(worstN).toBeGreaterThan(0.98); // central-difference normals
    expect(Math.abs(area - 4 * Math.PI) / (4 * Math.PI)).toBeLessThan(0.02);
  });

  it("an exact gradient gives exactly radial normals; colourAt / colour label the vertices", () => {
    const m = marchingTetrahedra(g, ball, 1, { gradient: (p) => [2 * p[0]!, 2 * p[1]!, 2 * p[2]!], colourAt: (p) => p[2]! });
    for (let v = 0; v < m.triangleCount * 3; v++) {
      const x = m.positions[v * 3]!, y = m.positions[v * 3 + 1]!, z = m.positions[v * 3 + 2]!, r = Math.hypot(x, y, z);
      expect(m.normals[v * 3]!).toBeCloseTo(x / r, 5); expect(m.normals[v * 3 + 2]!).toBeCloseTo(z / r, 5);
      expect(m.values![v]!).toBeCloseTo(z, 5);
    }
    const zs = sample(g, (_x, _y, z) => z);
    const m2 = marchingTetrahedra(g, ball, 1, { colour: zs });
    for (let v = 0; v < m2.triangleCount * 3; v++) expect(m2.values![v]!).toBeCloseTo(m2.positions[v * 3 + 2]!, 5); // linear along edges: exact
  });

  it("levels outside the range give no triangles; NaN cells are skipped", () => {
    expect(marchingTetrahedra(g, ball, -1).triangleCount).toBe(0);
    expect(marchingTetrahedra(g, ball, 100).triangleCount).toBe(0);
    const holed = Float64Array.from(ball);
    for (let p = 0; p < g.sampleCount; p++) if (g.point(p)[0]! > 0) holed[p] = NaN;
    const m = marchingTetrahedra(g, holed, 1);
    expect(m.triangleCount).toBeGreaterThan(0);
    for (let v = 0; v < m.triangleCount * 3; v++) expect(m.positions[v * 3]!).toBeLessThanOrEqual(0.125 + 1e-9);
  });
});

describe("exact projection and slices", () => {
  const x = { op: "coord", index: 0 } as const, y = { op: "coord", index: 1 } as const, z = { op: "coord", index: 2 } as const;
  const ball = buildScalarFieldData({ type: "symbolic", box: [[-2, 2], [-2, 2], [-2, 2]], expr: { op: "add", vals: [{ op: "square", val: x }, { op: "square", val: y }, { op: "square", val: z }] } }, 3);
  const g = sphereGrid(17);
  const vals = ball.sampleOn(g);

  it("projected vertices lie on the level set to ~1e-12 and normals are the exact gradient there", () => {
    const h = Math.hypot(...g.spacing);
    const grad = new SymbolicVectorFieldData({ k: "grad", s: { k: "arg", name: "f" } }, 3, { scalars: { f: ball }, vectors: {} });
    const m = marchingTetrahedra(g, vals, 1, { project: (p) => projectToLevel(ball, p, 1, h), gradient: (p) => grad.value(p) });
    expect(m.triangleCount).toBeGreaterThan(100);
    for (let v = 0; v < m.triangleCount * 3; v++) {
      const px = m.positions[v * 3]!, py = m.positions[v * 3 + 1]!, pz = m.positions[v * 3 + 2]!;
      expect(Math.abs(px * px + py * py + pz * pz - 1)).toBeLessThan(1e-6); // f32 storage of the exact point
      expect(m.normals[v * 3]!).toBeCloseTo(px, 5);
    }
  });

  it("sliceScalarField restricts a symbolic field to a plane with exact derivatives", () => {
    const s = sliceScalarField(ball, 1, 0.5); // y = 0.5 → x² + z² + 0.25 on (x, z)
    expect(s.dimCount).toBe(2);
    expect(s.box.intervals).toEqual([[-2, 2], [-2, 2]]);
    expect(s.value([1, 2])).toBeCloseTo(1 + 4 + 0.25, 12);
    expect(s.derivative(0).value([1, 2])).toBeCloseTo(2, 12);
    expect(s.derivative(1).value([1, 2])).toBeCloseTo(4, 12);
    expect(s.kind).toBe("symbolic");
    // exact contouring of the slice: the circle x² + z² = 0.75
    const g2 = new DenseGrid([25, 25], s.box);
    const r = contourField(s, g2, s.sampleOn(g2), 1, { tolerance: 1e-3 });
    expect(r.method).not.toBe("linear");
    for (const l of r.lines) for (let i = 0; i + 1 < l.length; i += 2) expect(Math.abs(Math.hypot(l[i]!, l[i + 1]!) - Math.sqrt(0.75))).toBeLessThan(1e-6);
  });
});

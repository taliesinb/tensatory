// Fused marching tetrahedra: the appended triangle set equals core's (as a multiset of vertices).

import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { DenseGrid, DenseScalarFieldData, SymbolicVectorFieldData, buildScalarFieldData, marchingTetrahedra, projectToLevel, type IsoMesh } from "@tensatory/core";
import { GpuBackend, allocMesh, fusedIsosurface, readMesh, sampleResident, uploadGrid, gpuStats } from "../src";

let gpu: GpuBackend | undefined;
beforeAll(async () => { gpu = await GpuBackend.create(); });
afterAll(async () => { gpu?.destroy(); await new Promise((r) => setTimeout(r, 50)); });

const x = { op: "coord", index: 0 } as const, y = { op: "coord", index: 1 } as const, z = { op: "coord", index: 2 } as const;
const ball = buildScalarFieldData({ type: "symbolic", box: [[-2, 2], [-2, 2], [-2, 2]], expr: { op: "add", vals: [{ op: "square", val: x }, { op: "square", val: y }, { op: "square", val: z }] } }, 3);
const gyroid = buildScalarFieldData({ type: "symbolic", box: [[-3, 3], [-3, 3], [-3, 3]], expr: { op: "add", vals: [
  { op: "mul", vals: [{ op: "sin", val: x }, { op: "cos", val: y }] }, { op: "mul", vals: [{ op: "sin", val: y }, { op: "cos", val: z }] }, { op: "mul", vals: [{ op: "sin", val: z }, { op: "cos", val: x }] }] } }, 3);
const gradOf = (f: typeof ball) => new SymbolicVectorFieldData({ k: "grad", s: { k: "arg", name: "f" } }, 3, { scalars: { f }, vectors: {} });

/** vertices sorted lexicographically by (position, normal, value), rounded to `digits` */
function canonical(m: IsoMesh, digits: number): string[] {
  const rows: string[] = [];
  const r = (v: number) => (Math.abs(v) < 0.5 * 10 ** -digits ? 0 : v).toFixed(digits);
  for (let v = 0; v < m.triangleCount * 3; v++) {
    rows.push([0, 1, 2].map((k) => r(m.positions[v * 3 + k]!)).join(",") + "|" + [0, 1, 2].map((k) => r(m.normals[v * 3 + k]!)).join(",") + "|" + r(m.values?.[v] ?? 0));
  }
  return rows.sort();
}
/** compare two canonical vertex lists allowing f32-scale differences */
function expectSame(a: IsoMesh, b: IsoMesh, digits: number): void {
  expect(b.triangleCount).toBe(a.triangleCount);
  const ca = canonical(a, digits), cb = canonical(b, digits);
  let mismatches = 0;
  for (let i = 0; i < ca.length; i++) if (ca[i] !== cb[i]) mismatches++;
  // rounding at the last digit may reorder a few near-tie rows: allow a small fraction
  expect(mismatches / Math.max(1, ca.length)).toBeLessThan(0.01);
}

describe("fused isosurface", () => {
  it("symbolic ball: exact-gradient normals and a colour field agree with core", async () => {
    if (!gpu) return;
    const g = new DenseGrid([21, 19, 17], ball.box);
    const values = await sampleResident(gpu, ball, g);
    const vals = ball.sampleOn(g);
    const grad = gradOf(ball);
    const colour = buildScalarFieldData({ type: "symbolic", box: ball.box.intervals, expr: { op: "mul", vals: [x, z] } }, 3);
    const kernel = fusedIsosurface(gpu, values, { field: ball, exact: false, colour });
    const mesh = allocMesh(gpu, kernel.capacity);
    await kernel.run(mesh, 1.3);
    const got = await readMesh(gpu, mesh);
    const want = marchingTetrahedra(g, vals, 1.3, { gradient: (p) => grad.value(p), colourAt: (p) => colour.value(p)! });
    expect(got.triangleCount).toBeGreaterThan(500);
    expectSame(want, got, 4);
    mesh.destroy(); values.destroy();
  });

  it("sampled gyroid: central-difference normals agree with core; NaN cells skipped", async () => {
    if (!gpu) return;
    const g = new DenseGrid([24, 22, 20], gyroid.box);
    const vals = Float64Array.from(gyroid.sampleOn(g));
    for (let p = 0; p < g.sampleCount; p++) if (g.point(p)[0]! > 2.2) vals[p] = NaN;
    const dense = new DenseScalarFieldData(g, vals);
    const values = uploadGrid(gpu, g, vals, 1);
    const kernel = fusedIsosurface(gpu, values);
    const mesh = allocMesh(gpu, kernel.capacity);
    await kernel.run(mesh, 0);
    const got = await readMesh(gpu, mesh);
    const want = marchingTetrahedra(g, dense.data, 0);
    expect(got.triangleCount).toBeGreaterThan(1000);
    for (let v = 0; v < got.triangleCount * 3; v++) expect(got.positions[v * 3]!).toBeLessThan(2.3);
    expectSame(want, got, 4);
    // a second level into the same (reset) mesh
    const { resetMesh } = await import("../src");
    resetMesh(gpu, mesh);
    await kernel.run(mesh, 0.7);
    const got2 = await readMesh(gpu, mesh);
    expectSame(marchingTetrahedra(g, dense.data, 0.7), got2, 4);
    mesh.destroy(); values.destroy();
  });

  it("exact projection: vertices on the unit sphere on both backends, and they agree", async () => {
    if (!gpu) return;
    const g = new DenseGrid([15, 14, 13], ball.box);
    const values = await sampleResident(gpu, ball, g);
    const vals = ball.sampleOn(g);
    const kernel = fusedIsosurface(gpu, values, { field: ball });
    const mesh = allocMesh(gpu, kernel.capacity);
    await kernel.run(mesh, 1);
    const got = await readMesh(gpu, mesh);
    const h = Math.hypot(...g.spacing), grad = gradOf(ball);
    const want = marchingTetrahedra(g, vals, 1, { project: (p) => projectToLevel(ball, p, 1, h), gradient: (p) => grad.value(p) });
    expect(got.triangleCount).toBe(want.triangleCount);
    for (let v = 0; v < got.triangleCount * 3; v++) {
      const r = Math.hypot(got.positions[v * 3]!, got.positions[v * 3 + 1]!, got.positions[v * 3 + 2]!);
      expect(Math.abs(r - 1)).toBeLessThan(2e-5); // f32 Newton
    }
    expectSame(want, got, 3);
    mesh.destroy(); values.destroy();
  });

  it("gpuStats works on a 3D resident grid", async () => {
    if (!gpu) return;
    const g = new DenseGrid([17, 17, 17], ball.box);
    const values = await sampleResident(gpu, ball, g);
    const st = await gpuStats(gpu, values);
    expect(st.min).toBeCloseTo(0, 5);
    expect(st.max).toBeCloseTo(12, 4);
    values.destroy();
  });
});

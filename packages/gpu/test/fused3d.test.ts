// Fused 3D streamlines agree with core's integration through the same trilinear vector grid.

import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { DenseGrid, DenseVectorFieldData, buildVectorFieldData, evenlySpacedStreamlines, integrateFromSeeds, streamlineSeeds, buildScalarFieldData } from "@tensatory/core";
import { GpuBackend, SEG3_FLOATS, allocSegments3, fusedStreamlines3, readSegments3, uploadGrid } from "../src";

let gpu: GpuBackend | undefined;
beforeAll(async () => { gpu = await GpuBackend.create(); });
afterAll(async () => { gpu?.destroy(); await new Promise((r) => setTimeout(r, 50)); });

const x = { op: "coord", index: 0 } as const, y = { op: "coord", index: 1 } as const, z = { op: "coord", index: 2 } as const;
const swirl = buildVectorFieldData({ type: "symbolicv", box: [[-1, 1], [-1, 1], [-1, 1]], expr: { op: "compv", coeffs: [{ op: "mul", vals: [-1, y] }, x, { op: "mul", vals: [0.3, { op: "add", vals: [1, { op: "mul", vals: [0.2, z] }] }] }] } }, 3);
const colour = buildScalarFieldData({ type: "symbolic", box: [[-1, 1], [-1, 1], [-1, 1]], expr: { op: "add", vals: [x, { op: "mul", vals: [2, z] }] } }, 3);

/** segments as [phase, arc, ax, ay, az, bx, by, bz, ca, cb] sorted by (phase, arc) */
function canonical(segs: Float32Array): number[][] {
  const rows: number[][] = [];
  for (let i = 0; i + SEG3_FLOATS <= segs.length; i += SEG3_FLOATS) rows.push([segs[i + 10]!, segs[i + 8]!, segs[i]!, segs[i + 1]!, segs[i + 2]!, segs[i + 4]!, segs[i + 5]!, segs[i + 6]!, segs[i + 3]!, segs[i + 7]!]);
  return rows.sort((a, b) => a[0]! - b[0]! || a[1]! - b[1]!);
}

describe("fused 3D streamlines", () => {
  it("stratified seeds both ways: same lines as core (trilinear grid, RK4), colour at the ends", async () => {
    if (!gpu) return;
    const g = new DenseGrid([17, 15, 13], swirl.box);
    const vals = swirl.sampleOn(g);
    const dense = new DenseVectorFieldData(g, vals);
    const grid = uploadGrid(gpu, g, vals, 3);
    const seeds = streamlineSeeds(swirl.box, 60, 4242);
    const opts = { maxSteps: 40, step: 0.03, sign: 1 as const, box: swirl.box, bidirectional: true };
    const kernel = fusedStreamlines3(gpu, grid, seeds, opts, colour);
    const segs = allocSegments3(gpu, kernel.capacity, true);
    await kernel.run(segs);
    const got = canonical(await readSegments3(gpu, segs));
    const lines = integrateFromSeeds(dense, seeds, opts);
    const want: number[][] = [];
    for (const l of lines) for (let i = 0; i + 5 < l.points.length; i += 3) {
      const a = l.points.subarray(i, i + 3), b = l.points.subarray(i + 3, i + 6);
      want.push([l.phase, (i / 3) * opts.step, a[0]!, a[1]!, a[2]!, b[0]!, b[1]!, b[2]!, colour.value(Array.from(a))!, colour.value(Array.from(b))!]);
    }
    want.sort((a, b) => a[0]! - b[0]! || a[1]! - b[1]!);
    expect(got.length).toBe(want.length);
    let worst = 0;
    for (let i = 0; i < want.length; i++) for (let k = 2; k < 10; k++) worst = Math.max(worst, Math.abs(got[i]![k]! - want[i]![k]!));
    expect(worst).toBeLessThan(3e-4);
    segs.destroy(); grid.destroy();
  });

  it("re-integrates a 3D Jobard–Lefer plan within its budgets", async () => {
    if (!gpu) return;
    const g = new DenseGrid([17, 15, 13], swirl.box);
    const vals = swirl.sampleOn(g);
    const dense = new DenseVectorFieldData(g, vals);
    const grid = uploadGrid(gpu, g, vals, 3);
    const opts = { count: 200, maxSteps: 50, step: 0.03, sign: 1 as const, box: swirl.box, bidirectional: false };
    const plan = evenlySpacedStreamlines(dense, { ...opts, mode: "evenly-spaced", seed: 3 });
    const kernel = fusedStreamlines3(gpu, grid, plan.seeds, opts);
    const segs = allocSegments3(gpu, kernel.capacity, true);
    await kernel.run(segs);
    const got = await readSegments3(gpu, segs);
    const wantSegs = plan.lines.reduce((s, l) => s + l.points.length / 3 - 1, 0);
    expect(got.length / SEG3_FLOATS).toBe(wantSegs);
    segs.destroy(); grid.destroy();
  });
});

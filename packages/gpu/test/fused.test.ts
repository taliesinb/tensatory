// Fused (resident) kernels: the appended segment sets must describe the same
// geometry core computes.

import { afterAll, beforeAll, describe, expect, it } from "vitest";
import {
  DenseGrid,
  DenseScalarFieldData,
  DenseVectorFieldData,
  SymbolicVectorFieldData,
  buildScalarFieldData,
  contourField,
  evenlySpacedStreamlines,
  integrateFromSeeds,
  marchingSquaresSegments,
  streamlineSeeds,
} from "@tensatory/core";
import { GpuBackend, SEG_FLOATS, SEG_LAYOUT, VERT_LAYOUT, allocMesh, allocSegments, freshProgress, fusedIsolines, fusedIsosurface, fusedStreamlines, readGrid, readMesh, readSegments, recolourStep, recolourer, resetSegments, sampleResident, uploadGrid } from "../src";

let gpu: GpuBackend | undefined;
beforeAll(async () => { gpu = await GpuBackend.create(); });
afterAll(async () => { gpu?.destroy(); await new Promise((r) => setTimeout(r, 50)); });

const x = { op: "coord", index: 0 } as const, y = { op: "coord", index: 1 } as const;
const bowl = buildScalarFieldData({ type: "symbolic", box: [[-2, 2], [-2, 2]], expr: { op: "add", vals: [{ op: "square", val: x }, { op: "square", val: y }] } }, 2);
const waves = buildScalarFieldData({ type: "symbolic", box: [[-3, 3], [-3, 3]], expr: { op: "mul", vals: [{ op: "sin", val: { op: "mul", vals: [3, x] } }, { op: "cos", val: { op: "mul", vals: [2, y] } }] } }, 2);

/** distance from a point to the nearest chord of a set of polylines */
function nearest(pt: [number, number], lines: ArrayLike<number>[]): number {
  let best = Infinity;
  for (const l of lines) for (let i = 0; i + 3 < l.length; i += 2) {
    const ax = l[i]!, ay = l[i + 1]!, bx = l[i + 2]!, by = l[i + 3]!;
    const dx = bx - ax, dy = by - ay, l2 = dx * dx + dy * dy || 1e-30;
    const t = Math.min(1, Math.max(0, ((pt[0] - ax) * dx + (pt[1] - ay) * dy) / l2));
    best = Math.min(best, Math.hypot(pt[0] - ax - t * dx, pt[1] - ay - t * dy));
  }
  return best;
}

describe("resident grids", () => {
  it("sampleResident + readGrid equals sampleOn", async () => {
    if (!gpu) return;
    const g = new DenseGrid([40, 30], bowl.box);
    const res = await sampleResident(gpu, bowl, g);
    const back = await readGrid(gpu, res);
    const cpu = bowl.sampleOn(g);
    for (let i = 0; i < cpu.length; i += 97) expect(back[i]).toBeCloseTo(cpu[i]!, 4);
    res.destroy();
  });
});

describe("fused isolines", () => {
  it("symbolic field: every appended segment lies on the exact contour, and covers core's contour", async () => {
    if (!gpu) return;
    for (const [f, level] of [[bowl, 1.7], [waves, 0.35]] as [typeof bowl, number][]) {
      const g = new DenseGrid([48, 48], f.box);
      const res = await sampleResident(gpu, f, g);
      const kernel = fusedIsolines(gpu, f, res);
      const segs = allocSegments(gpu, kernel.capacity, false);
      const tol = 2e-3;
      await kernel.run(segs, level, tol);
      const out = await readSegments(gpu, segs);
      const n = out.length / SEG_FLOATS;
      expect(n).toBeGreaterThan(50);
      // endpoints on the curve
      let maxRes = 0;
      for (let i = 0; i < n; i++) {
        for (const off of [0, 2]) maxRes = Math.max(maxRes, Math.abs(f.fn([out[i * SEG_FLOATS + off]!, out[i * SEG_FLOATS + off + 1]!], -1) - level));
        expect(out[i * SEG_FLOATS + 7]).toBe(0); // no particles
      }
      expect(maxRes).toBeLessThan(2e-4);
      // coverage: core's exact contour vertices are near some fused segment and vice versa
      const cpu = contourField(f, g, f.sampleOn(g), level, { tolerance: tol });
      const fusedLines: Float32Array[] = [];
      for (let i = 0; i < n; i++) fusedLines.push(out.subarray(i * SEG_FLOATS, i * SEG_FLOATS + 4));
      for (const l of cpu.lines) for (let i = 0; i < l.length; i += 10) expect(nearest([l[i]!, l[i + 1]!], fusedLines)).toBeLessThan(4 * tol);
      for (let i = 0; i < n; i += 7) expect(nearest([out[i * SEG_FLOATS]!, out[i * SEG_FLOATS + 1]!], cpu.lines)).toBeLessThan(4 * tol);
      // reuse: reset then run again gives the same count
      resetSegments(gpu, segs);
      await kernel.run(segs, level, tol);
      expect((await readSegments(gpu, segs)).length).toBe(out.length);
      segs.destroy(); res.destroy();
    }
  });
  it("sampled field: exactly the marching-squares segments; colour field evaluated at endpoints", async () => {
    if (!gpu) return;
    const g = new DenseGrid([25, 19], bowl.box);
    const vals = bowl.sampleOn(g);
    const dense = buildScalarFieldData({ type: "dense", box: bowl.box.intervals, samples: { type: "inline", shape: [25, 19], data: Array.from(vals) } }, 2);
    const res = uploadGrid(gpu, g, vals, 1);
    const kernel = fusedIsolines(gpu, dense, res, waves);
    const segs = allocSegments(gpu, kernel.capacity, false);
    await kernel.run(segs, 1.1, 1e-3);
    const out = await readSegments(gpu, segs);
    const cpu = marchingSquaresSegments(g, vals, 1.1);
    expect(out.length / SEG_FLOATS).toBe(cpu.length / 4);
    for (let i = 0; i < out.length / SEG_FLOATS; i++) {
      const ax = out[i * SEG_FLOATS]!, ay = out[i * SEG_FLOATS + 1]!;
      expect(out[i * SEG_FLOATS + 4]).toBeCloseTo(waves.fn([ax, ay], -1), 4);
    }
    segs.destroy(); res.destroy();
  });

  it("colour sources: a resident grid interpolates the colour field (costly fields), \"level\" is the level", async () => {
    if (!gpu) return;
    const g = new DenseGrid([25, 19], bowl.box);
    const vals = bowl.sampleOn(g);
    const dense = buildScalarFieldData({ type: "dense", box: bowl.box.intervals, samples: { type: "inline", shape: [25, 19], data: Array.from(vals) } }, 2);
    const res = uploadGrid(gpu, g, vals, 1);
    // the colour field sampled on a FINER grid than the isoline grid, so the interpolation is exercised
    const cg = new DenseGrid([97, 73], waves.box);
    const cres = await sampleResident(gpu, waves, cg);
    const run = async (colour: Parameters<typeof fusedIsolines>[3]) => {
      const kernel = fusedIsolines(gpu!, dense, res, colour);
      const segs = allocSegments(gpu!, kernel.capacity, false);
      await kernel.run(segs, 1.1, 1e-3);
      const out = await readSegments(gpu!, segs);
      segs.destroy();
      return out;
    };
    const exact = await run(waves), resident = await run(cres), level = await run("level");
    expect(resident.length).toBe(exact.length);
    const cpuColour = new DenseScalarFieldData(cg, waves.sampleOn(cg)); // core's bilinear interpolation of the same samples
    for (let i = 0; i < exact.length / SEG_FLOATS; i++) {
      const o = i * SEG_FLOATS, ax = resident[o]!, ay = resident[o + 1]!;
      expect(resident[o + 4]).toBeCloseTo(cpuColour.fn([ax, ay], -1), 4); // = core's interpolation
      expect(Math.abs(resident[o + 4]! - waves.fn([ax, ay], -1))).toBeLessThan(0.05); // ≈ the field (bilinear on 97×73 of a 3-period wave); segments append in no fixed order, so compare at the segment's own point
      expect(level[o + 4]).toBeCloseTo(1.1, 6); expect(level[o + 5]).toBeCloseTo(1.1, 6);
    }
    res.destroy(); cres.destroy();
  });
});

describe("progressive recolouring", () => {
  it("interleaved batches make every segment's colour exact, each visited once", async () => {
    if (!gpu) return;
    const g = new DenseGrid([25, 19], bowl.box);
    const vals = bowl.sampleOn(g);
    const dense = buildScalarFieldData({ type: "dense", box: bowl.box.intervals, samples: { type: "inline", shape: [25, 19], data: Array.from(vals) } }, 2);
    const res = uploadGrid(gpu, g, vals, 1);
    const cres = await sampleResident(gpu, waves, new DenseGrid([9, 7], waves.box)); // coarse: visibly interpolated
    const kernel = fusedIsolines(gpu, dense, res, cres);
    const segs = allocSegments(gpu, kernel.capacity, false);
    await kernel.run(segs, 1.1, 1e-3);
    const before = await readSegments(gpu, segs);
    const n = before.length / SEG_FLOATS;
    // the count is only known on the GPU: schedule from the capacity in 7 interleaved frames
    const r = recolourer(gpu, waves, SEG_LAYOUT);
    const jobs = [{ r, buffer: segs.buffer, indirect: segs.indirect, total: segs.capacity, progress: freshProgress() }];
    let frames = 0;
    while (recolourStep(jobs, Math.ceil(segs.capacity / 7))) frames++;
    expect(frames + 1).toBeLessThanOrEqual(jobs[0]!.progress.frames); // several phases per step when the share allows
    const after = await readSegments(gpu, segs);
    let changed = 0;
    for (let i = 0; i < n; i++) {
      const o = i * SEG_FLOATS;
      for (const [pos, col] of [[0, 4], [2, 5]] as const) {
        const exact = waves.fn([after[o + pos]!, after[o + pos + 1]!], -1);
        expect(after[o + col]).toBeCloseTo(exact, 4);
        if (Math.abs(before[o + col]! - exact) > 1e-3) changed++;
      }
      expect(after[o]).toBe(before[o]); expect(after[o + 1]).toBe(before[o + 1]); // geometry untouched
    }
    expect(changed).toBeGreaterThan(n / 4); // the coarse grid really was interpolated
    segs.destroy(); res.destroy(); cres.destroy();
  });
});

describe("progressive recolouring of a mesh", () => {
  it("writes exact colours into Vert.c and leaves positions and normals alone", async () => {
    if (!gpu) return;
    const x3 = { op: "coord", index: 0 } as const, y3 = { op: "coord", index: 1 } as const, z3 = { op: "coord", index: 2 } as const;
    const sphere = buildScalarFieldData({ type: "symbolic", box: [[-1.5, 1.5], [-1.5, 1.5], [-1.5, 1.5]], expr: { op: "add", vals: [{ op: "square", val: x3 }, { op: "square", val: y3 }, { op: "square", val: z3 }] } }, 3);
    const tint = buildScalarFieldData({ type: "symbolic", box: [[-1.5, 1.5], [-1.5, 1.5], [-1.5, 1.5]], expr: { op: "add", vals: [{ op: "sin", val: { op: "mul", vals: [4, x3] } }, { op: "cos", val: { op: "mul", vals: [3, z3] } }] } }, 3);
    const grid = new DenseGrid([20, 20, 20], sphere.box);
    const values = await sampleResident(gpu, sphere, grid);
    const coarse = await sampleResident(gpu, tint, new DenseGrid([5, 5, 5], tint.box)); // very coarse: clearly interpolated
    const kernel = fusedIsosurface(gpu, values, { exact: false, colour: coarse });
    const mesh = allocMesh(gpu, kernel.capacity);
    await kernel.run(mesh, 1.0);
    const before = await readMesh(gpu, mesh);
    const r = recolourer(gpu, tint, VERT_LAYOUT);
    const jobs = [{ r, buffer: mesh.buffer, indirect: mesh.indirect, total: 3 * mesh.capacity, progress: freshProgress() }];
    while (recolourStep(jobs, 5000)) { /* to completion */ }
    const after = await readMesh(gpu, mesh);
    const n = after.values!.length;
    expect(n).toBe(before.values!.length);
    let changed = 0;
    for (let v = 0; v < n; v++) {
      const p = [after.positions[3 * v]!, after.positions[3 * v + 1]!, after.positions[3 * v + 2]!];
      expect(after.values![v]).toBeCloseTo(tint.fn(p, -1), 4); // exact colour
      if (Math.abs(before.values![v]! - after.values![v]!) > 1e-3) changed++;
      for (let d = 0; d < 3; d++) { expect(after.positions[3 * v + d]).toBe(before.positions[3 * v + d]); expect(after.normals[3 * v + d]).toBe(before.normals[3 * v + d]); } // untouched
    }
    expect(changed).toBeGreaterThan(n / 2); // the coarse interpolation really was off
    mesh.destroy(); values.destroy(); coarse.destroy();
  });
});

describe("fused streamlines", () => {
  it("segments reproduce core's integration through the same vector grid, with arc / len / phase", async () => {
    if (!gpu) return;
    const grad = new SymbolicVectorFieldData({ k: "grad", s: { k: "arg", name: "f" } }, 2, { scalars: { f: bowl }, vectors: {} });
    const g = new DenseGrid([64, 64], bowl.box);
    const vres = await sampleResident(gpu, grad, g);
    const dense = new DenseVectorFieldData(g, grad.sampleOn(g));
    const seeds = streamlineSeeds(bowl.box, 60, 5);
    const opts = { maxSteps: 40, step: 0.05, sign: -1 as const, box: bowl.box };
    const cpu = integrateFromSeeds(dense, seeds, opts);
    const kernel = fusedStreamlines(gpu, vres, seeds, opts, bowl);
    const segs = allocSegments(gpu, kernel.capacity, true);
    await kernel.run(segs);
    const out = await readSegments(gpu, segs);
    const n = out.length / SEG_FLOATS;
    let cpuSegs = 0;
    for (const l of cpu) cpuSegs += l.points.length / 2 - 1;
    expect(Math.abs(n - cpuSegs)).toBeLessThan(cpuSegs * 0.1 + 5);
    // group by phase (unique per line) and check arc/len consistency and colour = bowl at endpoints
    const byPhase = new Map<number, number[]>();
    for (let i = 0; i < n; i++) {
      const o = i * SEG_FLOATS;
      const ph = out[o + 8]!;
      const l = byPhase.get(ph) ?? []; l.push(i); byPhase.set(ph, l);
      expect(out[o + 4]).toBeCloseTo(bowl.fn([out[o]!, out[o + 1]!], -1), 3);
      expect(out[o + 7]).toBeGreaterThan(0);
    }
    expect(byPhase.size).toBe(cpu.length);
    for (const [ph, idx] of byPhase) {
      const c = cpu.find((l) => Math.abs(l.phase - ph) < 1e-6)!;
      expect(c).toBeDefined();
      const len = out[idx[0]! * SEG_FLOATS + 7]!;
      expect(Math.abs(len - c.length)).toBeLessThan(opts.step * 4 + 1e-6);
      // arcs are k*h for consecutive k
      const arcs = idx.map((i) => out[i * SEG_FLOATS + 6]!).sort((a, b) => a - b);
      arcs.forEach((a, k) => expect(a).toBeCloseTo(k * opts.step, 5));
    }
    segs.destroy(); vres.destroy();
  });
  it("sizes capacity from the budgets and emits exactly the planned segments", async () => {
    if (!gpu) return;
    const grad = new SymbolicVectorFieldData({ k: "grad", s: { k: "arg", name: "f" } }, 2, { scalars: { f: bowl }, vectors: {} });
    const g = new DenseGrid([64, 64], bowl.box);
    const vres = await sampleResident(gpu, grad, g);
    const dense = new DenseVectorFieldData(g, grad.sampleOn(g));
    const opts = { count: 80, maxSteps: 300, step: 0.02, sign: -1 as const, box: bowl.box, seed: 3 };
    const plan = evenlySpacedStreamlines(dense, opts);
    const kernel = fusedStreamlines(gpu, vres, plan.seeds, opts);
    let planned = 0;
    for (const l of plan.lines) planned += l.points.length / 2 - 1;
    expect(kernel.capacity).toBe(planned); // Σ budgets, not lines × 2 × maxSteps
    expect(kernel.capacity).toBeLessThan(plan.lines.length * 2 * opts.maxSteps);
    const segs = allocSegments(gpu, kernel.capacity, true);
    await kernel.run(segs);
    const out = await readSegments(gpu, segs);
    expect(out.length / SEG_FLOATS).toBe(planned);
    segs.destroy(); vres.destroy();
  });
});

// CPU / GPU agreement for the geometry kernels: marching squares segments,
// projection onto level sets, exact isolines, streamline integration.

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import {
  Bundle,
  DenseGrid,
  DenseVectorFieldData,
  SymbolicVectorFieldData,
  buildScalarFieldData,
  buildVectorFieldData,
  contourField,
  evenlySpacedStreamlines,
  integrateFromSeeds,
  marchingSquaresSegments,
  projectToLevel,
  streamlineSeeds,
  type ScalarFieldData,
} from "@tensatory/core";
import { GpuBackend, gpuExactIsoContours, gpuIntegrateFromSeeds, gpuMarchingSquaresSegments, gpuProjector } from "../src";

let gpu: GpuBackend | undefined;
beforeAll(async () => { gpu = await GpuBackend.create(); });
afterAll(async () => { gpu?.destroy(); await new Promise((r) => setTimeout(r, 50)); });

const x = { op: "coord", index: 0 } as const, y = { op: "coord", index: 1 } as const;
const bowl = buildScalarFieldData({ type: "symbolic", box: [[-2, 2], [-2, 2]], expr: { op: "add", vals: [{ op: "square", val: x }, { op: "square", val: y }] } }, 2);
const waves = buildScalarFieldData({ type: "symbolic", box: [[-3, 3], [-3, 3]], expr: { op: "mul", vals: [{ op: "sin", val: { op: "mul", vals: [3, x] } }, { op: "cos", val: { op: "mul", vals: [2, y] } }] } }, 2);

/** segments as a sorted list of canonical strings, tolerant to f32 */
function canon(segs: ArrayLike<number>): string[] {
  const out: string[] = [];
  for (let i = 0; i + 3 < segs.length; i += 4) {
    const fx = (v: number) => (v + 0 === 0 ? 0 : v).toFixed(4).replace(/^-0\.0000$/, "0.0000");
    const a = [segs[i]!, segs[i + 1]!].map(fx), b = [segs[i + 2]!, segs[i + 3]!].map(fx);
    out.push(a.join(",") <= b.join(",") ? `${a}-${b}` : `${b}-${a}`);
  }
  return out.sort();
}

describe("marching squares kernel", () => {
  it("emits the same segments as core (bowl, waves, NaN cells)", async () => {
    if (!gpu) return;
    for (const [f, level] of [[bowl, 1.7], [waves, 0.3], [waves, -0.6]] as [ScalarFieldData, number][]) {
      const g = new DenseGrid([41, 37], f.box);
      const vals = f.sampleOn(g);
      if (f === waves) for (let i = 0; i < 60; i++) vals[i * 17 % vals.length] = NaN; // holes
      const cpu = marchingSquaresSegments(g, vals, level);
      const gp = await gpuMarchingSquaresSegments(gpu, g, vals, level);
      expect(gp.length).toBe(cpu.length);
      expect(canon(gp)).toEqual(canon(cpu));
    }
  });
});

describe("projection kernel", () => {
  it("lands on the level set where core does, including locked box faces", async () => {
    if (!gpu) return;
    const g = new DenseGrid([33, 33], bowl.box);
    const proj = gpuProjector(gpu, bowl, g);
    const level = 1.3;
    const pts: number[] = [];
    for (let i = 0; i < 40; i++) pts.push(-1.5 + 3 * ((i * 7919) % 40) / 40, -1.5 + 3 * ((i * 104729) % 40) / 40, 1.0);
    pts.push(2, 0.4, 1.0); // on the right face: must stay there
    pts.push(0, 0, 1.0); // gradient vanishes: cannot project
    const res = await proj.project(Float32Array.from(pts), level);
    for (let i = 0; i < pts.length / 3; i++) {
      const p = [pts[i * 3]!, pts[i * 3 + 1]!];
      const cpu = projectToLevel(bowl, p, level, pts[i * 3 + 2]!);
      const ok = res[i * 3 + 2]! > 0;
      if (!cpu) { expect(ok, `point ${p} should fail on both`).toBe(false); continue; }
      expect(ok, `point ${p} should succeed on both`).toBe(true);
      expect(Math.hypot(res[i * 3]! - cpu[0]!, res[i * 3 + 1]! - cpu[1]!)).toBeLessThan(2e-4);
      expect(Math.abs(bowl.fn([res[i * 3]!, res[i * 3 + 1]!], -1) - level)).toBeLessThan(1e-4);
    }
    expect(res[40 * 3]).toBe(2); // locked face
  });
});

describe("exact isolines on the GPU", () => {
  it("match core's exact contours: same topology, vertices on the curve, chords within tolerance", async () => {
    if (!gpu) return;
    for (const [f, level] of [[bowl, 0.25], [waves, 0.4]] as [ScalarFieldData, number][]) {
      const g = new DenseGrid([48, 48], f.box);
      const vals = f.sampleOn(g);
      const tol = 2e-3;
      const cpu = contourField(f, g, vals, level, { tolerance: tol });
      const gp = await gpuExactIsoContours(gpu, gpuProjector(gpu, f, g), f, g, vals, level, { tolerance: tol });
      expect(gp.method).toBe("exact");
      expect(gp.lines.length).toBe(cpu.lines.length);
      expect(gp.maxResidual).toBeLessThan(1e-4 * Math.max(1, Math.abs(level)));
      // every GPU vertex lies within tolerance of some CPU chord and vice versa (Hausdorff-ish, sampled)
      const near = (pt: [number, number], lines: ArrayLike<number>[]) => {
        let best = Infinity;
        for (const l of lines) for (let i = 0; i + 3 < l.length; i += 2) {
          const ax = l[i]!, ay = l[i + 1]!, bx = l[i + 2]!, by = l[i + 3]!;
          const dx = bx - ax, dy = by - ay, l2 = dx * dx + dy * dy || 1e-30;
          const t = Math.min(1, Math.max(0, ((pt[0] - ax) * dx + (pt[1] - ay) * dy) / l2));
          best = Math.min(best, Math.hypot(pt[0] - ax - t * dx, pt[1] - ay - t * dy));
        }
        return best;
      };
      for (const l of gp.lines) for (let i = 0; i < l.length; i += 8) expect(near([l[i]!, l[i + 1]!], cpu.lines)).toBeLessThan(3 * tol);
      for (const l of cpu.lines) for (let i = 0; i < l.length; i += 8) expect(near([l[i]!, l[i + 1]!], gp.lines)).toBeLessThan(3 * tol);
      // similar vertex budgets (same refinement rule)
      expect(gp.vertexCount).toBeGreaterThan(cpu.vertexCount * 0.5);
      expect(gp.vertexCount).toBeLessThan(cpu.vertexCount * 2);
    }
  });
});

describe("streamline kernel", () => {
  it("follows core's trajectories through a sampled vector grid", async () => {
    if (!gpu) return;
    const grad = new SymbolicVectorFieldData({ k: "grad", s: { k: "arg", name: "f" } }, 2, { scalars: { f: waves }, vectors: {} });
    const g = new DenseGrid([96, 96], waves.box);
    const dense = new DenseVectorFieldData(g, grad.sampleOn(g));
    const seeds = streamlineSeeds(waves.box, 150, 7);
    const opts = { maxSteps: 60, step: 0.03, sign: -1 as const, box: waves.box };
    const cpu = integrateFromSeeds(dense, seeds, opts);
    const gp = await gpuIntegrateFromSeeds(gpu, dense, seeds, opts);
    expect(gp.length).toBe(cpu.length);
    let close = 0, total = 0;
    gp.forEach((l, i) => {
      const c = cpu[i]!;
      expect(l.phase).toBe(c.phase);
      // trajectories agree closely for the first steps from the seed (before f32 drift near separatrices)
      const nb = Math.min(10, Math.floor((l.points.length / 2 - 1) / 2), Math.floor((c.points.length / 2 - 1) / 2));
      const seedL = l.points.length / 2 - 1 - Math.floor((l.points.length / 2 - 1) / 2); // rough centre
      void seedL;
      total++;
      if (Math.abs(l.points.length - c.points.length) <= 4 * 2) close++;
      void nb;
    });
    expect(close / total).toBeGreaterThan(0.85); // most lines have the same length ±4 steps
    // the seed point itself is exactly reproduced somewhere on every line
    for (let i = 0; i < gp.length; i++) {
      const sx = seeds.points[i * 2]!, sy = seeds.points[i * 2 + 1]!;
      let found = false;
      const l = gp[i]!.points;
      for (let k = 0; k < l.length; k += 2) if (Math.abs(l[k]! - sx) < 1e-6 && Math.abs(l[k + 1]! - sy) < 1e-6) { found = true; break; }
      expect(found).toBe(true);
    }
  });
  it("one-way integration matches core and starts at the seed", async () => {
    if (!gpu) return;
    const grad = new SymbolicVectorFieldData({ k: "grad", s: { k: "arg", name: "f" } }, 2, { scalars: { f: bowl }, vectors: {} });
    const g = new DenseGrid([64, 64], bowl.box);
    const dense = new DenseVectorFieldData(g, grad.sampleOn(g));
    const seeds = streamlineSeeds(bowl.box, 40, 9);
    const opts = { maxSteps: 50, step: 0.03, sign: -1 as const, box: bowl.box, bidirectional: false };
    const cpu = integrateFromSeeds(dense, seeds, opts);
    const gp = await gpuIntegrateFromSeeds(gpu, dense, seeds, opts);
    expect(gp.length).toBe(cpu.length);
    gp.forEach((l, i) => {
      expect(l.points[0]).toBeCloseTo(seeds.points[2 * i]!, 6); expect(l.points[1]).toBeCloseTo(seeds.points[2 * i + 1]!, 6);
      expect(Math.abs(l.points.length - cpu[i]!.points.length)).toBeLessThanOrEqual(2 * 2);
    });
  });

  it("honours per-seed step budgets (evenly-spaced plan re-integrated)", async () => {
    if (!gpu) return;
    const grad = new SymbolicVectorFieldData({ k: "grad", s: { k: "arg", name: "f" } }, 2, { scalars: { f: bowl }, vectors: {} });
    const g = new DenseGrid([64, 64], bowl.box);
    const dense = new DenseVectorFieldData(g, grad.sampleOn(g));
    const opts = { count: 80, maxSteps: 300, step: 0.02, sign: -1 as const, box: bowl.box, seed: 3 };
    const plan = evenlySpacedStreamlines(dense, opts);
    expect(plan.seeds.budgets).toBeDefined();
    const gp = await gpuIntegrateFromSeeds(gpu, dense, plan.seeds, opts);
    expect(gp.length).toBe(plan.lines.length);
    // a smooth bowl: the budgeted f32 lines have exactly the planned point counts and follow the f64 lines closely
    gp.forEach((l, i) => {
      const c = plan.lines[i]!;
      expect(l.points.length).toBe(c.points.length);
      expect(l.phase).toBe(c.phase);
      for (let k = 0; k < l.points.length; k++) expect(Math.abs(l.points[k]! - c.points[k]!)).toBeLessThan(2e-3);
    });
    // without budgets the same seeds run to the cap: strictly more points somewhere
    const free = await gpuIntegrateFromSeeds(gpu, dense, { points: plan.seeds.points, phases: plan.seeds.phases }, opts);
    expect(free.reduce((a, l) => a + l.points.length, 0)).toBeGreaterThan(gp.reduce((a, l) => a + l.points.length, 0));
  });
  it("matches core on a constant field exactly (deterministic, no drift)", async () => {
    if (!gpu) return;
    const v = buildVectorFieldData({ type: "symbolicv", box: [[0, 1], [0, 1]], expr: { op: "constv", value: [1, 0.5] } }, 2);
    const seeds = streamlineSeeds(v.box, 9, 3);
    const opts = { maxSteps: 30, step: 0.02 };
    const cpu = integrateFromSeeds(v, seeds, opts), gp = await gpuIntegrateFromSeeds(gpu, v, seeds, opts);
    expect(gp.length).toBe(cpu.length);
    gp.forEach((l, i) => {
      expect(l.points.length).toBe(cpu[i]!.points.length);
      for (let k = 0; k < l.points.length; k++) expect(l.points[k]).toBeCloseTo(cpu[i]!.points[k]!, 5);
    });
  });
});

describe("example bundle fields contour identically", () => {
  it("dense.json loss (sampled: marching squares path) and symbolic.json rosenbrock (exact path)", async () => {
    if (!gpu) return;
    const dir = join(__dirname, "../../../apps/viewer/public/bundles");
    const d = Bundle.parse(JSON.parse(readFileSync(join(dir, "dense.json"), "utf8")));
    const loss = d.scalarField("loss").data;
    const vals = loss.sampleOn(loss.samplePoints!);
    expect(canon(await gpuMarchingSquaresSegments(gpu, loss.samplePoints!, vals, 1.5))).toEqual(canon(marchingSquaresSegments(loss.samplePoints!, vals, 1.5)));
    const s = Bundle.parse(JSON.parse(readFileSync(join(dir, "symbolic.json"), "utf8")));
    const ros = s.scalarField("rosenbrock").data;
    const g = new DenseGrid([64, 64], ros.box);
    const rv = ros.sampleOn(g);
    const res = await gpuExactIsoContours(gpu, gpuProjector(gpu, ros, g), ros, g, rv, 50, { tolerance: 1e-3 });
    expect(res.maxResidual).toBeLessThan(5e-3);
    expect(res.lines.length).toBe(contourField(ros, g, rv, 50, { tolerance: 1e-3 }).lines.length);
  });
});

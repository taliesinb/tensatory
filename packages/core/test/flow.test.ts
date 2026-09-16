import { describe, expect, it } from "vitest";
import { Box, DenseGrid, SymbolicVectorFieldData, boxBlur, buildScalarFieldData, buildVectorFieldData, coverageStreamlines, evenlySpacedStreamlines, integrateFromSeeds, integrateStreamlines, planStreamlines, taubinSmooth } from "../src";

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

describe("streamline modes", () => {
  const bowl = buildScalarFieldData({ type: "symbolic", box: [[-1, 1], [-1, 1]], expr: { op: "add", vals: [{ op: "square", val: { op: "coord", index: 0 } }, { op: "square", val: { op: "coord", index: 1 } }] } }, 2);
  const grad = new SymbolicVectorFieldData({ k: "grad", s: { k: "arg", name: "f" } }, 2, { scalars: { f: bowl }, vectors: {} });
  /** nearest distance between a point and the vertices of the other lines */
  const nearestOther = (lines: { points: Float64Array }[], li: number, x: number, y: number): number => {
    let best = Infinity;
    lines.forEach((l, j) => { if (j === li) return; for (let k = 0; k < l.points.length; k += 2) best = Math.min(best, Math.hypot(l.points[k]! - x, l.points[k + 1]! - y)); });
    return best;
  };

  it("budgets cap the steps per direction", () => {
    const v = buildVectorFieldData({ type: "symbolicv", box: [[0, 1], [0, 1]], expr: { op: "constv", value: [1, 0] } }, 2);
    const seeds = { points: Float64Array.from([0.5, 0.5, 0.5, 0.2]), phases: Float64Array.from([0.1, 0.2]), budgets: Uint32Array.from([3, 5, 0, 1]) };
    const lines = integrateFromSeeds(v, seeds, { maxSteps: 100, step: 0.01 });
    expect(lines.map((l) => l.points.length / 2)).toEqual([9, 2]);
    expect(lines[0]!.points[0]).toBeCloseTo(0.47, 10);
    expect(lines[0]!.points[16]).toBeCloseTo(0.55, 10);
    expect(lines[1]!.points[0]).toBeCloseTo(0.5, 10);
  });

  it("one-way integration starts every line at its seed", () => {
    const v = buildVectorFieldData({ type: "symbolicv", box: [[0, 1], [0, 1]], expr: { op: "constv", value: [1, 0] } }, 2);
    const seeds = { points: Float64Array.from([0.5, 0.5, 0.2, 0.2]), phases: Float64Array.from([0.1, 0.2]) };
    const fwd = integrateFromSeeds(v, seeds, { maxSteps: 10, step: 0.01, bidirectional: false });
    fwd.forEach((l, i) => { expect(l.points.length / 2).toBe(11); expect(l.points[0]).toBeCloseTo(seeds.points[2 * i]!, 12); expect(l.points[20]).toBeCloseTo(seeds.points[2 * i]! + 0.1, 10); });
    // descending: the line still starts at the seed and runs the other way
    const back = integrateFromSeeds(v, seeds, { maxSteps: 10, step: 0.01, sign: -1, bidirectional: false });
    expect(back[0]!.points[0]).toBeCloseTo(0.5, 12); expect(back[0]!.points[20]).toBeCloseTo(0.4, 10);
    // evenly-spaced plans honour it too: budgets have no backward steps and re-integrate exactly
    const plan = evenlySpacedStreamlines(grad, { count: 60, maxSteps: 200, step: 0.01, sign: -1, seed: 5, bidirectional: false });
    for (let i = 0; i < plan.lines.length; i++) { expect(plan.seeds.budgets![2 * i]).toBe(0); expect(plan.lines[i]!.points[0]).toBe(plan.seeds.points[2 * i]); }
    const again = integrateFromSeeds(grad, plan.seeds, { maxSteps: 200, step: 0.01, sign: -1, bidirectional: false });
    expect(again.map((l) => l.points.length)).toEqual(plan.lines.map((l) => l.points.length));
  });

  it("evenly-spaced lines keep their separation and re-integrate from the budgets", () => {
    const opts = { count: 100, maxSteps: 400, step: 0.005, sign: -1 as const, seed: 4, mode: "evenly-spaced" as const };
    const plan = evenlySpacedStreamlines(grad, opts);
    const dSep = plan.separation, dTest = 0.5 * dSep;
    expect(dSep).toBeCloseTo(0.2, 12);
    expect(plan.lines.length).toBeGreaterThan(8);
    expect(plan.seeds.budgets!.length).toBe(2 * plan.lines.length);
    // every vertex is at least d_test - one step from every other line's vertices (a line stops at d_test)
    plan.lines.forEach((l, i) => { for (let k = 0; k < l.points.length; k += 2) expect(nearestOther(plan.lines, i, l.points[k]!, l.points[k + 1]!)).toBeGreaterThan(dTest - opts.step - 1e-9); });
    // seeds are at least d_sep from the lines that existed when they were placed: in particular from each other
    for (let i = 0; i < plan.lines.length; i++) for (let j = 0; j < i; j++) expect(Math.hypot(plan.seeds.points[2 * i]! - plan.seeds.points[2 * j]!, plan.seeds.points[2 * i + 1]! - plan.seeds.points[2 * j + 1]!)).toBeGreaterThan(dSep - 1e-9);
    // the whole box is covered: no point farther than ~d_sep from a line
    for (let gx = -0.9; gx <= 0.9; gx += 0.3) for (let gy = -0.9; gy <= 0.9; gy += 0.3) expect(nearestOther(plan.lines, -1, gx, gy)).toBeLessThan(dSep + 1e-9);
    // re-integration from the budgets reproduces the plan exactly
    const again = integrateFromSeeds(grad, plan.seeds, opts);
    expect(again.length).toBe(plan.lines.length);
    again.forEach((l, i) => { expect(l.points.length).toBe(plan.lines[i]!.points.length); expect(l.phase).toBe(plan.lines[i]!.phase); expect(Array.from(l.points)).toEqual(Array.from(plan.lines[i]!.points)); });
    // and the same plan comes out of the dispatcher
    expect(planStreamlines(grad, opts).lines.length).toBe(plan.lines.length);
  });

  it("coverage fill seeds the starved cells", () => {
    // a strongly divergent shear flow v = (1, 4y): lines fan out towards the corners, which stratified seeding leaves starved
    const v = buildVectorFieldData({ type: "symbolicv", box: [[0, 1], [-1, 1]], expr: { op: "compv", coeffs: [1, { op: "mul", vals: [4, { op: "coord", index: 1 }] }] } }, 2);
    const opts = { count: 32, maxSteps: 400, step: 0.01, seed: 2 };
    const strat = planStreamlines(v, opts);
    const cov = coverageStreamlines(v, opts);
    // vertices per seed cell (4 × 8 cells of side 0.25), starved = fewer than ¼ of the median occupied cell
    const starved = (lines: { points: Float64Array }[]): number => {
      const hits = new Array<number>(32).fill(0);
      for (const l of lines) for (let i = 0; i < l.points.length; i += 2) hits[Math.min(3, Math.floor(l.points[i]! * 4)) * 8 + Math.min(7, Math.floor((l.points[i + 1]! + 1) * 4))]!++;
      const occ = hits.filter((h) => h > 0).sort((a, b) => a - b);
      const threshold = Math.max(1, 0.25 * occ[occ.length >> 1]!);
      return hits.filter((h) => h < threshold).length;
    };
    expect(starved(strat.lines)).toBeGreaterThan(0);
    expect(starved(cov.lines)).toBeLessThan(starved(strat.lines));
    expect(cov.seeds.phases.length).toBeGreaterThan(strat.seeds.phases.length);
    expect(cov.lines.length).toBeGreaterThan(strat.lines.length);
    expect(cov.seeds.budgets).toBeUndefined();
    // all seeds re-integrate to the planned lines
    expect(integrateFromSeeds(v, cov.seeds, opts).length).toBe(cov.lines.length);
  });
});

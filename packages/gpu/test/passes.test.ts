// Resident grid passes vs core: statistics, box blur, Taubin-smoothed isolines.

import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { DenseGrid, boxBlur, buildScalarFieldData, computeStats, isoContours, taubinSmooth } from "@tensatory/core";
import { GpuBackend, SEG_FLOATS, allocSegments, blurResidentSync, gpuStats, readGrid, readSegments, sampleResident, smoothedIsolines, uploadGrid } from "../src";

let gpu: GpuBackend | undefined;
beforeAll(async () => { gpu = await GpuBackend.create(); });
afterAll(async () => { gpu?.destroy(); await new Promise((r) => setTimeout(r, 50)); });

const x = { op: "coord", index: 0 } as const, y = { op: "coord", index: 1 } as const;
const waves = buildScalarFieldData({ type: "symbolic", box: [[-3, 3], [-3, 3]], expr: { op: "mul", vals: [{ op: "sin", val: { op: "mul", vals: [3, x] } }, { op: "cos", val: { op: "mul", vals: [2, y] } }] } }, 2);
const bowl = buildScalarFieldData({ type: "symbolic", box: [[-2, 2], [-2, 2]], expr: { op: "add", vals: [{ op: "square", val: x }, { op: "square", val: y }] } }, 2);

describe("stats reduction", () => {
  it("min / max / positive min / mean agree with core (with NaN holes)", async () => {
    if (!gpu) return;
    const g = new DenseGrid([77, 61], waves.box);
    const vals = waves.sampleOn(g);
    for (let i = 0; i < 200; i++) vals[(i * 131) % vals.length] = NaN;
    const res = uploadGrid(gpu, g, vals, 1);
    const st = await gpuStats(gpu, res);
    const cpu = computeStats(vals);
    expect(st.min).toBeCloseTo(cpu.min, 5);
    expect(st.max).toBeCloseTo(cpu.max, 5);
    expect(st.mean).toBeCloseTo(cpu.mean, 4);
    expect(st.finite).toBe(cpu.finite);
    let pm = Infinity; for (const v of vals) if (v > 0 && v < pm) pm = v;
    expect(st.posMin).toBeCloseTo(pm, 6);
    res.destroy();
  });
  it("works on a vector grid channel and on large grids", async () => {
    if (!gpu) return;
    const g = new DenseGrid([512, 512], bowl.box);
    const res = await sampleResident(gpu, bowl, g);
    const st = await gpuStats(gpu, res);
    expect(st.min).toBeCloseTo(0, 3); // the grid has no sample exactly at the origin
    expect(st.max).toBeCloseTo(8, 4);
    expect(st.finite).toBe(512 * 512);
    res.destroy();
  });
});

describe("box blur", () => {
  it("equals core's boxBlur (edge-truncated means)", async () => {
    if (!gpu) return;
    const g = new DenseGrid([45, 31], waves.box);
    const vals = waves.sampleOn(g);
    const res = uploadGrid(gpu, g, vals, 1);
    for (const r of [1, 3]) {
      const blurred = blurResidentSync(gpu, res, r);
      const back = await readGrid(gpu, blurred);
      const cpu = boxBlur(g, vals, r);
      for (let i = 0; i < cpu.length; i += 7) expect(back[i]).toBeCloseTo(cpu[i]!, 5);
      blurred.destroy();
    }
    expect(blurResidentSync(gpu, res, 0)).toBe(res);
    res.destroy();
  });
});

describe("Taubin-smoothed isolines on the edge graph", () => {
  it("segment endpoints coincide with core's smoothed polyline vertices", async () => {
    if (!gpu) return;
    const g = new DenseGrid([40, 34], waves.box);
    const vals = waves.sampleOn(g);
    const res = uploadGrid(gpu, g, vals, 1);
    const kernel = smoothedIsolines(gpu, res, bowl);
    for (const [level, iterations] of [[0.3, 0], [0.3, 2], [-0.5, 4]] as [number, number][]) {
      const segs = allocSegments(gpu, kernel.capacity, false);
      kernel.dispatch(segs, level, iterations);
      const out = await readSegments(gpu, segs);
      const cpuLines = isoContours(g, vals, level).map((l) => taubinSmooth(l, iterations));
      const cpuPts: [number, number][] = [];
      for (const l of cpuLines) for (let i = 0; i < l.length; i += 2) cpuPts.push([l[i]!, l[i + 1]!]);
      let cpuSegs = 0; for (const l of cpuLines) cpuSegs += l.length / 2 - 1;
      const n = out.length / SEG_FLOATS;
      expect(n).toBe(cpuSegs);
      const nearest = (px: number, py: number) => { let b = Infinity; for (const [cx, cy] of cpuPts) b = Math.min(b, Math.hypot(px - cx, py - cy)); return b; };
      for (let i = 0; i < n; i++) {
        const o = i * SEG_FLOATS;
        expect(nearest(out[o]!, out[o + 1]!)).toBeLessThan(2e-4);
        expect(nearest(out[o + 2]!, out[o + 3]!)).toBeLessThan(2e-4);
        expect(out[o + 4]).toBeCloseTo(bowl.fn([out[o]!, out[o + 1]!], -1), 3); // colour at a
      }
      segs.destroy();
    }
    kernel.destroy(); res.destroy();
  });
});

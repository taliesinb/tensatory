import { describe, expect, it } from "vitest";
import { Box, DenseGrid, SymbolicScalarFieldData } from "@tensatory/core";
import { GpuBackend, gpuSampleOn } from "../src";

describe("dispatches beyond one workgroup dimension", () => {
  it("samples a 2048² grid (65536 workgroups of 64) correctly", async () => {
    const gpu = await GpuBackend.create();
    if (!gpu) return;
    const f = new SymbolicScalarFieldData({ k: "nary", op: "add", args: [{ k: "coord", index: 0 }, { k: "coord", index: 1 }] }, 2, undefined, new Box([0, 0], [1, 1]));
    const grid = new DenseGrid([2048, 2048], new Box([0, 0], [1, 1]));
    const v = await gpuSampleOn(gpu, f, grid);
    const cpu = f.sampleOn(grid);
    let worst = 0;
    for (let i = 0; i < v.length; i += 4097) worst = Math.max(worst, Math.abs(v[i]! - cpu[i]!));
    expect(worst).toBeLessThan(1e-5);
    expect(v[v.length - 1]).toBeCloseTo(2, 5); // the last point, in the second dispatch row
    gpu.destroy();
  });
});

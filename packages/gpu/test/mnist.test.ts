// The MNIST MLP bundle (apps/viewer/public/bundles/mnist-mlp/) on the GPU: 269k weights displaced along 3 PCA
// directions transpile within Safari's function-variable budget — the displaced 784×256 weight stays a LAZY array
// read from the storage buffer (nets.ts Arr kind "expr") — and agree with the CPU evaluator. But one lane evaluating
// 256 examples × 269k weights serially takes ~10 s per dispatch at ANY grid size (latency-bound; measured
// 2025-09: 256 examples 10 s, 1024 examples 40 s, the displacement reads ~17% of it), so the work budget
// (NET_MAX_WORK) keeps such fields off the per-thread path: `gpuTranspilable` is false and the viewer treats them as
// costly. The agreement / timing checks run with PERF=1 only.

import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { Bundle, DenseGrid, type ByteSource, type NetScalarFieldData } from "@tensatory/core";
import { GpuBackend, NET_MAX_FLOATS, NET_MAX_WORK, emitNetField, gpuSampleOn, gpuTranspilable, netFieldWork, setNetMaxWork } from "../src";

const DIR = join(__dirname, "../../../apps/viewer/public/bundles/mnist-mlp");
const src: ByteSource = {
  bytes: async (path) => {
    try { const b = await readFile(join(DIR, path)); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer; }
    catch (e) { if ((e as NodeJS.ErrnoException).code === "ENOENT") return null; throw e; }
  },
};
const bundleP = readFile(join(DIR, "bundle.json"), "utf8").then((s) => Bundle.load(JSON.parse(s), src));

let gpu: GpuBackend | undefined;
beforeAll(async () => { gpu = await GpuBackend.create(); });
afterAll(async () => { gpu?.destroy(); await new Promise((r) => setTimeout(r, 50)); });

describe("MNIST MLP bundle on the GPU", () => {
  it("fits in memory (the displaced weights are lazy, N is streamed) but exceeds the per-lane work budget", async () => {
    const b = await bundleP;
    for (const id of ["fastLoss2", "fastAcc2", "loss2", "loss3", "fastLoss3"]) {
      const fd = b.scalarField(id).data as NetScalarFieldData;
      const r = emitNetField("f", fd.field, { D: fd.dimCount, upload: () => 0, maxWork: Infinity }, [fd.field.output])!;
      expect(r.streamed, `${id} streams N`).toBe(true);
      expect(r.floats, `${id} floats`).toBeLessThan(NET_MAX_FLOATS);
      expect(r.code).not.toMatch(/array<f32, 200960>/); // W1t + Σ t D is never materialized
      expect(r.work, `${id} work`).toBeGreaterThan(NET_MAX_WORK);
      expect(gpuTranspilable(fd), `${id} is left to the CPU path`).toBe(false);
    }
    // 256 examples × (784·256 + 256·256 + 256·10) multiply-adds, roughly; the 1024-example fields 4× that
    expect(netFieldWork(b.scalarField("fastLoss2").data as NetScalarFieldData)).toBeGreaterThan(256 * 269_000);
    expect(netFieldWork(b.scalarField("loss2").data as NetScalarFieldData)).toBeGreaterThan(4 * 256 * 269_000);
  }, 60_000);

  it.runIf(process.env.PERF)("PERF: loss / accuracy (256 examples) agree with the CPU evaluator; timing", async () => {
    if (!gpu) return;
    const b = await bundleP;
    const fd = b.scalarField("fastLoss2").data as NetScalarFieldData, acc = b.scalarField("fastAcc2").data as NetScalarFieldData;
    const grid = new DenseGrid([2, 2], fd.box);
    setNetMaxWork(Infinity);
    const t0 = performance.now();
    const cpu = fd.sampleOn(grid);
    const t1 = performance.now();
    const gp = await gpuSampleOn(gpu, fd, grid);
    const t2 = performance.now();
    const gpa = await gpuSampleOn(gpu, acc, grid);
    setNetMaxWork(NET_MAX_WORK);
    console.log(`fastLoss2 2×2: cpu ${(t1 - t0).toFixed(0)} ms, gpu ${(t2 - t1).toFixed(0)} ms (incl. compile)`);
    for (let i = 0; i < cpu.length; i++) expect(gp[i]).toBeCloseTo(cpu[i]!, 3);
    const cpa = acc.sampleOn(grid);
    for (let i = 0; i < cpa.length; i++) expect(gpa[i]).toBeCloseTo(cpa[i]!, 6);
  }, 600_000);
});

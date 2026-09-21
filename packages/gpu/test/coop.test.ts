// The cooperative net kernel (src/coop.ts): one workgroup per grid point, the dataset axis streamed in tiles, arrays
// in workgroup memory. Checked against the CPU evaluator for iris (training-set nets, N = 120) and the MNIST MLP
// (N = 256 / 1024, hoisted first layer), values and gradients.

import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { Bundle, DenseGrid, type ByteSource, type NetScalarFieldData, type NetVectorFieldData } from "@tensatory/core";
import { GpuBackend, ProgramBuilder, buildSampleProgram, coopCapable, gpuSampleOn, readGrid, sampleResidentSync, type GpuProgram } from "../src";

const BUNDLES = join(__dirname, "../../../apps/viewer/public/bundles");
const dirSource = (dir: string): ByteSource => ({
  bytes: async (path) => {
    try { const b = await readFile(join(dir, path)); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer; }
    catch (e) { if ((e as NodeJS.ErrnoException).code === "ENOENT") return null; throw e; }
  },
});
const irisP = readFile(join(BUNDLES, "iris.json"), "utf8").then((s) => Bundle.parse(JSON.parse(s)));
const mnistP = readFile(join(BUNDLES, "mnist-mlp/bundle.json"), "utf8").then((s) => Bundle.load(JSON.parse(s), dirSource(join(BUNDLES, "mnist-mlp"))));

let gpu: GpuBackend | undefined;
beforeAll(async () => { gpu = await GpuBackend.create(); if (gpu) gpu.asyncCompile = false; });
afterAll(async () => { gpu?.destroy(); await new Promise((r) => setTimeout(r, 50)); });

/** run a cooperative program over its whole grid in one dispatch; returns the values */
async function runCoop(program: GpuProgram): Promise<Float32Array> {
  const n = program.sampleCount * program.channels;
  try {
    const { read: [o] } = await gpu!.runKernel({
      code: program.code,
      invocations: program.sampleCount * program.cooperative!.workgroupSize,
      workgroups: program.sampleCount,
      buffers: [{ role: "rw", size: n * 4, readback: true }, { role: "r", data: program.data }, { role: "r", data: new Uint32Array([0]) }],
    });
    return new Float32Array(o!, 0, n);
  } catch (e) {
    const diag = await gpu!.diagnostics(program.code).catch(() => []);
    throw new Error(`${e instanceof Error ? e.message : String(e)}\n${diag.join("\n")}`);
  }
}

const close = (got: ArrayLike<number>, want: ArrayLike<number>, rel: number, what: string) => {
  expect(got.length, what).toBe(want.length);
  for (let i = 0; i < want.length; i++) {
    const tol = rel * Math.max(1, Math.abs(want[i]!));
    expect(Math.abs(got[i]! - want[i]!), `${what}[${i}]: ${got[i]} vs ${want[i]}`).toBeLessThanOrEqual(tol);
  }
};

describe("cooperative kernel: iris training-set nets (N = 120)", () => {
  it("train/loss and objective agree with the CPU on a 5×5 grid; tile 8 divides 120", async () => {
    if (!gpu) return;
    const b = await irisP;
    for (const id of ["trainLoss2", "objective2", "trainAcc2"]) {
      const fd = b.scalarField(id).data as NetScalarFieldData;
      const grid = new DenseGrid([5, 5], fd.box);
      const program = new ProgramBuilder(grid).buildCooperative(fd);
      expect(program, `${id} is cooperative`).toBeDefined();
      expect(program!.cooperative!.tile).toBe(8);
      expect(program!.cooperative!.floats).toBeLessThan(4096);
      close(await runCoop(program!), fd.sampleOn(grid), id.includes("Acc") ? 1e-6 : 1e-4, id);
    }
  }, 120_000);

  it("the gradient program (a vector output) agrees with the CPU", async () => {
    if (!gpu) return;
    const b = await irisP;
    const fd = b.scalarField("trainLoss2").data as NetScalarFieldData;
    const g = fd.gradient() as NetVectorFieldData;
    const grid = new DenseGrid([4, 4], fd.box);
    const program = new ProgramBuilder(grid).buildCooperative(g);
    expect(program).toBeDefined();
    expect(program!.channels).toBe(2);
    close(await runCoop(program!), g.sampleOn(grid), 1e-3, "∇ train/loss");
  }, 120_000);
});

describe("cooperative kernel: the MNIST MLP", () => {
  it("fastLoss2 / fastAcc2 (N = 256) agree with the CPU at a 3×3 grid", async () => {
    if (!gpu) return;
    const b = await mnistP;
    for (const id of ["fastLoss2", "fastAcc2"]) {
      const fd = b.scalarField(id).data as NetScalarFieldData;
      const grid = new DenseGrid([3, 3], fd.box);
      const program = new ProgramBuilder(grid).buildCooperative(fd);
      expect(program, `${id} is cooperative`).toBeDefined();
      console.log(`${id}: tile ${program!.cooperative!.tile}, ${program!.cooperative!.floats} workgroup floats, work ${program!.cooperative!.work.toExponential(2)}, ${(program!.code.length / 1024).toFixed(1)} kB WGSL`);
      const t0 = performance.now();
      const got = await runCoop(program!);
      const t1 = performance.now();
      const want = fd.sampleOn(grid);
      const t2 = performance.now();
      console.log(`${id} 3×3: gpu ${(t1 - t0).toFixed(0)} ms (incl. compile), cpu ${(t2 - t1).toFixed(0)} ms`);
      close(got, want, id.includes("Acc") ? 1e-6 : 1e-3, id);
    }
  }, 300_000);

  it("the 3D fields and the gradient", async () => {
    if (!gpu) return;
    const b = await mnistP;
    const fd = b.scalarField("fastLoss3").data as NetScalarFieldData;
    const grid = new DenseGrid([2, 2, 2], fd.box);
    const program = new ProgramBuilder(grid).buildCooperative(fd);
    expect(program).toBeDefined();
    close(await runCoop(program!), fd.sampleOn(grid), 1e-3, "fastLoss3");
    const g = fd.gradient() as NetVectorFieldData;
    const gp = new ProgramBuilder(grid).buildCooperative(g);
    if (gp) {
      console.log(`∇fastLoss3: tile ${gp.cooperative!.tile}, ${gp.cooperative!.floats} floats, work ${gp.cooperative!.work.toExponential(2)}`);
      close(await runCoop(gp), g.sampleOn(grid), 1e-2, "∇fastLoss3");
    } else console.log("∇fastLoss3: not cooperative (stopgap: grid differences)");
  }, 300_000);
});

describe("cooperative kernel: timing", () => {
  it.runIf(process.env.PERF)("PERF: emitted MNIST kernels, µs per point by grid", async () => {
    if (!gpu) return;
    const b = await mnistP;
    console.log(`device: ${gpu.adapterInfo}; workgroup storage ${gpu.device.limits.maxComputeWorkgroupStorageSize} B`);
    for (const id of ["fastLoss2", "loss2"]) {
      const fd = b.scalarField(id).data as NetScalarFieldData;
      const rows: string[] = [];
      for (const n of [4, 16, 32]) {
        const grid = new DenseGrid([n, n], fd.box);
        const program = new ProgramBuilder(grid).buildCooperative(fd)!;
        const r0 = performance.now(); await runCoop(program); const compile = performance.now() - r0; // includes the first compile
        const t0 = performance.now(); await runCoop(program); const t1 = performance.now(); await runCoop(program); const t2 = performance.now();
        const ms = Math.min(t1 - t0, t2 - t1);
        rows.push(`${n}² (${grid.sampleCount} pts): ${ms.toFixed(0)} ms = ${(ms / grid.sampleCount * 1000).toFixed(0)} µs/pt${n === 4 ? ` (first run ${compile.toFixed(0)} ms)` : ""}`);
        if (n === 4) console.log(`${id}: tile ${program.cooperative!.tile}, ${program.cooperative!.floats} floats, ${program.code.length} B WGSL`);
      }
      console.log(`${id}: ${rows.join("; ")}`);
    }
  }, 600_000);
});

describe("cooperative kernel: integration (buildSampleProgram / gpuSampleOn / resident grids)", () => {
  it("gpuSampleOn of a big net is cooperative; an expression over it reads its resident values; its gradient is central differences", async () => {
    if (!gpu) return;
    const b = await mnistP;
    const loss = b.scalarField("fastLoss2").data as NetScalarFieldData;
    const grid = new DenseGrid([4, 3], loss.box);
    expect(coopCapable(loss)).toBe(true);
    expect(buildSampleProgram(loss, grid).cooperative).toBeDefined();
    const want = loss.sampleOn(grid);
    close(await gpuSampleOn(gpu, loss, grid), want, 1e-3, "fastLoss2 via gpuSampleOn");
    const log10 = b.scalarField("log10loss2").data; // log10 of the 1024-example loss: a pointwise field over a cooperative net
    const lg = new DenseGrid([2, 2], log10.box);
    const p = buildSampleProgram(log10, lg, (fd, g) => sampleResidentSync(gpu!, fd, g).buffer);
    expect(p.cooperative).toBeUndefined();
    expect(p.bindings?.length).toBe(1); // the net's resident values
    close(await gpuSampleOn(gpu, log10, lg), log10.sampleOn(lg), 1e-3, "log10(loss2)");
    // the gradient: exact autodiff on the CPU vs central differences of the resident grid on the GPU (one spacing)
    const g = loss.gradient();
    const gg = new DenseGrid([8, 8], loss.box);
    const gp = buildSampleProgram(g, gg, (fd, gr) => sampleResidentSync(gpu!, fd, gr).buffer);
    expect(gp.cooperative).toBeUndefined();
    expect(gp.code).toMatch(/fn dd/);
    const got = await gpuSampleOn(gpu, g, gg);
    const exact = g.sampleOn(gg);
    // interior points: the difference quotient over one spacing vs the exact derivative — agree to a few percent
    let worst = 0, n = 0;
    for (let i = 0; i < gg.sampleCount; i++) {
      const [x, y] = gg.gridPos(i);
      if (x === 0 || y === 0 || x === 7 || y === 7) continue;
      for (let d = 0; d < 2; d++) { const e = Math.abs(got[i * 2 + d]! - exact[i * 2 + d]!) / (Math.abs(exact[i * 2 + d]!) + 0.05); worst = Math.max(worst, e); n++; }
    }
    console.log(`∇ by central differences vs exact: worst relative deviation ${worst.toFixed(3)} over ${n} components`);
    expect(worst).toBeLessThan(0.25);
  }, 300_000);

  it("sampleResident of a cooperative field fills a resident grid readable by readGrid", async () => {
    if (!gpu) return;
    const b = await mnistP;
    const fd = b.scalarField("fastAcc3").data as NetScalarFieldData;
    const grid = new DenseGrid([3, 2, 2], fd.box);
    const g = sampleResidentSync(gpu, fd, grid);
    close(await readGrid(gpu, g), fd.sampleOn(grid), 1e-6, "fastAcc3 resident");
    g.destroy();
  }, 120_000);
});

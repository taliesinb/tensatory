// Net-backed fields on the GPU (nets.ts) must agree with the CPU reference
// evaluator: every array op through a small net, then the iris bundle's loss /
// accuracy fields and their finite-difference gradient.

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import type { ArrayExpr, BundleSpec, NetDefinitionSpec, NetSpec } from "@tensatory/schema";
import { Bundle, DenseGrid, SymbolicVectorFieldData, type NetScalarFieldData, type ScalarFieldData, type VectorFieldData } from "@tensatory/core";
import { GpuBackend, NET_MAX_FLOATS, buildSampleProgram, emitNetField, gpuSampleOn, gpuTranspilable, netFieldFloats } from "../src";

let gpu: GpuBackend | undefined;
beforeAll(async () => {
  gpu = await GpuBackend.create();
  if (!gpu) console.warn("no WebGPU adapter: GPU net tests skipped");
});
afterAll(async () => {
  gpu?.destroy();
  await new Promise((r) => setTimeout(r, 50));
});

function expectAgree(cpu: ArrayLike<number>, gpuVals: ArrayLike<number>, rtol = 2e-4, label = "") {
  expect(gpuVals.length).toBe(cpu.length);
  let scale = 0;
  for (let i = 0; i < cpu.length; i++) if (Number.isFinite(cpu[i]!)) scale = Math.max(scale, Math.abs(cpu[i]!));
  const atol = 1e-5 * (scale || 1);
  let worst = 0, worstI = -1;
  for (let i = 0; i < cpu.length; i++) {
    const a = cpu[i]!, b = gpuVals[i]!;
    if (Number.isNaN(a) || Number.isNaN(b)) { expect(Number.isNaN(a), `${label} NaN mismatch at ${i}`).toBe(Number.isNaN(b)); continue; }
    const err = Math.abs(a - b) / (atol + rtol * Math.abs(a));
    if (err > worst) { worst = err; worstI = i; }
  }
  expect(worst, `${label} worst at ${worstI}: cpu ${cpu[worstI]} gpu ${gpuVals[worstI]}`).toBeLessThanOrEqual(1);
}

async function agree(fd: ScalarFieldData | VectorFieldData, grid: DenseGrid, label: string, rtol?: number) {
  if (!gpu) return;
  const cpu = fd.sampleOn(grid);
  let gp: Float32Array;
  try { gp = await gpuSampleOn(gpu, fd, grid); } catch (e) { throw new Error(`${label}: ${(e as Error).message}\n${buildSampleProgram(fd, grid).code.slice(0, 4000)}`); }
  expectAgree(cpu, gp, rtol, label);
}

const box = [[-1.5, 1.6], [-1.2, 1.3]] as [number, number][]; // no grid point of 13 × 11 lands on 0 (ties / discontinuities)
const inline = (data: number[], shape = [data.length]) => ({ type: "inline", shape, data }) as const;

/** a field whose net maps the point p: [2] through `body` (a node expression over p and the constants) to a scalar */
function fieldOf(body: ArrayExpr, arrays: Record<string, ReturnType<typeof inline>> = {}, extraNodes: Record<string, ArrayExpr> = {}, nets: Record<string, NetSpec> = {}): ScalarFieldData {
  const net: NetDefinitionSpec = { type: "def", inputs: { p: [2] }, arrays, nodes: { ...extraNodes, r: body, s: { op: "reduce", fn: "sum", val: "r" } }, outputs: { s: [] } };
  const b = Bundle.parse({ tensatory: "0.1", manifolds: { m: { numDims: 2 } }, nets: { ...nets, f: net }, fields: { f: { kind: "scalar", data: { type: "net", net: "f", output: "s", box } } } } satisfies BundleSpec);
  return b.scalarField("f").data;
}

describe("wgsl nets: every array op agrees with the CPU evaluator", () => {
  const g = new DenseGrid([13, 11], fieldOf("p").box);
  const M = inline([0.51, -1.03, 2.07, 0.29, 1.53, -0.71], [3, 2]); // [3, 2]; incommensurate entries: no argmax ties on the grid
  const v3 = inline([1, -2, 0.5]);
  const A = inline([1, 2, 3, 4, 5, 6.5, 7, 8.3, 9, 10.1, 11, 12.7], [2, 2, 3]);
  const pA: ArrayExpr = { op: "reshape", val: "p", shape: [2, 1, 1] }; // p broadcast along A's leading axis
  const cases: [string, ArrayExpr, Record<string, ReturnType<typeof inline>>?][] = [
    ["identity", "p"],
    ["unary chain", { op: "tanh", val: { op: "sigmoid", val: { op: "square", val: "p" } } }],
    ["relu / gelu / softplus", { op: "add", vals: [{ op: "relu", val: "p" }, { op: "gelu", val: "p" }, { op: "softplus", val: "p" }] }],
    ["nary broadcast", { op: "mul", vals: ["M", "p", 2] }, { M }],
    ["binary", { op: "sub", vals: [{ op: "div", vals: ["M", { op: "add", vals: ["p", 3] }] }, { op: "pow", vals: [{ op: "abs", val: "p" }, 1.5] }] }, { M }],
    ["mod / atan2 / logBase", { op: "add", vals: [{ op: "mod", vals: ["p", 0.73] }, { op: "atan2", vals: ["p", 0.3] }, { op: "logBase", vals: [{ op: "add", vals: [{ op: "square", val: "p" }, 1] }, 3] }] }],
    ["compare / where / clamp", { op: "where", cond: { op: "gt", vals: ["p", 0] }, vals: [{ op: "clamp", val: "p", min: -0.5, max: 0.5 }, { op: "le", vals: ["M", "p"] }] }, { M }],
    ["mean / rms / min / max", { op: "add", vals: [{ op: "mean", vals: ["p", "M"] }, { op: "rms", vals: ["p", 1] }, { op: "min", vals: ["p", 0.2] }, { op: "max", vals: ["p", -0.2, "M"] }] }, { M }],
    ["matmul M·p", { op: "matmul", vals: ["M", "p"] }, { M }],
    ["matmul p·Mᵀ", { op: "matmul", vals: ["p", { op: "transpose", val: "M" }] }, { M }],
    ["matmul p·p", { op: "matmul", vals: ["p", "p"] }],
    ["matmul stacked", { op: "matmul", vals: ["A", "M"] }, { A, M }],
    ["einsum outer", { op: "einsum", subscripts: "i,j->ij", vals: ["p", "v"] }, { v: v3 }],
    ["einsum implicit", { op: "einsum", subscripts: "ij,j", vals: ["M", "p"] }, { M }],
    ["reduce sum axes", { op: "reduce", fn: "sum", val: { op: "mul", vals: ["A", pA] }, axes: [0, 2] }, { A }],
    ["reduce mean keepDims", { op: "reduce", fn: "mean", val: { op: "mul", vals: ["A", pA] }, axes: [1], keepDims: true }, { A }],
    ["reduce max / min / prod", { op: "add", vals: [{ op: "reduce", fn: "max", val: { op: "mul", vals: ["M", "p"] }, axes: [0] }, { op: "reduce", fn: "min", val: { op: "mul", vals: ["M", "p"] } }, { op: "reduce", fn: "prod", val: { op: "add", vals: ["M", "p"] }, axes: [0] }] }, { M }],
    ["logsumexp", { op: "reduce", fn: "logsumexp", val: { op: "mul", vals: ["M", "p"] }, axes: [1] }, { M }],
    ["argmax / argmin", { op: "add", vals: [{ op: "argmax", val: { op: "mul", vals: ["M", "p"] } }, { op: "argmin", val: { op: "mul", vals: ["M", "p"] }, axis: 1 }, { op: "argmin", val: { op: "mul", vals: ["A", pA] }, axis: 0 }] }, { M, A }],
    ["softmax", { op: "softmax", val: { op: "mul", vals: ["M", "p"] } }, { M }],
    ["logSoftmax axis 0", { op: "logSoftmax", val: { op: "mul", vals: ["M", "p"] }, axis: 0 }, { M }],
    ["reshape / transpose", { op: "transpose", val: { op: "mul", vals: [{ op: "reshape", val: "A", shape: [-1, 2] }, "p"] } }, { A }],
    ["transpose perm", { op: "reduce", fn: "sum", val: { op: "transpose", val: { op: "mul", vals: ["A", pA] }, perm: [2, 0, 1] }, axes: [1] }, { A }],
    ["concat", { op: "concat", vals: [{ op: "mul", vals: ["M", "p"] }, { op: "reshape", val: "p", shape: [1, 2] }], axis: 0 }, { M }],
    ["slice", { op: "mul", vals: [{ op: "slice", val: { op: "mul", vals: ["A", pA] }, axis: 2, start: 1 }, { op: "slice", val: { op: "slice", val: "A", axis: 0, step: -1 }, axis: 2, stop: -1 }] }, { A }],
    ["oneHot", { op: "mul", vals: [{ op: "oneHot", val: { op: "argmax", val: { op: "mul", vals: ["M", "p"] } }, size: 2 }, "M"] }, { M }],
    ["takeAlong", { op: "takeAlong", val: { op: "mul", vals: ["M", "p"] }, indices: { op: "reshape", val: { op: "argmax", val: { op: "mul", vals: ["M", "p"] } }, shape: [3, 1] }, axis: 1 }, { M }],
    ["stopGradient", { op: "stopGradient", val: { op: "square", val: "p" } }],
  ];
  for (const [label, body, arrays] of cases) {
    it(label, async () => { await agree(fieldOf(body, arrays ?? {}), g, label); });
  }

  it("einsum diagonals: extraction, embedding, and their gradients", async () => {
    const S = inline([1, 2, 3, 4], [2, 2]);
    await agree(fieldOf({ op: "einsum", subscripts: "ii,i->", vals: ["S", "p"] }, { S }), g, "diagonal");
    await agree(fieldOf({ op: "mul", vals: [{ op: "einsum", subscripts: "i->ii", vals: ["p"] }, "S"] }, { S }), g, "embedding");
    const f = fieldOf({ op: "einsum", subscripts: "i,j->iij", vals: ["p", "v"] }, { v: inline([1, 10, 0.3]) });
    await agree(f, g, "i,j->iij");
    await agree(f.derivative(0), g, "∂/∂x i,j->iij", 1e-3);
    const tr = fieldOf({ op: "einsum", subscripts: "ii->", vals: [{ op: "einsum", subscripts: "i,j->ij", vals: ["p", "p"] }] });
    await agree(tr.derivative(1).derivative(0), g, "∂² trace(p pᵀ)", 2e-3);
  });

  it("call: a net calling another (inlined)", async () => {
    const sq: NetSpec = { type: "def", inputs: { a: ["N"] }, nodes: { s: { op: "reduce", fn: "sum", val: { op: "square", val: "a" } } }, outputs: { s: [] } };
    await agree(fieldOf({ op: "call", net: "sq", inputs: { a: { op: "matmul", vals: ["M", "p"] } }, output: "s" }, { M }, {}, { sq }), g, "call");
  });

  it("inputs as coordinate expressions and a netv field", async () => {
    const b = Bundle.parse({
      tensatory: "0.1", manifolds: { m: { numDims: 2 } },
      nets: { rot: { type: "def", inputs: { q: [2], k: [] }, arrays: { R: inline([0, -1, 1, 0], [2, 2]) }, nodes: { v: { op: "mul", vals: [{ op: "matmul", vals: ["R", "q"] }, "k"] } }, outputs: { v: [2] } } },
      fields: {
        v: { kind: "vector", data: { type: "netv", net: "rot", output: "v", box, inputs: { q: { op: "add", vals: [{ op: "coordv" }, "c"] }, k: { op: "add", vals: [{ op: "coord", index: 0 }, 2] } }, arrays: { c: inline([0.5, -0.5]) } } },
      },
    } satisfies BundleSpec);
    await agree(b.vectorField("v").data, new DenseGrid([13, 11], b.vectorField("v").data.box), "netv");
  });
});

describe("iris bundle on the GPU", () => {
  const bundle = Bundle.parse(JSON.parse(readFileSync(join(__dirname, "../../../apps/viewer/public/bundles/iris.json"), "utf8")));

  it("the fields and their gradients fit Safari's 8192-byte function-variable limit", () => {
    const loss2 = bundle.scalarField("loss2").data as NetScalarFieldData;
    expect(gpuTranspilable(loss2)).toBe(true);
    expect(netFieldFloats(loss2)).toBeLessThan(1000); // 821: ANF + liveness reuse + in-place + fusion (was 2274)
    expect(netFieldFloats(loss2.gradient())).toBeLessThan(NET_MAX_FLOATS); // 1661
    expect(gpuTranspilable(loss2.derivative(1))).toBe(true);
    expect(loss2.costly).toBe(true); // still costly for CPU consumers
  });

  it("the emitted functions declare no more floats than they report", () => {
    const loss2 = bundle.scalarField("loss2").data as NetScalarFieldData;
    for (const f of [loss2.field, loss2.gradient().field]) {
      const r = emitNetField("x", f, { D: 2, upload: () => 0 })!;
      const declared = [...r.code.matchAll(/array<f32, (\d+)>/g)].reduce((n, m) => n + Number(m[1]), 0);
      expect(declared).toBe(r.floats);
      expect(declared * 4).toBeLessThanOrEqual(8192);
    }
  });

  it("loss and accuracy agree with the CPU evaluator (2D and 3D)", async () => {
    for (const id of ["loss2", "acc2"]) { const fd = bundle.scalarField(id).data; await agree(fd, new DenseGrid([48, 48], fd.box), id); }
    for (const id of ["loss3", "acc3"]) { const fd = bundle.scalarField(id).data; await agree(fd, new DenseGrid([12, 12, 12], fd.box), id); }
  });

  it("the gradient (autodiff on both sides) agrees to f32 precision; so does a second derivative", async () => {
    const loss = bundle.scalarField("loss2").data;
    const grad = new SymbolicVectorFieldData({ k: "grad", s: { k: "arg", name: "f" } }, 2, { scalars: { f: loss }, vectors: {} });
    await agree(grad, new DenseGrid([24, 24], loss.box), "∇loss", 1e-3);
    await agree(loss.derivative(0).derivative(1), new DenseGrid([16, 16], loss.box), "∂²loss/∂t0∂t1", 2e-3);
  });

  it("is fast: 256² points in one dispatch", async () => {
    if (!gpu) return;
    const loss = bundle.scalarField("loss2").data;
    const g = new DenseGrid([256, 256], loss.box);
    const t0 = performance.now();
    const vals = await gpuSampleOn(gpu, loss, g);
    const ms = performance.now() - t0;
    expect(vals.length).toBe(65536);
    expect(Number.isFinite(vals[0]!)).toBe(true);
    console.log(`iris loss on 256² via GPU: ${ms.toFixed(0)} ms (incl. compile)`);
  });
});

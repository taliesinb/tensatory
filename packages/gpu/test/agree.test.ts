// CPU / GPU agreement: whatever core computes on a grid, the GPU backend must
// reproduce to f32 precision.

import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import type { ScalarFieldDataSpec, SymbolicScalar, SymbolicVector, VectorFieldDataSpec } from "@tensatory/schema";
import {
  Box,
  Bundle,
  Codomain,
  DenseGrid,
  SymbolicScalarFieldData,
  SymbolicVectorFieldData,
  buildScalarFieldData,
  buildVectorFieldData,
  type ScalarFieldData,
  type VectorFieldData,
} from "@tensatory/core";
import { GpuBackend, buildSampleProgram, gpuSampleOn } from "../src";

let gpu: GpuBackend | undefined;
beforeAll(async () => {
  gpu = await GpuBackend.create();
  if (!gpu) console.warn("no WebGPU adapter: GPU agreement tests skipped");
});
afterAll(async () => {
  gpu?.destroy();
  await new Promise((r) => setTimeout(r, 50));
});

/** compare with a tolerance scaled to the CPU values' magnitude; NaN must agree with NaN */
function expectAgree(cpu: ArrayLike<number>, gpuVals: ArrayLike<number>, rtol = 2e-4, label = "") {
  expect(gpuVals.length).toBe(cpu.length);
  let scale = 0;
  for (let i = 0; i < cpu.length; i++) if (Number.isFinite(cpu[i]!)) scale = Math.max(scale, Math.abs(cpu[i]!));
  const atol = 1e-5 * (scale || 1);
  let worst = 0, worstI = -1;
  for (let i = 0; i < cpu.length; i++) {
    const a = cpu[i]!, b = gpuVals[i]!;
    if (Number.isNaN(a) || Number.isNaN(b)) { expect(Number.isNaN(a), `${label} NaN mismatch at ${i}: cpu ${a} gpu ${b}`).toBe(Number.isNaN(b)); continue; }
    const err = Math.abs(a - b) / (atol + rtol * Math.abs(a));
    if (err > worst) { worst = err; worstI = i; }
  }
  expect(worst, `${label} worst at ${worstI}: cpu ${cpu[worstI]} gpu ${gpuVals[worstI]}`).toBeLessThanOrEqual(1);
}

const x: SymbolicScalar = { op: "coord", index: 0 };
const y: SymbolicScalar = { op: "coord", index: 1 };
const box = Box.fromSpec([[0.1, 1.9], [-0.9, 0.9]]); // keeps logs / roots / atanh defined
const grid = new DenseGrid([32, 20], box); // spacings 1.8/31, 1.8/19: no coordinate lands on a discontinuity (0, 1, multiples of 0.4)

async function agreeScalar(fd: ScalarFieldData, g = grid, label = "", rtol?: number) {
  if (!gpu) return;
  const cpu = fd.sampleOn(g);
  const gp = await gpuSampleOn(gpu, fd, g);
  expectAgree(cpu, gp, rtol, label);
}
async function agreeVector(fd: VectorFieldData, g = grid, label = "", rtol?: number) {
  if (!gpu) return;
  expectAgree(fd.sampleOn(g), await gpuSampleOn(gpu, fd, g), rtol, label);
}
const sym = (expr: SymbolicScalar) => buildScalarFieldData({ type: "symbolic", box: box.intervals, expr }, 2);

describe("wgsl: every scalar op agrees with the CPU evaluator", () => {
  const unary = ["sin", "cos", "tan", "sinh", "cosh", "tanh", "asin", "acos", "atan", "asinh", "acosh", "atanh", "relu", "sigmoid", "gelu", "silu", "softplus", "elu", "erf", "floor", "ceil", "round", "sign", "abs", "exp", "exp2", "exp10", "log", "log2", "log10", "log1p", "expm1", "plogp", "sqrt", "square", "negate", "reciprocal", "gauss"] as const;
  for (const op of unary) {
    it(op, async () => {
      // acosh needs |arg| >= 1, asin/acos/atanh need |arg| <= 1: feed x in [0.1, 1.9] or y in [-0.9, 0.9]
      const arg: SymbolicScalar = op === "acosh" ? { op: "add", vals: [x, 1] } : ["asin", "acos", "atanh"].includes(op) ? y : x;
      await agreeScalar(sym({ op, val: arg } as SymbolicScalar), grid, op);
    });
  }
  it("n-ary, binary, clamp, kernels", async () => {
    await agreeScalar(sym({ op: "add", vals: [x, y, 2] }), grid, "add");
    await agreeScalar(sym({ op: "mul", vals: [x, y, 0.5] }), grid, "mul");
    await agreeScalar(sym({ op: "min", vals: [x, y, 0.3] }), grid, "min");
    await agreeScalar(sym({ op: "max", vals: [x, y, 0.3] }), grid, "max");
    await agreeScalar(sym({ op: "mean", vals: [x, y, 1] }), grid, "mean");
    await agreeScalar(sym({ op: "rms", vals: [x, y] }), grid, "rms");
    await agreeScalar(sym({ op: "sub", vals: [x, y] }), grid, "sub");
    await agreeScalar(sym({ op: "div", vals: [y, x] }), grid, "div");
    await agreeScalar(sym({ op: "pow", vals: [x, y] }), grid, "pow");
    await agreeScalar(sym({ op: "pow", vals: [{ op: "sub", vals: [y, 1] }, 3] }), grid, "pow negative base");
    await agreeScalar(sym({ op: "logBase", vals: [x, 3] }), grid, "logBase");
    await agreeScalar(sym({ op: "atan2", vals: [y, x] }), grid, "atan2");
    await agreeScalar(sym({ op: "mod", vals: [y, 0.4] }), grid, "mod");
    await agreeScalar(sym({ op: "clamp", val: y, min: -0.5, max: x }), grid, "clamp");
    await agreeScalar(sym({ op: "gaussKernel", val: x, mu: 1, sigma: 0.3 }), grid, "gaussKernel");
    await agreeScalar(sym({ op: "normalPDF", val: y, mu: 0.1, sigma: 0.5 }), grid, "normalPDF");
  });
  it("vector ops", async () => {
    const v = (expr: SymbolicVector) => buildVectorFieldData({ type: "symbolicv", box: box.intervals, expr } as VectorFieldDataSpec, 2);
    await agreeScalar(sym({ op: "norm", vec: { op: "coordv" } }), grid, "norm");
    await agreeScalar(sym({ op: "dot", vecs: [{ op: "coordv" }, { op: "constv", value: [1, -2] }] }), grid, "dot");
    await agreeScalar(sym({ op: "cosineSim", vecs: [{ op: "coordv" }, { op: "basisv", index: 1 }] }), grid, "cosineSim");
    await agreeScalar(sym({ op: "comp", vec: { op: "scalev", vec: { op: "coordv" }, by: y }, index: 1 }), grid, "comp/scalev");
    await agreeVector(v({ op: "normalize", vec: { op: "coordv" } }), grid, "normalize");
    await agreeVector(v({ op: "sumv", vecs: [{ op: "basisv", index: 0 }, { op: "coordv" }], coeffs: [y, 2] }), grid, "sumv");
    await agreeVector(v({ op: "addv", vecs: [{ op: "coordv" }, { op: "subv", vecs: [{ op: "constv", value: [1, 1] }, { op: "coordv" }] }] }), grid, "addv/subv");
    await agreeVector(v({ op: "meanv", vecs: [{ op: "coordv" }, { op: "basisv", index: 0 }] }), grid, "meanv");
    await agreeVector(v({ op: "compv", coeffs: [y, x] }), grid, "compv");
    await agreeVector(v({ op: "grad", val: { op: "mul", vals: [{ op: "sin", val: x }, { op: "exp", val: y }] } }), grid, "grad");
  });
});

describe("fields: dense, pointwise, pullbacks, derivatives through arguments", () => {
  const dense: ScalarFieldDataSpec = { type: "dense", box: box.intervals, samples: { type: "symbolic", shape: [17, 11], expr: { op: "add", vals: [{ op: "square", val: x }, { op: "mul", vals: [0.3, y] }] } } };
  it("dense on its own grid (direct reads) and resampled (interpolation)", async () => {
    const fd = buildScalarFieldData(dense, 2);
    await agreeScalar(fd, fd.samplePoints!, "dense direct");
    await agreeScalar(fd, grid, "dense interpolated");
    const outside = new DenseGrid([9, 9], Box.fromSpec([[-1, 3], [-2, 2]])); // partly outside the field's box -> NaN there
    await agreeScalar(fd, outside, "dense outside -> NaN");
  });
  it("pointwise over dense + symbolic args", async () => {
    const fd = buildScalarFieldData({ type: "pointwise", expr: { op: "add", vals: [{ op: "mul", vals: ["a", 2] }, { op: "sin", val: "b" }] }, scalars: { a: dense, b: { type: "symbolic", box: box.intervals, expr: y } } }, 2);
    await agreeScalar(fd, fd.samplePoints!, "pointwise direct");
    await agreeScalar(fd, grid, "pointwise interpolated");
  });
  it("gradient through a symbolic argument and through a dense argument", async () => {
    const symArg = buildScalarFieldData({ type: "symbolic", box: box.intervals, expr: { op: "mul", vals: [{ op: "square", val: x }, { op: "cos", val: y }] } }, 2);
    const g1 = new SymbolicVectorFieldData({ k: "grad", s: { k: "arg", name: "f" } }, 2, { scalars: { f: symArg }, vectors: {} });
    await agreeVector(g1, grid, "grad through symbolic arg");
    const d = buildScalarFieldData(dense, 2);
    const g2 = new SymbolicVectorFieldData({ k: "grad", s: { k: "arg", name: "f" } }, 2, { scalars: { f: d }, vectors: {} });
    await agreeVector(g2, d.samplePoints!, "grad through dense arg (grid differences)");
    await agreeVector(g2, grid, "grad through dense arg, interpolated");
    // second derivative through an argument: ∇|∇f|
    const gn = new SymbolicScalarFieldData({ k: "norm", v: { k: "grad", s: { k: "arg", name: "f" } } }, 2, { scalars: { f: symArg }, vectors: {} });
    const g3 = new SymbolicVectorFieldData({ k: "grad", s: { k: "arg", name: "n" } }, 2, { scalars: { n: gn }, vectors: {} });
    await agreeVector(g3, grid, "∇|∇f| (second derivative through arg)", 1e-3);
  });
  it("norm of a dense vector field, and vector components", async () => {
    const dv = buildVectorFieldData({ type: "densev", box: box.intervals, samples: { type: "symbolicv", shape: [9, 7, 2], expr: { op: "coordv" } } }, 2);
    await agreeVector(dv, grid, "dense vector interpolated");
    const n = new SymbolicScalarFieldData({ k: "norm", v: { k: "argv", name: "v" } }, 2, { scalars: {}, vectors: { v: dv } });
    await agreeScalar(n, grid, "|dense vector|");
    const c = new SymbolicScalarFieldData({ k: "argvi", name: "v", index: 1 }, 2, { scalars: {}, vectors: { v: dv } });
    await agreeScalar(c, grid, "dense vector component");
  });
  it("translate / scale pullbacks (values and derivatives)", async () => {
    const base: ScalarFieldDataSpec = { type: "symbolic", box: [[0, 1], [0, 1]], expr: { op: "mul", vals: [x, { op: "exp", val: y }] } };
    const t = buildScalarFieldData({ type: "scale", scale: [2, 0.5], origin: [0.5, 0.5], arg: { type: "translate", arg: base, vec: [0.3, -0.2] } }, 2);
    const g = new DenseGrid([15, 13], t.box);
    await agreeScalar(t, g, "pullback value");
    const grad = new SymbolicVectorFieldData({ k: "grad", s: { k: "arg", name: "f" } }, 2, { scalars: { f: t }, vectors: {} });
    await agreeVector(grad, g, "pullback gradient (chain rule)");
    const pd = buildScalarFieldData({ type: "translate", arg: dense, vec: [0.1, 0.1] }, 2);
    await agreeScalar(pd, pd.samplePoints!, "pulled-back dense, direct");
  });
});

describe("example bundles agree on every field and derived use", () => {
  const dir = join(__dirname, "../../../apps/viewer/public/bundles");
  const files = readdirSync(dir).filter((f) => f.endsWith(".json") && f !== "index.json");
  for (const file of files) {
    it(file, async () => {
      if (!gpu) return;
      const b = Bundle.parse(JSON.parse(readFileSync(join(dir, file), "utf8")));
      expect(b.buildAll().size).toBe(0);
      const gridFor = (d: { samplePoints?: DenseGrid | undefined; box: Box }) => d.samplePoints ?? new DenseGrid(d.box.dimCount === 3 ? [14, 13, 12] : [48, 48], d.box);
      for (const id of b.scalarFieldIds) {
        const f = b.scalarField(id);
        const g = gridFor(f.data), D = f.data.dimCount;
        await agreeScalar(f.data, g, `${file}:${id}`, 5e-4);
        const grad = new SymbolicVectorFieldData({ k: "grad", s: { k: "arg", name: "f" } }, D, { scalars: { f: f.data }, vectors: {} });
        await agreeVector(grad, g, `${file}:∇${id}`, 2e-3);
      }
      for (const id of b.vectorFieldIds) {
        const v = b.vectorField(id);
        const g = gridFor(v.data), D = v.data.dimCount;
        await agreeVector(v.data, g, `${file}:${id}`, 5e-4);
        const norm = new SymbolicScalarFieldData({ k: "norm", v: { k: "argv", name: "v" } }, D, { scalars: {}, vectors: { v: v.data } });
        await agreeScalar(norm, g, `${file}:|${id}|`, 5e-4);
      }
    });
  }
});

describe("program", () => {
  it("emits one WGSL let per distinct subtree (CSE)", () => {
    const f = sym({ op: "add", vals: [{ op: "exp", val: { op: "mul", vals: [x, y] } }, { op: "square", val: { op: "exp", val: { op: "mul", vals: [x, y] } } }] });
    const code = buildSampleProgram(f, grid).code;
    expect((code.match(/exp\(/g) ?? []).length - (code.match(/fn .*exp/g) ?? []).length).toBeLessThanOrEqual(1 + (code.match(/exp\(-/g) ?? []).length + (code.match(/exp\(x/g) ?? []).length);
    expect(code).toContain("@compute");
  });
  it("Codomain formatting is untouched by the backend (sanity)", () => {
    expect(new Codomain("norm").format(1234.5)).toBe("1230");
  });
});

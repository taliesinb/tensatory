// Reverse-mode autodiff (nets/autodiff.ts): the gradient program of every op
// agrees with central finite differences of the forward program.

import { describe, expect, it } from "vitest";
import type { ArrayExpr, BundleSpec, NetDefinitionSpec, NetSpec } from "@tensatory/schema";
import { Bundle, DenseGrid, NdArray, SymbolicVectorFieldData } from "../src";
import { readFileSync } from "node:fs";
import { join } from "node:path";

const arr = (data: number[], shape = [data.length]) => new NdArray(shape, Float64Array.from(data));
const inline = (data: number[], shape = [data.length]) => ({ type: "inline", shape, data }) as const;
const bundle = (nets: Record<string, NetSpec>, fields: BundleSpec["fields"] = {}) =>
  Bundle.parse({ tensatory: "0.1", manifolds: { m: { numDims: 2 } }, fields, nets } satisfies BundleSpec);

/**
 * A net `f` with inputs `x` (and constants), a scalar output `s = sum(body)`, and a weighted sum so that
 * every element of `body` matters: check d s / d x against central differences at a random point.
 */
function checkGrad(body: ArrayExpr, inputs: Record<string, number[]>, arrays: Record<string, ReturnType<typeof inline>> = {}, extra: Record<string, NetSpec> = {}, tol = 1e-5) {
  const net: NetDefinitionSpec = {
    type: "def", inputs: Object.fromEntries(Object.entries(inputs).map(([k, v]) => [k, [v.length]])), arrays,
    nodes: { r: body, s: { op: "reduce", fn: "sum", val: { op: "mul", vals: ["r", "r", 0.5] } } },
    outputs: { s: [] },
  };
  // s = ½ Σ r²  (so ds/dr = r, exercising the chain through `body` with a non-trivial upstream gradient)
  const names = Object.keys(inputs);
  const b = bundle({ ...extra, f: net, g: { type: "grad", net: "f", outputs: Object.fromEntries(names.map((k) => [`g_${k}`, { of: "s", wrt: k }])) } });
  const vals = Object.fromEntries(names.map((k) => [k, arr(inputs[k]!)]));
  const grads = b.net("g").evaluate(vals);
  const f = (v: Record<string, NdArray>) => b.net("f").evaluate(v).s!.data[0]!;
  for (const k of names) {
    const g = grads[`g_${k}`]!;
    expect([...g.shape]).toEqual([inputs[k]!.length]);
    for (let i = 0; i < inputs[k]!.length; i++) {
      const h = 1e-6 * (1 + Math.abs(inputs[k]![i]!));
      const plus = { ...vals, [k]: arr(inputs[k]!.map((x, j) => (j === i ? x + h : x))) };
      const minus = { ...vals, [k]: arr(inputs[k]!.map((x, j) => (j === i ? x - h : x))) };
      const fd = (f(plus) - f(minus)) / (2 * h);
      expect(Math.abs(g.data[i]! - fd), `d/d${k}[${i}]: autodiff ${g.data[i]} vs fd ${fd}`).toBeLessThan(tol * (1 + Math.abs(fd)));
    }
  }
}

const x = [0.7, -1.3, 0.4, 1.9, -0.2, 0.9];
const xp = [0.7, 1.3, 0.4, 1.9, 0.2, 0.9]; // positive (logs, roots, pow)
const M = inline([0.51, -1.03, 2.07, 0.29, 1.53, -0.71, 0.4, 1.1, -0.6, 0.35, 0.8, -1.2], [2, 6]);

describe("autodiff: every op vs finite differences", () => {
  const unary = ["sin", "cos", "tan", "sinh", "cosh", "tanh", "atan", "asinh", "relu", "sigmoid", "gelu", "silu", "softplus", "elu", "erf", "abs", "exp", "exp2", "exp10", "square", "negate", "gauss", "expm1"] as const;
  for (const op of unary) it(op, () => checkGrad({ op, val: "x" }, { x }));
  const positive = ["log", "log2", "log10", "log1p", "sqrt", "reciprocal", "plogp", "acosh"] as const;
  for (const op of positive) it(op, () => checkGrad({ op, val: op === "acosh" ? { op: "add", vals: ["x", 1] } : "x" }, { x: xp }));
  it("asin / acos / atanh (|x| < 1)", () => {
    const u = [0.3, -0.5, 0.1, 0.8, -0.2, 0.6];
    checkGrad({ op: "add", vals: [{ op: "asin", val: "x" }, { op: "acos", val: "x" }, { op: "atanh", val: "x" }] }, { x: u });
  });
  it("floor / ceil / round / sign: zero gradient", () => {
    const b = bundle({ f: { type: "def", inputs: { x: [3] }, nodes: { s: { op: "reduce", fn: "sum", val: { op: "add", vals: [{ op: "floor", val: "x" }, { op: "sign", val: "x" }] } } }, outputs: { s: [] } }, g: { type: "grad", net: "f", outputs: { gx: { of: "s", wrt: "x" } } } });
    expect(Array.from(b.net("g").evaluate({ x: arr([0.3, 1.7, -2.2]) }).gx!.data).map((v) => v + 0)).toEqual([0, 0, 0]);
  });

  it("nary: add / mul / min / max / mean / rms with broadcasting", () => {
    checkGrad({ op: "add", vals: ["x", "y", 2] }, { x, y: [0.5, -0.4, 0.3, 1.2, 0.8, -1.1] });
    checkGrad({ op: "mul", vals: ["x", "y", "M"] }, { x, y: [0.5, -0.4, 0.3, 1.2, 0.8, -1.1] }, { M });
    checkGrad({ op: "min", vals: ["x", "y"] }, { x, y: [0.5, -0.4, 0.3, 1.2, 0.8, -1.1] });
    checkGrad({ op: "max", vals: ["x", "M"] }, { x }, { M });
    checkGrad({ op: "mean", vals: ["x", "y", "M"] }, { x, y: [0.5, -0.4, 0.3, 1.2, 0.8, -1.1] }, { M });
    checkGrad({ op: "rms", vals: ["x", "y"] }, { x, y: [0.5, -0.4, 0.3, 1.2, 0.8, -1.1] });
  });
  it("binary: sub / div / pow / logBase / atan2 / mod", () => {
    const y = [0.5, -0.4, 0.3, 1.2, 0.8, -1.1];
    checkGrad({ op: "sub", vals: ["x", "M"] }, { x }, { M });
    checkGrad({ op: "div", vals: ["M", "y"] }, { y }, { M });
    checkGrad({ op: "pow", vals: ["x", "y"] }, { x: xp, y });
    checkGrad({ op: "logBase", vals: ["x", "y"] }, { x: xp, y: [2.5, 3.4, 1.3, 1.2, 4.8, 2.1] });
    checkGrad({ op: "atan2", vals: ["x", "y"] }, { x, y });
    checkGrad({ op: "mod", vals: ["x", 0.73] }, { x });
  });
  it("clamp / where / compare", () => {
    checkGrad({ op: "clamp", val: "x", min: -0.5, max: "y" }, { x, y: [0.5, 0.6, 0.3, 1.2, 0.8, 1.1] });
    checkGrad({ op: "where", cond: { op: "gt", vals: ["x", 0] }, vals: [{ op: "mul", vals: ["x", "y"] }, { op: "square", val: "y" }] }, { x, y: [0.5, -0.4, 0.3, 1.2, 0.8, -1.1] });
  });

  it("matmul / einsum (incl. broadcast letters and letters only in one operand)", () => {
    checkGrad({ op: "matmul", vals: ["M", "x"] }, { x }, { M });
    checkGrad({ op: "matmul", vals: ["x", { op: "transpose", val: "M" }] }, { x }, { M });
    checkGrad({ op: "matmul", vals: [{ op: "reshape", val: "x", shape: [2, 3] }, { op: "reshape", val: "y", shape: [3, 2] }] }, { x, y: [0.5, -0.4, 0.3, 1.2, 0.8, -1.1] });
    checkGrad({ op: "einsum", subscripts: "i,j->ij", vals: ["x", "y"] }, { x, y: [0.5, -0.4] });
    checkGrad({ op: "einsum", subscripts: "ij,j->i", vals: ["M", "x"] }, { x }, { M });
    checkGrad({ op: "einsum", subscripts: "ij->j", vals: [{ op: "reshape", val: "x", shape: [2, 3] }] }, { x });
    checkGrad({ op: "einsum", subscripts: "ij,kj->ik", vals: ["M", { op: "reshape", val: "x", shape: [1, 6] }] }, { x }, { M });
  });
  it("reductions", () => {
    const X: ArrayExpr = { op: "reshape", val: "x", shape: [2, 3] };
    for (const fn of ["sum", "mean", "max", "min", "prod", "logsumexp"] as const) {
      checkGrad({ op: "reduce", fn, val: X, axes: [1] }, { x });
      checkGrad({ op: "reduce", fn, val: X, axes: [0], keepDims: true }, { x });
      checkGrad({ op: "reduce", fn, val: X }, { x });
    }
  });
  it("softmax / logSoftmax", () => {
    const X: ArrayExpr = { op: "reshape", val: "x", shape: [2, 3] };
    checkGrad({ op: "softmax", val: X }, { x });
    checkGrad({ op: "logSoftmax", val: X, axis: 0 }, { x });
  });
  it("shape ops: reshape / transpose / concat / slice / takeAlong; oneHot & argmax are constant", () => {
    const X: ArrayExpr = { op: "reshape", val: "x", shape: [2, 3] };
    checkGrad({ op: "transpose", val: X }, { x });
    checkGrad({ op: "transpose", val: { op: "reshape", val: "x", shape: [1, 2, 3] }, perm: [2, 0, 1] }, { x });
    checkGrad({ op: "concat", vals: [X, "M", { op: "reshape", val: "y", shape: [2, 1] }], axis: 1 }, { x, y: [0.5, -0.4] }, { M });
    checkGrad({ op: "slice", val: X, axis: 1, start: 1 }, { x });
    checkGrad({ op: "slice", val: "x", axis: 0, step: -2 }, { x });
    checkGrad({ op: "slice", val: "x", axis: 0, start: 1, stop: 5, step: 2 }, { x });
    checkGrad({ op: "takeAlong", val: X, indices: { op: "reshape", val: "i", shape: [2, 2] }, axis: 1 }, { x, i: [2, 0, 1, 1] });
    checkGrad({ op: "takeAlong", val: X, indices: "I", axis: 0 }, { x }, { I: inline([1, 0, 1], [1, 3]) });
    checkGrad({ op: "mul", vals: [{ op: "oneHot", val: { op: "argmax", val: X }, size: 3 }, X] }, { x });
    const sg = bundle({ f: { type: "def", inputs: { x: [2] }, nodes: { s: { op: "reduce", fn: "sum", val: { op: "mul", vals: ["x", { op: "stopGradient", val: "x" }] } } }, outputs: { s: [] } }, g: { type: "grad", net: "f", outputs: { gx: { of: "s", wrt: "x" } } } });
    expect(Array.from(sg.net("g").evaluate({ x: arr([2, 3]) }).gx!.data)).toEqual([2, 3]); // d(x·sg(x))/dx = sg(x)
  });
  it("call: the callee's gradient net is called with the seed", () => {
    const sq: NetSpec = { type: "def", inputs: { a: ["N"] }, nodes: { s: { op: "reduce", fn: "sum", val: { op: "mul", vals: ["a", { op: "sin", val: "a" }] } } }, outputs: { s: [] } };
    checkGrad({ op: "call", net: "sq", inputs: { a: { op: "reshape", val: "x", shape: [2, 3] } }, output: "s" }, { x }, {}, { sq }); // batched call: [2] outputs
  });
  it("displace: gradient wrt the coefficients", () => {
    const bowl: NetSpec = { type: "def", inputs: { p: [2] }, nodes: { h: { op: "square", val: "p" }, f: { op: "reduce", fn: "sum", val: "h" } }, outputs: { f: [] } };
    const around: NetSpec = { type: "displace", net: { type: "bind", net: "bowl", bind: { p: inline([1, 2]) } }, directions: [{ arrays: { p: inline([1, 0]) } }, { arrays: { p: inline([0, 1]) } }] };
    checkGrad({ op: "call", net: "around", inputs: { t: "x" }, output: "f" }, { x: [0.3, -0.7] }, {}, { bowl, around });
  });
  it("einsum diagonals: extraction and embedding are each other's adjoints (the op set is closed)", () => {
    const A: ArrayExpr = { op: "reshape", val: "x", shape: [3, 3] };
    checkGrad({ op: "einsum", subscripts: "ii->", vals: [A] }, { x: [...x, 0.3, -0.8, 1.1] });                  // trace
    checkGrad({ op: "einsum", subscripts: "ii->i", vals: [{ op: "mul", vals: [A, "M3"] }] }, { x: [...x, 0.3, -0.8, 1.1] }, { M3: inline([0.5, -1, 2, 0.25, 1.5, -0.75, 0.9, 1.1, -0.4], [3, 3]) });
    checkGrad({ op: "einsum", subscripts: "i->ii", vals: ["y"] }, { y: [0.5, -0.4, 0.3] });                      // embedding
    checkGrad({ op: "einsum", subscripts: "i,j->iij", vals: ["y", "z"] }, { y: [0.5, -0.4], z: [1, 10, 0.3] });
    // and once more: the gradient of the gradient (the adjoint of an embedding is an extraction, and back)
    const b = bundle({
      f: { type: "def", inputs: { y: [2] }, nodes: { D: { op: "einsum", subscripts: "i->ii", vals: ["y"] }, s: { op: "reduce", fn: "sum", val: { op: "mul", vals: ["D", "D", "D"] } } }, outputs: { s: [] } },
      g: { type: "grad", net: "f", outputs: { gy: { of: "s", wrt: "y" } } },
      h: { type: "grad", net: "g", outputs: { Hv: { of: "gy", wrt: "y", seed: "v" } } },
    });
    // s = Σ y³ (only the diagonal is non-zero): gy = 3y², Hv = 6 y ⊙ v
    expect(Array.from(b.net("g").evaluate({ y: arr([2, 3]) }).gy!.data)).toEqual([12, 27]);
    expect(Array.from(b.net("h").evaluate({ y: arr([2, 3]), v: arr([1, -1]) }).Hv!.data)).toEqual([12, -18]);
  });
});

describe("grad nets", () => {
  const mlp: NetDefinitionSpec = {
    type: "def",
    inputs: { x: ["N", 3], y: ["N"], W1: [3, 4], b1: [4], W2: [4, 2], b2: [2] },
    nodes: {
      h: { op: "relu", val: { op: "add", vals: [{ op: "matmul", vals: ["x", "W1"] }, "b1"] } },
      logits: { op: "add", vals: [{ op: "matmul", vals: ["h", "W2"] }, "b2"] },
      logp: { op: "logSoftmax", val: "logits" },
      nll: { op: "negate", val: { op: "takeAlong", val: "logp", axis: -1, indices: { op: "reshape", val: "y", shape: ["N", 1] } } },
      loss: { op: "reduce", fn: "mean", val: "nll" },
    },
    outputs: { loss: [], logits: ["N", 2] },
  };
  const data = { x: inline([1, 0, 0.5, 0, 1, -0.5, 0.2, 0.3, 0.9], [3, 3]), y: inline([0, 1, 1]) };
  const W = { W1: arr([0.3, -0.2, 0.5, 0.1, 0.4, 0.7, -0.6, 0.2, -0.1, 0.8, 0.3, -0.4], [3, 4]), b1: arr([0.1, -0.1, 0.05, 0.2]), W2: arr([0.5, -0.5, 0.3, 0.2, -0.7, 0.4, 0.1, 0.6], [4, 2]), b2: arr([0.05, -0.05]) };
  const nets: Record<string, NetSpec> = {
    mlp,
    val: { type: "bind", net: "mlp", bind: data },
    grad: { type: "grad", net: "val", keep: ["loss"], outputs: { gW1: { of: "loss", wrt: "W1" }, gb1: { of: "loss", wrt: "b1" }, gW2: { of: "loss", wrt: "W2" }, gb2: { of: "loss", wrt: "b2" }, gh: { of: "loss", wrt: "h" } } },
  };

  it("d loss / d weights match finite differences; kept outputs pass through; internal node gradients too", () => {
    const b = bundle(nets);
    const out = b.net("grad").evaluate(W);
    expect(out.loss!.data[0]).toBeCloseTo(b.net("val").evaluate(W).loss!.data[0]!, 12);
    expect([...out.gh!.shape]).toEqual([3, 4]);
    const f = (w: typeof W) => b.net("val").evaluate(w).loss!.data[0]!;
    for (const k of ["W1", "b1", "W2", "b2"] as const) {
      const g = out[`g${k}`]!;
      expect([...g.shape]).toEqual([...W[k].shape]);
      for (let i = 0; i < g.size; i++) {
        const h = 1e-6;
        const plus = { ...W, [k]: new NdArray(W[k].shape, W[k].data.map((v, j) => (j === i ? v + h : v))) };
        const minus = { ...W, [k]: new NdArray(W[k].shape, W[k].data.map((v, j) => (j === i ? v - h : v))) };
        expect(g.data[i], `${k}[${i}]`).toBeCloseTo((f(plus) - f(minus)) / (2 * h), 6);
      }
    }
  });

  it("per-point gradients when the weights are batched", () => {
    const b = bundle(nets);
    const W1b = new NdArray([2, 3, 4], Float64Array.from([...W.W1.data, ...W.W1.data.map((v) => v * 0.5)]));
    const out = b.net("grad").evaluate({ ...W, W1: W1b });
    expect([...out.gW1!.shape]).toEqual([2, 3, 4]);
    expect([...out.loss!.shape]).toEqual([2]);
    const single = b.net("grad").evaluate({ ...W, W1: new NdArray([3, 4], W1b.data.slice(12)) });
    for (let i = 0; i < 12; i++) expect(out.gW1!.data[12 + i]).toBeCloseTo(single.gW1!.data[i]!, 12);
  });

  it("seeds: a VJP with a named seed and a Hessian-vector product", () => {
    const b = bundle({
      ...nets,
      j: { type: "grad", net: "val", outputs: { g: { of: "logits", wrt: "b2", seed: "v" } } },
      hvp: { type: "grad", net: "grad", outputs: { Hv: { of: "gb2", wrt: "b2", seed: "v" } } },
    });
    // VJP: v · d logits / d b2 = sum over examples of v (logits = ... + b2)
    const v = arr([1, 2, 3, 4, 5, 6], [3, 2]);
    const g = b.net("j").evaluate({ ...W, v }).g!;
    expect(Array.from(g.data)).toEqual([9, 12]);
    // HVP against finite differences of the gradient
    const vb = arr([0.3, -0.7]);
    const Hv = b.net("hvp").evaluate({ ...W, v: vb }).Hv!;
    const grad = (w: typeof W) => b.net("grad").evaluate(w).gb2!.data;
    const h = 1e-5;
    const gp = grad({ ...W, b2: arr([W.b2.data[0]! + h * 0.3, W.b2.data[1]! - h * 0.7]) });
    const gm = grad({ ...W, b2: arr([W.b2.data[0]! - h * 0.3, W.b2.data[1]! + h * 0.7]) });
    for (let i = 0; i < 2; i++) expect(Hv.data[i]).toBeCloseTo((gp[i]! - gm[i]!) / (2 * h), 5);
  });

  it("a gradient wrt an array that does not influence the output is zero", () => {
    const b = bundle({ f: { type: "def", inputs: { a: [2], c: [3] }, nodes: { s: { op: "reduce", fn: "sum", val: "a" } }, outputs: { s: [] } }, g: { type: "grad", net: "f", outputs: { gc: { of: "s", wrt: "c" } } } });
    expect(Array.from(b.net("g").evaluate({ a: arr([1, 2]), c: arr([1, 2, 3]) }).gc!.data)).toEqual([0, 0, 0]);
  });
});

describe("exact derivatives of net fields", () => {
  const b = Bundle.parse(JSON.parse(readFileSync(join(__dirname, "../../../apps/viewer/public/bundles/iris.json"), "utf8")));

  it("∂loss/∂t via autodiff matches central differences of the field", () => {
    const loss = b.scalarField("loss2").data;
    for (const p of [[0.1, -0.2], [0.5, 0.4], [-0.7, 0.9]]) {
      for (const d of [0, 1]) {
        const h = 1e-6;
        const q = (s: number) => p.map((v, i) => (i === d ? v + s : v));
        const fd = (loss.value(q(h))! - loss.value(q(-h))!) / (2 * h);
        expect(loss.derivative(d).value(p)).toBeCloseTo(fd, 6);
      }
    }
  });

  it("the gradient use sampled on a grid equals the autodiff gradient, and second derivatives compose", () => {
    const loss = b.scalarField("loss3").data;
    const grad = new SymbolicVectorFieldData({ k: "grad", s: { k: "arg", name: "f" } }, 3, { scalars: { f: loss }, vectors: {} });
    const grid = new DenseGrid([6, 6, 6], loss.box);
    const vals = grad.sampleOn(grid);
    const p = new Float64Array(3);
    for (const i of [0, 77, 215]) {
      grid.pointInto(i, p);
      for (let d = 0; d < 3; d++) expect(vals[3 * i + d]).toBeCloseTo(loss.derivative(d).value(p)!, 10);
    }
    // ∂²/∂t0∂t1 by two rewrites vs a difference of first derivatives
    const d01 = loss.derivative(0).derivative(1);
    const pt = [0.2, -0.3, 0.1];
    const h = 1e-5;
    const fd = (loss.derivative(0).value([0.2, -0.3 + h, 0.1])! - loss.derivative(0).value([0.2, -0.3 - h, 0.1])!) / (2 * h);
    expect(d01.value(pt)).toBeCloseTo(fd, 4);
  });

  it("inputs as coordinate expressions differentiate through the fold", () => {
    const bowl: NetSpec = { type: "def", inputs: { p: [2], k: [] }, nodes: { f: { op: "mul", vals: [{ op: "reduce", fn: "sum", val: { op: "square", val: "p" } }, "k"] } }, outputs: { f: [] } };
    const bb = bundle({ bowl }, {
      f: { kind: "scalar", data: { type: "net", net: "bowl", output: "f", inputs: { p: { op: "add", vals: [{ op: "coordv" }, "c"] }, k: { op: "add", vals: [{ op: "coord", index: 0 }, 2] } }, arrays: { c: inline([1, 1]) } } },
    });
    const f = bb.scalarField("f").data;
    // f = ((x+1)² + (y+1)²)(x + 2): ∂/∂x = 2(x+1)(x+2) + (x+1)² + (y+1)²
    const [x, y] = [0.3, 0.4]; // inside the default unit box
    expect(f.derivative(0).value([x, y])).toBeCloseTo(2 * (x + 1) * (x + 2) + (x + 1) ** 2 + (y + 1) ** 2, 10);
    expect(f.derivative(1).value([x, y])).toBeCloseTo(2 * (y + 1) * (x + 2), 10);
  });
});

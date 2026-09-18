import { describe, expect, it } from "vitest";
import type { ArrayExpr, BundleSpec, NetDefinitionSpec, NetSpec } from "@tensatory/schema";
import { Bundle, DenseGrid, NdArray, NetSchema, NotSupportedError, inferExpr, inferNet, type Shape } from "../src";

// the MLP of schema/nets.ts
const mlp: NetDefinitionSpec = {
  type: "def",
  inputs: { x: ["N", 784], y: ["N"], W1: [784, 128], b1: [128], W2: [128, 10], b2: [10] },
  nodes: {
    h: { op: "relu", val: { op: "add", vals: [{ op: "matmul", vals: ["x", "W1"] }, "b1"] } },
    logits: { op: "add", vals: [{ op: "matmul", vals: ["h", "W2"] }, "b2"] },
    logp: { op: "logSoftmax", val: "logits" },
    nll: { op: "negate", val: { op: "takeAlong", val: "logp", axis: -1, indices: { op: "reshape", val: "y", shape: ["N", 1] } } },
    loss: { op: "reduce", fn: "mean", val: "nll" },
    pred: { op: "argmax", val: "logits" },
    acc: { op: "reduce", fn: "mean", val: { op: "eq", vals: ["pred", "y"] } },
  },
  outputs: { loss: [], acc: [], logits: ["N", 10] },
};

const constant = (shape: number[]) => ({ type: "constant", shape, value: 0 }) as const;
const valSet = (n: number) => ({ x: constant([n, 784]), y: constant([n]) });
const weights = (g: number[] = []) => ({ W1: constant([...g, 784, 128]), b1: constant([...g, 128]), W2: constant([...g, 128, 10]), b2: constant([...g, 10]) });

const bundle = (nets: Record<string, NetSpec>, fields: BundleSpec["fields"] = {}): Bundle =>
  Bundle.parse({ tensatory: "0.1", manifolds: { pca: { numDims: 2 } }, fields, nets } satisfies BundleSpec);

// expression shapes in a scope of named arrays
const shapeOf = (e: ArrayExpr, names: Record<string, Shape>, axisNames: string[] = []) =>
  inferExpr(e, { names: new Map(Object.entries(names)), axisNames: new Set(axisNames), nets: { net: () => { throw new Error("no nets here"); } } });

describe("net syntax", () => {
  it("validates", () => {
    expect(NetSchema.safeParse(mlp).success).toBe(true);
    expect(NetSchema.safeParse({ type: "def", inputs: {}, outputs: {}, nodes: { a: { op: "nope", val: 1 } } }).success).toBe(false);
    expect(NetSchema.safeParse({ type: "bind", net: "mlp" }).success).toBe(false);
    expect(NetSchema.safeParse({ type: "grad", net: "mlp", outputs: { g: { of: "loss" } } }).success).toBe(false);
  });
});

describe("array expression shapes", () => {
  const S = { x: ["N", 784] as Shape, W: [784, 128] as Shape, b: [128] as Shape, v: [4] as Shape, M: [3, 4] as Shape, y: ["N"] as Shape };

  it("broadcasts elementwise ops", () => {
    expect(shapeOf({ op: "add", vals: [{ op: "matmul", vals: ["x", "W"] }, "b"] }, S)).toEqual(["N", 128]);
    expect(shapeOf({ op: "mul", vals: ["M", 2] }, S)).toEqual([3, 4]);
    expect(shapeOf({ op: "clamp", val: "M", min: 0, max: "v" }, S)).toEqual([3, 4]);
    expect(shapeOf({ op: "where", cond: { op: "gt", vals: ["v", 0] }, vals: ["v", 0] }, S)).toEqual([4]);
    expect(shapeOf({ op: "relu", val: 2 }, S)).toEqual([]);
    expect(() => shapeOf({ op: "add", vals: ["M", { op: "reshape", val: "v", shape: [4, 1] }] }, S)).toThrow(/do not broadcast/);
    expect(() => shapeOf({ op: "add", vals: ["x", "b"] }, S)).toThrow(/do not broadcast/);
    expect(() => shapeOf("nope", S)).toThrow(/unknown name/);
  });

  it("does not equate symbolic and numeric sizes", () => {
    expect(() => shapeOf({ op: "add", vals: ["y", "v"] }, S)).toThrow(/do not broadcast/);
    expect(() => shapeOf({ op: "add", vals: ["y", { op: "reshape", val: "v", shape: ["M"] }] }, { ...S, v: ["M"] }, ["M"])).toThrow(/do not broadcast/);
  });

  it("matmul follows numpy", () => {
    expect(shapeOf({ op: "matmul", vals: ["v", "v"] }, S)).toEqual([]);
    expect(shapeOf({ op: "matmul", vals: ["M", "v"] }, S)).toEqual([3]);
    expect(shapeOf({ op: "matmul", vals: [{ op: "transpose", val: "M" }, "M"] }, S)).toEqual([4, 4]);
    expect(() => shapeOf({ op: "matmul", vals: ["M", "M"] }, S)).toThrow(/inner sizes/);
    expect(() => shapeOf({ op: "matmul", vals: [1, "M"] }, S)).toThrow(/rank/);
  });

  it("einsum", () => {
    expect(shapeOf({ op: "einsum", subscripts: "ij,jk->ik", vals: ["x", "W"] }, S)).toEqual(["N", 128]);
    expect(shapeOf({ op: "einsum", subscripts: "ij,jk", vals: ["x", "W"] }, S)).toEqual(["N", 128]); // implicit output
    expect(shapeOf({ op: "einsum", subscripts: "ij->", vals: ["M"] }, S)).toEqual([]);
    expect(shapeOf({ op: "einsum", subscripts: "ij->ji", vals: ["M"] }, S)).toEqual([4, 3]);
    expect(shapeOf({ op: "einsum", subscripts: "i,j->ij", vals: ["v", "b"] }, S)).toEqual([4, 128]);
    expect(() => shapeOf({ op: "einsum", subscripts: "ij,ij->i", vals: ["M", "W"] }, S)).toThrow(/letter "i"/);
    expect(() => shapeOf({ op: "einsum", subscripts: "ij->k", vals: ["M"] }, S)).toThrow(/does not appear/);
    expect(shapeOf({ op: "einsum", subscripts: "i->ii", vals: ["v"] }, S)).toEqual([4, 4]); // diagonal embedding
    expect(shapeOf({ op: "einsum", subscripts: "ii->i", vals: [{ op: "einsum", subscripts: "i->ii", vals: ["v"] }] }, S)).toEqual([4]);
    expect(() => shapeOf({ op: "einsum", subscripts: "...ij->i", vals: ["M"] }, S)).toThrow(/not supported/);
    expect(() => shapeOf({ op: "einsum", subscripts: "ijk", vals: ["M"] }, S)).toThrow(/3 letters/);
  });

  it("reductions, argmax, softmax", () => {
    expect(shapeOf({ op: "reduce", fn: "sum", val: "x" }, S)).toEqual([]);
    expect(shapeOf({ op: "reduce", fn: "sum", val: "x", axes: [-1] }, S)).toEqual(["N"]);
    expect(shapeOf({ op: "reduce", fn: "max", val: "x", axes: [0], keepDims: true }, S)).toEqual([1, 784]);
    expect(shapeOf({ op: "reduce", fn: "logsumexp", val: "M", axes: [0, -2] }, S)).toEqual([4]); // deduplicated
    expect(() => shapeOf({ op: "reduce", fn: "sum", val: "M", axes: [2] }, S)).toThrow(/axis 2 out of range/);
    expect(shapeOf({ op: "argmax", val: "x" }, S)).toEqual(["N"]);
    expect(shapeOf({ op: "argmin", val: "M", axis: 0 }, S)).toEqual([4]);
    expect(shapeOf({ op: "softmax", val: "M" }, S)).toEqual([3, 4]);
    expect(() => shapeOf({ op: "logSoftmax", val: "M", axis: 5 }, S)).toThrow(/out of range/);
  });

  it("reshape with symbolic sizes", () => {
    expect(shapeOf({ op: "reshape", val: "y", shape: ["N", 1] }, S, ["N"])).toEqual(["N", 1]);
    expect(shapeOf({ op: "reshape", val: "x", shape: ["N", 28, 28] }, S, ["N"])).toEqual(["N", 28, 28]);
    expect(shapeOf({ op: "reshape", val: "x", shape: ["N", -1, 28] }, S, ["N"])).toEqual(["N", 28, 28]);
    expect(shapeOf({ op: "reshape", val: "M", shape: [-1] }, S)).toEqual([12]);
    expect(() => shapeOf({ op: "reshape", val: "x", shape: [-1, 28] }, S, ["N"])).toThrow(/symbolic sizes differ/);
    expect(() => shapeOf({ op: "reshape", val: "M", shape: [5, -1] }, S)).toThrow(/not a multiple/);
    expect(() => shapeOf({ op: "reshape", val: "M", shape: [5, 2] }, S)).toThrow(/12 elements/);
    expect(() => shapeOf({ op: "reshape", val: "M", shape: [-1, -1] }, S)).toThrow(/more than one -1/);
    expect(() => shapeOf({ op: "reshape", val: "y", shape: ["K"] }, S, ["N"])).toThrow(/unknown axis name "K"/);
  });

  it("transpose, concat, slice, oneHot, takeAlong", () => {
    expect(shapeOf({ op: "transpose", val: "x" }, S)).toEqual([784, "N"]);
    expect(shapeOf({ op: "transpose", val: "M", perm: [1, 0] }, S)).toEqual([4, 3]);
    expect(() => shapeOf({ op: "transpose", val: "M", perm: [0, 0] }, S)).toThrow(/not a permutation/);
    expect(shapeOf({ op: "concat", vals: ["M", "M"], axis: 0 }, S)).toEqual([6, 4]);
    expect(shapeOf({ op: "concat", vals: ["M", { op: "reshape", val: "v", shape: [1, 4] }], axis: -2 }, S)).toEqual([4, 4]);
    expect(() => shapeOf({ op: "concat", vals: ["M", "v"], axis: 0 }, S)).toThrow(/same rank/);
    expect(() => shapeOf({ op: "concat", vals: ["x", "x"], axis: 0 }, S)).toThrow(/symbolic axis "N"/);
    expect(shapeOf({ op: "slice", val: "M", axis: 1, start: 1 }, S)).toEqual([3, 3]);
    expect(shapeOf({ op: "slice", val: "M", axis: 1, start: -1 }, S)).toEqual([3, 1]);
    expect(shapeOf({ op: "slice", val: "M", axis: 1, step: 2 }, S)).toEqual([3, 2]);
    expect(shapeOf({ op: "slice", val: "M", axis: 1, step: -1 }, S)).toEqual([3, 4]);
    expect(shapeOf({ op: "slice", val: "M", axis: 1, start: 10 }, S)).toEqual([3, 0]);
    expect(shapeOf({ op: "slice", val: "x", axis: 1, stop: 10 }, S)).toEqual(["N", 10]);
    expect(() => shapeOf({ op: "slice", val: "x", axis: 0, stop: 10 }, S)).toThrow(/symbolic axis/);
    expect(shapeOf({ op: "oneHot", val: "y", size: 10 }, S)).toEqual(["N", 10]);
    expect(shapeOf({ op: "takeAlong", val: "M", indices: { op: "reshape", val: "v", shape: [1, 4] }, axis: 1 }, S)).toEqual([3, 4]);
    expect(() => shapeOf({ op: "takeAlong", val: "M", indices: "v", axis: 1 }, S)).toThrow(/same rank/);
    expect(shapeOf({ op: "stopGradient", val: "M" }, S)).toEqual([3, 4]);
  });

  it("coord leaves only in field context", () => {
    expect(() => shapeOf({ op: "coord", index: 0 }, S)).toThrow(/only allowed in the inputs of a net-backed field/);
    expect(() => shapeOf({ op: "coordv" }, S)).toThrow(/only allowed/);
  });
});

describe("net definitions", () => {
  it("infers the MLP's signature", () => {
    const sig = inferNet(mlp);
    expect(sig.inputs).toEqual(mlp.inputs);
    expect(sig.outputs).toEqual({ loss: [], acc: [], logits: ["N", 10] });
    expect(sig.nodes.h).toEqual(["N", 128]);
    expect(sig.nodes.nll).toEqual(["N", 1]);
    expect(sig.nodes.pred).toEqual(["N"]);
    expect(sig.batch).toEqual([]);
  });

  it("checks declared output shapes", () => {
    expect(() => inferNet({ ...mlp, outputs: { loss: [1] } })).toThrow(/declared \[1\] but has shape \[\]/);
    expect(() => inferNet({ ...mlp, outputs: { logits: ["N", 11] } })).toThrow(/declared/);
    expect(() => inferNet({ ...mlp, outputs: { nope: [] } })).toThrow(/neither an input nor a node/);
  });

  it("nodes may be declared in any order; cycles are errors", () => {
    const sig = inferNet({ type: "def", inputs: { a: [2] }, nodes: { c: { op: "add", vals: ["b", 1] }, b: { op: "square", val: "a" } }, outputs: { c: [2] } });
    expect(sig.outputs.c).toEqual([2]);
    expect(() => inferNet({ type: "def", inputs: {}, nodes: { a: { op: "add", vals: ["b", 1] }, b: { op: "square", val: "a" } }, outputs: {} })).toThrow(/depends on itself/);
  });

  it("one namespace per net", () => {
    expect(() => inferNet({ type: "def", inputs: { a: [2] }, nodes: { a: 1 }, outputs: {} })).toThrow(/both in inputs and nodes/);
    expect(() => inferNet({ type: "def", inputs: { a: [2] }, arrays: { a: constant([2]) }, outputs: {} })).toThrow(/both in inputs and arrays/);
    expect(() => inferNet({ type: "def", inputs: {}, arrays: { k: constant([2]) }, outputs: { k: [2] } })).toThrow(/constant array/);
  });

  it("constant arrays and unknown axis names", () => {
    const sig = inferNet({ type: "def", inputs: { a: ["N", 2] }, arrays: { m: constant([2]) }, nodes: { s: { op: "mul", vals: ["a", "m"] } }, outputs: { s: ["N", 2] } });
    expect(sig.outputs.s).toEqual(["N", 2]);
    expect(() => inferNet({ type: "def", inputs: { a: [4] }, nodes: { r: { op: "reshape", val: "a", shape: ["N", 2] } }, outputs: {} })).toThrow(/unknown axis name "N"/);
  });
});

describe("bind", () => {
  it("binds the validation set, fixing N and leaving the weights", () => {
    const b = bundle({ mlp, mlp_val: { type: "bind", net: "mlp", bind: valSet(50) } });
    const sig = b.net("mlp_val").signature;
    expect(Object.keys(sig.inputs)).toEqual(["W1", "b1", "W2", "b2"]);
    expect(sig.outputs).toEqual({ loss: [], acc: [], logits: [50, 10] });
    expect(sig.nodes.h).toEqual([50, 128]);
    expect(sig.batch).toEqual([]);
  });

  it("extra leading axes of a bound array become a batch prefix", () => {
    const b = bundle({ mlp, batched: { type: "bind", net: "mlp", bind: { ...valSet(50), ...weights([7]) } } });
    const sig = b.net("batched").signature;
    expect(sig.inputs).toEqual({});
    expect(sig.batch).toEqual([7]);
    // batch prefixes broadcast
    const b2 = bundle({ mlp, batched: { type: "bind", net: "mlp", bind: { ...valSet(50), W1: constant([7, 1, 784, 128]), b1: constant([5, 128]) } } });
    expect(b2.net("batched").signature.batch).toEqual([7, 5]);
    const b3 = bundle({ mlp, batched: { type: "bind", net: "mlp", bind: { W1: constant([7, 784, 128]), b1: constant([5, 128]) } } });
    expect(() => b3.net("batched")).toThrow(/do not broadcast/);
  });

  it("rejects wrong shapes, unknown inputs, inconsistent symbolic sizes", () => {
    expect(() => bundle({ mlp, v: { type: "bind", net: "mlp", bind: { x: constant([50, 783]) } } }).net("v")).toThrow(/axis 1 declared 784, got 783/);
    expect(() => bundle({ mlp, v: { type: "bind", net: "mlp", bind: { x: constant([50, 784]), y: constant([49]) } } }).net("v")).toThrow(/axis "N" is 50 elsewhere but 49 here/);
    expect(() => bundle({ mlp, v: { type: "bind", net: "mlp", bind: { z: constant([1]) } } }).net("v")).toThrow(/"z" is not an input/);
    expect(() => bundle({ mlp, v: { type: "bind", net: "mlp", bind: { b1: constant([]) } } }).net("v")).toThrow(/at least rank 1/);
    expect(() => bundle({ mlp, v: { type: "bind", net: "mlp", bind: { x: "val/x" } } }).net("v")).toThrow(NotSupportedError);
  });

  it("binding a bound net; unknown nets; cycles", () => {
    const b = bundle({
      mlp,
      a: { type: "bind", net: "mlp", bind: valSet(3) },
      c: { type: "bind", net: "a", bind: { W1: constant([784, 128]) } },
      loop1: { type: "bind", net: "loop2", bind: {} },
      loop2: { type: "bind", net: "loop1", bind: {} },
      dangling: { type: "bind", net: "nope", bind: {} },
    });
    expect(Object.keys(b.net("c").signature.inputs)).toEqual(["b1", "W2", "b2"]);
    expect(() => b.net("loop1")).toThrow(/cycle/);
    expect(() => b.net("dangling")).toThrow(/unknown net "nope"/);
    const errors = b.buildAll();
    expect([...errors.keys()].sort()).toEqual(["nets.dangling", "nets.loop1", "nets.loop2"]);
  });
});

describe("grad", () => {
  const nets: Record<string, NetSpec> = {
    mlp,
    mlp_val: { type: "bind", net: "mlp", bind: valSet(50) },
    mlp_val_grad: {
      type: "grad", net: "mlp_val", keep: ["loss", "acc"],
      outputs: { gW1: { of: "loss", wrt: "W1" }, gb1: { of: "loss", wrt: "b1" }, gW2: { of: "loss", wrt: "W2" }, gb2: { of: "loss", wrt: "b2" } },
    },
    hvp: { type: "grad", net: "mlp_val_grad", outputs: { HvW1: { of: "gW1", wrt: "W1", seed: "v" } } },
  };

  it("gradient net: same inputs, gradient outputs shaped like wrt, kept forward outputs", () => {
    const sig = bundle(nets).net("mlp_val_grad").signature;
    expect(Object.keys(sig.inputs)).toEqual(["W1", "b1", "W2", "b2"]);
    expect(sig.outputs).toEqual({ gW1: [784, 128], gb1: [128], gW2: [128, 10], gb2: [10], loss: [], acc: [] });
    expect(sig.nodes.loss).toEqual([]); // forward outputs become internal arrays
    expect(sig.nodes.h).toEqual([50, 128]);
  });

  it("gradient wrt an internal node", () => {
    const sig = bundle({ ...nets, gh: { type: "grad", net: "mlp_val", outputs: { gh: { of: "loss", wrt: "h" } } } }).net("gh").signature;
    expect(sig.outputs.gh).toEqual([50, 128]);
  });

  it("Hessian-vector product: a named seed becomes a new input", () => {
    const sig = bundle(nets).net("hvp").signature;
    expect(sig.inputs.v).toEqual([784, 128]);
    expect(sig.outputs).toEqual({ HvW1: [784, 128] });
  });

  it("seeds", () => {
    // fixed cotangent for a non-scalar output
    const sig = bundle({ ...nets, j: { type: "grad", net: "mlp_val", outputs: { g: { of: "logits", wrt: "W2", seed: constant([50, 10]) } } } }).net("j").signature;
    expect(sig.outputs.g).toEqual([128, 10]);
    // an existing input as the seed must have the output's shape
    expect(() => bundle({ ...nets, j: { type: "grad", net: "mlp_val", outputs: { g: { of: "logits", wrt: "W2", seed: "b2" } } } }).net("j")).toThrow(/seed "b2" has shape/);
    expect(() => bundle({ ...nets, j: { type: "grad", net: "mlp_val", outputs: { g: { of: "logits", wrt: "W2" } } } }).net("j")).toThrow(/only a scalar output/);
    expect(() => bundle({ ...nets, j: { type: "grad", net: "mlp_val", outputs: { g: { of: "loss", wrt: "W2", seed: "h" } } } }).net("j")).toThrow(/internal array/);
    expect(() => bundle({ ...nets, j: { type: "grad", net: "mlp_val", outputs: { g: { of: "logits", wrt: "W2", seed: constant([50, 11]) } } } }).net("j")).toThrow(/declared 10, got 11/);
  });

  it("grad with its own bind, and per-point gradients when the weights are batched", () => {
    const sig = bundle({ mlp, g: { type: "grad", net: "mlp", bind: { ...valSet(50), W1: constant([9, 784, 128]) }, outputs: { gW1: { of: "loss", wrt: "W1" } } } }).net("g").signature;
    expect(Object.keys(sig.inputs)).toEqual(["b1", "W2", "b2"]);
    expect(sig.batch).toEqual([9]);
    expect(sig.outputs.gW1).toEqual([784, 128]); // per batch element; the [9] goes in front at evaluation
  });

  it("rejects bad of / wrt / keep", () => {
    expect(() => bundle({ ...nets, g: { type: "grad", net: "mlp_val", outputs: { g: { of: "nope", wrt: "W1" } } } }).net("g")).toThrow(/not an output/);
    expect(() => bundle({ ...nets, g: { type: "grad", net: "mlp_val", outputs: { g: { of: "loss", wrt: "nope" } } } }).net("g")).toThrow(/neither an input nor an internal node/);
    // a bound input stays a valid wrt (gradient wrt the data at fixed weights)
    expect(bundle({ ...nets, g: { type: "grad", net: "mlp_val", outputs: { gx: { of: "loss", wrt: "x" } } } }).net("g").signature.outputs.gx).toEqual([50, 784]);
    expect(() => bundle({ ...nets, g: { type: "grad", net: "mlp_val", keep: ["h"], outputs: { g: { of: "loss", wrt: "W1" } } } }).net("g")).toThrow(/keep: "h" is not an output/);
    expect(() => bundle({ ...nets, g: { type: "grad", net: "mlp_val", keep: ["loss"], outputs: { loss: { of: "loss", wrt: "W1" } } } }).net("g")).toThrow(/both a kept output and a gradient output/);
  });
});

describe("call", () => {
  const sq: NetDefinitionSpec = { type: "def", inputs: { a: [3] }, nodes: { s: { op: "reduce", fn: "sum", val: { op: "square", val: "a" } } }, outputs: { s: [] } };

  it("calls a net by id or inline; extra leading axes batch", () => {
    const b = bundle({
      sq,
      user: {
        type: "def", inputs: { m: [5, 3], v: [3] },
        nodes: {
          rows: { op: "call", net: "sq", inputs: { a: "m" }, output: "s" },
          one: { op: "call", net: sq, inputs: { a: "v" }, output: "s" },
        },
        outputs: { rows: [5], one: [] },
      },
    });
    expect(b.net("user").signature.outputs).toEqual({ rows: [5], one: [] });
  });

  it("checks inputs and outputs of the callee", () => {
    const def = (inputs: Record<string, ArrayExpr>, output = "s"): NetSpec => ({ type: "def", inputs: { m: [5, 3] }, nodes: { r: { op: "call", net: "sq", inputs, output } }, outputs: { r: [5] } });
    expect(() => bundle({ sq, u: def({}) }).net("u")).toThrow(/missing inputs "a"/);
    expect(() => bundle({ sq, u: def({ a: "m", b: 1 }) }).net("u")).toThrow(/"b" is not an input/);
    expect(() => bundle({ sq, u: def({ a: "m" }, "t") }).net("u")).toThrow(/no output "t"/);
    expect(() => bundle({ sq, u: def({ a: { op: "slice", val: "m", axis: 1, stop: 2 } }) }).net("u")).toThrow(/axis 0 declared 3, got 2/);
  });

  it("a net may not call itself", () => {
    const b = bundle({ rec: { type: "def", inputs: { a: [3] }, nodes: { r: { op: "call", net: "rec", inputs: { a: "a" }, output: "r" } }, outputs: { r: [3] } } });
    expect(() => b.net("rec")).toThrow(/cycle/);
  });

  it("a called net's bound batch joins the caller's", () => {
    const b = bundle({
      sq,
      sqb: { type: "bind", net: sq, bind: { a: constant([4, 3]) } }, // batch [4], no inputs left
      u: { type: "def", inputs: {}, nodes: { r: { op: "call", net: "sqb", inputs: {}, output: "s" } }, outputs: { r: [4] } },
    });
    expect(b.net("u").signature.outputs.r).toEqual([4]);
  });
});

describe("displace", () => {
  const bowl: NetDefinitionSpec = { type: "def", inputs: { p: [2] }, nodes: { h: { op: "square", val: "p" }, f: { op: "reduce", fn: "sum", val: "h" } }, outputs: { f: [], h: [2] } };
  const inline = (data: number[], shape = [data.length]) => ({ type: "inline", shape, data }) as const;

  it("adds a [K] coefficient input; targets may be inputs, nodes, bound inputs or constants", () => {
    const b = bundle({
      bowl,
      around: { type: "displace", net: { type: "bind", net: "bowl", bind: { p: inline([1, 2]) } }, directions: [{ arrays: { p: inline([1, 0]) } }, { arrays: { p: inline([0, 1]) } }] },
      onNode: { type: "displace", net: "bowl", coeffs: "s", directions: [{ arrays: { h: inline([1, 1]) } }] },
      onConst: { type: "displace", net: { type: "def", inputs: {}, arrays: { c: inline([3]) }, nodes: { d: { op: "mul", vals: ["c", 2] } }, outputs: { d: [1] } }, directions: [{ arrays: { c: inline([1]) } }] },
    });
    expect(b.net("around").signature.inputs).toEqual({ t: [2] });
    expect(b.net("onNode").signature.inputs).toEqual({ p: [2], s: [1] });
    expect(b.net("onConst").signature.inputs).toEqual({ t: [1] });
  });

  it("rejects unknown targets, wrong shapes, name collisions", () => {
    expect(() => bundle({ bowl, d: { type: "displace", net: "bowl", directions: [{ arrays: { q: inline([1, 0]) } }] } }).net("d")).toThrow(/"q" is neither an input nor an internal array/);
    expect(() => bundle({ bowl, d: { type: "displace", net: "bowl", directions: [{ arrays: { p: inline([1, 0, 0]) } }] } }).net("d")).toThrow(/direction 0 of "p" has shape \[3\], "p" has \[2\]/);
    expect(() => bundle({ bowl, d: { type: "displace", net: "bowl", coeffs: "h", directions: [{ arrays: { p: inline([1, 0]) } }] } }).net("d")).toThrow(/collides/);
    expect(NetSchema.safeParse({ type: "displace", net: "bowl", directions: [] }).success).toBe(false);
    expect(NetSchema.safeParse({ type: "displace", net: "bowl", directions: [{ p: inline([1, 0]) }] }).success).toBe(false); // needs `arrays`
    expect(() => bundle({ bowl, d: { type: "displace", net: "bowl", directions: [{ arrays: {} }] } }).net("d")).toThrow(/names no arrays/);
  });

  it("norm / scale treat a direction as one vector over all its arrays", () => {
    // two arrays a: [2], c: [1] bound to (3, 4) and (0): the joint origin norm is 5
    const two: NetDefinitionSpec = { type: "def", inputs: { a: [2], c: [1] }, nodes: { s: { op: "reduce", fn: "sum", val: { op: "concat", vals: ["a", "c"], axis: 0 } } }, outputs: { s: [] } };
    const star = { type: "bind", net: "two", bind: { a: inline([3, 4]), c: inline([0]) } } as const;
    const at = (id: string, t: number[]) => bundle({ two, [id]: nets[id]! }).net(id).evaluate({ t: new NdArray([t.length], Float64Array.from(t)) }).s!.data[0]!;
    const nets: Record<string, NetSpec> = {
      raw: { type: "displace", net: star, directions: [{ arrays: { a: inline([1, 1]), c: inline([1]) } }] },
      scaled: { type: "displace", net: star, directions: [{ arrays: { a: inline([1, 1]), c: inline([1]) }, scale: 2 }] },
      unit: { type: "displace", net: star, directions: [{ arrays: { a: inline([0, 3]), c: inline([4]) }, norm: 1 }] }, // |(0,3,4)| = 5 -> (0, .6, .8)
      origin: { type: "displace", net: star, directions: [{ arrays: { a: inline([0, 3]), c: inline([4]) }, norm: "origin", scale: 0.5 }] }, // -> length 5, then halved
      partial: { type: "displace", net: star, directions: [{ arrays: { c: inline([2]) }, norm: "origin" }] }, // origin norm over `c` alone is 0
      onNode: { type: "displace", net: "two", directions: [{ arrays: { s: inline([1], []) }, norm: "origin" }] },
    };
    expect(at("raw", [1])).toBeCloseTo(7 + 3, 12);
    expect(at("scaled", [1])).toBeCloseTo(7 + 6, 12);
    expect(at("unit", [1])).toBeCloseTo(7 + 1.4, 12);
    expect(at("origin", [1])).toBeCloseTo(7 + 0.5 * 5 * 1.4, 12);
    expect(() => at("partial", [1])).toThrow(/target norm is 0/);
    expect(() => bundle({ two, onNode: nets.onNode! }).net("onNode").program).toThrow(/bound input or a baked array/);
  });
});

describe("evaluation", () => {
  const inline = (data: number[], shape = [data.length]) => ({ type: "inline", shape, data }) as const;
  const arr = (data: number[], shape = [data.length]) => new NdArray(shape, Float64Array.from(data));
  const run = (net: NetSpec, inputs: Record<string, NdArray>, nets: Record<string, NetSpec> = {}) => bundle({ ...nets, it: net }).net("it").evaluate(inputs);
  const one = (e: ArrayExpr, inputs: Record<string, [number[], number[]]>): NdArray => {
    const spec: NetDefinitionSpec = { type: "def", inputs: Object.fromEntries(Object.entries(inputs).map(([k, [, s]]) => [k, s])), nodes: { r: e }, outputs: {} };
    (spec.outputs as Record<string, number[]>).r = [...inferNet(spec).nodes.r!] as number[];
    return run(spec, Object.fromEntries(Object.entries(inputs).map(([k, [d, s]]) => [k, arr(d, s)]))).r!;
  };
  const near = (a: NdArray, shape: number[], data: number[]) => {
    expect([...a.shape]).toEqual(shape);
    data.forEach((x, i) => expect(a.data[i]).toBeCloseTo(x, 10));
  };

  it("elementwise with broadcasting", () => {
    near(one({ op: "add", vals: ["a", "b"] }, { a: [[1, 2, 3], [3]], b: [[10], []] }), [3], [11, 12, 13]);
    near(one({ op: "sub", vals: ["a", "b"] }, { a: [[1, 2, 3, 4, 5, 6], [2, 3]], b: [[1, 2, 3], [3]] }), [2, 3], [0, 0, 0, 3, 3, 3]);
    near(one({ op: "relu", val: "a" }, { a: [[-1, 2], [2]] }), [2], [0, 2]);
    near(one({ op: "where", cond: { op: "gt", vals: ["a", 0] }, vals: ["a", 0] }, { a: [[-1, 2], [2]] }), [2], [0, 2]);
    near(one({ op: "clamp", val: "a", min: 0, max: 1 }, { a: [[-1, 0.5, 2], [3]] }), [3], [0, 0.5, 1]);
    near(one({ op: "eq", vals: ["a", "b"] }, { a: [[1, 2], [2]], b: [[1, 3], [2]] }), [2], [1, 0]);
    near(one({ op: "mean", vals: ["a", "b", 6] }, { a: [[0], []], b: [[3], []] }), [], [3]);
  });

  it("matmul and einsum", () => {
    near(one({ op: "matmul", vals: ["A", "B"] }, { A: [[1, 2, 3, 4], [2, 2]], B: [[5, 6, 7, 8], [2, 2]] }), [2, 2], [19, 22, 43, 50]);
    near(one({ op: "matmul", vals: ["A", "v"] }, { A: [[1, 2, 3, 4], [2, 2]], v: [[1, 1], [2]] }), [2], [3, 7]);
    near(one({ op: "matmul", vals: ["v", "A"] }, { A: [[1, 2, 3, 4], [2, 2]], v: [[1, 1], [2]] }), [2], [4, 6]);
    near(one({ op: "matmul", vals: ["v", "v"] }, { v: [[3, 4], [2]] }), [], [25]);
    near(one({ op: "einsum", subscripts: "ij->ji", vals: ["A"] }, { A: [[1, 2, 3, 4], [2, 2]] }), [2, 2], [1, 3, 2, 4]);
    near(one({ op: "einsum", subscripts: "ii->", vals: ["A"] }, { A: [[1, 2, 3, 4], [2, 2]] }), [], [5]);
    near(one({ op: "einsum", subscripts: "ii->i", vals: ["A"] }, { A: [[1, 2, 3, 4], [2, 2]] }), [2], [1, 4]);
    near(one({ op: "einsum", subscripts: "i->ii", vals: ["v"] }, { v: [[3, 4], [2]] }), [2, 2], [3, 0, 0, 4]); // the diagonal, zero elsewhere
    near(one({ op: "einsum", subscripts: "i,j->iij", vals: ["v", "w"] }, { v: [[3, 4], [2]], w: [[1, 10], [2]] }), [2, 2, 2], [3, 30, 0, 0, 0, 0, 4, 40]);
    near(one({ op: "einsum", subscripts: "i,j", vals: ["v", "w"] }, { v: [[1, 2], [2]], w: [[10, 20, 30], [3]] }), [2, 3], [10, 20, 30, 20, 40, 60]);
  });

  it("reductions, argmax, softmax", () => {
    const A: [number[], number[]] = [[1, 2, 3, 4, 5, 6], [2, 3]];
    near(one({ op: "reduce", fn: "sum", val: "A" }, { A }), [], [21]);
    near(one({ op: "reduce", fn: "mean", val: "A", axes: [1] }, { A }), [2], [2, 5]);
    near(one({ op: "reduce", fn: "max", val: "A", axes: [0], keepDims: true }, { A }), [1, 3], [4, 5, 6]);
    near(one({ op: "reduce", fn: "prod", val: "A", axes: [-1] }, { A }), [2], [6, 120]);
    near(one({ op: "reduce", fn: "logsumexp", val: "A", axes: [1] }, { A }), [2], [Math.log(Math.E + Math.E ** 2 + Math.E ** 3), Math.log(Math.E ** 4 + Math.E ** 5 + Math.E ** 6)]);
    near(one({ op: "argmax", val: "A" }, { A }), [2], [2, 2]);
    near(one({ op: "argmin", val: "A", axis: 0 }, { A }), [3], [0, 0, 0]);
    const sm = one({ op: "softmax", val: "A" }, { A });
    expect(sm.data[0]! + sm.data[1]! + sm.data[2]!).toBeCloseTo(1, 12);
    expect(sm.data[2]! / sm.data[0]!).toBeCloseTo(Math.E ** 2, 10);
    const lsm = one({ op: "logSoftmax", val: "A" }, { A });
    expect(Math.exp(lsm.data[3]!) + Math.exp(lsm.data[4]!) + Math.exp(lsm.data[5]!)).toBeCloseTo(1, 12);
  });

  it("shape ops", () => {
    const A: [number[], number[]] = [[1, 2, 3, 4, 5, 6], [2, 3]];
    near(one({ op: "reshape", val: "A", shape: [3, -1] }, { A }), [3, 2], [1, 2, 3, 4, 5, 6]);
    near(one({ op: "transpose", val: "A" }, { A }), [3, 2], [1, 4, 2, 5, 3, 6]);
    near(one({ op: "concat", vals: ["A", "A"], axis: 1 }, { A }), [2, 6], [1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6]);
    near(one({ op: "concat", vals: ["A", { op: "reshape", val: "v", shape: [1, 3] }], axis: 0 }, { A, v: [[7, 8, 9], [3]] }), [3, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9]);
    near(one({ op: "slice", val: "A", axis: 1, start: 1 }, { A }), [2, 2], [2, 3, 5, 6]);
    near(one({ op: "slice", val: "A", axis: 1, step: -1 }, { A }), [2, 3], [3, 2, 1, 6, 5, 4]);
    near(one({ op: "slice", val: "A", axis: 0, start: -1 }, { A }), [1, 3], [4, 5, 6]);
    near(one({ op: "oneHot", val: "v", size: 3 }, { v: [[0, 2], [2]] }), [2, 3], [1, 0, 0, 0, 0, 1]);
    near(one({ op: "takeAlong", val: "A", indices: "i", axis: 1 }, { A, i: [[2, 0], [2, 1]] }), [2, 1], [3, 4]);
    near(one({ op: "takeAlong", val: "A", indices: "i", axis: 0 }, { A, i: [[1, 0, 1], [1, 3]] }), [1, 3], [4, 2, 6]);
  });

  it("implicit batch: extra leading axes vmap, prefixes broadcast, declared ranks decide alignment", () => {
    const def = (inputs: Record<string, number[]>, r: ArrayExpr, out: number[]): NetDefinitionSpec => ({ type: "def", inputs, nodes: { r }, outputs: { r: out } });
    // declared a: [3], b: [] — passing b with batch [2] must give [2, 3], not treat [3] and [2] as one axis
    near(run(def({ a: [3], b: [] }, { op: "add", vals: ["a", "b"] }, [3]), { a: arr([1, 2, 3]), b: arr([10, 20]) }).r!, [2, 3], [11, 12, 13, 21, 22, 23]);
    // batched matmul: G=2 weight matrices against one vector
    near(run(def({ v: [2], W: [2, 2] }, { op: "matmul", vals: ["v", "W"] }, [2]), { v: arr([1, 1]), W: arr([1, 0, 0, 1, 2, 0, 0, 2], [2, 2, 2]) }).r!, [2, 2], [1, 1, 2, 2]);
    // reductions, argmax, reshape, transpose act on declared axes only
    near(run(def({ A: [2] }, { op: "reduce", fn: "sum", val: "A" }, []), { A: arr([1, 2, 3, 4], [2, 2]) }).r!, [2], [3, 7]);
    near(run(def({ A: [2] }, { op: "argmax", val: "A" }, []), { A: arr([1, 9, 5, 2], [2, 2]) }).r!, [2], [1, 0]);
    near(run(def({ A: [2] }, { op: "transpose", val: { op: "reshape", val: "A", shape: [2, 1] } }, [1, 2]), { A: arr([1, 2, 3, 4], [2, 2]) }).r!, [2, 1, 2], [1, 2, 3, 4]);
    near(run(def({ A: [2] }, { op: "softmax", val: "A" }, [2]), { A: arr([0, 0, 1, 1], [2, 2]) }).r!, [2, 2], [0.5, 0.5, 0.5, 0.5]);
    near(run(def({ A: [2] }, { op: "concat", vals: ["A", "A"], axis: 0 }, [4]), { A: arr([1, 2, 3, 4], [2, 2]) }).r!, [2, 4], [1, 2, 1, 2, 3, 4, 3, 4]);
    near(run(def({ A: [2], i: [1] }, { op: "takeAlong", val: "A", indices: "i", axis: 0 }, [1]), { A: arr([1, 2, 3, 4], [2, 2]), i: arr([1]) }).r!, [2, 1], [2, 4]);
    // batch prefixes broadcast (1 stretches) ...
    near(run(def({ a: [], b: [] }, { op: "mul", vals: ["a", "b"] }, []), { a: arr([1, 2, 3], [3, 1]), b: arr([10, 100], [1, 2]) }).r!, [3, 2], [10, 100, 20, 200, 30, 300]);
    // ... and prefixes that do not broadcast are an error
    expect(() => run(def({ a: [], b: [] }, { op: "add", vals: ["a", "b"] }, []), { a: arr([1, 2, 3]), b: arr([1, 2]) })).toThrow(/do not broadcast/);
  });

  it("bind, call and symbolic sizes", () => {
    const sq: NetDefinitionSpec = { type: "def", inputs: { a: ["N"] }, nodes: { s: { op: "reduce", fn: "sum", val: { op: "square", val: "a" } }, n: { op: "reshape", val: "a", shape: ["N", 1] } }, outputs: { s: [], n: ["N", 1] } };
    near(run(sq, { a: arr([3, 4]) }).s!, [], [25]);
    near(run(sq, { a: arr([3, 4]) }).n!, [2, 1], [3, 4]);
    near(run({ type: "bind", net: "sq", bind: { a: inline([1, 2, 3]) } }, {}, { sq }).s!, [], [14]);
    const user: NetDefinitionSpec = { type: "def", inputs: { m: [2, 3] }, nodes: { rows: { op: "call", net: "sq", inputs: { a: "m" }, output: "s" } }, outputs: { rows: [2] } };
    near(run(user, { m: arr([1, 2, 3, 4, 5, 6], [2, 3]) }, { sq }).rows!, [2], [14, 77]);
    expect(() => run(sq, { a: arr([1, 2], [1, 2]), zz: arr([1]) })).toThrow(/not an input/);
    expect(() => run(sq, {})).toThrow(/input "a" not given/);
    near(run({ type: "grad", net: "sq", outputs: { g: { of: "s", wrt: "a" } } }, { a: arr([3, 4]) }, { sq }).g!, [2], [6, 8]); // d sum(a²) / da = 2a
  });

  it("the MLP forward pass agrees with a hand computation", () => {
    // 2 examples, 3 features, 2 hidden, 2 classes
    const tiny: NetDefinitionSpec = { ...mlp, inputs: { x: ["N", 3], y: ["N"], W1: [3, 2], b1: [2], W2: [2, 2], b2: [2] }, outputs: { loss: [], acc: [], logits: ["N", 2] } };
    const out = run(tiny, {
      x: arr([1, 0, 0, 0, 1, 0], [2, 3]), y: arr([0, 1]),
      W1: arr([1, -1, 0, 1, 0, 0], [3, 2]), b1: arr([0, 0.5]), W2: arr([1, 0, 0, 1], [2, 2]), b2: arr([0, 0]),
    });
    // h = relu(x W1 + b1) = [[1, 0], [0, 1.5]]; logits = h W2 + b2 = h
    near(out.logits!, [2, 2], [1, 0, 0, 1.5]);
    const nll = (l: number[], y: number) => -(l[y]! - Math.log(l.reduce((s, v) => s + Math.exp(v), 0)));
    expect(out.loss!.data[0]).toBeCloseTo((nll([1, 0], 0) + nll([0, 1.5], 1)) / 2, 12);
    expect(out.acc!.data[0]).toBe(1);
  });
});

describe("net-backed fields", () => {
  const inline = (data: number[], shape = [data.length]) => ({ type: "inline", shape, data }) as const;
  const bowl: NetDefinitionSpec = { type: "def", inputs: { p: [2] }, nodes: { h: { op: "square", val: "p" }, f: { op: "reduce", fn: "sum", val: "h" } }, outputs: { f: [], h: [2] } };
  const build = (data: unknown, kind: "scalar" | "vector" = "scalar", nets: Record<string, NetSpec> = { bowl }) => {
    const b = bundle(nets, { f: { kind, data } as BundleSpec["fields"][string] });
    const errors = b.buildAll();
    if (errors.size) throw errors.get("f") ?? [...errors.values()][0];
    return b;
  };
  const fail = (data: unknown, kind: "scalar" | "vector" = "scalar", nets?: Record<string, NetSpec>) => {
    try { build(data, kind, nets); } catch (e) { return e as Error; }
    throw new Error("expected an error");
  };

  it("a net with one remaining input takes the point by default", () => {
    const f = build({ type: "net", net: "bowl", output: "f", box: [[-1, 1], [-1, 1]] }).scalarField("f").data;
    expect(f.kind).toBe("symbolic");
    expect(f.box.a).toEqual([-1, -1]);
    expect(f.value([0.3, 0.4])).toBeCloseTo(0.25, 12);
    expect(f.value([2, 0])).toBeUndefined(); // outside the box
    // batched sampling agrees with per-point evaluation
    const grid = new DenseGrid([5, 7], f.box);
    const vals = f.sampleOn(grid);
    const p = new Float64Array(2);
    for (let i = 0; i < grid.sampleCount; i++) { grid.pointInto(i, p); expect(vals[i]).toBeCloseTo(f.fn(p, -1), 12); }
    // finite-difference derivative
    expect(f.derivative(0).value([0.3, 0.4])).toBeCloseTo(0.6, 5);
    expect(f.derivative(1).derivative(1).value([0.3, 0.4])).toBeCloseTo(2, 3);
  });

  it("a vector field from a [D] output", () => {
    const rot: NetSpec = { type: "def", inputs: { p: [2] }, arrays: { R: inline([0, -1, 1, 0], [2, 2]) }, nodes: { v: { op: "matmul", vals: ["R", "p"] } }, outputs: { v: [2] } };
    const v = build({ type: "netv", net: rot, output: "v" }, "vector").vectorField("f").data;
    expect(v.value([0.5, 0.25])).toEqual([-0.25, 0.5]);
    const grid = new DenseGrid([3, 3], v.box);
    const vals = v.sampleOn(grid);
    expect(vals.length).toBe(18);
    expect(vals[2 * 2]).toBeCloseTo(-1, 12); // grid point 2 is (0, 1); R·(0, 1) = (-1, 0)
    expect(vals[2 * 2 + 1]).toBeCloseTo(0, 12);
    expect(v.component(0).value([0.5, 0.25])).toBeCloseTo(-0.25, 12);
  });

  it("the loss landscape pattern: bind the origin, displace along directions, the coefficients are the point", () => {
    // f(t) = |(1, 2) + t0 (1, 0) + t1 (0, 1)|^2
    const b = build({ type: "net", net: "around", output: "f", box: [[-1, 1], [-1, 1]] }, "scalar", {
      bowl,
      around: { type: "displace", net: { type: "bind", net: "bowl", bind: { p: inline([1, 2]) } }, directions: [{ arrays: { p: inline([1, 0]) } }, { arrays: { p: inline([0, 1]) } }] },
    });
    const f = b.scalarField("f").data;
    expect(f.value([0, 0])).toBeCloseTo(5, 12);
    expect(f.value([0.5, -1])).toBeCloseTo(1.5 ** 2 + 1, 12);
  });

  it("displacing a node, on a 1-D manifold", () => {
    const b = Bundle.parse({
      tensatory: "0.1", manifolds: { line: { numDims: 1 } },
      nets: {
        bowl,
        around: { type: "displace", net: { type: "bind", net: "bowl", bind: { p: inline([1, 2]) } }, directions: [{ arrays: { h: inline([1, 1]) } }] },
      },
      fields: { f: { kind: "scalar", data: { type: "net", net: "around", output: "f", box: [[-2, 2]] } } },
    } satisfies BundleSpec);
    expect(b.buildAll().size).toBe(0);
    expect(b.scalarField("f").data.value([0.5])).toBeCloseTo(6, 12);
  });

  it("inputs as expressions of the coordinates over named arrays", () => {
    // p = c + coord0 · d  with c = (1, 1), d = (1, 0):  f = (1 + x)^2 + 1
    const f = build({
      type: "net", net: "bowl", output: "f",
      arrays: { c: inline([1, 1]), d: inline([1, 0]) },
      inputs: { p: { op: "add", vals: ["c", { op: "mul", vals: ["d", { op: "coord", index: 0 }] }] } },
    }).scalarField("f").data;
    expect(f.value([0.5, 0.9])).toBeCloseTo(1.5 ** 2 + 1, 12);
    // constant inputs: the same value everywhere
    const g = build({ type: "net", net: "bowl", output: "f", inputs: { p: "c" }, arrays: { c: inline([1, 1]) } }).scalarField("f").data;
    expect(g.value([0.2, 0.7])).toBe(2);
    expect(Array.from(g.sampleOn(new DenseGrid([2, 2], g.box)))).toEqual([2, 2, 2, 2]);
  });

  it("checks inputs, outputs and batches", () => {
    const mlpNets = { mlp, mlp_val: { type: "bind", net: "mlp", bind: valSet(50) } as NetSpec };
    expect(fail({ type: "net", net: "mlp_val", output: "loss" }, "scalar", mlpNets).message).toMatch(/exactly one remaining input .* this one has "W1", "b1", "W2", "b2"/);
    expect(fail({ type: "net", net: { type: "bind", net: "bowl", bind: { p: inline([1, 2]) } }, output: "f" }).message).toMatch(/this one has none/);
    expect(fail({ type: "net", net: { type: "def", inputs: { p: [3] }, outputs: { p: [3] } }, output: "p" }).message).toMatch(/declared 3, got 2/);
    expect(fail({ type: "net", net: "bowl", output: "h" }).message).toMatch(/has shape \[2\], a scalar field needs \[\]/);
    expect(fail({ type: "netv", net: "bowl", output: "f" }, "vector").message).toMatch(/a vector field needs \[2\]/);
    expect(fail({ type: "net", net: "bowl", output: "nope" }).message).toMatch(/no output "nope"/);
    expect(fail({ type: "net", net: "bowl", output: "f", inputs: { p: "c" }, arrays: { c: inline([1, 1, 1, 1], [2, 2]) } }).message).toMatch(/add batch axes \[2\]/);
    expect(fail({ type: "net", net: { type: "bind", net: "bowl", bind: { p: inline([1, 2, 3, 4], [2, 2]) } }, output: "f" }).message).toMatch(/bound arrays add batch axes \[2\]/);
    expect(fail({ type: "net", net: "bowl", output: "f", inputs: { p: { op: "coord", index: 2 } } }).message).toMatch(/index 2 out of range for 2/);
    expect(fail({ type: "net", net: "bowl", output: "f", inputs: { q: { op: "coordv" } } }).message).toMatch(/"q" is not an input/);
    expect(fail({ type: "net", net: "bowl", output: "f", inputs: {} }).message).toMatch(/"p" are not given/);
  });
});

// Hoisting (nets/hoist.ts): a contraction of a constant with a displaced constant distributes into folded constants
// plus a t-linear term, other constant-only nodes fold, dead constants are pruned; the hoisted program evaluates to
// the same values (and gradients) as the original, and the MNIST first layer is gone from the per-point work.

import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import type { BundleSpec, NetSpec } from "@tensatory/schema";
import { Bundle, NdArray, evaluate, exprNames, gradProgram, hoistProgram, type NetScalarFieldData, type Program } from "../src";
import { dirSource } from "./helpers";

const rnd = (shape: number[], seed: number) => ({ type: "random", shape, dist: { type: "gaussian", seed } }) as const;

/** a 2-layer net on a fixed dataset, θ* displaced along 2 random directions in its weights */
const spec: BundleSpec = {
  tensatory: "0.1",
  manifolds: { plane: { numDims: 2 } },
  nets: {
    net: {
      type: "def",
      inputs: { x: ["N", 6], W1: [6, 5], b1: [5], W2: [5, 3] },
      nodes: {
        xs: { op: "mul", vals: ["x", 0.5] },
        h: { op: "tanh", val: { op: "add", vals: [{ op: "matmul", vals: ["xs", "W1"] }, "b1"] } },
        out: { op: "matmul", vals: ["h", "W2"] },
        loss: { op: "reduce", fn: "mean", val: { op: "square", val: "out" } },
      },
      outputs: { loss: [] },
    },
    star: { type: "bind", net: "net", bind: { x: rnd([7, 6], 1), W1: rnd([6, 5], 2), b1: rnd([5], 3), W2: rnd([5, 3], 4) } },
    disp: { type: "displace", net: "star", directions: [
      { arrays: { W1: rnd([6, 5], 11), b1: rnd([5], 12), W2: rnd([5, 3], 13) } },
      { arrays: { W1: rnd([6, 5], 21), b1: rnd([5], 22), W2: rnd([5, 3], 23) } },
    ] } satisfies NetSpec,
  },
  fields: { f: { kind: "scalar", domain: "plane", data: { type: "net", net: "disp", output: "loss", box: [[-1, 1], [-1, 1]] } } },
};

const pts = [[0, 0], [0.3, -0.7], [-1, 0.5], [0.9, 0.9]];
const at = (prog: Program, nets: NetScalarFieldData["field"]["nets"], out: string, p: number[]) =>
  evaluate(prog, { [Object.keys(prog.inputs)[0]!]: new NdArray([1, 2], Float64Array.from(p)) }, { nets })[out]!.data;

describe("hoisting", () => {
  const b = Bundle.parse(spec);
  const fd = b.scalarField("f").data as NetScalarFieldData;
  const original = b.net("disp").program; // the field's program before hoisting: its sole input is t = the point
  const hoisted = hoistProgram(original, fd.field.nets);

  it("distributes the first layer's matmul, folds the constants and prunes the rest", () => {
    expect(fd.field.program).toBe(hoisted); // the field uses the hoisted program (memoized)
    const nodeNames = hoisted.nodes.map((n) => n.name);
    const exprOf = (n: string) => hoisted.nodes.find((x) => x.name === n)!.expr;
    // xs folded, W1__disp gone: h reads folded constants A = xs·W1 and XD = stack_k xs·D1_k through t
    expect(nodeNames).not.toContain("xs");
    expect(nodeNames).not.toContain("W1__disp");
    expect(Object.keys(hoisted.consts)).not.toContain("x");
    expect(Object.keys(hoisted.consts)).not.toContain("W1");
    expect(Object.keys(hoisted.consts)).not.toContain("W1__dispd");
    const hDeps = [...exprNames(exprOf("h"))];
    expect(hDeps.some((d) => d.endsWith("__h"))).toBe(true);
    expect(hDeps.some((d) => d.endsWith("__hd"))).toBe(true);
    expect(hoisted.consts[hDeps.find((d) => d.endsWith("__hd"))!]!.arr.shape).toEqual([2, 7, 5]);
    // the second layer's matmul is NOT distributed: h depends on the point
    expect(nodeNames).toContain("W2__disp");
    expect(JSON.stringify(exprOf("out"))).toContain("W2__disp");
    expect(hoisted.inputs).toEqual(original.inputs);
  });

  it("evaluates to the same values and the same gradient", () => {
    for (const p of pts) {
      expect(at(hoisted, fd.field.nets, "loss", p)[0]).toBeCloseTo(at(original, fd.field.nets, "loss", p)[0]!, 12);
    }
    const t = Object.keys(original.inputs)[0]!;
    const g0 = gradProgram(original, [{ name: "g", of: "loss", wrt: t }], [], fd.field.nets, []);
    const g1 = gradProgram(hoisted, [{ name: "g", of: "loss", wrt: t }], [], fd.field.nets, []);
    for (const p of pts) {
      const a = at(g0, fd.field.nets, "g", p), c = at(g1, fd.field.nets, "g", p);
      expect(c.length).toBe(2);
      for (let i = 0; i < 2; i++) expect(c[i]).toBeCloseTo(a[i]!, 10);
    }
  });

  it("folds a fully bound program down to its outputs", () => {
    const star = b.net("star").program;
    const h = hoistProgram(star, fd.field.nets);
    expect(h.nodes).toEqual([]); // no input: everything is a constant, the output included
    expect(Object.keys(h.consts)).toEqual(["loss"]);
    expect(evaluate(h, {}, { nets: fd.field.nets })["loss"]!.data[0]).toBeCloseTo(evaluate(star, {}, { nets: fd.field.nets })["loss"]!.data[0]!, 12);
  });
});

describe("hoisting the MNIST MLP", () => {
  const DIR = join(__dirname, "../../../apps/viewer/public/bundles/mnist-mlp");
  const bundleP = readFile(join(DIR, "bundle.json"), "utf8").then((s) => Bundle.load(JSON.parse(s), dirSource(DIR)));

  it("removes x, W1t and the first layer's matmul; values agree with the unhoisted program", async () => {
    const b = await bundleP;
    const fd = b.scalarField("fastLoss2").data as NetScalarFieldData;
    const original = b.net("mlp_pca2_256").program;
    const t0 = performance.now();
    const hoisted = fd.field.program;
    const ms = performance.now() - t0;
    console.log(`hoisting mlp_pca2_256 (N = 256): ${ms.toFixed(0)} ms`);
    expect(hoisted).not.toBe(original);
    const consts = Object.keys(hoisted.consts);
    expect(consts).not.toContain("x");
    expect(consts).not.toContain("W1t");
    expect(consts).not.toContain("W1t__dispd");
    expect(consts).toContain("y");
    expect(consts).toContain("W2t__dispd");
    const h1 = hoisted.nodes.find((n) => n.name === "h1")!;
    expect(JSON.stringify(h1.expr)).not.toContain("matmul");
    const hd = consts.find((c) => c.endsWith("__hd"))!;
    expect(hoisted.consts[hd]!.arr.shape).toEqual([2, 256, 256]);
    expect(hoisted.consts[hd]!.shape).toEqual([2, "N", 256]);
    for (const p of [[0, 0], [-5, 1.5]]) {
      expect(at(hoisted, fd.field.nets, "loss", p)[0]).toBeCloseTo(at(original, fd.field.nets, "loss", p)[0]!, 9);
      expect(at(hoisted, fd.field.nets, "acc", p)[0]).toBeCloseTo(at(original, fd.field.nets, "acc", p)[0]!, 12);
    }
  }, 120_000);
});

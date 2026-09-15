import { describe, expect, it } from "vitest";
import type { SymbolicScalar, SymbolicVector } from "@tensatory/schema";
import {
  SymbolicScalarSchema,
  compileScalar,
  compileVector,
  diffScalar,
  emptyEnv,
  gradient,
  normalizeScalar,
  normalizeVector,
  pureContext,
  SpecError,
} from "../src";

const evalS = (e: SymbolicScalar, p: number[], consts: Record<string, number> = {}) => {
  const ast = normalizeScalar(e, emptyEnv(p.length, consts));
  return compileScalar(ast, pureContext(p.length))(p, -1);
};
const evalV = (e: SymbolicVector, p: number[], consts: Record<string, number> = {}) => {
  const ast = normalizeVector(e, emptyEnv(p.length, consts));
  return Array.from(compileVector(ast, pureContext(p.length))(p, -1, new Float64Array(p.length)));
};

const x: SymbolicScalar = { op: "coord", index: 0 };
const y: SymbolicScalar = { op: "coord", index: 1 };

describe("symbolic scalar", () => {
  it("validates syntax", () => {
    expect(SymbolicScalarSchema.safeParse({ op: "add", vals: [x, 1] }).success).toBe(true);
    expect(SymbolicScalarSchema.safeParse({ op: "nope", vals: [x] }).success).toBe(false);
    expect(SymbolicScalarSchema.safeParse({ op: "sub", vals: [x] }).success).toBe(false);
  });

  it("evaluates literals, coords, consts, names", () => {
    expect(evalS(3, [0])).toBe(3);
    expect(evalS(x, [2, 5])).toBe(2);
    expect(evalS("k", [0], { k: 7 })).toBe(7);
    expect(evalS({ op: "const", name: "k" }, [0], { k: 7 })).toBe(7);
    expect(() => evalS("zz", [0])).toThrow(SpecError);
    expect(() => evalS({ op: "coord", index: 3 }, [0, 0])).toThrow(SpecError);
  });

  it("evaluates arithmetic", () => {
    expect(evalS({ op: "add", vals: [{ op: "square", val: x }, { op: "square", val: y }] }, [3, 4])).toBe(25);
    expect(evalS({ op: "sub", vals: [x, y] }, [3, 4])).toBe(-1);
    expect(evalS({ op: "div", vals: [x, y] }, [3, 4])).toBe(0.75);
    expect(evalS({ op: "pow", vals: [x, 3] }, [2])).toBe(8);
    expect(evalS({ op: "mod", vals: [-1, 3] }, [0])).toBe(2);
    expect(evalS({ op: "clamp", val: x, min: 0, max: 1 }, [5])).toBe(1);
    expect(evalS({ op: "min", vals: [x, y, 2] }, [3, 4])).toBe(2);
    expect(evalS({ op: "rms", vals: [3, 4] }, [0])).toBeCloseTo(Math.sqrt(12.5));
    expect(evalS({ op: "gaussKernel", val: x, mu: 1, sigma: 2 }, [1])).toBe(1);
    expect(evalS({ op: "normalPDF", val: x, mu: 0, sigma: 1 }, [0])).toBeCloseTo(1 / Math.sqrt(2 * Math.PI));
  });

  it("evaluates vector ops", () => {
    expect(evalS({ op: "norm", vec: { op: "coordv" } }, [3, 4])).toBe(5);
    expect(evalS({ op: "dot", vecs: [{ op: "coordv" }, { op: "basisv", index: 1 }] }, [3, 4])).toBe(4);
    expect(evalS({ op: "comp", vec: { op: "scalev", vec: { op: "coordv" }, by: 2 }, index: 0 }, [3, 4])).toBe(6);
    const n = evalV({ op: "normalize", vec: { op: "constv", value: [3, 4] } }, [0, 0]);
    expect(n[0]).toBeCloseTo(0.6);
    expect(n[1]).toBeCloseTo(0.8);
    expect(evalV({ op: "sumv", vecs: [{ op: "basisv", index: 0 }, { op: "basisv", index: 1 }], coeffs: [x, y] }, [3, 4])).toEqual([3, 4]);
    expect(evalV({ op: "compv", coeffs: [y, x] }, [3, 4])).toEqual([4, 3]);
    expect(() => evalV({ op: "constv", value: [1, 2, 3] }, [0, 0])).toThrow(SpecError);
  });
});

describe("symbolic differentiation", () => {
  const numGrad = (f: (p: number[]) => number, p: number[], h = 1e-6) =>
    p.map((_, k) => {
      const a = [...p], b = [...p];
      a[k]! += h; b[k]! -= h;
      return (f(a) - f(b)) / (2 * h);
    });

  const cases: [string, SymbolicScalar, number[]][] = [
    ["x^2 + y^2", { op: "add", vals: [{ op: "square", val: x }, { op: "square", val: y }] }, [0.3, -0.7]],
    ["sin(x) * exp(y)", { op: "mul", vals: [{ op: "sin", val: x }, { op: "exp", val: y }] }, [0.3, -0.7]],
    ["x / y", { op: "div", vals: [x, y] }, [0.3, -0.7]],
    ["x ^ y", { op: "pow", vals: [x, y] }, [1.3, 0.7]],
    ["tanh(x y)", { op: "tanh", val: { op: "mul", vals: [x, y] } }, [0.3, -0.7]],
    ["|p|", { op: "norm", vec: { op: "coordv" } }, [0.3, -0.7]],
    ["gaussKernel", { op: "gaussKernel", val: x, mu: y, sigma: 0.5 }, [0.3, -0.7]],
    ["normalPDF", { op: "normalPDF", val: x, mu: 0.1, sigma: y }, [0.3, 0.7]],
    ["cosineSim", { op: "cosineSim", vecs: [{ op: "coordv" }, { op: "constv", value: [1, 2] }] }, [0.3, -0.7]],
    ["gelu", { op: "gelu", val: x }, [0.3, 0]],
    ["silu", { op: "silu", val: x }, [-0.4, 0]],
    ["min(x, y)", { op: "min", vals: [x, y] }, [0.3, -0.7]],
    ["clamp", { op: "clamp", val: x, min: y, max: 1 }, [-0.9, -0.7]],
    ["logBase", { op: "logBase", vals: [x, y] }, [2.5, 3.5]],
    ["atan2", { op: "atan2", vals: [x, y] }, [0.3, -0.7]],
    ["|grad(x^2 y)|", { op: "norm", vec: { op: "grad", val: { op: "mul", vals: [{ op: "square", val: x }, y] } } }, [0.3, -0.7]],
  ];

  for (const [name, e, p] of cases) {
    it(`d(${name}) matches finite differences`, () => {
      const D = p.length;
      const ast = normalizeScalar(e, emptyEnv(D));
      const f = compileScalar(ast, pureContext(D));
      const g = compileVector(gradient(ast, D), pureContext(D))(p, -1, new Float64Array(D));
      const ng = numGrad((q) => f(q, -1), p);
      for (let k = 0; k < D; k++) expect(g[k]).toBeCloseTo(ng[k]!, 5);
    });
  }

  it("simplifies constants", () => {
    const ast = normalizeScalar({ op: "add", vals: [{ op: "mul", vals: [2, x]}, 3, 4] }, emptyEnv(1));
    expect(diffScalar(ast, 0, 1)).toEqual({ k: "const", value: 2 });
    const c = normalizeScalar({ op: "add", vals: [{ op: "sqrt", val: 16 }, 1] }, emptyEnv(1));
    expect(c).toEqual({ k: "const", value: 5 });
  });
});

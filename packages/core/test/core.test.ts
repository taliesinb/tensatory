import { describe, expect, it } from "vitest";
import type { BundleSpec, ScalarFieldDataSpec } from "@tensatory/schema";
import {
  Box,
  Bundle,
  SymbolicScalarFieldData,
  SymbolicVectorFieldData,
  DenseGrid,
  NotSupportedError,
  SpecError,
  buildArray,
  buildScalarFieldData,
  buildVectorFieldData,
  computeStats,
  isoContours,
  marchingSquaresSegments,
  Codomain,
} from "../src";

describe("geometry", () => {
  it("Box basics", () => {
    const b = Box.fromSpec([[-1, 1], [0, 2]]);
    expect(b.size).toEqual([2, 2]);
    expect(b.center).toEqual([0, 1]);
    expect(b.contains([0, 0])).toBe(true);
    expect(b.contains([0, -0.1])).toBe(false);
    expect(b.translate([1, 1]).intervals).toEqual([[0, 2], [1, 3]]);
    expect(b.scale([0, 0], [2, 0.5]).intervals).toEqual([[-2, 2], [0, 1]]);
    expect(() => Box.fromSpec([[1, 0]])).toThrow(SpecError);
  });

  it("DenseGrid is row-major with the last axis fastest", () => {
    const g = new DenseGrid([3, 2], Box.fromSpec([[0, 1], [0, 10]]));
    expect(g.sampleCount).toBe(6);
    expect(g.strides).toEqual([2, 1]);
    expect(g.point(1)).toEqual([0, 10]);
    expect(g.point(2)).toEqual([0.5, 0]);
    expect(g.gridPos(5)).toEqual([2, 1]);
    expect(g.locate([0.5, 5])).toEqual([1, 0.5]);
    expect(g.locate([2, 0])).toBeUndefined();
  });
});

describe("arrays", () => {
  it("builds inline / constant / oneHot / manyHot", () => {
    const a = buildArray({ type: "inline", shape: [2, 2], data: [1, 2, 3, 4] });
    expect(a.get(1, 0)).toBe(3);
    expect(a.get(-1, -1)).toBe(4);
    expect(a.part([1]).toNested()).toEqual([3, 4]);
    expect(() => buildArray({ type: "inline", shape: [2, 2], data: [1] })).toThrow(SpecError);
    expect(buildArray({ type: "constant", shape: [3], value: 7 }).toNested()).toEqual([7, 7, 7]);
    expect(buildArray({ type: "oneHot", shape: [3], pos: -1 }).toNested()).toEqual([0, 0, 1]);
    expect(buildArray({ type: "oneHot", shape: [2, 3], pos: [0, 2], hot: 5 }).toNested()).toEqual([[5, 0, 0], [0, 0, 5]]);
    expect(buildArray({ type: "manyHot", shape: [4], pos: [0, 2] }).toNested()).toEqual([1, 0, 1, 0]);
    expect(buildArray({ type: "manyHot", shape: [2, 2], pos: [[0, 1], []], cold: -1 }).toNested()).toEqual([[1, 1], [-1, -1]]);
  });

  it("builds symbolic arrays from cell indices", () => {
    const a = buildArray({ type: "symbolic", shape: [2, 3], expr: { op: "add", vals: [{ op: "mul", vals: [{ op: "coord", index: 0 }, 10] }, { op: "coord", index: 1 }] } });
    expect(a.toNested()).toEqual([[0, 1, 2], [10, 11, 12]]);
    const c = buildArray({ type: "symbolic", shape: [3], origin: [1], expr: { op: "square", val: { op: "coord", index: 0 } } });
    expect(c.toNested()).toEqual([1, 0, 1]);
    const v = buildArray({ type: "symbolicv", shape: [2, 2, 2], expr: { op: "coordv" } });
    expect(v.toNested()).toEqual([[[0, 0], [0, 1]], [[1, 0], [1, 1]]]);
    expect(() => buildArray({ type: "symbolicv", shape: [2, 2, 3], expr: { op: "coordv" } })).toThrow(SpecError);
    expect(() => buildArray({ type: "handle", shape: [2], path: "x" })).toThrow(NotSupportedError);
  });
});

describe("field data", () => {
  it("dense scalar: exact at samples, interpolated between", () => {
    const f = buildScalarFieldData({ type: "dense", samples: { type: "inline", shape: [2, 2], data: [0, 1, 2, 3] } }, 2);
    expect(f.kind).toBe("sampled");
    expect(f.value([0, 0])).toBe(0);
    expect(f.value([1, 1])).toBe(3);
    expect(f.value([0.5, 0.5])).toBe(1.5);
    expect(f.value([0.25, 0])).toBe(0.5);
    expect(f.value([2, 0])).toBeUndefined();
    const s = f.stats();
    expect(s.min).toBe(0); expect(s.max).toBe(3); expect(s.mean).toBe(1.5);
    // gradient by grid differences: d/dx = 2, d/dy = 1
    expect(f.partial(0)([0, 0], 0)).toBe(2);
    expect(f.partial(1)([0, 0], 0)).toBe(1);
  });

  it("symbolic scalar over a box, sampled onto a grid", () => {
    const f = buildScalarFieldData({ type: "symbolic", box: [[-1, 1], [-1, 1]], expr: { op: "add", vals: [{ op: "square", val: { op: "coord", index: 0 } }, { op: "square", val: { op: "coord", index: 1 } }] } }, 2);
    expect(f.kind).toBe("symbolic");
    expect(f.value([0.5, 0.5])).toBe(0.5);
    const g = new DenseGrid([3, 3], f.box);
    expect(Array.from(f.sampleOn(g))).toEqual([2, 1, 2, 1, 0, 1, 2, 1, 2]);
    expect(f.partial(0)([0.5, 0], -1)).toBe(1);
  });

  it("pointwise: inherits sample points, rejects mismatched supports", () => {
    const dense: ScalarFieldDataSpec = { type: "dense", samples: { type: "inline", shape: [2, 2], data: [0, 1, 2, 3] } };
    const f = buildScalarFieldData({ type: "pointwise", expr: { op: "mul", vals: ["a", 2, "k"] }, consts: { k: 0.5 }, scalars: { a: dense } }, 2);
    expect(f.kind).toBe("sampled");
    expect(f.samplePoints?.size).toEqual([2, 2]);
    expect(Array.from(f.sampleOn(f.samplePoints!))).toEqual([0, 1, 2, 3]);
    const other: ScalarFieldDataSpec = { type: "dense", samples: { type: "inline", shape: [3, 3], data: new Array(9).fill(0) } };
    expect(() => buildScalarFieldData({ type: "pointwise", expr: { op: "add", vals: ["a", "b"] }, scalars: { a: dense, b: other } }, 2)).toThrow(/identical discrete supports/);
    expect(() => buildScalarFieldData({ type: "pointwise", expr: "a", consts: { a: 1 }, scalars: { a: dense } }, 2)).toThrow(/bound in both/);
    // pointwise over only symbolic args stays symbolic, box = intersection
    const s1: ScalarFieldDataSpec = { type: "symbolic", box: [[0, 2], [0, 2]], expr: { op: "coord", index: 0 } };
    const s2: ScalarFieldDataSpec = { type: "symbolic", box: [[1, 3], [1, 3]], expr: { op: "coord", index: 1 } };
    const g = buildScalarFieldData({ type: "pointwise", expr: { op: "add", vals: ["p", "q"] }, scalars: { p: s1, q: s2 } }, 2);
    expect(g.kind).toBe("symbolic");
    expect(g.box.intervals).toEqual([[1, 2], [1, 2]]);
    expect(g.value([1.5, 2])).toBe(3.5);
  });

  it("gradient through arguments", () => {
    const dense: ScalarFieldDataSpec = { type: "dense", samples: { type: "symbolic", shape: [11, 11], expr: { op: "mul", vals: [{ op: "coord", index: 0 }, 0.1] } }, box: [[0, 1], [0, 1]] }; // f = x
    const v = buildVectorFieldData({ type: "pointwisev", expr: { op: "grad", val: { op: "mul", vals: ["a", "a"] } }, scalars: { a: dense } }, 2); // grad(x^2) = (2x, 0)
    expect(v.kind).toBe("sampled");
    const at = v.value([0.5, 0.5])!;
    expect(at[0]).toBeCloseTo(1, 6);
    expect(at[1]).toBeCloseTo(0, 6);
    const onGrid = v.sampleOn(v.samplePoints!);
    expect(onGrid[(5 * 11 + 5) * 2]).toBeCloseTo(1, 6);
  });

  it("translate / scale pull back the domain and the support", () => {
    const base: ScalarFieldDataSpec = { type: "symbolic", box: [[0, 1], [0, 1]], expr: { op: "coord", index: 0 } };
    const t = buildScalarFieldData({ type: "translate", arg: base, vec: [10, 0] }, 2);
    expect(t.box.intervals).toEqual([[10, 11], [0, 1]]);
    expect(t.value([10.25, 0.5])).toBeCloseTo(0.25);
    const s = buildScalarFieldData({ type: "scale", arg: { type: "dense", samples: { type: "inline", shape: [2, 2], data: [0, 1, 2, 3] } }, scale: 2 }, 2);
    expect(s.box.intervals).toEqual([[0, 2], [0, 2]]);
    expect(s.samplePoints!.box.intervals).toEqual([[0, 2], [0, 2]]);
    expect(s.value([2, 2])).toBe(3);
    expect(s.partial(0)([0, 0], 0)).toBe(1); // inner slope 2, halved
  });

  it("dense vector", () => {
    const v = buildVectorFieldData({ type: "densev", samples: { type: "symbolicv", shape: [3, 3, 2], expr: { op: "coordv" } } }, 2);
    expect(v.value([1, 0.5])).toEqual([2, 1]);
    expect(() => buildVectorFieldData({ type: "densev", samples: { type: "constant", shape: [3, 3], value: 0 } }, 2)).toThrow(SpecError);
  });
});

describe("stats & codomain", () => {
  it("computeStats handles non-finite values", () => {
    const s = computeStats([1, -2, 0, NaN, Infinity, 3]);
    expect(s).toMatchObject({ total: 6, finite: 4, nan: 1, posInf: 1, pos: 3, neg: 1, zero: 1, min: -2, max: 3, mean: 0.5 });
  });
  it("codomain param mapping respects log/flip", () => {
    const c = new Codomain("celoss");
    expect(c.min).toBe(0); expect(c.flip).toBe(true); expect(c.log).toBe("e");
    expect(c.toParam(1, 1, 100)).toBe(1);
    expect(c.toParam(10, 1, 100)).toBeCloseTo(0.5);
    expect(c.fromParam(c.toParam(37, 1, 100), 1, 100)).toBeCloseTo(37);
    expect(new Codomain({ min: -1, max: 1 }).clip(5)).toBe(1);
  });
});

describe("marching squares", () => {
  const g = new DenseGrid([21, 21], Box.fromSpec([[-1, 1], [-1, 1]]));
  const r2 = Float64Array.from({ length: g.sampleCount }, (_, i) => { const [x, y] = g.point(i) as [number, number]; return x * x + y * y; });

  it("contours of x^2+y^2 at 0.25 form one closed loop of radius ~0.5", () => {
    const segs = marchingSquaresSegments(g, r2, 0.25);
    expect(segs.length).toBeGreaterThan(0);
    const lines = isoContours(g, r2, 0.25);
    expect(lines.length).toBe(1);
    const l = lines[0]!;
    expect(l[0]).toBeCloseTo(l[l.length - 2]!); expect(l[1]).toBeCloseTo(l[l.length - 1]!); // closed
    for (let i = 0; i < l.length; i += 2) expect(Math.hypot(l[i]!, l[i + 1]!)).toBeCloseTo(0.5, 1);
  });

  it("open contour of x at 0.05 spans the box", () => {
    const xs = Float64Array.from({ length: g.sampleCount }, (_, i) => g.point(i)[0]!);
    const lines = isoContours(g, xs, 0.05);
    expect(lines.length).toBe(1);
    const ys = Array.from(lines[0]!).filter((_, i) => i % 2 === 1);
    expect(Math.min(...ys)).toBeCloseTo(-1); expect(Math.max(...ys)).toBeCloseTo(1);
    expect(isoContours(g, xs, 5)).toEqual([]);
  });
});

describe("bundle", () => {
  const spec: BundleSpec = {
    tensatory: "0.1",
    name: "test",
    manifolds: { plane: { numDims: 2, dimNames: ["x", "y"] } },
    fields: {
      bowl: { kind: "scalar", codomain: "norm", data: { type: "symbolic", box: [[-2, 2], [-2, 2]], expr: { op: "norm", vec: { op: "coordv" } } } },
      twice: { kind: "scalar", data: { type: "pointwise", expr: { op: "mul", vals: ["bowl", 2] }, scalars: { bowl: "bowl" } } },
      g: { kind: "vector", data: { type: "pointwisev", expr: { op: "grad", val: "bowl" }, scalars: { bowl: "bowl" } } },
      loop1: { kind: "scalar", data: { type: "pointwise", expr: "a", scalars: { a: "loop2" } } },
      loop2: { kind: "scalar", data: { type: "pointwise", expr: "a", scalars: { a: "loop1" } } },
      wrongKind: { kind: "scalar", data: { type: "pointwise", expr: "a", scalars: { a: "g" } } },
    },
    pointSets: { origin: { points: [[0, 0]], labels: ["0"] } },
  };

  it("parses, resolves references lazily, detects cycles", () => {
    const b = Bundle.parse(spec);
    expect(b.defaultManifold!.dimNames).toEqual(["x", "y"]);
    expect(b.scalarField("twice").data.value([1, 0])).toBe(2);
    expect(b.scalarField("bowl").codomain.name).toBe("norm");
    const grad = b.vectorField("g").data.value([0, 1.5])!;
    expect(grad[0]).toBeCloseTo(0); expect(grad[1]).toBeCloseTo(1);
    expect(() => b.field("loop1")).toThrow(/cycle/);
    expect(() => b.field("wrongKind")).toThrow(/vector field, expected a scalar/);
    expect(() => b.field("nope")).toThrow(/unknown field/);
    const errors = b.buildAll();
    expect([...errors.keys()].sort()).toEqual(["loop1", "loop2", "wrongKind"]);
    expect(b.pointSets.get("origin")!.points).toEqual([[0, 0]]);
  });

  it("rejects malformed json with a path", () => {
    expect(() => Bundle.parse({ tensatory: "0.1", fields: { f: { kind: "scalar", data: { type: "symbolic", expr: { op: "bogus" } } } } })).toThrow(SpecError);
    expect(() => Bundle.parse({ tensatory: "9", fields: {} })).toThrow(/tensatory/);
  });

  it("infers the manifold when none is declared", () => {
    const b = Bundle.parse({ tensatory: "0.1", fields: { f: { kind: "scalar", data: { type: "dense", samples: { type: "constant", shape: [2, 3], value: 1 } } } } });
    expect(b.defaultManifold!.numDims).toBe(2);
  });
});

describe("higher derivatives through arguments", () => {
  it("∇|∇f| of a symbolic field is exact (second derivatives of the argument)", () => {
    const x = { op: "coord", index: 0 } as const, y = { op: "coord", index: 1 } as const;
    const f = buildScalarFieldData({ type: "symbolic", box: [[-2, 2], [-2, 2]], expr: { op: "add", vals: [{ op: "square", val: x }, { op: "mul", vals: [3, { op: "square", val: y }] }] } }, 2); // x² + 3y²
    const gn = new SymbolicScalarFieldData({ k: "norm", v: { k: "grad", s: { k: "arg", name: "f" } } }, 2, { scalars: { f }, vectors: {} }); // |∇f| = sqrt(4x² + 36y²)
    const g = new SymbolicVectorFieldData({ k: "grad", s: { k: "arg", name: "n" } }, 2, { scalars: { n: gn }, vectors: {} });
    const [px, py] = [0.5, 0.25];
    const n = Math.sqrt(4 * px * px + 36 * py * py);
    const v = g.value([px, py])!;
    expect(v[0]).toBeCloseTo((4 * px) / n, 8);
    expect(v[1]).toBeCloseTo((36 * py) / n, 8);
    expect(g.kind).toBe("symbolic");
  });
  it("second derivatives of sampled data fall back to finite differences", () => {
    const dense = buildScalarFieldData({ type: "dense", box: [[0, 1], [0, 1]], samples: { type: "symbolic", shape: [41, 41], expr: { op: "square", val: { op: "mul", vals: [{ op: "coord", index: 0 }, 0.025] } } } }, 2); // x²
    const d2 = dense.derivative(0).derivative(0);
    expect(d2.kind).toBe("sampled");
    expect(d2.fn([0.5, 0.5], d2.samplePoints!.pos([20, 20]))).toBeCloseTo(2, 6);
    expect(d2.value([0.5, 0.5])).toBeCloseTo(2, 2);
  });
});

describe("formatReal", () => {
  it("uses compact superscript scientific notation", async () => {
    const { formatReal } = await import("../src");
    expect(formatReal(6.2384e-5)).toBe("6.24·10⁻⁵");
    expect(formatReal(1.53e-10)).toBe("1.53·10⁻¹⁰");
    expect(formatReal(2512.3)).toBe("2510");
    expect(formatReal(25123)).toBe("2.51·10⁴");
    expect(formatReal(99999)).toBe("1·10⁵");
    expect(formatReal(0.883)).toBe("0.883");
    expect(formatReal(-0.000123)).toBe("-1.23·10⁻⁴");
    expect(formatReal(0)).toBe("0");
  });
});

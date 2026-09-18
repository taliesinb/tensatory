import { describe, expect, it } from "vitest";
import type { ScalarDistributionSpec } from "@tensatory/schema";
import { SizedArraySchema, SpecError, buildArray, hashSeed } from "../src";

const N = 20000;
const draw = (dist: ScalarDistributionSpec, n = N) =>
  buildArray({ type: "random", shape: [n], dist }).data as Float64Array;
const mean = (xs: Float64Array) => xs.reduce((a, b) => a + b, 0) / xs.length;
const variance = (xs: Float64Array) => { const m = mean(xs); return xs.reduce((a, b) => a + (b - m) ** 2, 0) / xs.length; };

describe("random arrays", () => {
  it("parses and validates", () => {
    expect(SizedArraySchema.safeParse({ type: "random", shape: [2], dist: { type: "gaussian", seed: 1 } }).success).toBe(true);
    expect(SizedArraySchema.safeParse({ type: "random", shape: [2], dist: { type: "gaussian", seed: "d0", scale: 0.1 }, widget: { label: "d₀" } }).success).toBe(true);
    expect(SizedArraySchema.safeParse({ type: "random", shape: [2], dist: { type: "gaussian" } }).success).toBe(false); // seed is required (may be null)
    expect(SizedArraySchema.safeParse({ type: "random", shape: [2], dist: { type: "gaussian", seed: null, scale: 0 } }).success).toBe(false);
    expect(SizedArraySchema.safeParse({ type: "random", shape: [2], dist: { type: "studentT", seed: 0 } }).success).toBe(false); // df required
    expect(() => buildArray({ type: "random", shape: [2], dist: { type: "integers", seed: 0, lo: 3, hi: 3 } })).toThrow(SpecError);
    expect(() => buildArray({ type: "random", shape: [2], dist: { type: "discrete", seed: 0, values: [1, 2], probs: [1] } })).toThrow(SpecError);
  });

  it("is a pure function of seed and cell index", () => {
    const a = draw({ type: "gaussian", seed: 42 }, 100);
    const b = draw({ type: "gaussian", seed: 42 }, 100);
    expect(Array.from(b)).toEqual(Array.from(a));
    const c = draw({ type: "gaussian", seed: 43 }, 100);
    expect(Array.from(c)).not.toEqual(Array.from(a));
    // a longer array keeps the prefix; a reshaped array keeps the flat order
    const longer = draw({ type: "gaussian", seed: 42 }, 200);
    expect(Array.from(longer.subarray(0, 100))).toEqual(Array.from(a));
    const grid = buildArray({ type: "random", shape: [10, 10], dist: { type: "gaussian", seed: 42 } }).data as Float64Array;
    expect(Array.from(grid)).toEqual(Array.from(a));
    // string seeds hash, numbers of any size are accepted
    expect(hashSeed("d0")).not.toBe(hashSeed("d1"));
    expect(Array.from(draw({ type: "uniform", seed: "d0" }, 5))).toEqual(Array.from(draw({ type: "uniform", seed: "d0" }, 5)));
    expect(Array.from(draw({ type: "uniform", seed: "d0" }, 5))).not.toEqual(Array.from(draw({ type: "uniform", seed: "d1" }, 5)));
    expect(draw({ type: "uniform", seed: 2 ** 40 + 7 }, 3)).toHaveLength(3);
    // null: unseeded but fixed within the process
    expect(Array.from(draw({ type: "uniform", seed: null }, 5))).toEqual(Array.from(draw({ type: "uniform", seed: null }, 5)));
  });

  it("has the right moments and supports", () => {
    const tol = 0.03;
    const u = draw({ type: "uniform", seed: 1 });
    expect(Math.min(...u)).toBeGreaterThanOrEqual(-1); expect(Math.max(...u)).toBeLessThan(1);
    expect(mean(u)).toBeCloseTo(0, 1); expect(variance(u)).toBeCloseTo(1 / 3, 1);
    const u2 = draw({ type: "uniform", seed: 1, loc: 0.5, scale: 0.5 });
    expect(Math.min(...u2)).toBeGreaterThanOrEqual(0); expect(Math.max(...u2)).toBeLessThan(1);

    const g = draw({ type: "gaussian", seed: 2, loc: 3, scale: 2 });
    expect(Math.abs(mean(g) - 3)).toBeLessThan(tol * 2); expect(Math.abs(variance(g) - 4)).toBeLessThan(0.15);

    const l = draw({ type: "laplace", seed: 3, scale: 0.5 });
    expect(Math.abs(mean(l))).toBeLessThan(tol); expect(Math.abs(variance(l) - 2 * 0.25)).toBeLessThan(0.05);

    const e = draw({ type: "exponential", seed: 4, scale: 2 });
    expect(Math.min(...e)).toBeGreaterThanOrEqual(0);
    expect(Math.abs(mean(e) - 2)).toBeLessThan(0.06); expect(Math.abs(variance(e) - 4)).toBeLessThan(0.2);

    const t = draw({ type: "studentT", seed: 5, df: 5 });
    expect(Math.abs(mean(t))).toBeLessThan(0.05); expect(Math.abs(variance(t) - 5 / 3)).toBeLessThan(0.15); // var = df / (df - 2)

    const b = draw({ type: "bernoulli", seed: 6, p: 0.25, hot: 1, cold: -1 });
    expect(new Set(b)).toEqual(new Set([1, -1]));
    expect(Math.abs(mean(b) - (0.25 - 0.75))).toBeLessThan(tol);

    const d = draw({ type: "discrete", seed: 7, values: [10, 20, 30], probs: [1, 1, 2] });
    expect(new Set(d)).toEqual(new Set([10, 20, 30]));
    expect(Math.abs(d.filter((x) => x === 30).length / N - 0.5)).toBeLessThan(tol);

    const i = draw({ type: "integers", seed: 8, lo: -2, hi: 3 });
    expect(new Set(i)).toEqual(new Set([-2, -1, 0, 1, 2]));
    expect(Math.abs(mean(i))).toBeLessThan(0.05);
  });
});

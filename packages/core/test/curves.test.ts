// Curves (schema/curves.ts, notes/curves.md): symbolic, sampled (linear / cubic / step, times, velocities, closed),
// flow (RK4 against exact solutions), pushforwards, bundle integration, handles and slicing.

import { describe, expect, it } from "vitest";
import type { BundleSpec, InlineArraySpec } from "@tensatory/schema";
import { Bundle, FlowCurveData, SampledCurveData, SpecError, collectHandles, sliceSpec } from "../src";

const close = (a: ArrayLike<number>, b: ArrayLike<number>, digits = 9) => { expect(a.length).toBe(b.length); for (let i = 0; i < a.length; i++) expect(a[i]).toBeCloseTo(b[i]!, digits); };
const T = { op: "coord", index: 0 } as const;
const bundle = (curves: BundleSpec["curves"], extra: Partial<BundleSpec> = {}): Bundle =>
  Bundle.parse({ tensatory: "0.1", manifolds: { m: { numDims: 2 }, v: { numDims: 3 } }, defaultManifold: "m", fields: {}, ...extra, curves } satisfies BundleSpec);

describe("symbolic curves", () => {
  const b = bundle({
    circle: { data: { type: "symbolic", interval: [0, 2 * Math.PI], expr: { op: "compv", coeffs: [{ op: "cos", val: T }, { op: "sin", val: T }] } }, param: { name: "θ", unit: "rad" } },
    helix: { domain: "v", data: { type: "symbolic", interval: [0, 4], expr: { op: "compv", coeffs: [{ op: "cos", val: T }, { op: "sin", val: T }, { op: "mul", vals: ["k", T] }] }, consts: { k: 0.5 } } },
  });
  it("evaluates, differentiates exactly, and knows its interval and parameter", () => {
    expect(b.buildAll().size).toBe(0);
    const c = b.curve("circle");
    expect(c.data.kind).toBe("symbolic");
    expect(c.data.interval).toEqual([0, 2 * Math.PI]);
    expect(c.param).toEqual({ name: "θ", unit: "rad", codomain: undefined });
    close(c.data.point(Math.PI / 2)!, [0, 1]);
    close(c.data.velocity(Math.PI / 2)!, [-1, 0]);
    expect(c.data.point(7)).toBeUndefined();
    expect(c.data.arcLength()).toBeCloseTo(2 * Math.PI, 3);
    const h = b.curve("helix").data;
    expect(h.dimCount).toBe(3);
    close(h.point(4)!, [Math.cos(4), Math.sin(4), 2]);
    const pl = h.polyline(1, 3, 5);
    expect(pl.length).toBe(15);
    close(pl.subarray(0, 3), [Math.cos(1), Math.sin(1), 0.5]);
    close(pl.subarray(12, 15), [Math.cos(3), Math.sin(3), 1.5]);
  });
  it("rejects a wrong dimension and an empty interval", () => {
    expect(bundle({ c: { data: { type: "symbolic", interval: [0, 1], expr: { op: "compv", coeffs: [T, T, T] } } } }).buildAll().get("curves.c")!.message).toMatch(/3 components, expected 2/);
    expect(bundle({ c: { data: { type: "symbolic", interval: [0, 1], expr: { op: "compv", coeffs: [T, { op: "coord", index: 1 }] } } } }).buildAll().get("curves.c")!.message).toMatch(/only use coordinate 0/);
    expect(bundle({ c: { data: { type: "symbolic", interval: [0, 1], expr: { op: "coordv" } } } }).buildAll().get("curves.c")!.message).toMatch(/no meaning/);
    expect(bundle({ c: { data: { type: "symbolic", interval: [1, 1], expr: { op: "compv", coeffs: [T, T] } } } }).buildAll().get("curves.c")!.message).toMatch(/t0 < t1/);
  });
});

describe("sampled curves", () => {
  const pts: InlineArraySpec = { type: "inline", shape: [4, 2], data: [0, 0, 1, 1, 2, 0, 3, 1] };
  it("linear: times default to the index, samples are the polyline vertices", () => {
    const c = bundle({ z: { data: { type: "sampled", points: pts }, param: { name: "step" } } }).curve("z").data as SampledCurveData;
    expect(c.kind).toBe("sampled");
    expect(c.interval).toEqual([0, 3]);
    expect([...c.sampleTimes]).toEqual([0, 1, 2, 3]);
    close(c.point(0.5)!, [0.5, 0.5]);
    close(c.point(2.25)!, [2.25, 0.25]);
    close(c.velocity(0.5)!, [1, 1]);
    close(c.velocity(1.5)!, [1, -1]);
    close(c.polyline(), [0, 0, 1, 1, 2, 0, 3, 1]);
    close(c.polyline(0.5, 2.5), [0.5, 0.5, 1, 1, 2, 0, 2.5, 0.5]);
    expect(c.point(3.5)).toBeUndefined();
    expect(c.arcLength()).toBeCloseTo(3 * Math.SQRT2, 12);
  });
  it("explicit times (array or interval), labels, step interpolation", () => {
    const b = bundle({
      a: { data: { type: "sampled", points: pts, times: { type: "inline", shape: [4], data: [0, 10, 20, 40] }, labels: ["a", "b", "c", "d"] } },
      u: { data: { type: "sampled", points: pts, times: [1, 4] } },
      s: { data: { type: "sampled", points: pts, interp: "step" } },
    });
    expect(b.buildAll().size).toBe(0);
    const a = b.curve("a").data as SampledCurveData;
    expect(a.interval).toEqual([0, 40]);
    close(a.point(30)!, [2.5, 0.5]);
    expect(a.labels).toEqual(["a", "b", "c", "d"]);
    const u = b.curve("u").data;
    expect(u.interval).toEqual([1, 4]);
    close(u.point(2)!, [1, 1]);
    const s = b.curve("s").data;
    close(s.point(0.99)!, [0, 0]);
    close(s.point(1)!, [1, 1]);
    close(s.velocity(0.5)!, [0, 0]);
    expect(s.polyline().length).toBeGreaterThan(8); // a staircase has both ends of every jump
  });
  it("cubic: Hermite with given velocities interpolates a parabola exactly; Catmull–Rom otherwise", () => {
    const N = 5, P: number[] = [], V: number[] = [];
    for (let i = 0; i < N; i++) { const t = i; P.push(t, t * t); V.push(1, 2 * t); }
    const b = bundle({
      h: { data: { type: "sampled", interp: "cubic", points: { type: "inline", shape: [N, 2], data: P }, velocities: { type: "inline", shape: [N, 2], data: V } } },
      cr: { data: { type: "sampled", interp: "cubic", points: { type: "inline", shape: [N, 2], data: P } } },
    });
    const h = b.curve("h").data;
    for (const t of [0.3, 1.5, 2.7, 3.9]) { close(h.point(t)!, [t, t * t], 10); close(h.velocity(t)!, [1, 2 * t], 10); }
    const cr = b.curve("cr").data;
    close(cr.point(2)!, [2, 4]); // samples are hit exactly
    expect(Math.abs(cr.point(1.5)![1]! - 2.25)).toBeLessThan(0.1); // between, close to the parabola
    expect(h.polyline(0, 4, 64).length).toBeGreaterThanOrEqual(64 * 2);
  });
  it("closed: the interval gains one step and wraps back to the first point", () => {
    const c = bundle({ c: { data: { type: "sampled", points: { type: "inline", shape: [4, 2], data: [1, 0, 0, 1, -1, 0, 0, -1] }, closed: true } } }).curve("c").data;
    expect(c.interval).toEqual([0, 4]);
    close(c.point(4)!, [1, 0]);
    close(c.point(3.5)!, [0.5, -0.5]);
  });
  it("rejects mismatched shapes and non-increasing times", () => {
    const e = (curves: BundleSpec["curves"]) => bundle(curves).buildAll().get("curves.c")!.message;
    expect(e({ c: { data: { type: "sampled", points: { type: "inline", shape: [4, 3], data: new Array(12).fill(0) } } } })).toMatch(/\[N, 2\]/);
    expect(e({ c: { data: { type: "sampled", points: pts, times: { type: "inline", shape: [3], data: [0, 1, 2] } } } })).toMatch(/times must be a vector|3 entries for 4/);
    expect(e({ c: { data: { type: "sampled", points: pts, times: { type: "inline", shape: [4], data: [0, 2, 1, 3] } } } })).toMatch(/strictly increasing/);
    expect(e({ c: { data: { type: "sampled", points: pts, labels: ["x"] } } })).toMatch(/labels has 1/);
  });
});

describe("flow curves", () => {
  const fields: BundleSpec["fields"] = {
    // ẋ = -x: x(t) = x0 e^{-t}
    decay: { kind: "vector", data: { type: "symbolicv", expr: { op: "scalev", vec: { op: "coordv" }, by: -1 }, box: [[-3, 3], [-3, 3]] } },
    // rotation ẋ = (-y, x): circles
    rot: { kind: "vector", data: { type: "symbolicv", expr: { op: "compv", coeffs: [{ op: "negate", val: { op: "coord", index: 1 } }, { op: "coord", index: 0 }] }, box: [[-3, 3], [-3, 3]] } },
    // a bowl: gradient descent from (1, 1) follows -(2x, 2y): x(t) = e^{-2t}
    bowl: { kind: "scalar", data: { type: "symbolic", expr: { op: "add", vals: [{ op: "square", val: { op: "coord", index: 0 } }, { op: "square", val: { op: "coord", index: 1 } }] }, box: [[-2, 2], [-2, 2]] } },
  };
  it("integrates a vector field forwards and backwards (RK4 vs the exact solution)", () => {
    const b = bundle({
      d: { data: { type: "flow", field: "decay", start: [1, 2], interval: [0, 2], method: { step: 0.01 } } },
      r: { data: { type: "flow", field: "rot", start: [1, 0], interval: [-1, 1], method: { step: 0.005 } } },
    }, { fields });
    expect(b.buildAll().size).toBe(0);
    const d = b.curve("d").data as FlowCurveData;
    expect(d.kind).toBe("symbolic");
    close(d.point(0)!, [1, 2]);
    close(d.point(1)!, [Math.exp(-1), 2 * Math.exp(-1)], 8);
    close(d.point(2)!, [Math.exp(-2), 2 * Math.exp(-2)], 8);
    close(d.velocity(1)!, [-Math.exp(-1), -2 * Math.exp(-1)], 8); // ẋ = F(x)
    const r = b.curve("r").data;
    close(r.point(-1)!, [Math.cos(1), -Math.sin(1)], 8);
    close(r.point(0.5)!, [Math.cos(0.5), Math.sin(0.5)], 8);
    close(r.point(0)!, [1, 0]);
    expect(r.polyline().length / 2).toBeGreaterThan(300); // the integration steps are the vertices
  });
  it("a scalar field means its gradient: descent / ascent", () => {
    const b = bundle({
      down: { data: { type: "flow", field: "bowl", start: [1, 1], interval: [0, 1], method: { step: 0.005 } } },
      up: { data: { type: "flow", field: "bowl", start: [0.1, 0], interval: [0, 1], dir: "ascending", method: { step: 0.005 } } },
    }, { fields });
    expect(b.buildAll().size).toBe(0);
    close(b.curve("down").data.point(1)!, [Math.exp(-2), Math.exp(-2)], 7);
    close(b.curve("up").data.point(1)!, [0.1 * Math.exp(2), 0], 7);
  });
  it("an inline vector field spec, a default step, and errors", () => {
    const b = bundle({
      f: { data: { type: "flow", field: fields.rot!.data as never, start: [0, 1], interval: [0, Math.PI] } },
      bad: { data: { type: "flow", field: "rot", start: [0, 0, 1], interval: [0, 1] } },
      tol: { data: { type: "flow", field: "rot", start: [0, 1], interval: [0, 1], method: { tol: 1e-6 } } },
    }, { fields });
    const errors = b.buildAll();
    expect([...errors.keys()].sort()).toEqual(["curves.bad", "curves.tol"]);
    expect(errors.get("curves.bad")!.message).toMatch(/expected 2 components/);
    expect(errors.get("curves.tol")!.message).toMatch(/method.tol/);
    close(b.curve("f").data.point(Math.PI)!, [0, -1], 6);
  });
});

describe("pushforwards and references", () => {
  it("translate / scale move the points (and scale the velocity); curves refer to curves; cycles are errors", () => {
    const b = bundle({
      base: { data: { type: "sampled", points: { type: "inline", shape: [2, 2], data: [0, 0, 1, 2] } } },
      moved: { data: { type: "translate", arg: "base", vec: [10, 20] } },
      big: { data: { type: "scale", arg: { type: "translate", arg: "base", vec: [1, 0] }, origin: [1, 0], scale: [2, 3] } },
      loop: { data: { type: "translate", arg: "loop", vec: [0, 0] } },
    });
    const errors = b.buildAll();
    expect([...errors.keys()]).toEqual(["curves.loop"]);
    expect(errors.get("curves.loop")!.message).toMatch(/cycle/);
    const moved = b.curve("moved").data;
    expect(moved.kind).toBe("sampled");
    close(moved.point(0.5)!, [10.5, 21]);
    close(moved.velocity(0.5)!, [1, 2]);
    const big = b.curve("big").data;
    close(big.point(1)!, [1 + 2 * (2 - 1), 3 * 2]);
    close(big.velocity(0.5)!, [2, 6]);
    expect(b.curves.map((c) => c.id)).toEqual(["base", "moved", "big"]);
  });
  it("domain defaults and checks like fields; info like everything else", () => {
    const b = bundle({ c: { domain: "v", data: { type: "sampled", points: { type: "inline", shape: [1, 3], data: [1, 2, 3] } }, summary: "one point", name: "dot" } });
    expect(b.buildAll().size).toBe(0);
    const c = b.curve("c");
    expect(c.domain.id).toBe("v");
    expect(c.info).toEqual({ name: "dot", summary: "one point", details: undefined });
    expect(() => bundle({ c: { domain: "nope", data: { type: "symbolic", interval: [0, 1], expr: { op: "coordv" } } } }).curve("c")).toThrow(SpecError);
  });
});

describe("curves in bundles", () => {
  it("collectHandles finds the arrays of sampled and flow curves", () => {
    const spec: BundleSpec = {
      tensatory: "0.1", manifolds: { m: { numDims: 2 } }, fields: {},
      curves: {
        a: { data: { type: "sampled", points: "traj.npy", times: { type: "handle", path: "t.npy" }, velocities: "v.npz/g" } },
        b: { data: { type: "flow", field: { type: "densev", samples: { type: "handle", path: "F.npy", shape: [4, 4, 2] } }, start: { type: "handle", path: "x0.npy", shape: [2] }, interval: [0, 1] } },
        c: { data: { type: "scale", arg: { type: "translate", arg: "a", vec: "vec.npy" }, scale: 2, origin: { type: "handle", path: "o.npy" } } },
      },
    };
    expect([...collectHandles(spec).keys()].sort()).toEqual(["F.npy", "o.npy", "t.npy", "traj.npy", "v.npz/g", "vec.npy", "x0.npy"]);
  });
  it("slicing an N-D space projects sampled curves and drops the others with a reason", () => {
    const spec: BundleSpec = {
      tensatory: "0.1", manifolds: { p: { numDims: 4 } }, fields: {},
      curves: {
        s: { domain: "p", data: { type: "sampled", points: { type: "inline", shape: [2, 4], data: [0, 1, 2, 3, 4, 5, 6, 7] }, velocities: { type: "inline", shape: [2, 4], data: [1, 1, 1, 1, 2, 2, 2, 2] } } },
        y: { domain: "p", data: { type: "symbolic", interval: [0, 1], expr: { op: "coordv" } } },
      },
    };
    const r = sliceSpec(spec, "p", [1, 3]);
    expect(r.dropped).toEqual({ "curves.y": "symbolic curves are not sliced yet" });
    expect(r.spec.curves!.s!.data).toEqual({ type: "sampled", points: { type: "inline", shape: [2, 2], data: [1, 3, 5, 7] }, velocities: { type: "inline", shape: [2, 2], data: [1, 1, 2, 2] } });
    expect(new Bundle(r.spec).buildAll().size).toBe(0);
  });
});

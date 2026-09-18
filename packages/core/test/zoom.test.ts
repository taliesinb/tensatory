import { describe, expect, it } from "vitest";
import type { BundleSpec } from "@tensatory/schema";
import { Bundle, zoomBoxes, zoomable } from "../src";

const spec = {
  tensatory: "0.1",
  manifolds: { p: { numDims: 2 }, q: { numDims: 1 } },
  fields: {
    f: { kind: "scalar", domain: "p", data: { type: "symbolic", expr: { op: "coord", index: 0 }, box: [[-1, 3], [0, 2]] } },
    g: { kind: "scalar", domain: "p", data: { type: "symbolic", expr: 1 } }, // unit box
    d: { kind: "scalar", domain: "p", data: { type: "dense", samples: { type: "constant", shape: [2, 2], value: 1 }, box: [[0, 1], [0, 1]] } },
    s: { kind: "scalar", domain: "p", data: { type: "pointwise", expr: { op: "add", vals: ["f", "h"] }, scalars: { f: "f", h: { type: "symbolic", expr: 2, box: [[-1, 3], [0, 2]] } } } },
    t: { kind: "scalar", domain: "p", data: { type: "translate", vec: [1, 1], arg: { type: "symbolic", expr: 0, box: [[0, 2], [0, 2]] } } },
    o: { kind: "scalar", domain: "q", data: { type: "symbolic", expr: 0, box: [[0, 1]] } },
  },
} as unknown as BundleSpec;

describe("box zoom", () => {
  it("scales symbolic boxes of one manifold around their centres, leaves sampled data and other manifolds alone", () => {
    expect(zoomable(spec, "p")).toBe(true);
    const z = new Bundle(zoomBoxes(spec, "p", 1.5));
    expect(z.scalarField("f").data.box.intervals).toEqual([[-2, 4], [-0.5, 2.5]]);
    expect(z.scalarField("g").data.box.intervals).toEqual([[-0.25, 1.25], [-0.25, 1.25]]);
    expect(z.scalarField("d").data.box.intervals).toEqual([[0, 1], [0, 1]]);
    expect(z.scalarField("s").data.box.intervals).toEqual([[-2, 4], [-0.5, 2.5]]); // both arguments zoomed
    expect(z.scalarField("t").data.box.intervals).toEqual([[0.5, 3.5], [0.5, 3.5]]); // inner [0,2]² → [-.5,2.5]², translated by 1
    expect(z.scalarField("o").data.box.intervals).toEqual([[0, 1]]);
    expect(z.scalarField("f").data.value([3.5, 2.4])).toBe(3.5); // evaluable in the widened box
    expect(zoomBoxes(spec, "p", 1)).toBe(spec);
  });

  it("works with the implicit default manifold and reports unzoomable spaces", () => {
    const one = { tensatory: "0.1", fields: { f: { kind: "scalar", data: { type: "symbolic", expr: 0, box: [[0, 2], [0, 2]] } } } } as unknown as BundleSpec;
    expect(new Bundle(zoomBoxes(one, "default", 2)).scalarField("f").data.box.intervals).toEqual([[-1, 3], [-1, 3]]);
    const sampled = { tensatory: "0.1", fields: { d: spec.fields.d } , manifolds: { p: { numDims: 2 } } } as unknown as BundleSpec;
    expect(zoomable(sampled, "p")).toBe(false);
  });
});

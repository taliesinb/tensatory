// Slices (bundle/slice.ts): an N-D manifold restricted to an axis-aligned 2D / 3D subspace through its origin, as
// a spec rewrite. Checked against the N-D fields evaluated at the embedded points.

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import type { BundleSpec } from "@tensatory/schema";
import { Bundle, DenseGrid, sliceSpec, sliceable } from "../src";

const dir = join(__dirname, "../../../apps/viewer/public/bundles");
const nd = JSON.parse(readFileSync(join(dir, "symbolic-nd.json"), "utf8")) as BundleSpec;
const embed = (p: readonly number[], dims: readonly number[], origin: readonly number[]): number[] => {
  const q = [...origin];
  dims.forEach((d, a) => { q[d] = p[a]!; });
  return q;
};
const probe = (k: number): number[][] => (k === 2 ? [[0.3, -0.7], [-1.1, 0.4], [0.9, 0.9], [0, 0]] : [[0.3, -0.7, 0.5], [-1.1, 0.4, -0.2], [0.9, 0.9, -0.9], [0, 0, 0]]);

describe("sliceSpec", () => {
  it("sliceable: 3 < D <= 8", () => {
    for (const m of Object.keys(nd.manifolds!)) expect(sliceable(nd, m)).toBe(true);
    const d2 = JSON.parse(readFileSync(join(dir, "symbolic2d.json"), "utf8")) as BundleSpec;
    expect(sliceable(d2, "plane")).toBe(false);
  });

  for (const [manifold, dims] of [["gauss4", [0, 1]], ["gauss4", [1, 2, 3]], ["quad4", [0, 3]], ["mix5", [0, 1, 2]], ["mix5", [2, 4]], ["rosen5", [0, 1]], ["rosen5", [1, 3, 4]], ["waves5", [0, 2, 3]]] as const) {
    it(`${manifold} sliced along [${dims}] agrees with the N-D fields at embedded points (scalars, vector projections, full-gradient norms)`, () => {
      const full = new Bundle(nd);
      const { spec, dropped } = sliceSpec(nd, manifold, dims);
      expect(dropped).toEqual({});
      const sliced = Bundle.parse(spec);
      expect(sliced.buildAll().size).toBe(0);
      const m = sliced.manifolds.get(manifold)!;
      expect(m.numDims).toBe(dims.length);
      expect(m.spec.origin).toBeUndefined();
      const origin = nd.manifolds![manifold]!.origin ?? new Array<number>(full.manifolds.get(manifold)!.numDims).fill(0);
      for (const id of full.scalarFieldIds.filter((f) => full.scalarField(f).domain.id === manifold)) {
        const a = full.scalarField(id).data, b = sliced.scalarField(id).data;
        expect(b.dimCount).toBe(dims.length);
        for (const p of probe(dims.length)) {
          const q = embed(p, dims, origin);
          expect(b.value(p), `${id} at ${p}`).toBeCloseTo(a.value(q)!, 9);
          // derivatives along the slice are the N-D partials along those dimensions
          const da = a.derivative(dims[0]!).value(q)!;
          if (Number.isFinite(da)) expect(b.derivative(0).value(p), `∂${id} at ${p}`).toBeCloseTo(da, 7); // (∂|∇p| is NaN at the peak)
        }
      }
      for (const id of full.vectorFieldIds.filter((f) => full.vectorField(f).domain.id === manifold)) {
        const a = full.vectorField(id).data, b = sliced.vectorField(id).data;
        for (const p of probe(dims.length)) {
          const va = a.value(embed(p, dims, origin))!, vb = b.value(p)!;
          expect(vb.length).toBe(dims.length);
          dims.forEach((d, i) => expect(vb[i], `${id}[${d}] at ${p}`).toBeCloseTo(va[d]!, 8));
        }
      }
      // point sets are projected; other manifolds untouched
      for (const [id, ps] of Object.entries(spec.pointSets ?? {})) {
        const src = nd.pointSets![id]!;
        if (src.domain !== manifold) { expect(ps).toEqual(src); continue; }
        expect(ps.points).toEqual(src.points.map((p) => dims.map((d) => p[d]!)));
      }
    });
  }

  it("the slice of the symbolic-nd bundle is itself a bundle every other tool accepts (parses, stats finite, isolines)", () => {
    const { spec } = sliceSpec(nd, "gauss4", [0, 1]);
    const b = Bundle.parse(spec);
    const p = b.scalarField("gauss4/p").data;
    const g = new DenseGrid([33, 33], p.box); // odd: a sample at the origin
    const vals = p.sampleOn(g);
    expect(vals.every(Number.isFinite)).toBe(true);
    expect(Math.max(...vals)).toBeCloseTo(1, 3); // the peak lies on the slice
    expect(b.scalarField("gauss4/gradnorm").data.value([0, 0])).toBeCloseTo(0, 9);
  });

  it("dims are sorted and deduplicated; bad dims / counts are errors", () => {
    const a = sliceSpec(nd, "gauss4", [2, 0, 2]).spec, b = sliceSpec(nd, "gauss4", [0, 2]).spec;
    expect(a).toEqual(b);
    expect(() => sliceSpec(nd, "gauss4", [0, 4])).toThrow(/out of range/);
    expect(() => sliceSpec(nd, "gauss4", [0, 1, 2, 3])).toThrow(/between 1 and 3/);
    expect(() => sliceSpec(nd, "nope", [0, 1])).toThrow(/unknown manifold/);
  });

  it("dense data is sliced at the grid sample nearest the origin; pullbacks move with it; the flow survives when its field does", () => {
    // a 4D 3×3×3×3 grid whose value is 100 i0 + 10 i1 + i2 + 0.1 i3 (grid indices), box [0,1]^4, origin (0, 0.5, 1, 0.3)
    const data: number[] = [];
    for (let i0 = 0; i0 < 3; i0++) for (let i1 = 0; i1 < 3; i1++) for (let i2 = 0; i2 < 3; i2++) for (let i3 = 0; i3 < 3; i3++) data.push(100 * i0 + 10 * i1 + i2 + 0.1 * i3);
    const spec: BundleSpec = {
      tensatory: "0.1",
      manifolds: { m: { numDims: 4, origin: [0, 0.5, 1, 0.3], flow: "v" } },
      fields: {
        g: { kind: "scalar", domain: "m", data: { type: "dense", samples: { type: "inline", shape: [3, 3, 3, 3], data }, box: [[0, 1], [0, 1], [0, 1], [0, 1]] } },
        t: { kind: "scalar", domain: "m", data: { type: "translate", arg: "g", vec: [0.5, 0, 0, 0] } },
        v: { kind: "vector", domain: "m", data: { type: "symbolicv", box: [[0, 1], [0, 1], [0, 1], [0, 1]], expr: { op: "coordv" } } },
      },
    };
    const { spec: s, dropped } = sliceSpec(spec, "m", [0, 3]);
    expect(dropped).toEqual({});
    const b = Bundle.parse(s);
    expect(b.buildAll().size).toBe(0);
    const g = b.scalarField("g").data;
    expect(g.samplePoints!.size).toEqual([3, 3]);
    // fixed dims: i1 = round(0.5 * 2) = 1, i2 = 2 -> 10 + 2 = 12 baked in
    expect(g.value([0, 0])).toBeCloseTo(12, 9);
    expect(g.value([1, 1])).toBeCloseTo(200 + 12 + 0.2, 9);
    const t = b.scalarField("t").data;
    expect(t.box.intervals).toEqual([[0.5, 1.5], [0, 1]]);
    expect(t.value([0.5, 0])).toBeCloseTo(12, 9);
    expect(b.vectorField("v").data.value([0.25, 0.75])).toEqual([0.25, 0.75]);
    expect(b.manifolds.get("m")!.flow).toBe("v");
    // an origin outside the sampled box cannot be sliced
    expect(sliceSpec({ ...spec, manifolds: { m: { numDims: 4, origin: [0, 2, 0, 0] } } }, "m", [0, 3]).dropped.g).toMatch(/outside the sampled box/);
  });

  it("a derivative along a fixed dimension of a sampled argument cannot be sliced and drops the field with a reason", () => {
    const spec: BundleSpec = {
      tensatory: "0.1",
      manifolds: { m: { numDims: 4 } },
      fields: {
        g: { kind: "scalar", domain: "m", data: { type: "dense", samples: { type: "constant", shape: [2, 2, 2, 2], value: 1 }, box: [[0, 1], [0, 1], [0, 1], [0, 1]] } },
        gn: { kind: "scalar", domain: "m", data: { type: "pointwise", expr: { op: "norm", vec: { op: "grad", val: "g" } }, scalars: { g: "g" } } },
        along: { kind: "scalar", domain: "m", data: { type: "pointwise", expr: { op: "comp", vec: { op: "grad", val: "g" }, index: 1 }, scalars: { g: "g" } } },
      },
    };
    const { spec: s, dropped } = sliceSpec(spec, "m", [0, 1]);
    expect(Object.keys(dropped)).toEqual(["gn"]);
    expect(dropped.gn).toMatch(/fixed dimension/);
    expect(Bundle.parse(s).buildAll().size).toBe(0); // `along` (∂g/∂x₁, along the slice) stays a pointwise over the sliced g
    expect(s.fields.along!.data).toMatchObject({ type: "pointwise", scalars: { g: "g" } });
  });

  it("net fields: the iris 3D space sliced to (t₀, t₂) equals the 3D loss at t₁ = 0", () => {
    const irisSpec = JSON.parse(readFileSync(join(dir, "iris.json"), "utf8")) as BundleSpec;
    const full = Bundle.parse(irisSpec);
    const { spec, dropped } = sliceSpec(irisSpec, "rand3", [0, 2]);
    expect(dropped).toEqual({});
    const b = Bundle.parse(spec);
    const errors = b.buildAll();
    expect([...errors.entries()].map(([id, e]) => `${id}: ${e.message}`)).toEqual([]);
    const a = full.scalarField("loss3").data, s = b.scalarField("loss3").data;
    expect(s.dimCount).toBe(2);
    for (const p of [[0.2, -0.3], [-0.5, 0.4], [0, 0]]) expect(s.value(p)).toBeCloseTo(a.value([p[0]!, 0, p[1]!])!, 9);
    expect(s.derivative(1).value([0.2, -0.3])).toBeCloseTo(a.derivative(2).value([0.2, 0, -0.3])!, 7);
  });
});

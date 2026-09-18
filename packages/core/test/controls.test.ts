import { describe, expect, it } from "vitest";
import type { BundleSpec, NetSpec } from "@tensatory/schema";
import { Bundle, NdArray, adjustSpec, controlRows } from "../src";

const rnd = (shape: number[], seed: string, widget?: object): object => ({ type: "random", shape, dist: { type: "gaussian", seed }, ...(widget ? { widget } : {}) });
const inline = (data: number[]) => ({ type: "inline", shape: [data.length], data }) as const;

// sum(a) + sum(c) around a bound origin, displaced along two random directions
const two: NetSpec = { type: "def", inputs: { a: [2], c: [3] }, nodes: { s: { op: "reduce", fn: "sum", val: { op: "concat", vals: ["a", "c"], axis: 0 } } }, outputs: { s: [] } };
const spec = {
  tensatory: "0.1",
  manifolds: { m: { numDims: 3 } },
  nets: {
    two,
    star: { type: "bind", net: "two", bind: { a: inline([3, 4]), c: inline([0, 0, 0]) } },
    rand: {
      type: "displace", net: "star",
      directions: [
        { name: "d₀", widget: { id: "d0" }, norm: "origin", arrays: { a: rnd([2], "d0/a"), c: rnd([3], "d0/c") } },
        { name: "d₁", widget: {}, arrays: { a: rnd([2], "d1/a", { label: "d₁ a", scaleRange: [0.5, 2] }) } },
        { arrays: { c: rnd([3], "d2/c") } }, // no widget: no row
      ],
    },
  },
  fields: {
    s: { kind: "scalar", domain: "m", data: { type: "net", net: "rand", output: "s", box: [[-1, 1], [-1, 1], [-1, 1]] } },
    noise: { kind: "scalar", domain: "m", data: { type: "dense", samples: rnd([4, 4, 4], "noise", { label: "noise", scaleSteps: 10 }), box: [[0, 1], [0, 1], [0, 1]] } },
  },
} as unknown as BundleSpec;

describe("controls", () => {
  it("parses and lists rows in spec order", () => {
    const b = Bundle.parse(spec);
    expect([...b.buildAll().entries()]).toEqual([]);
    const rows = controlRows(b.spec);
    // spec order after parsing (the zod schema lists fields before nets)
    expect(rows.map((r) => [r.id, r.label, r.kind, r.hasScale, r.members.length])).toEqual([
      ["fields.noise.data.samples", "noise", "array", true, 1],
      ["d0", "d₀", "direction", true, 2],
      ["nets.rand.directions.1", "d₁", "direction", true, 1],
      ["nets.rand.directions.1.arrays.a", "d₁ a", "array", true, 1],
    ]);
    expect(rows[1]!.members).toEqual(["nets.rand.directions.0.arrays.a", "nets.rand.directions.0.arrays.c"]);
    expect(rows[3]!.scaleRange).toEqual([0.5, 2]);
    expect(rows[0]!.scaleSteps).toBe(10);
    expect(rows[1]!.scaleRange).toEqual([0.01, 100]);
  });

  it("adjustments re-salt members and multiply scales; unadjusted rows are untouched", () => {
    const b = Bundle.parse(spec);
    const at = (s: BundleSpec, t: number[]) => new Bundle(s).net("rand").evaluate({ t: new NdArray([t.length], Float64Array.from(t)) }).s!.data[0]!;
    const base = at(b.spec, [1, 0, 0]);
    expect(adjustSpec(b.spec, {})).toEqual(b.spec);
    expect(adjustSpec(b.spec, { d0: {} })).toEqual(b.spec);
    // reseed d0: a different direction of the same length (norm "origin" = 5)
    const reseeded = adjustSpec(b.spec, { d0: { seed: 7 } });
    expect(at(reseeded, [1, 0, 0])).not.toBeCloseTo(base, 6);
    expect(at(reseeded, [0, 1, 0])).toBeCloseTo(at(b.spec, [0, 1, 0]), 12); // d₁ untouched
    expect(adjustSpec(b.spec, { d0: { seed: 7 } })).toEqual(reseeded); // deterministic
    // scale d0 by 2 doubles the displacement
    const scaled = adjustSpec(b.spec, { d0: { scale: 2 } });
    expect(at(scaled, [1, 0, 0]) - 7).toBeCloseTo(2 * (base - 7), 10);
    // an array row inside a direction row: both salts apply, the array row's scale multiplies dist.scale
    const both = adjustSpec(b.spec, { "nets.rand.directions.1": { seed: 1 }, "nets.rand.directions.1.arrays.a": { seed: 2, scale: 3 } });
    const dir1 = (s: BundleSpec) => (s.nets!.rand as Extract<NetSpec, { type: "displace" }>).directions[1]!.arrays.a as { dist: { seed: unknown; scale?: number } };
    expect(dir1(both).dist.seed).not.toBe("d1/a");
    expect(dir1(both).dist.scale).toBe(3);
    expect(dir1(adjustSpec(b.spec, { "nets.rand.directions.1": { seed: 1 } })).dist.seed).not.toBe(dir1(both).dist.seed);
    // unwidgeted direction 2 cannot be addressed
    expect(adjustSpec(b.spec, { "nets.rand.directions.2": { seed: 1, scale: 2 } })).toEqual(b.spec);
    // the dense field's array row
    const noisy = adjustSpec(b.spec, { "fields.noise.data.samples": { seed: 3, scale: 0.5 } });
    const sample = (s: BundleSpec) => new Bundle(s).scalarField("noise").data.sampleOn(new Bundle(s).scalarField("noise").data.samplePoints!);
    expect(Array.from(sample(noisy))).not.toEqual(Array.from(sample(b.spec)));
    expect(new Bundle(noisy).spec.fields.noise!.data).toMatchObject({ samples: { dist: { scale: 0.5 } } });
  });
});

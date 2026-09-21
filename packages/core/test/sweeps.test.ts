// Sweeps (notes/sweeps.md §2): the root schema, `common` merging, faceting over the records, the nearest member
// on a key change, structural signatures, and loading members inline / by path (with the member's own sidecars).

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import type { BundleSpec, CommonSpec, SweepSpec } from "@tensatory/schema";
import { Bundle, SINGLE_MEMBER, SpecError, Sweep, facets, mapSource, mergeCommon, nearestMember, rootKind, shortHash, signatureOf } from "../src";
import { dirSource, jsonBytes, npyBytes } from "./helpers";

const DEMO_DIR = join(__dirname, "../../../apps/viewer/public/bundles/sweep-demo");
const demoJson = () => JSON.parse(readFileSync(join(DEMO_DIR, "sweep.json"), "utf8")) as unknown;
const demo = () => Sweep.parse(demoJson(), dirSource(DEMO_DIR));

const plane = (expr: unknown): BundleSpec => ({
  tensatory: "0.1",
  fields: { f: { kind: "scalar", data: { type: "symbolic", box: [[-1, 1], [-1, 1]], expr: expr as never } } },
});

/** the dependent-structure example of the note: mlp members have `num_layers`, the convnet does not */
const mlpSweep: SweepSpec = {
  tensatory: "0.2",
  keys: { arch: { kind: "nominal", values: ["mlp", "convnet", "resnet"] }, lr: { kind: "ordinal", codomain: "log" } },
  members: {
    "mlp-3-s0": { record: { arch: "mlp", num_layers: 3, lr: 1e-3, seed: 0 }, bundle: plane({ op: "coord", index: 0 }) },
    "mlp-3-s1": { record: { arch: "mlp", num_layers: 3, lr: 1e-3, seed: 1 }, bundle: plane({ op: "coord", index: 0 }) },
    "mlp-4-s0": { record: { arch: "mlp", num_layers: 4, lr: 1e-2, seed: 0 }, bundle: plane({ op: "coord", index: 0 }) },
    "conv-s0": { record: { arch: "convnet", lr: 3e-4, seed: 0 }, bundle: plane({ op: "coord", index: 1 }) },
  },
};

describe("sweep schema", () => {
  it("tells roots apart by version", () => {
    expect(rootKind(demoJson())).toBe("sweep");
    expect(rootKind({ tensatory: "0.1", fields: {} })).toBe("bundle");
    expect(rootKind({ tensatory: "0.3" })).toBeUndefined();
    expect(rootKind(null)).toBeUndefined();
  });

  it("validates the root, the keys and every inline member with the bundle schema", () => {
    expect(() => Sweep.validate({ tensatory: "0.1", fields: {} })).toThrow(SpecError);
    expect(() => Sweep.validate({ tensatory: "0.2", members: {} })).not.toThrow();
    expect(() => Sweep.validate({ ...mlpSweep, keys: { arch: { kind: "weird" } } })).toThrow(/kind|invalid/i);
    expect(() => Sweep.validate({ tensatory: "0.2", members: { a: { record: {}, bundle: { tensatory: "0.1" } } } })).toThrow(/fields/);
    expect(() => Sweep.validate({ tensatory: "0.2", members: { a: { record: { x: [1] }, bundle: "a.json" } } })).toThrow(SpecError);
  });

  it("refuses external arrays in `common` (handles belong to the members)", () => {
    const spec = {
      tensatory: "0.2",
      common: { fields: { g: { kind: "scalar", data: { type: "dense", samples: { type: "handle", path: "g.npy", shape: [2, 2] } } } } },
      members: {},
    };
    expect(() => Sweep.validate(spec)).toThrow(/common.*g\.npy/);
  });
});

describe("mergeCommon", () => {
  it("merges per record and per id, the member winning; scalars fall back to common", () => {
    const common: CommonSpec = {
      name: "common name", summary: "common summary",
      manifolds: { p: { numDims: 2 }, q: { numDims: 3 } },
      fields: { shared: { kind: "scalar" as const, name: "from common", data: { type: "symbolic" as const, box: [[0, 1], [0, 1]], expr: 1 } } },
      nets: {},
    };
    const member: BundleSpec = {
      tensatory: "0.1", name: "member",
      manifolds: { p: { numDims: 2, dimNames: ["a", "b"] } },
      fields: { own: { kind: "scalar", data: { type: "symbolic", box: [[0, 1], [0, 1]], expr: 2 } }, shared: { kind: "scalar", name: "from member", data: { type: "symbolic", box: [[0, 1], [0, 1]], expr: 3 } } },
    };
    const m = mergeCommon(common, member);
    expect(m.name).toBe("member");
    expect(m.summary).toBe("common summary");
    expect(m.manifolds).toEqual({ p: { numDims: 2, dimNames: ["a", "b"] }, q: { numDims: 3 } });
    expect(Object.keys(m.fields)).toEqual(["own", "shared"]); // the member's ids first, then what common adds
    expect(m.fields.shared!.name).toBe("from member");
    expect(m.nets).toEqual({});
    expect(m.curves).toBeUndefined();
    expect(mergeCommon(undefined, member)).toBe(member);
    expect(member.fields.shared!.name).toBe("from member"); // inputs untouched
  });
});

describe("facets", () => {
  it("lists keys in declaration then discovery order, hides keys the current member lacks", () => {
    expect(facets(mlpSweep).map((f) => f.key)).toEqual(["arch", "lr", "num_layers", "seed"]);
    expect(facets(mlpSweep, "conv-s0").map((f) => f.key)).toEqual(["arch", "lr", "seed"]);
  });

  it("orders values by `keys.values`, then discovery (nominal) or sorted (ordinal); listed-but-absent values have no members", () => {
    const [arch, lr, layers, seed] = facets(mlpSweep);
    expect(arch!.kind).toBe("nominal");
    expect(arch!.values.map((v) => [v.value, v.members])).toEqual([["mlp", ["mlp-3-s0", "mlp-3-s1", "mlp-4-s0"]], ["convnet", ["conv-s0"]], ["resnet", []]]);
    expect(lr!.kind).toBe("ordinal"); // declared
    expect(lr!.values.map((v) => v.value)).toEqual([3e-4, 1e-3, 1e-2]);
    expect(layers!.kind).toBe("ordinal"); // inferred: all numbers
    expect(layers!.values.map((v) => v.value)).toEqual([3, 4]);
    expect(seed!.varying).toBe(true);
    expect(facets({ ...mlpSweep, keys: undefined }).map((f) => f.key)).toEqual(["arch", "num_layers", "lr", "seed"]);
  });

  it("marks a value direct when a member with it agrees with the current member on every other shared key", () => {
    const from = (id: string, key: string) => Object.fromEntries(facets(mlpSweep, id).find((f) => f.key === key)!.values.map((v) => [String(v.value), v.direct]));
    // from mlp-3-s0: seed 1 exists with the same arch / layers / lr → direct; arch convnet differs in lr → not direct
    expect(from("mlp-3-s0", "seed")).toEqual({ "0": true, "1": true });
    expect(from("mlp-3-s0", "arch")).toEqual({ mlp: true, convnet: false, resnet: false });
    expect(from("mlp-3-s0", "num_layers")).toEqual({ "3": true, "4": false }); // 4 layers only with lr 1e-2
    // from the convnet: num_layers is not a shared key, so the mlp members compare on arch (differs) only when arch is the facet — direct
    expect(from("conv-s0", "arch")).toEqual({ mlp: false, convnet: true, resnet: false }); // lr differs everywhere
    expect(from("conv-s0", "seed")).toEqual({ "0": true, "1": false });
    // without a current member everything present is direct
    expect(facets(mlpSweep).every((f) => f.values.every((v) => v.direct === v.members.length > 0))).toBe(true);
  });

  it("on the demo sweep: no saddle at k = 2, 3D only for the k = 1 bowl", () => {
    const s = demo();
    expect(s.keys).toEqual(["shape", "k", "dims"]);
    expect(s.hasFacets).toBe(true);
    const at = (id: string, key: string) => Object.fromEntries(s.facets(id).find((f) => f.key === key)!.values.map((v) => [String(v.value), v.direct]));
    expect(at("bowl-k1", "shape")).toEqual({ bowl: true, saddle: true });
    expect(at("bowl-k2", "shape")).toEqual({ bowl: true, saddle: false });
    expect(at("bowl-k1", "dims")).toEqual({ "2": true, "3": true });
    expect(at("bowl-k4", "dims")).toEqual({ "2": true, "3": false });
    expect(at("bowl-k1-3d", "k")).toEqual({ "1": true, "2": false, "4": false });
    expect(s.facets("bowl-k1").find((f) => f.key === "k")!.name).toBe("sharpness");
  });

  it("attribute keys (per-member measurements) never count when comparing records and are never facets", () => {
    const spec: SweepSpec = {
      tensatory: "0.2",
      keys: { n_params: { attribute: true }, test_acc: { attribute: true, codomain: "fraction" } },
      members: {
        "mlp-pca": { record: { model: "mlp", dirs: "pca", n_params: 269322, test_acc: 0.976 }, bundle: plane(1) },
        "mlp-rnd": { record: { model: "mlp", dirs: "random", n_params: 269322, test_acc: 0.976 }, bundle: plane(1) },
        "conv-pca": { record: { model: "convnet", dirs: "pca", n_params: 56394, test_acc: 0.984 }, bundle: plane(1) },
      },
    };
    const model = facets(spec, "conv-pca").find((f) => f.key === "model")!;
    expect(model.values.map((v) => [v.value, v.direct])).toEqual([["mlp", true], ["convnet", true]]); // n_params / test_acc differ but do not count
    expect(facets(spec, "conv-pca").filter((f) => f.attribute).map((f) => [f.key, f.varying])).toEqual([["n_params", true], ["test_acc", true]]);
    expect(nearestMember(spec, "conv-pca", "model", "mlp")).toBe("mlp-pca");
    const s = new Sweep(spec, { bytes: async () => null });
    expect(s.hasFacets).toBe(true);
    // only attributes varying: no facets to show
    expect(new Sweep({ ...spec, members: { a: spec.members["mlp-pca"]!, b: { ...spec.members["mlp-pca"]!, record: { ...spec.members["mlp-pca"]!.record, test_acc: 0.9 } } } }, { bytes: async () => null }).hasFacets).toBe(false);
  });

  it("nearestMember: the fewest differing shared keys, then the most shared keys, then sweep order", () => {
    expect(nearestMember(mlpSweep, "mlp-3-s0", "seed", 1)).toBe("mlp-3-s1");
    expect(nearestMember(mlpSweep, "mlp-3-s1", "num_layers", 4)).toBe("mlp-4-s0");
    expect(nearestMember(mlpSweep, "conv-s0", "arch", "mlp")).toBe("mlp-3-s0"); // seed 0 agrees for s0 members; first wins
    expect(nearestMember(mlpSweep, "mlp-4-s0", "arch", "convnet")).toBe("conv-s0");
    expect(nearestMember(mlpSweep, "mlp-3-s0", "arch", "resnet")).toBeUndefined();
    expect(nearestMember(mlpSweep, undefined, "seed", 0)).toBe("mlp-3-s0");
    const s = demo();
    expect(s.nearest("bowl-k2", "shape", "saddle")).toBe("saddle-k1");
    expect(s.nearest("saddle-k4", "dims", 3)).toBe("bowl-k1-3d");
  });
});

describe("members", () => {
  it("loads inline and by-path members with `common` merged in, and builds them", async () => {
    const s = demo();
    expect(s.memberIds).toHaveLength(6);
    expect(s.info?.summary).toMatch(/toy sweep/);
    const k2 = await s.member("bowl-k2");
    expect(k2.buildAll().size).toBe(0);
    expect(k2.name).toBe("bowl k = 2");
    expect(Object.keys(k2.spec.fields)).toEqual(["f", "gradNorm"]);
    expect(k2.scalarField("f").data.value([1, 0])).toBeCloseTo(2, 12);
    expect(k2.scalarField("gradNorm").data.value([1, 0])).toBeCloseTo(4, 12); // |∇ 2(x² + y²)| at (1, 0)
    expect(k2.scalarField("gradNorm").domain.id).toBe("plane");
    const s4 = await s.member("saddle-k4");
    expect(s4.buildAll().size).toBe(0);
    expect(s4.scalarField("f").data.value([0, 1])).toBeCloseTo(-4, 12);
    const v = await s.member("bowl-k1-3d");
    expect(v.buildAll().size).toBe(0);
    expect(v.scalarField("gradNorm").domain.id).toBe("volume");
    expect(await s.member("bowl-k2")).toBe(k2); // cached
  });

  it("a by-path member's sidecars are relative to ITS document", async () => {
    const vals = new Float32Array([0, 1, 2, 3, 4, 5]);
    const src = mapSource({
      "runs/a/bundle.json": jsonBytes({
        tensatory: "0.1",
        fields: { g: { kind: "scalar", data: { type: "dense", box: [[0, 1], [0, 1]], samples: { type: "handle", path: "vals.npy", shape: [2, 3] } } } },
      }),
      "runs/a/vals.npy": npyBytes(vals, [2, 3]),
    });
    const s = Sweep.parse({ tensatory: "0.2", common: { fields: { twice: { kind: "scalar", data: { type: "pointwise", expr: { op: "mul", vals: [2, "g"] }, scalars: { g: "g" } } } } }, members: { a: { record: { run: "a" }, bundle: "runs/a/bundle.json" } } }, src);
    const b = await s.member("a");
    expect(b.buildAll().size).toBe(0);
    const g = b.scalarField("g").data;
    expect([...g.sampleOn(g.samplePoints!)]).toEqual([...vals]);
    expect(b.scalarField("twice").data.value([1, 1])).toBeCloseTo(10, 6);
  });

  it("a missing or malformed member document is a SpecError naming it, and can be retried", async () => {
    const files: Record<string, Uint8Array> = {};
    const s = Sweep.parse({ tensatory: "0.2", members: { a: { record: {}, bundle: "a/bundle.json" }, b: { record: {}, bundle: "b.json" } } }, mapSource(files));
    await expect(s.member("a")).rejects.toThrow(/a\/bundle\.json.*not found/);
    files["a/bundle.json"] = jsonBytes(plane(1));
    expect((await s.member("a")).fieldIds).toEqual(["f"]);
    files["b.json"] = new TextEncoder().encode("{ not json");
    await expect(s.member("b")).rejects.toThrow(/b\.json.*not JSON/);
    files["b.json"] = jsonBytes({ tensatory: "0.1" });
    await expect(s.member("b")).rejects.toThrow(/b\.json.*fields/);
    await expect(s.member("zzz")).rejects.toThrow(/unknown member/);
  });

  it("a lone bundle is a one-member sweep without facets", async () => {
    const b = Bundle.parse(plane(1));
    const s = Sweep.single(b);
    expect(s.memberIds).toEqual([SINGLE_MEMBER]);
    expect(s.hasFacets).toBe(false);
    expect(s.facets(SINGLE_MEMBER)).toEqual([]);
    expect(await s.member(SINGLE_MEMBER)).toBe(b);
  });
});

describe("structural signature", () => {
  it("is shared by members with the same spaces and fields, regardless of values, curves or point sets", async () => {
    const s = demo();
    const sig = async (id: string) => signatureOf((await s.member(id)).spec);
    expect(await sig("bowl-k1")).toBe(await sig("saddle-k4"));
    expect(await sig("bowl-k1")).not.toBe(await sig("bowl-k1-3d"));
    expect(await sig("bowl-k1")).toBe("scalar f:2;scalar gradNorm:2;space plane:2;space volume:3");
    const withCurve: BundleSpec = { ...plane(1), curves: { c: { data: { type: "symbolic", interval: [0, 1], expr: { op: "compv", coeffs: [{ op: "coord", index: 0 }, 0] } } } }, pointSets: { p: { points: [[0, 0]] } } };
    expect(signatureOf(withCurve)).toBe(signatureOf(plane(1)));
    expect(signatureOf(plane(1))).toBe("scalar f:2;space default:2"); // the implicit manifold, its dimension inferred from the box
    expect(signatureOf({ tensatory: "0.1", fields: { g: { kind: "vector", data: { type: "symbolicv", expr: { op: "coordv" } } } } })).toBe("space default:?;vector g:?");
  });

  it("shortHash is stable and short", () => {
    expect(shortHash("scalar f:2;space plane:2")).toBe(shortHash("scalar f:2;space plane:2"));
    expect(shortHash("a")).not.toBe(shortHash("b"));
    expect(shortHash("scalar f:2;space plane:2").length).toBeLessThanOrEqual(7);
  });
});

// External arrays: the loaders (.bin / .npy / .npz / zarr v2 / v3) against fixtures written by numpy and
// zarr-python (fixtures/handles/make.py), `part` with kept axes, collectHandles over a bundle, and a bundle whose
// fields are backed by every format (Bundle.load through a readFile ByteSource).

import { join } from "node:path";
import { readFile } from "node:fs/promises";
import { describe, expect, it } from "vitest";
import { dirSource } from "./helpers";
import type { BundleSpec } from "@tensatory/schema";
import type { ArrayData, RawArray } from "../src";
import {
  Bundle, DenseGrid, DenseScalarFieldData, SpecError, NotSupportedError,
  applyPart, collectHandles, formatOf, loadArray, loadArrays, mapSource, partShape, rebaseSource, sizedShape, zarrSplit,
} from "../src";

const DIR = join(__dirname, "fixtures/handles");

const src = dirSource(DIR);
const load = (path: string, hint?: { shape?: number[]; dtype?: RawArray["dtype"] }) => loadArray(src, path, hint, ["t"]);

/** row-major cell index of `shape` as a flat typed array of the STORED type — what `ramp` in make.py wrote */
type Ctor = { from(src: Iterable<number>): ArrayData };
const ramp = (shape: number[], f: (i: number) => number = (i) => i, T: Ctor = Float32Array) => T.from(Array.from({ length: shape.reduce((a, b) => a * b, 1) }, (_, i) => f(i)));
const VOL = [4, 5, 6, 2];

describe("format by path", () => {
  it("tells the reader from the path", () => {
    expect(formatOf("a/b.bin")).toBe("bin");
    expect(formatOf("w.NPY")).toBe("npy");
    expect(formatOf("data.npz/foo/bar")).toBe("npz");
    expect(formatOf("vol.zarr")).toBe("zarr");
    expect(formatOf("store.zarr/g/arr")).toBe("zarr");
    expect(formatOf("data.npz")).toBeUndefined(); // an archive is not an array
    expect(formatOf("readme.txt")).toBeUndefined();
    expect(zarrSplit("a/store.zarr/g/arr")).toEqual({ root: "a/store.zarr", node: "g/arr" });
    expect(zarrSplit("store.zarr")).toEqual({ root: "store.zarr", node: "" });
  });
  it("refuses unknown formats and missing files", async () => {
    await expect(load("readme.txt")).rejects.toThrow(/cannot tell the format/);
    await expect(load("missing.npy")).rejects.toThrow(/not found/);
    await expect(load("stored.npz/nope")).rejects.toThrow(/no member "nope".*members: a, nested\/b/);
  });
});

describe(".npy", () => {
  it("reads every dtype, both endiannesses, Fortran order, scalars and v2 headers", async () => {
    expect(await load("f4.npy")).toEqual({ shape: [2, 3], dtype: "float32", data: ramp([2, 3]) });
    expect(await load("f8.npy")).toEqual({ shape: [2, 3], dtype: "float64", data: ramp([2, 3], (i) => i / 8, Float64Array) });
    expect(await load("i4.npy")).toEqual({ shape: [5], dtype: "int32", data: ramp([5], (i) => i - 2, Int32Array) });
    expect(await load("i8.npy")).toEqual({ shape: [3], dtype: "int64", data: ramp([3], (i) => (i - 1) * 2 ** 40, Float64Array) }); // no 64-bit integer array: numbers
    expect(await load("u1.npy")).toEqual({ shape: [2, 2], dtype: "uint8", data: ramp([2, 2], undefined, Uint8Array) });
    expect(await load("u2.npy")).toEqual({ shape: [2, 2], dtype: "uint16", data: ramp([2, 2], (i) => i * 300, Uint16Array) });
    expect(await load("bool.npy")).toEqual({ shape: [2, 2], dtype: "bool", data: Uint8Array.of(1, 0, 0, 1) });
    expect(await load("bigendian.npy")).toEqual({ shape: [2, 3], dtype: "float32", data: ramp([2, 3]) });
    expect(await load("fortran.npy")).toEqual({ shape: [2, 3, 4], dtype: "float64", data: ramp([2, 3, 4], undefined, Float64Array) });
    expect(await load("scalar.npy")).toEqual({ shape: [], dtype: "float32", data: Float32Array.of(42.5) });
    expect(await load("v2.npy")).toEqual({ shape: [3], dtype: "float32", data: ramp([3]) });
  });
  it("validates the bundle's hint", async () => {
    await expect(load("f4.npy", { shape: [3, 2] })).rejects.toThrow(/has shape \[2,3\] but the bundle declares \[3,2\]/);
    await expect(load("f4.npy", { dtype: "float64" })).rejects.toThrow(/is float32 but the bundle declares float64/);
    expect((await load("f4.npy", { shape: [2, 3], dtype: "float32" })).shape).toEqual([2, 3]);
  });
});

describe(".bin", () => {
  it("needs a shape, defaults to float32, checks the byte count", async () => {
    await expect(load("vol.bin")).rejects.toThrow(/needs a declared `shape`/);
    expect(await load("vol.bin", { shape: VOL })).toEqual({ shape: VOL, dtype: "float32", data: ramp(VOL) });
    expect(await load("vec.bin", { shape: [3], dtype: "float64" })).toEqual({ shape: [3], dtype: "float64", data: Float64Array.of(1.5, -2.5, 3.5) });
    await expect(load("vec.bin", { shape: [3] })).rejects.toThrow(/holds 24 bytes but shape \[3\] of float32 needs 12/);
  });
});

describe(".npz", () => {
  it("reads stored and deflated members, nested names, with or without .npy", async () => {
    expect(await load("stored.npz/a")).toEqual({ shape: [2, 2], dtype: "float32", data: ramp([2, 2]) });
    expect(await load("stored.npz/nested/b")).toEqual({ shape: [3], dtype: "int32", data: ramp([3], undefined, Int32Array) });
    expect(await load("stored.npz/nested/b.npy")).toEqual({ shape: [3], dtype: "int32", data: ramp([3], undefined, Int32Array) });
    expect(await load("deflated.npz/a")).toEqual({ shape: [10, 10], dtype: "float64", data: ramp([10, 10], undefined, Float64Array) });
    expect(await load("deflated.npz/w")).toEqual({ shape: [3, 2], dtype: "float32", data: ramp([3, 2]) });
  });
});

describe("zarr", () => {
  it("v2: uncompressed edge chunks, zlib, gzip, all-fill, Fortran chunks, scalars, '/' separator", async () => {
    expect(await load("v2.zarr/vol")).toEqual({ shape: VOL, dtype: "float32", data: ramp(VOL) });
    expect(await load("v2.zarr/zlib")).toEqual({ shape: [7, 5], dtype: "float64", data: ramp([7, 5], undefined, Float64Array) });
    expect(await load("v2.zarr/gzip")).toEqual({ shape: [6], dtype: "int32", data: ramp([6], undefined, Int32Array) });
    expect(await load("v2.zarr/fill")).toEqual({ shape: [4, 4], dtype: "float32", data: new Float32Array(16).fill(7.5) });
    expect(await load("v2.zarr/fortran")).toEqual({ shape: [3, 4], dtype: "float64", data: ramp([3, 4], undefined, Float64Array) });
    expect(await load("v2.zarr/scalar")).toEqual({ shape: [], dtype: "float64", data: Float64Array.of(3.25) });
    expect(await load("v2sep.zarr")).toEqual({ shape: [5, 3], dtype: "float32", data: ramp([5, 3]) });
  });
  it("v3: bytes + gzip with 'c/' keys and edge chunks, big-endian int64, NaN fill", async () => {
    expect(await load("v3.zarr/vol")).toEqual({ shape: VOL, dtype: "float32", data: ramp(VOL) });
    expect(await load("v3.zarr/raw")).toEqual({ shape: [5], dtype: "int64", data: ramp([5], (i) => i - 2, Float64Array) });
    const fill = await load("v3.zarr/fill");
    expect(fill.shape).toEqual([3]);
    expect([...fill.data].every(Number.isNaN)).toBe(true);
  });
  it("refuses what it cannot decode, by name", async () => {
    await expect(load("zstd2.zarr")).rejects.toThrow(NotSupportedError);
    await expect(load("zstd2.zarr")).rejects.toThrow(/compressor "zstd" is not supported/);
    await expect(load("zstd.zarr")).rejects.toThrow(/codec "zstd" is not supported/);
    await expect(load("v2.zarr")).rejects.toThrow(/is not a zarr array/); // the group
    await expect(load("v2.zarr/nope")).rejects.toThrow(/is not a zarr array/);
  });
});

describe("part", () => {
  const raw: RawArray = { shape: [2, 3, 4], dtype: "float64", data: ramp([2, 3, 4]) };
  it("keeps null axes and drops fixed ones", () => {
    expect(partShape([4, 5, 6, 2], [null, null, null, 0])).toEqual([4, 5, 6]);
    expect(partShape([4, 5, 6, 2], [1])).toEqual([5, 6, 2]);
    expect(partShape([4, 5, 6, 2], [1, null, 2])).toEqual([5, 2]);
    expect(partShape([4, 5], undefined)).toEqual([4, 5]);
    expect(() => partShape([4], [0, 0])).toThrow(SpecError);
    expect(sizedShape({ type: "handle", path: "x.npy", shape: [4, 5, 6, 2], part: [null, null, null, 0] })).toEqual([4, 5, 6]);
    expect(sizedShape({ type: "inline", shape: [2, 2], data: [1, 2, 3, 4] })).toEqual([2, 2]);
  });
  it("gathers: leading (contiguous), trailing, mixed, negative, a single cell, identity", () => {
    expect(applyPart(raw, [1]).toNested()).toEqual([[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]);
    expect(applyPart(raw, [null, null, 3]).toNested()).toEqual([[3, 7, 11], [15, 19, 23]]);
    expect(applyPart(raw, [null, -1]).toNested()).toEqual([[8, 9, 10, 11], [20, 21, 22, 23]]);
    expect(applyPart(raw, [1, null, 0]).toNested()).toEqual([12, 16, 20]);
    expect(applyPart(raw, [1, 2, 3]).toNested()).toEqual(23);
    expect(applyPart(raw, [null, null]).shape).toEqual([2, 3, 4]);
    expect(applyPart(raw, undefined).data).toBe(raw.data); // no copy when nothing is fixed
    expect(() => applyPart(raw, [2])).toThrow(/index 2 out of range/);
  });
  it("permutes the kept axes (numpy transpose semantics)", () => {
    expect(partShape([2, 3, 4], undefined, [2, 0, 1])).toEqual([4, 2, 3]);
    expect(partShape([2, 3, 4, 5], [null, null, null, 1], [2, 1, 0])).toEqual([4, 3, 2]);
    // full reversal: out[k][j][i] = raw[i][j][k] = (i*3 + j)*4 + k
    const t = applyPart(raw, undefined, [2, 1, 0]);
    expect(t.shape).toEqual([4, 3, 2]);
    expect(t.get(3, 1, 1)).toBe((1 * 3 + 1) * 4 + 3);
    expect(t.get(0, 2, 0)).toBe(2 * 4);
    // a channel then a swap of the two kept axes: out[j][i] = raw[i][j][3]
    const s = applyPart(raw, [null, null, 3], [1, 0]);
    expect(s.toNested()).toEqual([[3, 15], [7, 19], [11, 23]]);
    expect(applyPart(raw, undefined, [0, 1, 2]).data).toBe(raw.data); // the identity permutation is free
    expect(() => applyPart(raw, undefined, [0, 1])).toThrow(/not a permutation of the 3 kept axes/);
    expect(() => applyPart(raw, undefined, [0, 0, 1])).toThrow(/not a permutation/);
    expect(sizedShape({ type: "handle", path: "v.bin", shape: [40, 30, 20, 2], part: [null, null, null, 0], axes: [2, 1, 0] })).toEqual([20, 30, 40]);
  });
});

describe("collectHandles", () => {
  it("finds sized and bare handles in fields, pullbacks, stats and nets, merging hints", async () => {
    const spec = Bundle.validate(JSON.parse(await readFile(join(DIR, "bundle.json"), "utf8")));
    const h = collectHandles(spec);
    expect([...h.keys()].sort()).toEqual(["deflated.npz/a", "deflated.npz/w", "i4.npy", "missing.npy", "v2.zarr/vol", "v3.zarr/vol", "vec.bin", "vol.bin", "vol.npy"]);
    expect(h.get("vol.bin")).toEqual({ shape: [4, 5, 6, 2], dtype: "float32" });
    expect(h.get("deflated.npz/w")).toEqual({});
    expect(h.get("vec.bin")).toEqual({ shape: [3], dtype: "float64" });
  });
  it("rejects one path declared with two shapes", () => {
    const spec: BundleSpec = {
      tensatory: "0.1",
      fields: {
        a: { kind: "scalar", data: { type: "dense", samples: { type: "handle", path: "x.npy", shape: [2, 2] } } },
        b: { kind: "scalar", data: { type: "dense", samples: { type: "handle", path: "x.npy", shape: [4] } } },
      },
    };
    expect(() => collectHandles(spec)).toThrow(/declared with shape \[4\] here and \[2,2\] elsewhere/);
  });
  it("walks nested net specs and call expressions", () => {
    const spec: BundleSpec = {
      tensatory: "0.1",
      manifolds: { m: { numDims: 1 } },
      fields: {
        f: {
          kind: "scalar",
          data: {
            type: "net", output: "o",
            net: {
              type: "grad",
              net: { type: "bind", net: { type: "def", inputs: { x: [1], w: [1] }, nodes: { o: { op: "call", net: { type: "def", inputs: { a: [1] }, arrays: { k: { type: "handle", path: "k.npy", shape: [1] } }, nodes: { o: { op: "mul", vals: ["a", "k"] } }, outputs: { o: [1] } }, inputs: { a: { op: "mul", vals: ["x", "w"] } }, output: "o" } }, outputs: { o: [1] } }, bind: { w: "w.npy" } },
              outputs: { g: { of: "o", wrt: "x", seed: { type: "handle", path: "seed.npy" } }, h: { of: "o", wrt: "x", seed: "v" } },
            },
            arrays: { c: "c.npy" },
          },
        },
      },
    };
    expect([...collectHandles(spec).keys()].sort()).toEqual(["c.npy", "k.npy", "seed.npy", "w.npy"]);
  });
});

describe("Bundle.load", () => {
  const specP = readFile(join(DIR, "bundle.json"), "utf8").then((s) => JSON.parse(s) as unknown);

  it("backs fields with every format; the same volume through four stores agrees", async () => {
    const progress: number[] = [];
    const bundle = await Bundle.load(await specP, src, { onProgress: (p) => progress.push(p.done) });
    expect(progress).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9]);
    const errors = bundle.buildAll();
    expect([...errors.keys()]).toEqual(["broken"]);
    expect(errors.get("broken")!.message).toMatch(/fields\.broken\.data\.samples: external array "missing\.npy" failed to load: .*file "missing\.npy" not found/);

    const ch = (x: number, y: number, z: number, c: number) => ((x * 5 + y) * 6 + z) * 2 + c;
    const bin0 = bundle.scalarField("bin0").data as DenseScalarFieldData;
    expect(bin0.kind).toBe("sampled");
    expect(bin0.data).toBeInstanceOf(Float32Array); // the stored type is kept: no widening to f64
    expect(bin0.samplePoints).toEqual(new DenseGrid([4, 5, 6], bin0.box));
    expect(bin0.value([0, 0, 0])).toBe(ch(0, 0, 0, 0));
    expect(bin0.value([3, 4, 5])).toBe(ch(3, 4, 5, 0));
    expect(bin0.value([2, 1, 3])).toBe(ch(2, 1, 3, 0));
    expect(bundle.scalarField("npy1").data.value([2, 1, 3])).toBe(ch(2, 1, 3, 1));
    expect(bundle.scalarField("zarr2").data.value([2, 1, 3])).toBe(ch(2, 1, 3, 1));
    expect(bundle.scalarField("zarr3").data.value([2, 1, 3])).toBe(ch(2, 1, 3, 0));
    expect(bundle.scalarField("slab").data.value([1, 1])).toBe(99);
    // translate by vec.bin = (1.5, -2.5, 3.5): shifted(p) = bin0(p - vec)
    const shifted = bundle.scalarField("shifted").data;
    expect(shifted.box.a).toEqual([1.5, -2.5, 3.5]);
    expect(shifted.value([3.5, -1.5, 6.5])).toBe(ch(2, 1, 3, 0));
    // the net bound to deflated.npz/w (rows [0,1],[2,3],[4,5]): s(x) = x0 + 5 x1 + 9 x2
    const lin = bundle.scalarField("lin").data;
    expect(lin.kind).toBe("symbolic");
    expect(lin.value([1, 2, 3])).toBeCloseTo(1 + 10 + 27, 9);
    expect(bundle.net("lin_b").signature.inputs).toEqual({ x: [3] });
  });

  it("built without sidecars, every handle-backed field fails on its own and the rest work", async () => {
    const bundle = Bundle.parse(await specP);
    const errors = bundle.buildAll();
    expect([...errors.keys()].sort()).toEqual(["bin0", "broken", "lin", "nets.lin_b", "npy1", "shifted", "slab", "zarr2", "zarr3"]);
    expect(errors.get("bin0")!.message).toMatch(/external array "vol\.bin" was not loaded \(the bundle was built without its sidecar files\)/);
    expect(errors.get("lin")!.message).toMatch(/nets\.lin_b\.bind\.W: external array "deflated\.npz\/w" was not loaded/);
  });

  it("loadArrays + rebaseSource + mapSource", async () => {
    const files = { "sub/w.npy": new Uint8Array(await readFile(join(DIR, "f4.npy"))) };
    const spec: BundleSpec = { tensatory: "0.1", manifolds: { m: { numDims: 2 } }, fields: { f: { kind: "scalar", data: { type: "dense", samples: { type: "handle", path: "w.npy", shape: [2, 3] } } } } };
    const arrays = await loadArrays(spec, rebaseSource(mapSource(files), "sub"));
    expect(arrays.raw("w.npy", []).shape).toEqual([2, 3]);
    expect(() => arrays.raw("other.npy", ["p"])).toThrow(/external array "other\.npy" was not loaded/);
    const b = new Bundle(spec, arrays);
    expect(b.buildAll().size).toBe(0);
    expect(b.scalarField("f").data.value([1, 1])).toBe(5);
  });
});

// Fused glyphs: the appended arrow segments must be exactly core's (as a multiset — threads append in any order),
// in 2D (Seg) and 3D (Seg3), with the colour field at each glyph's point and the same normalizing maximum.

import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { Box, SymbolicVectorFieldData, arrowGlyphs, buildScalarFieldData, buildVectorFieldData, latticeIn, latticePoints, maxNorm } from "@tensatory/core";
import { GpuBackend, SEG3_FLOATS, SEG_FLOATS, allocSegments, allocSegments3, fusedGlyphs, readSegments, readSegments3, resetSegments } from "../src";

let gpu: GpuBackend | undefined;
beforeAll(async () => { gpu = await GpuBackend.create(); });
afterAll(async () => { gpu?.destroy(); await new Promise((r) => setTimeout(r, 50)); });

const x = { op: "coord", index: 0 } as const, y = { op: "coord", index: 1 } as const;
const bowl = buildScalarFieldData({ type: "symbolic", box: [[-2, 2], [-2, 2]], expr: { op: "add", vals: [{ op: "square", val: x }, { op: "square", val: y }] } }, 2);
const waves = buildScalarFieldData({ type: "symbolic", box: [[-2, 2], [-2, 2]], expr: { op: "mul", vals: [{ op: "sin", val: { op: "mul", vals: [3, x] } }, { op: "cos", val: { op: "mul", vals: [2, y] } }] } }, 2);

/** canonical multiset key of a segment's endpoints (rounded to f32-ish precision), direction-insensitive */
const segKey = (a: number[], b: number[]) => {
  const r = (v: number) => v.toFixed(4);
  const ka = a.map(r).join(","), kb = b.map(r).join(",");
  return ka < kb ? `${ka}|${kb}` : `${kb}|${ka}`;
};
function cpuKeys(lines: Float64Array[], D: number): Map<string, number> {
  const m = new Map<string, number>();
  for (const l of lines) for (let i = 0; i + 2 * D - 1 < l.length; i += D) {
    const k = segKey(Array.from(l.subarray(i, i + D)), Array.from(l.subarray(i + D, i + 2 * D)));
    m.set(k, (m.get(k) ?? 0) + 1);
  }
  return m;
}

describe("fused glyphs", () => {
  it("2D: the same arrows as core on a hex lattice, coloured at the glyph point, maximum read back", async () => {
    if (!gpu) return;
    const grad = new SymbolicVectorFieldData({ k: "grad", s: { k: "arg", name: "f" } }, 2, { scalars: { f: bowl }, vectors: {} });
    const lattice = latticeIn(new Box([-1.5, -1.2], [1.8, 1.7]), 0.21, bowl.box.a);
    expect(lattice.pointCount).toBeGreaterThan(100);
    const kernel = fusedGlyphs(gpu, grad, waves);
    expect(kernel.capacityFor(lattice)).toBe(3 * lattice.pointCount);
    const segs = allocSegments(gpu, kernel.capacityFor(lattice), false);
    await kernel.run(segs, lattice);
    const out = await readSegments(gpu, segs);
    const n = out.length / SEG_FLOATS;
    // core: the same lattice, the same field, the same normalization
    const pts = latticePoints(lattice);
    const vectors = new Float64Array(lattice.pointCount * 2);
    for (let i = 0; i < lattice.pointCount; i++) vectors.set(grad.value([pts[2 * i]!, pts[2 * i + 1]!])!, 2 * i);
    const cpu = arrowGlyphs(pts, vectors, 2, lattice.spacing);
    expect(n).toBe(cpu.lines.length / 2 * 3); // 3 segments per glyph (the origin, a zero vector, has none)
    expect(await kernel.readMaxNorm()).toBeCloseTo(maxNorm(vectors, 2), 4);
    const want = cpuKeys(cpu.lines, 2), got = new Map<string, number>();
    for (let i = 0; i < n; i++) {
      const o = i * SEG_FLOATS;
      const k = segKey([out[o]!, out[o + 1]!], [out[o + 2]!, out[o + 3]!]);
      got.set(k, (got.get(k) ?? 0) + 1);
      expect(out[o + 4]).toBe(out[o + 5]); // one colour per glyph
      expect(out[o + 7]).toBe(0); // no particles
    }
    let missing = 0;
    for (const [k, c] of want) if ((got.get(k) ?? 0) !== c) missing++;
    expect(missing).toBeLessThan(want.size * 0.01 + 1); // f32 rounding may flip a 4-decimal key on rare segments
    // colour: waves at the glyph's point, which is the shaft's midpoint (barbs are told apart by their midpoints)
    const shafts = new Map<string, number>();
    for (let k = 0; k < cpu.lines.length; k += 2) { const l = cpu.lines[k]!; const p = cpu.point[k]!; shafts.set(`${(0.5 * (l[0]! + l[2]!)).toFixed(3)},${(0.5 * (l[1]! + l[3]!)).toFixed(3)}`, waves.fn([pts[2 * p]!, pts[2 * p + 1]!], -1)); }
    let coloured = 0;
    for (let i = 0; i < n; i++) {
      const o = i * SEG_FLOATS;
      const c = shafts.get(`${(0.5 * (out[o]! + out[o + 2]!)).toFixed(3)},${(0.5 * (out[o + 1]! + out[o + 3]!)).toFixed(3)}`);
      if (c !== undefined) { expect(out[o + 4]).toBeCloseTo(c, 3); coloured++; }
    }
    expect(coloured).toBeGreaterThan(cpu.lines.length / 4);
    // reuse: reset + run appends the same count; a different (finer) lattice through the same kernel appends more
    resetSegments(gpu, segs);
    await kernel.run(segs, lattice);
    expect((await readSegments(gpu, segs)).length).toBe(out.length);
    const fine = latticeIn(bowl.box, 0.1, bowl.box.a);
    const segs2 = allocSegments(gpu, kernel.capacityFor(fine), false);
    await kernel.run(segs2, fine);
    const fp = latticePoints(fine);
    let nonzero = 0;
    for (let i = 0; i < fine.pointCount; i++) if (Math.hypot(...grad.value([fp[2 * i]!, fp[2 * i + 1]!])!) > 0) nonzero++;
    expect((await readSegments(gpu, segs2)).length / SEG_FLOATS).toBe(3 * nonzero);
    // head through the same kernel: exactly core's segments, 2 per glyph
    resetSegments(gpu, segs);
    await kernel.run(segs, lattice, "head");
    const o2 = await readSegments(gpu, segs);
    const cpu2 = arrowGlyphs(pts, vectors, 2, lattice.spacing, { style: "head" });
    expect(o2.length / SEG_FLOATS).toBe(cpu2.lines.length * 2);
    const w2 = cpuKeys(cpu2.lines, 2), g2 = new Map<string, number>();
    for (let i = 0; i < o2.length / SEG_FLOATS; i++) { const o = i * SEG_FLOATS; const k = segKey([o2[o]!, o2[o + 1]!], [o2[o + 2]!, o2[o + 3]!]); g2.set(k, (g2.get(k) ?? 0) + 1); }
    let miss = 0;
    for (const [k, c] of w2) if ((g2.get(k) ?? 0) !== c) miss++;
    expect(miss).toBeLessThan(w2.size * 0.01 + 1);
    // triangle: one filled record per glyph — base in a / b, apex in (arc, len) — equal to core's triangles
    resetSegments(gpu, segs);
    await kernel.run(segs, lattice, "triangle");
    const o3 = await readSegments(gpu, segs);
    const cpu3 = arrowGlyphs(pts, vectors, 2, lattice.spacing, { style: "triangle" });
    expect(o3.length / SEG_FLOATS).toBe(cpu3.triangles.length / 6);
    const triKey = (t: ArrayLike<number>, o: number) => [t[o], t[o + 1], t[o + 2], t[o + 3], t[o + 4], t[o + 5]].map((v) => v!.toFixed(4)).join(",");
    const w3 = new Set<string>();
    for (let i = 0; i < cpu3.triangles.length; i += 6) w3.add(triKey(cpu3.triangles, i));
    let found = 0;
    for (let i = 0; i < o3.length / SEG_FLOATS; i++) { const o = i * SEG_FLOATS; if (w3.has(triKey([o3[o]!, o3[o + 1]!, o3[o + 2]!, o3[o + 3]!, o3[o + 6]!, o3[o + 7]!], 0))) found++; }
    expect(found).toBeGreaterThan(w3.size * 0.99 - 1);
    segs.destroy(); segs2.destroy(); kernel.destroy();
  });
  it("3D: Seg3 arrows on an FCC lattice equal core's", async () => {
    if (!gpu) return;
    const v = buildVectorFieldData({ type: "symbolicv", box: [[-1, 1], [-1, 1], [-1, 1]], expr: { op: "compv", coeffs: [{ op: "coord", index: 1 }, { op: "negate", val: { op: "coord", index: 0 } }, { op: "mul", vals: [0.5, { op: "coord", index: 2 }] }] } }, 3);
    const lattice = latticeIn(v.box, 0.3);
    expect(lattice.cosets.length).toBe(4);
    const kernel = fusedGlyphs(gpu, v);
    const segs = allocSegments3(gpu, kernel.capacityFor(lattice), false);
    await kernel.run(segs, lattice);
    const out = await readSegments3(gpu, segs);
    const n = out.length / SEG3_FLOATS;
    const pts = latticePoints(lattice);
    const vectors = new Float64Array(lattice.pointCount * 3);
    for (let i = 0; i < lattice.pointCount; i++) vectors.set(v.value([pts[3 * i]!, pts[3 * i + 1]!, pts[3 * i + 2]!])!, 3 * i);
    const cpu = arrowGlyphs(pts, vectors, 3, lattice.spacing);
    expect(n).toBe(cpu.lines.length / 2 * 3);
    const want = cpuKeys(cpu.lines, 3), got = new Map<string, number>();
    for (let i = 0; i < n; i++) {
      const o = i * SEG3_FLOATS;
      const k = segKey([out[o]!, out[o + 1]!, out[o + 2]!], [out[o + 4]!, out[o + 5]!, out[o + 6]!]);
      got.set(k, (got.get(k) ?? 0) + 1);
    }
    let missing = 0;
    for (const [k, c] of want) if ((got.get(k) ?? 0) !== c) missing++;
    expect(missing).toBeLessThan(want.size * 0.01 + 1);
    segs.destroy(); kernel.destroy();
  });
});

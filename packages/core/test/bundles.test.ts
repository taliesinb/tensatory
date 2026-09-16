import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { Bundle, DenseGrid, isoContours } from "../src";

const dir = join(__dirname, "../../../apps/viewer/public/bundles");
const files = readdirSync(dir).filter((f) => f.endsWith(".json") && f !== "index.json");

describe("example bundles", () => {
  for (const f of files) {
    it(`${f} parses, builds every field, samples and contours`, () => {
      const b = Bundle.parse(JSON.parse(readFileSync(join(dir, f), "utf8")));
      const errors = b.buildAll();
      expect([...errors.entries()].map(([id, e]) => `${id}: ${e.message}`)).toEqual([]);
      for (const id of b.scalarFieldIds) {
        const fd = b.scalarField(id).data;
        const D = fd.box.dimCount;
        const grid = fd.samplePoints ?? new DenseGrid(Array(D).fill(D === 2 ? 40 : 12), fd.box);
        const vals = fd.sampleOn(grid);
        const st = fd.stats();
        expect(Number.isFinite(st.min), `${id} has finite min`).toBe(true);
        if (D === 2) {
          const lines = isoContours(grid, vals, 0.5 * (st.min + st.max));
          expect(Array.isArray(lines)).toBe(true);
        }
      }
      for (const id of b.vectorFieldIds) {
        const fd = b.vectorField(id).data;
        const v = fd.value(fd.box.center);
        expect(v?.length).toBe(fd.box.dimCount);
      }
    });
  }

  it("dense.json: exact gradient and FD gradient agree away from edges", () => {
    const b = Bundle.parse(JSON.parse(readFileSync(join(dir, "dense.json"), "utf8")));
    const err = b.scalarField("gradError").data;
    const g = err.samplePoints!;
    const vals = err.sampleOn(g);
    let interiorMax = 0;
    for (let i = 2; i < g.size[0]! - 2; i++) for (let j = 2; j < g.size[1]! - 2; j++) interiorMax = Math.max(interiorMax, vals[g.pos([i, j])]!);
    expect(interiorMax).toBeLessThan(0.5); // spacing 0.125, second derivatives O(10): central differences within ~0.3
    expect(b.scalarField("loss").exactGradient?.id).toBe("lossGrad");
    expect(b.scalarField("shifted").data.samplePoints!.box.intervals).toEqual([[-0.5, 1], [-0.5, 1]]);
  });
});

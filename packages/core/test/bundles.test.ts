import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { Bundle, DenseGrid, infoOf, isoContours } from "../src";

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
    }, 30_000); // iris: 16 net fields, half of them over the 120-example training set, sampled on the CPU
  }

  it("dynamical-systems.json: every space's flow is a vector field on it; derived Lyapunov derivatives are ≤ 0", () => {
    const b = Bundle.parse(JSON.parse(readFileSync(join(dir, "dynamical-systems.json"), "utf8")));
    expect(b.buildAll().size).toBe(0);
    for (const m of b.manifolds.values()) {
      expect(m.flow, `${m.id} declares a flow`).toBeDefined();
      expect(b.vectorField(m.flow!).domain).toBe(m);
    }
    // ∇H·F for the damped pendulum is −γ y²: never positive
    const hdot = b.scalarField("pendulum/Hdot").data;
    const g = new DenseGrid([41, 31], hdot.box);
    for (const v of hdot.sampleOn(g)) expect(v).toBeLessThanOrEqual(1e-9);
    // the Van der Pol cycle is a closed ordered point set on which F is tangent to the polyline (a coarse check)
    const cyc = b.pointSets.get("vanderpol/cycle")!;
    expect(cyc.ordered).toBe(true);
    expect(cyc.points[0]).toEqual(cyc.points[cyc.points.length - 1]);
  });

  it("summary / details: `info` of bundles, spaces and fields, and index.json mirrors every bundle's name and summary", () => {
    const index = JSON.parse(readFileSync(join(dir, "index.json"), "utf8")) as { file: string; name?: string; summary?: string }[];
    expect(index.map((e) => e.file).sort()).toEqual([...files].sort());
    for (const e of index) {
      const b = Bundle.parse(JSON.parse(readFileSync(join(dir, e.file), "utf8")));
      expect(e.name, `${e.file} name`).toBe(b.name);
      expect(e.summary, `${e.file} summary`).toBe(b.spec.summary);
      expect(b.info?.summary, `${e.file} has a summary`).toBeDefined();
      for (const m of b.manifolds.values()) if (m.info?.summary) expect(m.info.summary).not.toContain("\n"); // one line
    }
    const b = Bundle.parse(JSON.parse(readFileSync(join(dir, "dynamical-systems.json"), "utf8")));
    expect(b.manifolds.get("lorenz")!.info?.summary).toMatch(/^σ = 10/);
    const F = b.vectorField("lorenz/F").info!;
    expect(F.name).toBe("F");
    expect(F.summary).toMatch(/^ẋ = σ\(y − x\)/);
    expect(F.details!.length).toBeGreaterThan(F.summary!.length);
    expect(b.vectorField("gradient/gradV").info?.details).toBeUndefined(); // summary only
    expect(infoOf("x", {})).toBeUndefined();
    expect(infoOf("x", { summary: "  " })).toBeUndefined();
    expect(infoOf("x", { details: "d" })).toEqual({ name: "x", summary: undefined, details: "d" });
  });

  it("a manifold flow that is not a vector field on it is reported under manifolds.<id>", () => {
    const spec = {
      tensatory: "0.1",
      manifolds: { p: { numDims: 2, flow: "s" }, q: { numDims: 2, flow: "v" } },
      fields: {
        s: { kind: "scalar", domain: "p", data: { type: "symbolic", box: [[0, 1], [0, 1]], expr: { op: "coord", index: 0 } } },
        v: { kind: "vector", domain: "p", data: { type: "symbolicv", box: [[0, 1], [0, 1]], expr: { op: "coordv" } } },
      },
    };
    const errors = Bundle.parse(spec).buildAll();
    expect([...errors.keys()].sort()).toEqual(["manifolds.p", "manifolds.q"]);
  });

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

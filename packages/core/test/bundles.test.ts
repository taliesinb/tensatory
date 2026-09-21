import { existsSync, readFileSync, readdirSync, statSync } from "node:fs";
import { dirname, join } from "node:path";
import { describe, expect, it } from "vitest";
import { Bundle, DenseGrid, Sweep, infoOf, isoContours, rootKind } from "../src";
import { dirSource } from "./helpers";

const dir = join(__dirname, "../../../apps/viewer/public/bundles");
/**
 * every document: a top-level `<name>.json` (a bundle), or a directory's `bundle.json` (a bundle with sidecar arrays)
 * or `sweep.json` (a sweep whose members live inline or in subdirectories — those are not documents of their own here)
 */
const files = readdirSync(dir).flatMap((f) => {
  if (f.endsWith(".json")) return f === "index.json" ? [] : [f];
  if (!statSync(join(dir, f)).isDirectory()) return [];
  return ["bundle.json", "sweep.json"].filter((doc) => existsSync(join(dir, f, doc))).map((doc) => `${f}/${doc}`);
});

/** the sidecar files beside a document, as the viewer would fetch them */
const sourceFor = (file: string) => dirSource(join(dir, dirname(file)));
const readJson = (file: string) => JSON.parse(readFileSync(join(dir, file), "utf8")) as unknown;
const load = (file: string) => Bundle.load(readJson(file), sourceFor(file));
/** a document's bundles: the bundle itself, or every member of a sweep (by id) */
async function bundlesOf(file: string): Promise<[string, Bundle][]> {
  const json = readJson(file);
  if (rootKind(json) !== "sweep") return [[file, await load(file)]];
  const s = Sweep.parse(json, sourceFor(file));
  return Promise.all(s.memberIds.map(async (id) => [`${file} · ${id}`, await s.member(id)] as [string, Bundle]));
}
/** what an index entry mirrors: a bundle's or a sweep's own name and summary */
const infoOfDoc = (file: string): { name: string; summary: string | undefined } => {
  const json = readJson(file);
  return rootKind(json) === "sweep" ? (({ name, spec }) => ({ name, summary: spec.summary }))(Sweep.parse(json, sourceFor(file))) : (({ name, spec }) => ({ name, summary: spec.summary }))(Bundle.parse(json));
};

/** bundles whose live fields cost ~0.1 s per point on the CPU (the MNIST MLP): built here, sampled in their own test */
const HEAVY = new Set(["mnist-mlp/bundle.json"]);

describe("example bundles", () => {
  for (const f of files) {
    it(`${f} parses, builds every field${HEAVY.has(f) ? "" : ", samples and contours"}`, async () => {
      for (const [label, b] of await bundlesOf(f)) {
      const errors = b.buildAll();
      expect([...errors.entries()].map(([id, e]) => `${label} ${id}: ${e.message}`)).toEqual([]);
      if (HEAVY.has(f)) return;
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
    // the Van der Pol cycle is a closed sampled curve: it wraps back to its first point
    const cyc = b.curve("vanderpol/cycle").data;
    expect(cyc.kind).toBe("sampled");
    expect(cyc.point(cyc.t1)).toEqual(cyc.point(cyc.t0));
    // the Hopf cycle is the unit circle written as an expression; the Lorenz attractor a flow curve that integrates
    // F live and stays inside the box over its whole interval
    const hopf = b.curve("hopf/cycle").data;
    expect(Math.hypot(...hopf.point(1.234)!)).toBeCloseTo(1, 12);
    const lorenz = b.curve("lorenz/attractor");
    expect(lorenz.data.kind).toBe("symbolic");
    const pts = lorenz.data.polyline();
    expect(pts.length / 3).toBeGreaterThan(1000);
    const box = b.scalarField("lorenz/trap").data.box;
    let inside = 0;
    for (let i = 0; i < pts.length; i += 3) if (box.contains([pts[i]!, pts[i + 1]!, pts[i + 2]!], 1e-9)) inside++;
    expect(inside / (pts.length / 3)).toBeGreaterThan(0.99);
    // the pendulum orbit ends captured near the well at θ = 0
    const end = b.curve("pendulum/orbit").data.point(45)!;
    expect(Math.abs(end[0]!)).toBeLessThan(0.5);
  });

  it("summary / details: `info` of bundles, spaces and fields, and index.json mirrors every bundle's name and summary", async () => {
    const index = JSON.parse(readFileSync(join(dir, "index.json"), "utf8")) as { file: string; name?: string; summary?: string }[];
    // every indexed bundle exists; the heavy ones are deliberately unindexed (the viewer cannot show them yet)
    expect(index.map((e) => e.file).sort()).toEqual(files.filter((f) => !HEAVY.has(f)).sort());
    for (const e of index) {
      const info = infoOfDoc(e.file);
      expect(e.name, `${e.file} name`).toBe(info.name);
      expect(e.summary, `${e.file} summary`).toBe(info.summary);
      expect(info.summary, `${e.file} has a summary`).toBeDefined();
      for (const [, b] of await bundlesOf(e.file)) for (const m of b.manifolds.values()) if (m.info?.summary) expect(m.info.summary).not.toContain("\n"); // one line
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

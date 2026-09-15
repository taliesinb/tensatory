import { it } from "vitest";
import { readFileSync } from "node:fs";
import { Bundle, DenseGrid, SymbolicVectorFieldData, SymbolicScalarFieldData, contourField, integrateStreamlines } from "../src";
import type { SExpr, VExpr } from "../src/symbolic/ast";
import { diffScalar } from "../src/symbolic/diff";

const isNode = (x: unknown): x is SExpr | VExpr => !!x && typeof x === "object" && "k" in (x as object);
const count = (e: SExpr | VExpr): number => {
  let n = 1;
  for (const v of Object.values(e as Record<string, unknown>)) {
    if (Array.isArray(v)) { for (const x of v) if (isNode(x)) n += count(x); }
    else if (isNode(v)) n += count(v);
  }
  return n;
};
const time = (label: string, f: () => unknown, reps = 1) => { const t0 = performance.now(); let r; for (let i = 0; i < reps; i++) r = f(); console.log(label.padEnd(44), ((performance.now() - t0) / reps).toFixed(1), "ms"); return r; };

it.runIf(process.env.PERF)("perf: |∇ mixture| pipeline (PERF=1 to run)", { timeout: 60000 }, () => {
  const b = Bundle.parse(JSON.parse(readFileSync(__dirname + "/../../../apps/viewer/public/bundles/symbolic.json", "utf8")));
  const m = b.scalarField("mixture").data as SymbolicScalarFieldData;
  const gn = b.scalarField("mixtureGradNorm").data as SymbolicScalarFieldData;
  console.log("nodes: mixture", count(m.ast), " |∇m| (pointwise ast)", count(gn.ast));
  const d0 = diffScalar(gn.ast, 0, 2);
  console.log("nodes: d|∇m|/dx (through arg)", count(d0));
  // the fully inlined derivative (what actually gets evaluated once args resolve)
  const inl = new SymbolicScalarFieldData({ k: "norm", v: { k: "grad", s: m.ast } }, 2, undefined, m.box);
  console.log("nodes: |∇m| inlined", count(inl.ast), " d/dx", count(diffScalar(inl.ast, 0, 2)), " d²/dx²", count(diffScalar(diffScalar(inl.ast, 0, 2), 0, 2)));

  const grid = new DenseGrid([128, 128], gn.box);
  time("sample |∇m| 128²", () => gn.sampleOn(grid));
  const dx = time("derivative(0) construct+compile", () => gn.derivative(0)) as SymbolicScalarFieldData;
  const p = new Float64Array(2);
  const at = (i: number) => { p[0] = -2 + (i % 100) * 0.04; p[1] = -2 + Math.floor(i / 100) * 0.4; return p; };
  time("eval |∇m| x1000", () => { for (let i = 0; i < 1000; i++) gn.fn(at(i), -1); });
  time("eval d|∇m|/dx x1000", () => { for (let i = 0; i < 1000; i++) dx.fn(at(i), -1); });
  const vals = gn.sampleOn(grid);
  const st = gn.stats();
  time("exact isolines, 1 level", () => contourField(gn, grid, vals, 0.5 * (st.min + st.max), { tolerance: 6 / 128 * 0.25 }));
  time("exact isolines, 4 levels", () => { for (let k = 1; k <= 4; k++) contourField(gn, grid, vals, st.min + (k / 5) * (st.max - st.min), { tolerance: 6 / 128 * 0.25 }); });
  const g = new SymbolicVectorFieldData({ k: "grad", s: { k: "arg", name: "n" } }, 2, { scalars: { n: gn }, vectors: {} });
  const o = new Float64Array(2);
  time("eval ∇|∇m| x1000", () => { for (let i = 0; i < 1000; i++) g.fn(at(i), -1, o); });
  time("streamlines 1000 lines x100 steps", () => integrateStreamlines(g, { count: 1000, maxSteps: 100, step: 6 / 128 / 2, sign: -1, seed: 1 }));
  time("streamlines 200 lines x100 steps", () => integrateStreamlines(g, { count: 200, maxSteps: 100, step: 6 / 128 / 2, sign: -1, seed: 1 }));
});

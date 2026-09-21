// Writes apps/viewer/public/bundles/sweep-demo/: a tiny SWEEP (notes/sweeps.md §2) of symbolic members for
// exercising the record row, faceting and options-by-signature without any real data:
//
//   node tools/sweep-demo/build.mjs
//
// Members: k·(x² + y²) for k ∈ {1, 2, 4}, k·(x² − y²) for k ∈ {1, 4} (a non-Cartesian record set: no saddle at k = 2),
// and one 3D bowl (k = 1) — a different structural signature. `common` carries the two manifolds and the derived
// |∇f|. Every member is inline except `saddle k = 4`, written as its own document in a subdirectory so member
// loading by path (and sidecar paths relative to the MEMBER document) is exercised by the tests and the viewer.

import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const outDir = join(dirname(fileURLToPath(import.meta.url)), "../../apps/viewer/public/bundles/sweep-demo");
mkdirSync(join(outDir, "saddle-k4"), { recursive: true });

const coord = (i) => ({ op: "coord", index: i });
const sq = (i) => ({ op: "square", val: coord(i) });
const box = (D) => Array.from({ length: D }, () => [-2, 2]);

/** k·(x² ± y² [+ z²]) */
function member(shape, k, D) {
  const terms = D === 3 ? [sq(0), shape === "saddle" ? { op: "negate", val: sq(1) } : sq(1), sq(2)] : [sq(0), shape === "saddle" ? { op: "negate", val: sq(1) } : sq(1)];
  const sign = shape === "saddle" ? "−" : "+";
  const formula = D === 3 ? `${k === 1 ? "" : `${k}·`}(x² ${sign} y² + z²)` : `${k === 1 ? "" : `${k}·`}(x² ${sign} y²)`;
  return {
    tensatory: "0.1",
    name: `${shape} k = ${k}${D === 3 ? " (3D)" : ""}`,
    summary: `f = ${formula}: a ${shape} of sharpness ${k}${D === 3 ? " in three dimensions" : ""}`,
    fields: {
      f: {
        kind: "scalar", name: "f", domain: D === 3 ? "volume" : "plane",
        summary: formula,
        data: { type: "symbolic", box: box(D), expr: k === 1 ? { op: "add", vals: terms } : { op: "mul", vals: [k, { op: "add", vals: terms }] } },
      },
    },
  };
}

const record = (shape, k, dims) => ({ shape, k, dims });

const saddleK4 = member("saddle", 4, 2);
writeFileSync(join(outDir, "saddle-k4/bundle.json"), JSON.stringify(saddleK4, null, 1) + "\n");

const sweep = {
  tensatory: "0.2",
  name: "sweep demo",
  summary: "A toy sweep: bowls and saddles k·(x² ± y²) of sharpness k ∈ {1, 2, 4}, one of them in 3D — for the record row, faceting and options that survive a member switch.",
  details: [
    "Six symbolic members with flat metadata records over three keys. `shape` and `k` vary independently except that there is no saddle at k = 2, and only the k = 1 bowl exists in 3D — so the record set is not a grid, which is what the greyed values in the record row show: a greyed value exists in the sweep, but switching to it changes another key as well.",
    "The 2D members share one structural signature (space `plane`, fields `f` and `gradNorm`), so slot selections, levels and the view carry over when flipping `shape` or `k`; the 3D member has its own.",
    "`saddle k = 4` is a member by path (saddle-k4/bundle.json); the others are inline. `common` contributes the manifolds and |∇f|.",
  ].join("\n\n"),
  keys: {
    shape: { kind: "nominal", values: ["bowl", "saddle"], summary: "bowl: x² + y², saddle: x² − y²" },
    k: { name: "sharpness", kind: "ordinal", codomain: "log", summary: "the factor in front: f = k·(x² ± y²)" },
    dims: { kind: "ordinal", summary: "the dimension of the space the member lives in" },
  },
  common: {
    manifolds: {
      plane: { name: "plane", numDims: 2, dimNames: ["x", "y"], summary: "the 2D members' space" },
      volume: { name: "volume", numDims: 3, dimNames: ["x", "y", "z"], summary: "the 3D member's space" },
    },
    fields: {
      gradNorm: { kind: "scalar", name: "|∇f|", codomain: "norm", summary: "the gradient norm of f, derived symbolically (from `common`)", data: { type: "pointwise", expr: { op: "norm", vec: { op: "grad", val: "f" } }, scalars: { f: "f" } } },
    },
  },
  members: {
    "bowl-k1": { record: record("bowl", 1, 2), bundle: member("bowl", 1, 2) },
    "bowl-k2": { record: record("bowl", 2, 2), bundle: member("bowl", 2, 2) },
    "bowl-k4": { record: record("bowl", 4, 2), bundle: member("bowl", 4, 2) },
    "saddle-k1": { record: record("saddle", 1, 2), bundle: member("saddle", 1, 2) },
    "saddle-k4": { record: record("saddle", 4, 2), bundle: "saddle-k4/bundle.json" },
    "bowl-k1-3d": { record: record("bowl", 1, 3), bundle: member("bowl", 1, 3) },
  },
};
// the 3D bowl's `gradNorm` from common lives on `plane` unless it says otherwise: give the field a domain per member
for (const m of Object.values(sweep.members)) if (typeof m.bundle === "object") m.bundle.defaultManifold = m.record.dims === 3 ? "volume" : "plane";
saddleK4.defaultManifold = "plane";
writeFileSync(join(outDir, "saddle-k4/bundle.json"), JSON.stringify(saddleK4, null, 1) + "\n");
writeFileSync(join(outDir, "sweep.json"), JSON.stringify(sweep, null, 1) + "\n");
console.log(`wrote ${outDir}/sweep.json (${Object.keys(sweep.members).length} members)`);

// Turns a volume of the loss-landscape prototype (~/projects/loss-landscape/viewer/data/<name>.json + <name>.bin)
// into a Tensatory bundle DIRECTORY: apps/viewer/public/bundles/<out>/bundle.json beside a copy of the raw volume,
// which the bundle refers to through `handle` array specs — the first bundles with external arrays.
//
//   node tools/loss-landscape/build.mjs mnist_convnet_pca [out-name] [--data <dir>]
//
// The prototype stores each volume headerless: float32 with the channels interleaved and x FASTEST —
// `data[((z·ny + y)·nx + x)·C + c]` (main.js `cvox = x + nx·(y + ny·z)`), i.e. row-major of shape (nz, ny, nx, C) —
// channels ["log10_loss", "accuracy"], the grid spanning `axes` (the box, in the coordinates of the `dirs`
// directions, θ* at the origin), the optimizer `trajectory` as points of that space. Here that becomes: one
// manifold per volume (`pca`: dimWeights = explained variance; `random`: none), a dense scalar field per channel —
// each a `handle` into the same `vol.bin` with `part: [null, null, null, c]` selecting the channel and
// `axes: [2, 1, 0]` turning the kept (z, y, x) into (x, y, z) — a derived `loss` (10^log10_loss, `celoss`
// codomain), the trajectory as an ordered point set and θ* as a labelled point. The file itself is copied unchanged
// (512 KB for 40³, 2 MB for 64³).
//
// Then add the bundle to apps/viewer/public/bundles/index.json by hand (`"file": "<out>/bundle.json"`).

import { copyFileSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { homedir } from "node:os";
import { fileURLToPath } from "node:url";

const args = process.argv.slice(2);
const dataIdx = args.indexOf("--data");
const dataDir = dataIdx >= 0 ? args.splice(dataIdx, 2)[1] : join(homedir(), "projects/loss-landscape/viewer/data");
const [name, outName = name.replace(/_/g, "-")] = args;
if (!name) { console.error("usage: node tools/loss-landscape/build.mjs <volume-name> [out-name] [--data <dir>]"); process.exit(2); }

const meta = JSON.parse(readFileSync(join(dataDir, `${name}.json`), "utf8"));
const { dims, channels, axes, trajectory, explained_variance: ev, model, dataset, dirs, n_params, eval_n } = meta;
if (meta.dtype !== "float32") throw new Error(`unexpected dtype ${meta.dtype}`);
const D = dims.length, C = channels.length;
const shape = [...dims].reverse().concat(C); // stored (z, y, x, c)
const perm = Array.from({ length: D }, (_, i) => D - 1 - i); // kept (z, y, x) -> (x, y, z)
const dimNames = dirs === "pca" ? Array.from({ length: D }, (_, i) => `pc${i + 1}`) : Array.from({ length: D }, (_, i) => `d${i}`);

const outDir = join(dirname(fileURLToPath(import.meta.url)), "../../apps/viewer/public/bundles", outName);
mkdirSync(outDir, { recursive: true });
copyFileSync(join(dataDir, `${name}.bin`), join(outDir, "vol.bin"));

const range = (c) => meta[`${c}_range`];
const channelField = (c, k, extra) => ({
  kind: "scalar",
  name: c.replace(/_/g, " "),
  domain: dirs,
  ...extra,
  data: {
    type: "dense",
    samples: { type: "handle", path: "vol.bin", shape, part: [...new Array(D).fill(null), k], axes: perm, dtype: "float32" },
    ...(range(c) ? { stats: { extrema: { min: range(c)[0], max: range(c)[1] } } } : {}),
    box: axes,
  },
});

const fields = {};
channels.forEach((c, k) => {
  fields[c] = channelField(c, k, {
    codomain: c === "accuracy" ? "fraction" : "lin",
    summary: c === "log10_loss"
      ? `log₁₀ of the ${dataset} cross-entropy loss of the ${model} on ${eval_n} evaluation examples, sampled on a ${dims.join("×")} grid`
      : c === "accuracy" ? `classification accuracy on the same ${eval_n} examples` : `channel ${k} of the collected volume`,
  });
});
if (channels.includes("log10_loss")) {
  fields.loss = {
    kind: "scalar",
    name: "loss",
    domain: dirs,
    codomain: "celoss",
    summary: "the cross-entropy loss itself, 10^log10_loss (pointwise from the stored channel)",
    data: { type: "pointwise", expr: { op: "pow", vals: [10, "l"] }, scalars: { l: "log10_loss" } },
  };
}

const bundle = {
  tensatory: "0.1",
  name: `${model} on ${dataset}: ${dirs} landscape`,
  summary: `Loss and accuracy of a trained ${model} (${n_params.toLocaleString("en")} parameters) around θ* in the ${dirs === "pca" ? "top-3 PCA subspace of its optimizer trajectory" : "span of 3 random directions"}: a ${dims.join("×")} float32 volume stored beside the bundle (vol.bin), each channel a \`handle\` into it.`,
  details: [
    `Collected by the loss-landscape prototype (~/projects/loss-landscape): the model was trained on ${dataset}, then the`,
    `loss and accuracy were evaluated on ${eval_n} held-out examples at every point of a ${dims.join("×")} grid in the`,
    dirs === "pca"
      ? `3D subspace spanned by the top principal components of the parameter trajectory (explained variance ${ev.map((v) => (100 * v).toFixed(1) + "%").join(", ")}).`
      : `3D subspace spanned by three random directions (filter-normalized).`,
    `θ* (the final model) is at the origin; the box is the span of the trajectory with a margin. The stored channel is`,
    `log₁₀(loss); \`loss\` recovers the loss itself as a pointwise field. The raw volume is the prototype's own`,
    `\`${name}.bin\` — headerless float32, channels interleaved, x fastest: row-major (z, y, x, channel) — so the`,
    `bundle reads it through \`{ "type": "handle", "path": "vol.bin", "shape": [${shape}], "part": [null, null, null, c],`,
    `"axes": [${perm}] }\`: the part picks the channel, the axes permutation puts x first.`,
  ].join("\n"),
  manifolds: {
    [dirs]: {
      name: dirs === "pca" ? "PCA directions" : "random directions",
      numDims: D,
      dimNames,
      ...(ev ? { dimWeights: ev } : {}),
      summary: dirs === "pca" ? `coefficients along the top-${D} principal components of the optimizer trajectory (θ* at 0)` : `coefficients along ${D} random directions (θ* at 0)`,
    },
  },
  defaultManifold: dirs,
  fields,
  pointSets: {
    ...(trajectory ? { trajectory: { domain: dirs, points: trajectory, ordered: true, name: "optimizer trajectory" } } : {}),
    theta: { domain: dirs, points: [new Array(D).fill(0)], labels: ["θ*"], name: "θ*" },
  },
};

writeFileSync(join(outDir, "bundle.json"), JSON.stringify(bundle, null, 1) + "\n");
console.log(`wrote ${join(outDir, "bundle.json")} + vol.bin (${shape.join("×")} float32)`);

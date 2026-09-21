// Turns the volumes of the loss-landscape prototype (~/projects/loss-landscape/viewer/data/<name>.json + <name>.bin)
// into the SWEEP apps/viewer/public/bundles/loss-landscape/ (notes/sweeps.md §2 "the trial dataset as a sweep"):
//
//   node tools/loss-landscape/build.mjs [--data <dir>]
//
// Every volume found in the data directory becomes a MEMBER directory `<name-with-dashes>/` holding `bundle.json`,
// `record.json` (its flat metadata record) and a copy of the raw volume `vol.bin` that the bundle refers to through
// `handle` array specs. Two sweep documents are written beside them: `sweep.json` lists the members whose volumes are
// small enough to be committed (the 40³ ConvNet pair, 512 KB each), `sweep-all.json` every member found (the 64³ MLP
// volumes are 2 MB each and stay local: their directories and sweep-all.json are gitignored). `sweep.json` is in
// bundles/index.json; the full one loads through ?bundle=loss-landscape/sweep-all.json.
//
// The prototype stores each volume headerless: float32 with the channels interleaved and x FASTEST —
// `data[((z·ny + y)·nx + x)·C + c]` (main.js `cvox = x + nx·(y + ny·z)`), i.e. row-major of shape (nz, ny, nx, C) —
// channels ["log10_loss", "accuracy"], the grid spanning `axes` (the box, in the coordinates of the `dirs`
// directions, θ* at the origin), the optimizer `trajectory` as points of that space. Here that becomes: one
// manifold per volume (`pca`: dimWeights = explained variance; `random`: none), a dense scalar field per channel —
// each a `handle` into the same `vol.bin` with `part: [null, null, null, c]` selecting the channel and
// `axes: [2, 1, 0]` turning the kept (z, y, x) into (x, y, z) — the trajectory as a sampled CURVE (parameter: the
// snapshot index) and θ* as a labelled point. The derived `loss` (10^log10_loss, `celoss` codomain) lives in the
// sweep's `common`. The file itself is copied unchanged (512 KB for 40³, 2 MB for 64³).

import { copyFileSync, existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { homedir } from "node:os";
import { fileURLToPath } from "node:url";

const args = process.argv.slice(2);
const dataIdx = args.indexOf("--data");
const dataDir = dataIdx >= 0 ? args.splice(dataIdx, 2)[1] : join(homedir(), "projects/loss-landscape/viewer/data");
if (args.length) { console.error("usage: node tools/loss-landscape/build.mjs [--data <dir>]"); process.exit(2); }

const sweepDir = join(dirname(fileURLToPath(import.meta.url)), "../../apps/viewer/public/bundles/loss-landscape");

/** the prototype's test accuracies, from its training logs (logs/<dataset>_<model>.log: "test acc 0.9760") */
const TEST_ACC = { mnist_mlp: 0.976, mnist_convnet: 0.9838 };
/** volumes small enough to commit (512 KB): the members of the committed sweep.json */
const COMMITTED = new Set(["mnist_convnet_pca", "mnist_convnet_random"]);

/** build one member directory from a prototype volume; returns its record */
function buildMember(name) {
  const meta = JSON.parse(readFileSync(join(dataDir, `${name}.json`), "utf8"));
  const { dims, channels, axes, trajectory, explained_variance: ev, model, dataset, dirs, n_params, eval_n } = meta;
  if (meta.dtype !== "float32") throw new Error(`${name}: unexpected dtype ${meta.dtype}`);
  const D = dims.length, C = channels.length;
  const shape = [...dims].reverse().concat(C); // stored (z, y, x, c)
  const perm = Array.from({ length: D }, (_, i) => D - 1 - i); // kept (z, y, x) -> (x, y, z)
  const dimNames = dirs === "pca" ? Array.from({ length: D }, (_, i) => `pc${i + 1}`) : Array.from({ length: D }, (_, i) => `d${i}`);
  const outName = name.replace(/_/g, "-");
  const outDir = join(sweepDir, outName);
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

  const bundle = {
    tensatory: "0.1",
    name: `${model} · ${dirs}`,
    summary: `Loss and accuracy of the trained ${model} (${n_params.toLocaleString("en")} parameters) around θ* in the ${dirs === "pca" ? "top-3 PCA subspace of its optimizer trajectory" : "span of 3 random directions"}: a ${dims.join("×")} float32 volume stored beside the bundle (vol.bin), each channel a \`handle\` into it.`,
    details: [
      `Collected by the loss-landscape prototype (~/projects/loss-landscape): the model was trained on ${dataset}, then the`,
      `loss and accuracy were evaluated on ${eval_n} held-out examples at every point of a ${dims.join("×")} grid in the`,
      dirs === "pca"
        ? `3D subspace spanned by the top principal components of the parameter trajectory (explained variance ${ev.map((v) => (100 * v).toFixed(1) + "%").join(", ")}).`
        : `3D subspace spanned by three random directions (filter-normalized).`,
      `θ* (the final model) is at the origin; the box is ${dirs === "pca" ? "the span of the trajectory with a margin" : "[−1.5, 1.5]³"}. The stored channel is`,
      `log₁₀(loss); the sweep's common \`loss\` recovers the loss itself as a pointwise field. The raw volume is the prototype's own`,
      `\`${name}.bin\` — headerless float32, channels interleaved, x fastest: row-major (z, y, x, channel) — so the`,
      `bundle reads it through \`{ "type": "handle", "path": "vol.bin", "shape": [${shape}], "part": [null, null, null, c],`,
      `"axes": [${perm}] }\`: the part picks the channel, the axes permutation puts x first.`,
    ].join(" "),
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
      theta: { domain: dirs, points: [new Array(D).fill(0)], labels: ["θ*"], name: "θ*" },
    },
    ...(trajectory
      ? {
          curves: {
            trajectory: {
              domain: dirs,
              name: "optimizer trajectory",
              param: { name: "snapshot" },
              summary: `the ${trajectory.length} parameter snapshots of the training run, projected onto the ${dirs} directions; the last one is θ*`,
              data: { type: "sampled", points: { type: "inline", shape: [trajectory.length, D], data: trajectory.flat() } },
            },
          },
        }
      : {}),
  };
  writeFileSync(join(outDir, "bundle.json"), JSON.stringify(bundle, null, 1) + "\n");

  // the record: coordinates of the sweep (dataset / model / dirs / seed) and attributes of the member
  const record = {
    dataset, model, dirs, seed: 0,
    n_params, eval_n, grid: dims[0],
    ...(TEST_ACC[`${dataset}_${model}`] !== undefined ? { test_acc: TEST_ACC[`${dataset}_${model}`] } : {}),
  };
  writeFileSync(join(outDir, "record.json"), JSON.stringify(record, null, 1) + "\n");
  console.log(`wrote ${outName}/bundle.json + record.json + vol.bin (${shape.join("×")} float32)`);
  return { id: outName, record, path: `${outName}/bundle.json` };
}

const volumes = readdirSync(dataDir).filter((f) => f.endsWith(".json") && f !== "index.json" && existsSync(join(dataDir, f.replace(/\.json$/, ".bin")))).map((f) => f.replace(/\.json$/, "")).sort();
if (!volumes.length) { console.error(`no volumes (<name>.json + <name>.bin) in ${dataDir}`); process.exit(1); }
const members = volumes.map(buildMember);

/** the sweep document over `ms` */
const sweepDoc = (ms, everything) => ({
  tensatory: "0.2",
  name: everything ? "loss-landscape prototype runs (all)" : "loss-landscape prototype runs",
  summary: `MNIST loss / accuracy volumes of the loss-landscape prototype around each trained model's θ*, along its top-3 PCA directions or 3 random ones${everything ? ", for both the MLP and the ConvNet" : " (the ConvNet; the MLP members are local: ?bundle=loss-landscape/sweep-all.json)"} — a sweep: flip \`dirs\` with the camera held to see the same minimum in two frames.`,
  details: [
    "One run of the prototype's run_all.sh (September 2025, seed 0): an MLP 784-256-256-10 (269 322 parameters, test accuracy 0.976) and a 3-conv ConvNet (56 394 parameters, 0.984) trained on MNIST with SGD (lr 0.05, momentum 0.9, weight decay 5e-4, 3 epochs), then log₁₀(loss) and accuracy evaluated on a fixed held-out subset at every point of a 3D grid around θ*: 64³ (N = 1024) for the MLP, 40³ (N = 384) for the ConvNet.",
    "Two kinds of directions per model: `pca` — the top three principal components of the 48-snapshot optimizer trajectory, the box the trajectory's span with a margin, the trajectory itself a curve — and `random` — three filter-normalized random directions over [−1.5, 1.5]³ (the directions were never saved, so these volumes cannot be related to parameter space).",
    "Members are the prototype's own volumes read in place (`handle` arrays into the interleaved .bin); `common` adds `loss` = 10^log10_loss under the `celoss` codomain. The volume members share one structural signature (`log10_loss`, `accuracy`, `loss` on a 3D space), so slots, levels, camera and colormaps carry over when flipping `model` or `dirs`; the pca members also carry the trajectory curve and θ*.",
    "Attributes of a member (`parameters`, `eval n`, `grid`, `test acc`) are shown with the record and never count as a change. notes/sweeps.md lists what a rerun of the collection script should produce for the members to be first-class (raw loss, self-describing arrays, saved directions, several seeds).",
  ].join("\n\n"),
  keys: {
    dataset: { kind: "nominal", values: ["mnist", "fmnist", "cifar10"], summary: "the training / evaluation dataset" },
    model: { kind: "nominal", values: ["mlp", "convnet"], summary: "mlp: 784-256-256-10 ReLU MLP; convnet: three conv layers" },
    dirs: { kind: "nominal", values: ["pca", "random"], summary: "pca: the top-3 principal components of the optimizer trajectory; random: three filter-normalized random directions" },
    seed: { kind: "nominal", summary: "the training seed (only 0 was run)" },
    n_params: { name: "parameters", attribute: true, summary: "the model's parameter count" },
    eval_n: { name: "eval n", attribute: true, summary: "held-out examples the loss and accuracy were evaluated on at each grid point" },
    grid: { attribute: true, summary: "grid points per axis of the volume" },
    test_acc: { name: "test acc", attribute: true, codomain: "fraction", summary: "test-set accuracy of the trained model (θ*)" },
  },
  common: {
    fields: {
      loss: {
        kind: "scalar",
        name: "loss",
        codomain: "celoss",
        summary: "the cross-entropy loss itself, 10^log10_loss (pointwise from the stored channel; from the sweep's `common`)",
        data: { type: "pointwise", expr: { op: "pow", vals: [10, "l"] }, scalars: { l: "log10_loss" } },
      },
    },
  },
  members: Object.fromEntries(ms.map((m) => [m.id, { record: m.record, bundle: m.path }])),
});

const committed = members.filter((m) => COMMITTED.has(m.id.replace(/-/g, "_")));
writeFileSync(join(sweepDir, "sweep.json"), JSON.stringify(sweepDoc(committed, false), null, 1) + "\n");
writeFileSync(join(sweepDir, "sweep-all.json"), JSON.stringify(sweepDoc(members, true), null, 1) + "\n");
console.log(`wrote sweep.json (${committed.length} members) and sweep-all.json (${members.length} members) in ${sweepDir}`);

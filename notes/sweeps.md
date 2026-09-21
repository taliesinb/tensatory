# External arrays and sweeps

Two related designs for the bundle format. §1 (external arrays) is **built**
as designed here, with one addition found on contact with the data (`axes`,
see the end of §1); the current description is in
[bundle-schema.md](bundle-schema.md) "External arrays". §2 (sweeps) is
**built** as designed, with the differences recorded in "As built" at the end
of §2. Ordering against other work: [roadmap.md](roadmap.md).

## 1. External arrays (`handle` specs)

### The concrete case

The loss-landscape prototype stores each volume as `name.json` (dims,
channels, axes, trajectory, …) plus `name.bin`: headerless `float32`,
`40×40×40×2`, **channel-interleaved** and **x fastest** —
`data[((z·40 + y)·40 + x)·2 + c]` (this note first assumed x slowest; the
prototype's `cvox = x + nx·(y + ny·z)` says otherwise), channels
`['log10_loss', 'accuracy']`. Two things the phase-1 `SizedArrayHandleSpec`
could not say about it:

1. a raw file has no shape of its own, so the bundle must give the shape
   *before* `part` is applied (today `shape` is documented as the shape after);
2. a channel is an index on the **last** axis, and `part: AxisPos[]` fixes
   leading indices only.

### Schema change (small, backward compatible)

```ts
export type ArrayPart = (AxisPos | null)[];   // integer: fix that axis and drop it; null: keep it; trailing nulls may be omitted
export type Dtype = "float32" | "float64" | "int32" | "uint8";  // short list, extend on demand

export type SizedArrayHandleSpec = {
  type: "handle";
  path: ArrayPath;
  shape: ArrayShape;   // the STORED shape (what the file holds; validated against npy / zarr metadata). Result shape = shape minus fixed axes
  part?: ArrayPart;
  dtype?: Dtype;       // required knowledge for .bin (defaults float32); validation for self-describing stores
};
```

Today's `part: [0]` keeps its meaning exactly, so no `BUNDLE_VERSION` bump.
The channel of the convnet volume is
`{ "type": "handle", "path": "vol.bin", "shape": [40,40,40,2], "part": [null,null,null,0] }`;
a planar `(C, N)` layout is `part: [0]`; a cell is a `part` fixing every axis
(so `CellSpec` is redundant and should go). `applyPart` is one generic strided
gather shared by every format.

Deliberately **not** added: ranges / strides in `part` (subsampling is the
adaptive resolution's job, windows are the field's `box`), axis reordering
(a separate future `order` on `dense` if a Fortran-order dataset appears),
byte offsets / endianness / several arrays per raw file (one array per
`.bin`; two channels are two handles into the same file, fetched once).

### Runtime: bytes in the environment, formats in core, build stays synchronous

The whole build pipeline is synchronous and lazy (`Bundle.field(id)` with
cycle detection → `buildScalarFieldData` → `buildArray` → `NdArray`), with
~70 test call sites and a synchronous viewer `setBundle`. Bundles are small
(0.5–2 MB), so I/O is a **prefetch phase** and the build is untouched.

```ts
// packages/core/src/arrays/load.ts  — isomorphic: no fetch, no fs
export interface ByteSource { bytes(path: string): Promise<ArrayBuffer>; }   // path relative to the bundle document
export interface RawArray { shape: number[]; dtype: Dtype; data: Float32Array | Float64Array | Int32Array | Uint8Array; }

/** decode one stored array (before `part`); the format follows the path */
export function loadArray(src: ByteSource, path: ArrayPath, hint: { shape?: number[]; dtype?: Dtype }): Promise<RawArray>;
//   .bin        needs hint.shape (SpecError otherwise); byteLength must equal product × itemsize
//   .npy        parse the header, validate against the hint
//   .npz/member bytes("x.npz") → inflate the member (DecompressionStream "deflate-raw") → npy
//   .zarr/…     NotSupportedError until written; asks the ByteSource for .zarray and chunk keys — the reason the
//               primitive is "bytes by path" and not "array by path"
```

```ts
// packages/core/src/arrays/spec.ts
export function collectHandles(spec: BundleSpec): Map<ArrayPath, { shape?: number[]; dtype?: Dtype }>;
//   pure walk over fields[*].data (samples, sparse points, translate.vec, scale.origin, stats arrays);
//   dedups by path; conflicting hints for one path → SpecError
export interface ArrayResolver { raw(path: ArrayPath, at: string[]): RawArray; }   // sync, mirrors FieldResolver
export const noArrays: ArrayResolver;   // throws SpecError("array … was not loaded")
export function buildArray(spec: SizedArraySpec, arrays: ArrayResolver = noArrays, path?: string[]): NdArray;
//   "handle": arrays.raw → check shape → applyPart → NdArray (widened to Float64Array; the GPU re-narrows to f32 anyway)
//   resolvePoint gets the same resolver, so array-backed translate.vec / scale.origin stop being NotSupported for free

// packages/core/src/bundle/bundle.ts
class Bundle {
  constructor(spec: BundleSpec, arrays: ArrayResolver = noArrays);   // unchanged for every existing caller
  static parse(json: unknown, arrays?: ArrayResolver): Bundle;        // sync, as today
  static async load(json: unknown, src: ByteSource): Promise<Bundle>; // validate → collectHandles → loadArray each → construct
}
```

`buildAll()`'s per-field error collection already turns a missing sidecar or
a shape mismatch into one red line in the bundle panel instead of a dead
bundle. Environment implementations are trivial: the viewer's
`fetch(new URL(path, bundleUrl))`, a `readFile(join(dir, path))` for tests, a
`Map` for unit tests. The local-upload path (`pickFile`) has no sidecars and
uses `Bundle.parse` — handles fail per field with a clear message.

A bundle with sidecars is a directory (`bundles/<name>/bundle.json` +
`vol.bin`); `index.json` points at the JSON; `path` is relative to that
document, so the same directory serves from Vite, a static host or a future
Tensatory server.

Deferred: lazy per-field loading (an async `Bundle.prepare(fieldIds)` filling
the same resolver map — the interface already allows it), abort signals,
cross-bundle byte caching. (The sketch's "widened to Float64Array" did not
survive: `ArrayData` now spans the typed arrays and loaded elements keep
their stored type.)

### As built

`packages/core/src/arrays/load.ts` (readers), `arrays/spec.ts`
(`ArrayResolver`, `applyPart`, `buildAnyArray`), `bundle/handles.ts`
(`collectHandles`, `loadArrays`), `Bundle.load`. Differences from the sketch:

* `ByteSource.bytes` returns `null` for a missing file (a zarr chunk that was
  never written is all `fill_value`; a 404 is not an error there).
* One addition to the schema: `axes`, a permutation of the kept axes after
  `part` (numpy `transpose`), because the stored order of the prototype's
  volumes is `(z, y, x, c)` and the field wants `(x, y, z)`. This is the axis
  reordering the sketch deferred — it belongs on the handle, not on `dense`,
  since it is a property of how the file was written.
* zarr v3 is read as well as v2 (both are two JSON shapes over the same chunk
  loop); blosc / zstd are refused by name until a decoder is worth its size.
* Bare-path arrays (`bind: { "x": "val/x.npy" }`) work: `collectHandles` knows
  which positions are arrays, and shape inference asks the resolver.
* A failed load is remembered per path and rethrown on use, so `buildAll()`
  attributes it to the fields concerned instead of failing the bundle.

## 2. Sweeps: many members, one document

### The problem

A hyperparameter sweep, or many independently seeded runs, is many bundles
sharing a metadata vocabulary. The structure is **dependent**: an `mlp`
member with `num_layers = 4` has four per-layer spaces / fields that a
`convnet` member lacks, while every member has the global landscape. The
member set is not a Cartesian product of key values.

### The move: encode terms, discover structure

The dependent typing is the **producer's** problem: the exporter knows a
4-layer MLP yields 4 layer spaces and simply emits them. The format
represents *terms*; the UI discovers structure by unifying over the terms it
was given. A schema that tried to say "if `arch = mlp` then …" would be
overengineering and wrong at the first residual branch.

A sweep is **a set of members, each with a flat metadata record**:

```jsonc
{
  "tensatory": "0.2",
  "name": "mnist sweep",
  "keys": {                                   // optional: how to read / display record columns
    "arch":          { "kind": "nominal", "values": ["mlp", "convnet"] },
    "lr":            { "kind": "ordinal", "codomain": "log" },
    "training_seed": { "kind": "nominal" },
    "num_layers":    { "kind": "ordinal" }    // only present on mlp members — fine
  },
  "common": {                                 // a partial BundleSpec merged into every member (member wins, per id)
    "manifolds": { "pca": { "numDims": 3, "dimNames": ["p0", "p1", "p2"] } },
    "fields":    { "sharpness": { "kind": "scalar", "data": { "type": "pointwise", "expr": "…", "vectors": { "g": "lossGrad" } } } }
  },
  "members": {
    "mlp-3-s0": { "record": { "arch": "mlp", "num_layers": 3, "lr": 1e-3, "training_seed": 0 }, "bundle": "mlp-3-s0/bundle.json" },
    "mlp-3-s1": { "record": { "arch": "mlp", "num_layers": 3, "lr": 1e-3, "training_seed": 1 }, "bundle": "mlp-3-s1/bundle.json" },
    "conv-s0":  { "record": { "arch": "convnet", "lr": 3e-4, "training_seed": 0 },              "bundle": { "…": "inline BundleSpec" } }
  }
}
```

A member *is* today's `BundleSpec`, inline or as a path relative to the sweep
document — which plugs straight into the `ByteSource` design: members and
their sidecars are fetched lazily, only when selected. A lone bundle is a
sweep with one member and an empty record, so the viewer loads one document
type.

### Records, not a tree

The instinct is `arch → num_layers → seed` nesting. A tree as the *storage*
shape bakes in one key order, while the interesting navigations cut across it
("same seed, vary arch"). Records are order-free and **every hierarchy is a
derived view**: pick a key order, group. The one thing a tree gives — shared
definitions between siblings — is `common`. If per-arch commons become
necessary, `common` can become a list of `{ "when": { "arch": "mlp" }, … }`
partials without touching anything else; that is the escape hatch, not the
starting point.

The non-Cartesian shape is handled by **faceted navigation**: one control per
key that *varies across the members consistent with the current selection*;
choosing `arch = convnet` makes `num_layers` disappear and restricts `lr` to
values that exist. No dependent-type knowledge anywhere — a filter over rows,
`facets(records, partialSelection)` as a pure core function.

### UI: the precedent is spaces

The spaces work established "fields, point sets, slot selections, view /
camera belong to a space" and "matrix columns absent in a space are hidden
(`SlotDef.present`)". A member is one level above space: the bundle panel
gains a **record row** (a discrete control per varying key, fixed keys as
text) above the `space` picker; everything below lists *this member's* spaces
and fields, unchanged.

What survives a member switch: persisted options are keyed by **structural
signature** — the sorted set of `(fieldId, kind, domain.numDims)` plus space
ids — rather than by member. Two seeds of one architecture have identical
signatures, so slots, iso levels, camera and colormaps carry over verbatim
(flipping through seeds shows the same picture). A different architecture
shares a subset: resolve slot by slot — a field id present in the new member
keeps its slot, one absent goes `NONE`, the rule `SlotDef.present` applies
today. Thus the "type" of a member is its discovered signature, "subtyping" is
set inclusion of field ids, and nobody declares either.

### Per-layer structure inside a member

Two different things hide in "more layers ⇒ more spaces / fields":

* **Per-layer spaces** (a layer's parameter subspace, its own manifold): the
  space picker already handles a member with five manifolds. Nothing to add.
* **Per-layer fields on one space** (layer-wise gradient norms on the global
  plane): they multiply matrix rows. Handle by **convention, not schema**:
  field ids with `/` segments (`layers/2/gradNorm`, as `ArrayPath` already
  does), matrix rows grouped by leading segment, groups collapsible. Add an
  optional `groups` map for display names / ordering only if a real need
  appears.

### Two things that come free

1. **Hyperparameters are 1-D codomains.** `keys.lr.codomain: "log"` reuses
   the legend machinery: log-scaled discrete slider, compact formatting.
2. **A sweep is a sparse sampled field on hyperparameter space.** Records are
   points in a manifold whose dims are the numeric keys; a per-member scalar
   summary (centre loss, final accuracy) gives values at those points — the
   `sparse` data type that has sat unimplemented since phase 1 gets its first
   customer. Not to be built now; it indicates the record model points the
   right way.

### Deferred

* Cross-member references (`pointwise` over `seed0/loss` and `seed1/loss`
  for difference maps, averages over seeds): the field-id string leaves room
  for a `member/field` form; it forces loading several members and extends
  the "identical sample points" rule across members, so wait for the first
  difference view.
* A varying key as an animation axis / small multiples (a ▶ strip like iso
  levels); no format change needed.
* Conditional `common`, as above.

### Cost

Schema: a `SweepSpec` root (`keys`, `common`, `members`),
`MemberSpec = { record, bundle: ArrayPath | BundleSpec }`, a `KeySpec` reusing
`CodomainSpec`. Core: `Sweep.parse`, `member(id) → Promise<Bundle>` (merge
`common`, `Bundle.load` with a `ByteSource` rebased to the member's
directory), `signatureOf(bundle)`, `facets(...)`. Viewer: the record row,
options keyed by `sweep + signature + space`, slot resolution on member
switch. The dependent-type problem never appears in code: the exporter emits
terms and the viewer unifies them.

### Implementation pointers (for whoever builds §2)

Where the seams are today, so the work does not start by rediscovering them:

**Schema / core**

* `BundleSchema` pins `tensatory: z.literal(BUNDLE_VERSION)` (`"0.1"`), so a
  sweep root with `"0.2"` needs its own zod root (`SweepSchema`) and a
  dispatcher that reads the version first; `Bundle.validate(json)` is the
  reusable half. Members that are inline `BundleSpec`s validate with
  `BundleSchema` as they are.
* Loading is `Bundle.load(json, src, { onProgress })` =
  `Bundle.validate` → `loadArrays(spec, src)` → `new Bundle(spec, arrays)`.
  A member by path is `fetch(json)` + `Bundle.load(json,
  rebaseSource(src, dirname(memberPath)))` — `rebaseSource(src, dir)`
  (`core/src/arrays/load.ts`) already exists for exactly this. `ByteSource.bytes`
  returns `null` for a missing file (not an error).
* `common` merging is per top-level record and per id (member wins):
  `manifolds`, `fields`, `pointSets`, `curves`, `nets`, and the scalar
  `name` / `summary` / `details`. Merge BEFORE `collectHandles` /
  `loadArrays`, since `common` may carry handles too (paths then resolve
  relative to the SWEEP document, not the member's directory — decide and
  document; the simplest rule is "a `common` handle path is relative to the
  sweep document" and a rebased source per member for the rest).
* `collectHandles` (`core/src/bundle/handles.ts`) is a typed walk with one
  `switch` per spec kind (fields, nets, curves); it does not need to know
  about sweeps if members are merged into plain `BundleSpec`s first.
* The viewer's rebuild chain is `adjustedBundle(parsed)`: `adjustSpec` →
  `sliceSpec(spec, m, dims, undefined, parsed.arrays)` → `zoomBoxes` →
  `new Bundle(spec, parsed.arrays)`. A member `Bundle` flows through it
  unchanged; the resolver is carried on `Bundle.arrays`.

**Viewer (`apps/viewer/src/main.ts`)**

* `bundleList` is `bundles/index.json` (`{ file, name?, summary? }[]`),
  populated at boot into `pickSel` (`#pickBundle`); `chooseBundle(i)` and
  `pickSel.onchange` call `loadBundle(file)`, which fetches `bundles/<file>`,
  `Bundle.load`s it with `fetchSource(url)` (paths relative to the document
  URL, 404 → null) and calls `setBundle(parsed, file, wantSpace)`.
  `?bundle=<file>` picks at boot. A sweep root should become another kind of
  index entry (or be detected by its version) and `loadBundle` grow a branch;
  member switching is a second `setBundle` with the member's `Bundle`.
* `setBundle` sets `state.bundleFile` / `baseSpec` / `arrays`, reads the
  per-bundle options (`readOpts()` under `optsKey()` =
  `tensatory.opts.<bundleFile>`), rebuilds, `buildAll()`s into
  `buildErrors`, builds the Controls pane (`controls.build(controlRows(...))`),
  binds the bundle ⓘ (`bindInfoIcon($("bundleInfo"), bundle.info)`),
  fills the space picker and calls `setSpace`. `setSpace` builds the curves
  pane, computes `usable` (fields on the space that built), applies
  `loadOpts()` and the slot defaults. Options keyed by STRUCTURAL SIGNATURE
  (the design's point) means replacing `optsKey()`'s `bundleFile` with
  `sweep + signatureOf(bundle)` for members — `Opts` is one JSON object per
  key with `ui` / `ui3` / `maps` / `intervals` / `space` / `spaces[space]`
  (sel, view, dir, res, camera) / `controls` / `boxZoom` / `slice` / `curves`.
* A one-alternative table picker (`picker.ts`) is a plain label; a record row
  (one discrete control per varying key) belongs in the `bundle` panel
  (`#picker`, rows are `.prow`), above the `space` row; the slice row
  (`#sliceRow`, a `.ch.multi` flipper bar) is the closest existing widget.
  Picker rows show `optionText(info)` in their summary column.
* Slot resolution on a member switch: `state.lockedSel` / `state.sel` are
  keyed by slot (`SLOTS`), values are field ids or `NONE`; `isUsable(id)`
  says whether an id exists on the current space. `SlotDef.present` hides
  columns absent in a space.
* Local upload (`#pickFile`, `multiple`) builds a `ByteSource` over the picked
  files by name / `webkitRelativePath`; a sweep picked locally would need its
  member documents among them.

**Tests**

* `core/test/bundles.test.ts` discovers every `bundles/*.json` and every
  `bundles/<dir>/bundle.json`, `Bundle.load`s each with a readFile
  `ByteSource`, builds and samples every field (the `HEAVY` set is built but
  not sampled), and asserts `index.json` mirrors exactly the non-heavy
  documents' `file` / `name` / `summary`. A sweep document in `bundles/`
  will trip both unless the test learns the `"0.2"` root.
* The readFile `ByteSource` (`dirSource`) is copied in `bundles.test.ts`,
  `handles.test.ts`, `mnist.test.ts` and `gpu/test/mnist.test.ts`; a fifth
  copy is the moment to move it into a shared test helper.
* Bundle directories with sidecars: `mnist-convnet-pca/` (512 KB) is a good
  small member; `mnist-mlp/` is heavy and unindexed. A sweep fixture wants
  2–3 tiny members with a shared `common` and a non-Cartesian record set (one
  key absent on some members) so `facets` is exercised.

### Worked example: the loss-landscape trial dataset as a sweep

What the prototype actually produced (`~/projects/loss-landscape`, one run of
`run_all.sh`, September 2025; **one seed**, `--seed 0`; the loop over
`{mnist, fmnist, cifar10} × {mlp, convnet}` only completed for MNIST, and
there is no ResNet or "ReSST" anywhere — just an MLP 784-256-256-10 and a
3-conv ConvNet):

| member | dataset | model | dirs | what exists | box |
|---|---|---|---|---|---|
| `mnist_mlp_pca` | mnist | mlp (269 322 params, test acc 0.976) | pca | 64³ volume (`log10_loss`, `accuracy`; N = 1024), trajectory (48), θ*, `.pt` checkpoint | from the trajectory |
| `mnist_mlp_random` | mnist | mlp | random | 64³ volume, no trajectory | [−1.5, 1.5]³ |
| `mnist_convnet_pca` | mnist | convnet (56 394 params, test acc 0.984) | pca | 40³ volume (N = 384), trajectory, θ*, `.pt` | from the trajectory |
| `mnist_convnet_random` | mnist | convnet | random | 40³ volume | [−1.5, 1.5]³ |
| `fmnist_convnet_{pca,random}` | fmnist | convnet | pca / random | only the 2D `.npy` planes (25 × 25, raw loss); no checkpoint, so the pca plane's box is lost (the random one is [−1, 1]²) | — |
| `mnist_{mlp,convnet}_{pca,random}` 2D | mnist | both | both | 31 × 31 `.npy` planes from `landscape.py` (same θ*, N = 2048) | random [−1, 1]²; pca recomputable from `.pt` |

So "multiple seeds, architectures" is half true: **architectures and direction
kinds vary, seeds do not**. The dataset is a 2 × 2 grid of volumes plus a
ragged fringe (a dataset with planes but no volumes; 2D planes beside 3D
volumes for the same θ*), which is exactly the non-Cartesian shape the record
model was designed for. Generating more seeds is cheap if wanted (MLP: ~2 s
per epoch, a 64³ volume ~2 min on the Apple GPU; `landscape.py --seed k` then
`volume.py --ckpt`) and would make `training_seed` a real key.

**As a sweep document** (`bundles/loss-landscape/sweep.json`):

```jsonc
{
  "tensatory": "0.2",
  "name": "loss-landscape prototype runs",
  "keys": {
    "dataset": { "kind": "nominal", "values": ["mnist", "fmnist"] },
    "model":   { "kind": "nominal", "values": ["mlp", "convnet"] },
    "dirs":    { "kind": "nominal", "values": ["pca", "random"] },
    "seed":    { "kind": "nominal" },
    "test_acc": { "kind": "ordinal", "codomain": "fraction" }      // a per-member summary: a sparse field on the record space
  },
  "common": {
    "fields": {
      "loss": { "kind": "scalar", "codomain": "celoss", "data": { "type": "pointwise", "expr": { "op": "pow", "vals": [10, "l"] }, "scalars": { "l": "log10_loss" } } }
    }
  },
  "members": {
    "mnist_mlp_pca":        { "record": { "dataset": "mnist", "model": "mlp",     "dirs": "pca",    "seed": 0, "test_acc": 0.976, "n_params": 269322 }, "bundle": "mnist-mlp-pca/bundle.json" },
    "mnist_mlp_random":     { "record": { "dataset": "mnist", "model": "mlp",     "dirs": "random", "seed": 0, "test_acc": 0.976, "n_params": 269322 }, "bundle": "mnist-mlp-random/bundle.json" },
    "mnist_convnet_pca":    { "record": { "dataset": "mnist", "model": "convnet", "dirs": "pca",    "seed": 0, "test_acc": 0.984, "n_params": 56394 },  "bundle": "mnist-convnet-pca/bundle.json" },
    "mnist_convnet_random": { "record": { "dataset": "mnist", "model": "convnet", "dirs": "random", "seed": 0, "test_acc": 0.984, "n_params": 56394 },  "bundle": "mnist-convnet-random/bundle.json" },
    "fmnist_convnet_random": { "record": { "dataset": "fmnist", "model": "convnet", "dirs": "random", "seed": 0 }, "bundle": "fmnist-convnet-random/bundle.json" }
  }
}
```

Each member is what `tools/loss-landscape/build.mjs` already emits (a
manifold `pca` or `random`, dense `log10_loss` / `accuracy` handles into the
prototype's `.bin`, the trajectory curve and θ* for pca members) — the tool
gains a `--member` mode that writes the record beside it, and a 2D variant
that wraps a `.npy` plane. `common` holds what every member repeats
(`loss` = 10^log10_loss; the codomains). The **structural signatures**: the
four volume members share `{log10_loss, accuracy, loss}` on a 3D space, so
slots, iso levels, camera and colormaps carry over verbatim when flipping
`model` or `dirs`; the pca members additionally have the `trajectory` curve
and θ* (a superset — the curve row appears / disappears, nothing else
changes); the fmnist member is a 2D space, so the arm switches and only the
2D options apply.

**UI for this example**: the bundle panel gains a **record row** above
`space`: `dataset` [mnist | fmnist], `model` [mlp | convnet], `dirs`
[pca | random] as three choice flippers (the `#sliceRow` bar is the widget),
`seed 0` and `test acc 0.976 · 269k params` as plain text since they do not
vary (or vary only with the others). Faceting: picking `fmnist` greys `mlp`
and `pca` (no such members); picking `mlp` greys `fmnist`. Switching a
member is a `setBundle` of that member's `Bundle` with options loaded under
`sweep + signature`; the space picker shows the member's single space as a
label (a one-alternative table picker). The sweep's own ⓘ carries the
description above. The
natural first "cross-member" view is not a difference map (deferred) but the
one the record already affords: flipping `dirs` between pca and random for
the same θ* with the camera held — the paper's two views of one minimum,
side by side in time.

The local-storage key for member options becomes
`tensatory.opts.<sweepFile>#<signature>`; a lone bundle keeps
`tensatory.opts.<file>`.

### Rerunning the collection: what the script should produce for a sweep

`volume.py` was written for a viewer without codomains, handles or curves;
rerunning it (cheap: MLP ~2 s / epoch, a 64³ volume ~2 min on the Apple GPU)
is the chance to make the members first-class. In order of value:

1. **Raw `loss`, not `log10_loss`.** The log was the prototype viewer's
   substitute for a codomain; here `celoss` (log e, flipped, nats) does it as
   a display hint, values stay exact, and the `pointwise` `10^x` field goes
   away. Store `loss` and `accuracy` (float32).
2. **Self-describing arrays, one per channel**: an `.npz` with `loss [nx, ny,
   nz]`, `accuracy [nx, ny, nz]` in x-slowest order (plain numpy indexing,
   `arr[ix, iy, iz]`), instead of the headerless interleaved `.bin` that
   needs `shape`, `part`, `axes` and `dtype` in every handle. Then a handle
   is `{ "type": "handle", "path": "arrays.npz/loss" }`.
3. **Emit the Tensatory bundle from Python**, member + record, instead of
   converting afterwards (`tools/loss-landscape/build.mjs` becomes
   unnecessary): `bundle.json` with `summary` / `details`, the manifold
   (`dimWeights` = explained variance for pca), the fields with codomains,
   θ* as a point set, the trajectory as a `sampled` curve with `param.name =
   "snapshot"`, and beside it `record.json` = `{ dataset, model, dirs, seed,
   lr, epochs, batch_size, momentum, weight_decay, n_params, test_acc,
   final_train_loss, eval_n, grid, explained_variance?, wall_s }` for the
   sweep document to gather. One source of truth for the metadata. For MLP
   members also emit the **`net` def** (the script has the `nn.Module` in
   hand; a `.pt` state_dict is only named tensors, the architecture and
   nonlinearities are not in it — `tools/mnist/export.py` writes the def by
   hand today) and assert that def reproduces the model's own loss at θ*.
   Nothing reads `.pt` in TypeScript and nothing should: export `.npz` (or
   safetensors) from Python.
4. **Save the directions**, always. The `random` volumes' directions were
   never written, so those volumes cannot be related to parameter space at
   all; pca directions are recomputable only while the `.pt` exists. Write
   `dirs/d0`, `dirs/d1`, `dirs/d2` (per layer, as `tools/mnist/export.py`
   does) and `theta/*` into the member's `.npz`. With them an MLP member can
   carry live `net` fields later (the cooperative kernel), and the affine
   frame of roadmap 4 has its data.
5. **Project the trajectory onto the random directions too**, and record
   the loss / accuracy AT each snapshot (48 forward passes: nothing):
   `traj/coords [T, 3]`, `traj/loss [T]`, `traj/acc [T]`, `traj/step [T]`.
   That is the `along` field of curves.md ("loss along the SGD trajectory")
   with the data already there, and `times` for the curve in real steps.
6. **Directional gradients as a channel.** `torch.func.vmap` over
   `grad(loss)` gives ∂L/∂a, ∂L/∂b, ∂L/∂c at every grid point for about the
   cost of a second forward pass: store `grad [nx, ny, nz, 3]` and declare
   it as the loss field's `exactGradient` (a `densev` field). Streamlines and
   glyphs then follow the true gradient instead of finite differences of the
   grid, which is what the prototype's `grad_*` channels were meant to be
   and never were.
7. **Seeds**: `--seed k` for k in 0..2 at least, so `training_seed` is a
   real key and the record set is not a 2 × 2 grid. Keep the eval subset
   FIXED across seeds and models (it already is: `randperm(60000, seed 0)`)
   so members are comparable; consider `eval_n` = 512 for both models so a
   signature-sharing pair also shares its sample.
8. **One grid size per dimension for all members** (48³ is a good middle:
   MLP 64³ was 2 MB per channel, ConvNet 40³ was compute-bound). Same box
   convention (random: [−1.5, 1.5]³ in filter-normalized units; pca: the
   trajectory's span padded 20 %, third axis ≥ 35 % of the longest).
9. **Drop the separate 2D planes.** A pca plane is the z = 0 slice of the pca
   volume (its directions are the first two of the top-3), a random plane a
   slice of the random volume; the viewer crops. Fewer members, no
   lost-box problem (the fmnist planes without a checkpoint).
10. **Fill or keep the fringe deliberately.** fmnist for both models makes
    `dataset` a real key; cifar10 (the loop's third dataset, never run) is a
    natural "convnet only" fringe that keeps the record set non-Cartesian —
    which the faceted UI should be seen handling.

Not worth doing: storing anything the viewer computes (isolines, stats
beyond `extrema`), or float64 volumes (f32 is what the GPU reads).

### As built

`schema/sweep.ts` (`SweepSpec`, `MemberSpec`, `KeySpec`, `CommonSpec`,
`SWEEP_VERSION = "0.2"`, `RootSpec = BundleSpec | SweepSpec`);
`packages/core/src/bundle/sweep.ts` (`SweepSchema`, `rootKind`,
`mergeCommon`, `facets` / `nearestMember` / `membersWhere`, `signatureOf` /
`shortHash`, class `Sweep`); `apps/viewer/src/recordPane.ts` and the
`loadBundle` / `setMember` / `setBundle` chain in `main.ts`;
`tools/loss-landscape/build.mjs` (the trial dataset) and
`tools/sweep-demo/build.mjs` (a six-member symbolic sweep for the UI);
`packages/core/test/sweeps.test.ts`. Differences from the design above:

* **No handles in `common`** — `Sweep.validate` refuses them (`SpecError`
  under `common`), so the "relative to which document" question never
  arises. Members by path resolve their sidecars relative to THEIR document
  (`rebaseSource`); inline members relative to the sweep document.
* **`keys.<k>.attribute`**: a key can be a per-member MEASUREMENT (test
  accuracy, parameter count, evaluation-set size) rather than a coordinate of
  the sweep. Attributes are shown with the record, never become a flipper and
  never count when records are compared — without this, `model → mlp` would
  never be a "direct" switch because `n_params` differs too. This is the
  "sparse field on the record space" of the design, as metadata for now.
* **Faceting is relative to the current member, and every existing value is
  reachable.** A value is `direct` when some member carrying it agrees with
  the current member on every other (non-attribute) key they share; it is
  shown tinted. A value no member differs only in is still clickable, shown
  plain, and jumps to the NEAREST member (`nearestMember`: the fewest other
  keys changed, then the most shared keys, then sweep order) with the status
  line saying what else changed (`dirs → random, model → convnet`). A value
  listed in `keys.values` that no member has is disabled. Keys absent from
  the current member's record are hidden (the convnet has no `num_layers`).
  Ordinal keys (declared, or every value a number) are sorted; nominal keys
  keep `keys.values` order then discovery order.
* **The structural signature is read off the spec**, not the built bundle:
  sorted `space <id>:<dims>` + `<kind> <fieldId>:<dims>`; curves and point
  sets are excluded on purpose (a trajectory present in one member only must
  not reset the camera). Options live under
  `tensatory.opts.<sweepFile>#<shortHash(signature)>`; the last member under
  `tensatory.member.<sweepFile>`; `?member=<id>` in the URL. Note the
  signature INCLUDES space ids, so `pca` and `random` members of the trial
  dataset do NOT share it (their manifolds differ, as do their boxes and
  scales); what carries across a member switch into a signature with no
  saved options for the space is the CARRY: slot ids that exist in the new
  member keep their slot, the 2D view / 3D camera carry over for the same
  space, and into a different 3D space the camera holds its ORIENTATION and
  re-fits its distance to the new box (`View3D.holdOrientation`) — "the same
  minimum in two frames" without a 40× box mismatch.
* `Sweep.single(bundle)` wraps a lone bundle as a one-member sweep
  (`SINGLE_MEMBER`); the viewer does not use it yet (a lone bundle keeps
  `tensatory.opts.<file>` and no record rows), it is there for the day the
  viewer holds one document type.
* zod's union errors are descended (`specErrorOf` in `core/src/errors.ts`),
  so an inline member missing `fields` says so instead of "Invalid input".
* The trial dataset: `bundles/loss-landscape/sweep.json` (indexed) lists the
  two ConvNet members (512 KB each, committed); `sweep-all.json` (gitignored,
  with the `mnist-mlp-*/` directories, 2 MB volumes) lists all four and loads
  through `?bundle=loss-landscape/sweep-all.json` — an unindexed `?bundle=`
  joins the picker for the session. `mnist-convnet-pca/` moved into the
  sweep. `record.json` sits beside each member's `bundle.json` as the
  collection script of the wishlist above would write it.
* `bundles.test.ts` discovers `<dir>/sweep.json` roots and builds every
  member; the members' own `bundle.json`s in subdirectories are not
  discovered as documents. `dirSource` lives in `test/helpers.ts`.

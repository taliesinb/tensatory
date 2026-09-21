# External arrays and sweeps

Two related designs for the bundle format. §1 (external arrays) is **built**
as designed here, with one addition found on contact with the data (`axes`,
see the end of §1); the current description is in
[bundle-schema.md](bundle-schema.md) "External arrays". §2 (sweeps) is not
built. Ordering against other work: [roadmap.md](roadmap.md).

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

# Roadmap

Rough order; each item is independent enough to be picked up alone. Where the
bundle format is concerned, the state of play is: `schema/` still describes
the full phase‑1 design and the runtime implements exactly its inline subset
(see [bundle-schema.md](bundle-schema.md)); `handle`, `sparse` / `sparsev`,
array-backed points and `mappings.ts` parse (or are commented out) but build
to `NotSupportedError`. Nothing outside JSON has ever been loaded.

## Bundle format

1. **Raw `.bin` array backend** (`handle` specs) — the first real storage
   backend and the first edit to `schema/arrays.ts` since phase 1. Design in
   [sweeps.md §1](sweeps.md): `part` gains `null` ("keep this axis") so a
   handle can address a channel of the loss-landscape's channel-interleaved
   `(N, C)` arrays, `shape` becomes the stored shape, `dtype` is added;
   bytes come from an environment-provided `ByteSource`, formats are decoded
   in core, arrays are prefetched so the build stays synchronous (a bundle
   becomes a JSON + sidecar directory). Then the existing loss-landscape
   volumes (`~/projects/loss-landscape/viewer/data/*.json + .bin`) become real
   bundles: manifold `params` (n_params) + manifold `pca` (3D, `dimWeights` =
   explained variance), `log10_loss` / `accuracy` dense fields with `celoss` /
   `fraction` codomains, `grad_*` as the `exactGradient`, the trajectory and
   θ* as point sets (the affine injection between the manifolds waits for 4).
2. **npz / npy / zarr backends** (same `handle` vocabulary and `ByteSource`;
   `path` semantics per store are already sketched in `schema/arrays.ts`),
   **sparse supports** (`sparse` / `sparsev`; needs a sampled-data
   representation that is not a grid). Array-backed `PointSpec` /
   `VectorSpec` fall out of 1.
3. **Sweeps** ([sweeps.md §2](sweeps.md)): a `SweepSpec` root holding
   members (each a `BundleSpec`, inline or by path) with flat metadata
   records and a merged `common` partial; faceted navigation over the
   records in the bundle panel, options keyed by structural signature so a
   view survives switching seeds. Depends on 1 for lazily fetched members.
4. **Charts / affine frames** (`schema/mappings.ts`, currently commented
   out): fields on a low-dimensional frame inside a high-dimensional
   parameter manifold; 1-forms vs vectors under pullback.
5. **Nets** ([nets.md](nets.md)): the schema is implemented end to end —
   shape inference, CPU reference evaluator, WebGPU transpiler, autodiff
   (`grad` as a program rewrite; exact derivatives of net fields), the iris
   example checked against PyTorch, CPU / GPU agreement. Next: streaming the
   dataset axis in the transpiler so nets larger than function-scope memory
   run on the GPU, elementwise fusion in the emitter, and an MLP-on-MNIST
   bundle (needs 1 for the weights and validation set).
6. **Server-side computation**: a Tensatory server that materializes fields on
   demand (e.g. a Torch script sampling a new grid), with the same
   `FieldDataSpec` vocabulary; and client-side computation via third-party JS
   (WebGPU training).
7. **Schema housekeeping** when 1 lands: drop `CellSpec` (a `part` fixing
   every axis is a cell), drop or use `VectorStatistics` and the phantom
   `ArraySpec<_N>` parameter, decide whether `stats.quantiles` / `histogram`
   (parsed, never read) stay. The `part` change is backward compatible, so
   no version bump; the sweep root is `"0.2"`.

## Geometry and rendering

8. **Interval-arithmetic quadtree seeding** for exact isolines — an interval
   evaluator over the expression tree, prune cells whose enclosure excludes
   the level, subdivide the rest; removes the seed-grid topology limitation
   ([isolines.md](isolines.md)). Generalizes to an octree for 3D.
9. **3D, second round** ([3d.md](3d.md) "Next"): fused surface smoothing
   (welded connectivity on the device — Taubin is CPU-only today), temporal
   supersampling, depth-tested points, per-level colouring through the I_V
   colormap when no I_C is set, colouring the face outlines by I_C.
10. **GPU backend** ([gpu.md](gpu.md) "Next stages"): `timestamp-query` for
   real GPU frame times in the adaptive resolution instead of vsync-quantized
   rAF intervals; reuse same-sized resident buffers across frames (a moving
   crop allocates and frees its grids every frame).
11. **Glyph rescaling** ([glyphs.md](glyphs.md)): the arrows are normalized
   linearly against the longest vector sampled, which leaves most of a
   heavy-tailed field (gradient norms spanning orders of magnitude) as dots —
   add a `scale` choice (linear / log / rank or quantile), and perhaps a
   "comet" glyph (width tapering along the shaft, which the line pipelines'
   particle ramp already provides).
12. **CPU path**: a Web Worker for contouring / integration in `compute = cpu`
   mode; `diff` emitting shared references rather than copies (smaller trees
   before CSE); compiling value + gradient as one function for projection
   ([performance.md](performance.md)).

## Done since phase 1

Kept here so the list above reads against what exists: spaces and the 3D arm
(marching tetrahedra CPU + GPU, exact ∇f projection, crop with face outlines,
3D streamlines, CPU Taubin, WebGPU renderer with weighted-blended OIT,
depth-tested lines), static vector-field glyphs on hex / FCC lattices in both
arms, the full GPU path (WGSL transpiler, fused kernels,
drawIndirect, statistics / blur / smoothing passes, `compute` × `render`
modes), adaptive resolution with a memory cap, colormap interval selections
on every legend bar. None of it required a change to the bundle format.

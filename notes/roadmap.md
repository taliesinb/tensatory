# Roadmap

Rough order; each item is independent enough to be picked up alone. Where the
bundle format is concerned, the state of play is: `schema/` describes the
full design; the runtime implements the inline arrays, the `handle` array
backend (`.bin` / `.npy` / `.npz` / zarr v2 + v3, see
[bundle-schema.md](bundle-schema.md) "External arrays"), array-backed points
and vectors; `sparse` / `sparsev` and `mappings.ts` parse (or are commented
out) but build to `NotSupportedError`.

## Bundle format

1. ~~Raw `.bin` array backend~~ and ~~npz / npy / zarr backends~~ — done
   ([sweeps.md §1](sweeps.md) as designed, plus `axes`: the prototype's
   volumes turned out to be stored x-fastest, `(z, y, x, c)`, so a handle
   also permutes its kept axes). `apps/viewer/public/bundles/loss-landscape/mnist-convnet-pca/`
   (`tools/loss-landscape/build.mjs`, now a sweep member) is the first bundle
   directory: manifold `pca` (3D, `dimWeights` = explained variance),
   `log10_loss` / `accuracy` as channel handles into the prototype's own
   `.bin`, the sweep's common pointwise `loss` (`celoss`), trajectory curve
   and θ*. Left from the original plan: the `params` manifold and the affine
   injection into it (waits for 4); the MLP volumes (2 MB each) are built
   into the sweep but not committed. Loader follow-ups: blosc / zstd codecs
   (zarr-python's defaults; a WASM decoder or a `zarr.js`-style dependency),
   lazy per-field loading (`Bundle.prepare(fieldIds)` filling the same
   resolver), abort signals, cross-bundle byte caching, zip64. (Stored
   element types are kept — `ArrayData` covers the JS typed arrays — so no
   widening on load; the CPU compute type stays f64 on purpose: it is the
   reference the f32 GPU is tested against.)
2. **Sparse supports** (`sparse` / `sparsev`; needs a sampled-data
   representation that is not a grid). Their `points` / `samples` arrays
   already load.
2b. ~~Curves~~ — built ([curves.md](curves.md)): `curves` in the bundle,
   symbolic / sampled / flow / pushforward data, the viewer's curves panel
   (per-curve toggle and interval slider), the bundles migrated off
   `pointSets.ordered`. Left: `along` / `velocity` as field data on the
   curve's parameter interval (loss along the SGD trajectory, ⟨∇f, γ'⟩),
   slicing of non-sampled curves, adaptive stepping for `flow`, a t
   scrubber with the cursor pane reading the fields at γ(t), a 1-D arm.
3. ~~Sweeps~~ — built ([sweeps.md §2](sweeps.md) "As built"): the `"0.2"`
   root with members (inline or by path, fetched on selection) and flat
   records, `common` merged per id (no handles in it), `keys` with
   `attribute` for per-member measurements; the viewer's record rows
   (faceting relative to the current member, nearest-member jumps), options
   keyed by sweep + structural signature, slots / view / camera orientation
   carried across a switch. `bundles/loss-landscape/` (the prototype's
   volumes; the MLP pair local only) and `bundles/sweep-demo/`. Left: a
   varying key as an animation axis / small multiples, cross-member fields
   (difference maps, seed averages), conditional `common`, a sparse field
   over the record space from the attributes, the collection-script rerun
   (sweeps.md "Rerunning the collection").
4. **Charts / affine frames** (`schema/mappings.ts`, currently commented
   out): fields on a low-dimensional frame inside a high-dimensional
   parameter manifold; 1-forms vs vectors under pullback.
5. **Nets** ([nets.md](nets.md)): the schema is implemented end to end —
   shape inference, CPU reference evaluator, WebGPU transpiler, autodiff
   (`grad` as a program rewrite; exact derivatives of net fields), the iris
   example checked against PyTorch, CPU / GPU agreement; streaming of the
   dataset axis, elementwise fusion and lazy arrays in the emitter. The
   **MNIST MLP bundle** exists (`bundles/mnist-mlp/`, `tools/mnist/export.py`:
   269k weights and the eval set as `.npz` members, CPU-checked against
   PyTorch) but is not indexed: one GPU lane per point takes ~10 s per
   dispatch at any grid size (nets.md "the MNIST MLP experiment"), so the
   work budget leaves it to the CPU path, which is two minutes per rung.
   Next: a **cooperative kernel** — a workgroup per grid point splitting the
   matmuls, writing a resident values grid — then hoisting the
   point-independent first layer, then index the bundle.
6. **Server-side computation**: a Tensatory server that materializes fields on
   demand (e.g. a Torch script sampling a new grid), with the same
   `FieldDataSpec` vocabulary; and client-side computation via third-party JS
   (WebGPU training).
7. **Schema housekeeping**: `CellSpec` is gone (a `part` fixing every axis
   is a cell); still open: drop or use `VectorStatistics` and the phantom
   `ArraySpec<_N>` parameter, decide whether `stats.quantiles` / `histogram`
   (parsed and, as handles, loaded — but never read) stay. The `part` /
   `axes` / `dtype` additions were backward compatible, so no version bump;
   the sweep root is `"0.2"` (built).

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

# Roadmap

Rough order; each item is independent enough to be picked up alone.

1. **Raw `.bin` array backend** (`handle` specs) — trivial (`float32`, known
   shape) and it turns the existing loss-landscape volumes
   (`~/projects/loss-landscape/viewer/data/*.json + .bin`) into real bundles:
   manifold `params` (n_params) + manifold `pca` (3D, `dimWeights` = explained
   variance), an affine injection between them, `log10_loss` / `accuracy`
   dense fields with `celoss` / `fraction` codomains, `grad_*` as the
   `exactGradient`, the trajectory and θ* as point sets. Channel-interleaved
   `(N, C)` storage needs a slice/axis notion in `part`.
2. **Interval-arithmetic quadtree seeding** for exact isolines — an interval
   evaluator over the expression tree, prune cells whose enclosure excludes
   the level, subdivide the rest; removes the seed-grid topology limitation.
   Generalizes to an octree for 3D.
3. **3D** — done in its first form (`notes/3d.md`): spaces, marching
   tetrahedra on CPU and GPU, exact-gradient normals, WebGPU renderer with
   weighted-blended OIT. Still to do: crop planes, isolines on the box faces,
   3D streamlines, vertex projection onto the isosurface, Taubin smoothing
   (needs welded connectivity), depth-tested box / trajectories, per-level
   colouring through the I_V colormap when no I_C is set.
4. **npz / npy / zarr backends** and sparse supports.
5. **Charts / affine frames** (`schema/mappings.ts`): fields on a
   low-dimensional frame inside a high-dimensional parameter manifold; 1-forms
   vs vectors under pullback.
6. **Server-side computation**: a Tensatory server that materializes fields on
   demand (e.g. a Torch script sampling a new grid), with the same
   `FieldDataSpec` vocabulary; and client-side computation via third-party JS
   (WebGPU training).
7. **GPU path** (stage 1 done: `packages/gpu`, see [gpu.md](gpu.md)):
   next, contouring and integration as compute passes, then a WebGPU
   renderer.
8. **Workers** for contouring and integration; **CSE improvements** in
   `diff`.

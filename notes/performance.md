# Performance

Everything is CPU JavaScript so far: grids, samples and streamlines are packed
`Float64Array`s (row-major, numpy layout); rendering is Canvas 2D (`ImageData`
raster, `Path2D` strokes). No WebGL / WebGPU, textures or compute shaders.
The expression language is closed precisely so it can be transpiled to WGSL
later; see [roadmap.md](roadmap.md).

## The case study: `|∇ mixture|`

Selecting the derived field `|∇ mixture|` (`norm(grad m)` of a 3-Gaussian
mixture) as C / I_V / S_∇ needs its gradient — a second derivative of the
mixture through the argument. Measured with `PERF=1 pnpm test`
(`packages/core/test/perf.test.ts`), 128² grid, viewer-like settings:

| | before | after |
|---|---|---|
| evaluate `\|∇m\|` at a point | 2 µs | 1.1 µs |
| evaluate ∇`\|∇m\|` | 29 µs | 9.4 µs |
| exact isolines, 4 levels (one frame) | 370 ms | 111 ms (exact) / a few ms (rough) |
| streamlines 1k lines × 100 steps | 15 s | 0.4 s incl. sampling, in the viewer |

## What was done

1. **CSE in the compiler** (`symbolic/compile.ts`). Differentiation produces
   trees with massive repetition (31 → 2 616 → 26 334 nodes for value, first
   and second derivative). Subtrees are hash-consed by structural key and
   compiled once; shared subtrees get a per-point memo (last coordinates +
   pos → value). Exact, safe through nested argument evaluation.
2. **Sample-then-integrate streamlines** (`main.ts` `integrableVector`). A
   symbolic vector field is sampled once onto a fixed grid (128 in 2D, 64 in
   3D; cached) and streamlines integrate through the bilinear interpolant, as the
   3D prototype did: O(grid) instead of O(lines × steps × 4) evaluations.
3. **Progressive isolines**. While the level is moving (animation, or within
   200 ms of a slider change) symbolic fields are contoured with plain
   marching squares; once settled, a frame swaps in the exact projected lines.
4. **Adaptive resolution** ([resolution.md](resolution.md)): the grid is a
   feedback loop with a `moving` tier holding 30 fps while levels animate and
   a `settled` tier bounded by one recomputation's latency and the memory
   cap; resident sets are sized from measured triangle / segment counts.
5. Smaller: `derivative(dim)` cached per field datum; isoline results cached
   by (field, grid, levels, tolerance, colour); sampled values cached per
   (use, grid); streamline sets LRU-cached by (field, count, steps, sign,
   step, box) with colours recomputed cheaply.

## Where time still goes

* Exact contouring at rest: ~30 ms per level for derivative-heavy fields
  (Newton projection of ~1.6k vertices × (fn + 2 partials)). Fine at rest;
  the rough-while-moving rule hides it during animation.
* Sampling derived fields on 128²: 20–150 ms, cached per grid.
* Rendering 10k streamlines: ~20 ms/frame (binned `Path2D` strokes).

## Headroom

* Make `diff` emit shared references rather than copies (smaller trees before
  CSE), and compile value + gradient as one function for projection.
* A Web Worker for contouring / integration to keep the main thread free.
* GPU: transpile expressions to WGSL; sample, raster and (in 3D) march on the
  GPU; upload grids as textures / storage buffers.

## Nets

See [gpu.md](gpu.md) stage 8. Two viewer-side lessons from the iris bundle:
a fused kernel's cache key must contain only what changes its CODE (the
colormap selection was in the isoline key and compiled a shader per drag
event), and exact projection of a net-backed field is latency-bound (~40 ms
per level however few vertices) — so the fused path shows marching squares
while the level moves and projects once it settles. `window.__tensatory.frames`
logs the last 60 frames' JS ms / dispatches / pipelines built for this kind of
investigation; `?isoexact=0` disables 2D exact isolines for comparison.


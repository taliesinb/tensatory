# Adaptive resolution and the memory cap

The grid the fields are sampled on — 2D: raster, isoline seeds; 3D: the
isosurface cube — is not a control. `apps/viewer/src/autores.ts` picks it by
feedback, one controller per arm (`autoRes2`, `autoRes3` in `main.ts`), over
a ladder of resolutions (×√2 per step, ≈ ×2.8 cells in 3D):

* 2D: 32 · 48 · 64 · 96 · 128 · 192 · 256 · 384 · 512 · 768 · 1024 · 1536 · 2048
* 3D: 16 · 24 · 32 · 48 · 64 · 96 · 128 · 192 · 256

## Two tiers

* **moving** — used while the picture is recomputed every frame: the ▶
  animation, dragging `value` / `split`, panning / zooming in 2D (the 2D grid
  follows the view box) and dragging a crop range in 3D (the volume grid
  follows the cropped box). Bounded by the frame budget: 30 fps
  (`FRAME_BUDGET_MS = 33.4`, stepping up only when the next step is predicted
  under 26 ms).
* **settled** — used once the levels stop (`ISO_SETTLE_MS`, the same 200 ms
  the 2D isolines wait before swapping rough lines for exact ones). Bounded by
  the latency of ONE recomputation (`SETTLED_BUDGET_MS = 200`; a step is
  undone above 1.5× that), the memory cap and the ladder.

Both start at the bottom of the ladder (3D 16, 2D 64) for a new space and
ramp one step per measured computation; the last good pair is remembered per
bundle + space (`spaces[id].res` in `tensatory.opts.<file>`) so a reload
starts there. `moving ≤ settled` always; a moving step up carries the settled
tier along (a budget met every frame means one recomputation is fine), and
until the moving tier has a window of its own samples it sits two steps
below the settled tier (≈ one recomputation's cost spread over six frames).
`?res=N` / `?res3=N` pin a resolution and switch the loop off.

The current grid and the segments / triangles drawn are the `iso res` row of
the `system` panel (closed by default: end users rarely need it); the
`memory` row there shows `used/cap` in MB and the loop's last decision in a
terse form so it fits one line: `mov` / `set` = the tier, `↑` / `↓` a step
with its reason (`~41ms` predicted, `44ms` measured, `probe`, `draw 40ms`,
`mem cap`), `set@192 →256 ~1016MB` = holding at 192, the next step's
predicted cost.

Whenever the loop acts, a glyph flashes in the top-right corner of the
viewport (`#resFlash`: ↑ / ↓ a tier step, ⟳ a remeasure; its tooltip is the
decision) so a feedback adjustment can be told from any other stutter — an
orbit at rest draws resident geometry and never triggers one.

## Measurement

`frame()` in `main.ts` reports every rendered frame to the controller at the
NEXT rAF: `ms` = the rAF interval after the frame (it includes the GPU stall:
the compositor cannot present until the queued compute and render passes are
done) or the frame's JS time, whichever is larger; `recomputed` = the frame's
key (grid, levels, slots, options, view box, crop) changed or the backend
counted dispatches (`GpuBackend.dispatches`); `compiled` = a compute
pipeline was built (`pipelinesBuilt`) — such a frame is not representative,
and for a settled recomputation the controller asks for a **remeasure**
(`onRemeasure` → `fused.redo()` / `view3d.redo()`: the same geometry is
dispatched again without compiles and timed cleanly) — unless the frame's
JS time (`jsMs`) alone already exceeds the settled budget: a compile stalls
the GPU, not the main thread, so a long JS time is real CPU work (a `costly`
net field sampled on the dispatch grid, whose fallback reader bakes its grid
and so compiles at every step) and the sample is judged as slow at once,
without the remeasure that would only hit the caches.

Frame times are quantized by vsync (17 / 33 / 50 ms), so a 60 fps window
says nothing about headroom. The moving tier therefore **probes**: one step
up, validated by the next window (the 75th percentile of 8 frames, so a
quarter of the frames may hiccup); a miss steps back and remembers the failed
step for that context (fields, levels, options — `resCtx()`) for 60 s. When
the window is above one vsync the next step is extrapolated by (n′/n)ᴰ
instead. Render-only frames (orbiting, particles) that miss the budget step
the displayed tier down too — triangle count is a render cost.

## Sets sized from measured complexity

Meshes and segment sets are appended by fused kernels through an atomic
counter. The kernels now count every record even when the set is full (the
capacity travels in the params buffer, `bitcast<u32>`, and is the set's real
size), so the indirect buffer holds the TRUE count. After every dispatch the
count is read back (`GpuBackend.readCounter`, 16 bytes, one in flight per set;
`Counted<S>` in `view3d.ts`, `gpuFused.ts`):

* `count > capacity` is a detected overflow: the set is reallocated for the
  true count and dispatched again next frame (`invalidate()`).
* per **family** (field, colour, options — not the grid) a `Complexity`
  record keeps the running maximum with a 60 s half-life decay, so a cycle of
  animated levels keeps its peak; a new set is sized `records × (n′/n)ᵉ ×
  1.5` (e = 2 for surfaces, 1 for isolines; floor 16k records, ceiling the
  kernel's worst case and the device's binding limit).

This replaced the worst-case allocations (`cells × 8` segments in 2D — 1.3 GB
per level at 2048²; `clamp(cells / 2, 64k, 1M)` triangles in 3D) with what
the field actually produces (3 MB for four exact isolines at 2048²; the gyroid
at 192³ really needs ~1M triangles per level).

## Memory

`GpuBackend.createBuffer` accounts every resident allocation in
`bytesAllocated` (meshes, segment sets, grids, smoothing scratch, `keep`
outputs); the viewer's caches (`Cache` in `cache.ts`: LRU with byte
accounting and a per-frame touch stamp) add their JS arrays (sampler results,
3D CPU value grids). `memoryNow()` = device bytes + JS bytes; the
`system` panel shows it live next to the controller's last decision.

The cap (`mem cap` flipper in the system panel: 256 / 512 / 1024 / 2048 MB,
default 1024; `tensatory.memcap`, `?memcap=`) acts three ways:

1. after every frame `governMemory` trims the caches to 90 % of it, least
   recently used first, never an entry the frame just used (`Cache.trim`);
2. a resolution step whose predicted working set exceeds 85 % of it is not
   taken (the working set = the frame's live entries, split into ∝ nᴰ grids
   and ∝ nᴰ⁻¹ meshes / lines for extrapolation);
3. a frame whose working set alone exceeds the cap steps its tier down.

Kernels are keyed by the identity of the grid they read (`uidOf(values)`),
since a trimmed grid that is rebuilt gets a new buffer and the kernel holding
the old one would dispatch against a destroyed buffer.

## Device limits met on the way

* `maxComputeWorkgroupsPerDimension` = 65535: one dimension of 64-wide groups
  covers 4.19 M invocations — a 2048² grid is one workgroup over, any 3D grid
  past 160³ far over, and the dispatch fails validation (nothing computed,
  and before this the frame *looked* fast). `GpuBackend` lays every dispatch
  out in two dimensions and rewrites the entry point to rebuild the linear
  index (`linearize`), so kernels keep their 1D `id.x`.
* `maxStorageBufferBindingSize` defaults to 128 MB; a gyroid level at 192³ is
  142 MB. The device is requested with the adapter's limits (up to 2 GB), and
  set capacities are also clamped to `maxBufferBytes`.
* **No kernel bakes its grid any more.** A `DenseGrid` travels as a 20-float
  header (`GRID_FLOATS`, `packGrid` / `gridWgsl` / `bakedGrid` in `wgsl.ts`;
  ints as bits) in the params or data buffer: the sampling programs
  (`ProgramBuilder` — header at the start of the packed data, so the
  index → point map and the fallback reader are grid-free; a dense field's own
  support stays baked, it is intrinsic to the field), the marching-squares
  kernels (`marchingSquaresWgsl(GridRef)`: fused isolines, read-back isolines,
  Taubin passes), blur, plane sampler / slicer, the stats reduction and the
  mesh kernel. One pipeline per (field, options) serves every grid, so a
  resolution step, a 2D pan or a 3D crop drag compiles nothing: measured 0
  compiles / frame and 60 fps while sliding a crop band that rebuilds a 96³
  volume, its meshes and six face grids every frame (it was 5 compiles and
  185 ms / frame with baked grids). One consequence of runtime grid values:
  `edgeOf` in the Taubin seed pass classifies a vertex by the nearest cell
  side instead of an exact `y0 + hy` comparison, since fast-math may contract
  that expression differently in two places.

## Streamlines

The grid streamlines are measured in (step = ½ cell, `length` in steps,
`tail` in cells) is FIXED — 128 in 2D (`STREAM_N`), 64 in 3D
(`View3D.STREAM_N`); 32 / 16 for `costly` fields (net-backed, CPU-evaluated;
see [nets.md](nets.md)) — or the vector field's own grid: tying it to the
adaptive resolution would change the lines' lengths with the tier.

## Measuring a settled recompute (GPU completion)

Frame times come from rAF intervals, which works for the moving tier (a
sustained rate) but not for the settled tier's single recompute sample:
WebGPU submits are asynchronous and the browser lets a backlog build for a
frame or two before it blocks presenting, so a 5 s remesh was measured as the
10 ms until the next rAF and the stall landed on a later, render-only frame
the controller ignores — 256³ nets were "accepted" and every pause froze for
seconds. A settled-tier recompute is now timed by `onSubmittedWorkDone`
(`pendingFrame.awaitingGpu` in main.ts; frames rendered meanwhile are not
samples). Consequences: the first sample at a rung is often the net's
one-time CACHE FILL of that values grid (0.6 / 2.2 / 5.9 s at 128 / 192 /
256³ for iris), which is exactly what the remeasure-first policy is for — the
remeasure hits the cache and measures the remesh alone (~40–70 ms at 256³ in
Chrome); a confirmed failure descends to the rung the sample PREDICTS meets
the budget (cost ∝ n^dims) instead of one rung per multi-second recompute.
Render-only frames carrying progressive-recolour batches are not reported
(`Recolour.busy`: they once read as slow draws); recomputed frames always are.
The settled tier is `stable` (holding) at the top of the ladder or below a
failed step even without a recomputed sample — a paused frame at the moving
tier's grid never recomputes — and a hold decision re-renders once (the
render before it may have been refused for recolouring).


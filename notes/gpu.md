# The GPU backend

`packages/gpu` (`@tensatory/gpu`). A WebGPU implementation of the same
computations core does on the CPU, living alongside it. Stage 1 (done):
sampling any field on a `DenseGrid` with a compute shader, with tests that the
two backends agree. Later stages: contouring / integration on the GPU, and a
shader-based renderer.

## Pieces

| file | role |
|---|---|
| `src/wgsl.ts` | `FunctionEmitter`: normalized AST → one WGSL function `fn f(p: vecD<f32>, pos: i32) -> f32 \| vecD`. Emits SSA (`let tN = …`) keyed by structural subtree key, so every distinct subtree is computed once (the GPU counterpart of core's CSE memo). `PRELUDE` holds helpers WGSL lacks (`erf_`, `gelu_`, `pow_` with JS semantics for negative bases, `mod_`, `round_` = half-up, `nan_()` — WGSL rejects NaN literals). `expandGrad` replaces `grad` nodes by their explicit gradient before emission. |
| `src/program.ts` | `ProgramBuilder`: field data → a complete compute program for one dispatch grid. Symbolic data is transpiled, with its arguments bound recursively; dense data is uploaded and read through a generated reader (direct `pos` read when the dispatch grid *is* its support, multilinear interpolation otherwise, NaN outside its box — the CPU semantics exactly); derivatives of dense arguments are computed by core (grid differences) and uploaded; pullbacks become coordinate maps with the chain-rule factor; anything else (closure-backed data) is sampled by core on the dispatch grid and uploaded. **All uploads are packed into one storage buffer** (binding 1) with offsets — WebGPU allows only 8 storage buffers per stage, and derived fields easily need more readers than that. |
| `src/device.ts` | `GpuBackend`: finds `navigator.gpu` in a browser or Dawn's node bindings (`webgpu` package) in node (device requested with the adapter's buffer limits, up to 2 GB); explicit bind-group layout (out + data); pipeline cache by shader code; validation error scopes; `run(program)` → `Float32Array`. Kernels use a 1D `id.x`; dispatches over 65535 workgroups are laid out in 2D and the entry point rewritten (`linearize`). `createBuffer` accounts resident allocations (`bytesAllocated`); `dispatches` / `pipelinesBuilt` counters and `readCounter` (a set's true record count) feed the viewer's adaptive resolution ([resolution.md](resolution.md)). |
| `src/sample.ts` | `gpuSampleOn(backend, field, grid)`: the counterpart of `field.sampleOn(grid)`. |
| `src/nets.ts` | `emitNetField`: a net-backed field ([nets.md](nets.md)) → one WGSL field function. One thread evaluates the whole net for its point, so the function has no batch: every array has its declared per-example shape and lives in a function-scope `var a: array<f32, N>` (constants in the shared `data` buffer); ops are nested loops with an inner accumulation for `einsum` / `reduce`; elementwise trees are one loop; `call` inlines the callee; `reshape` aliases. Emitted from the A-normal form with elementwise fusion, best-fit array reuse by liveness and in-place updates, bounded by `NET_MAX_FLOATS` (2000 floats: Safari's 8192-byte limit); `gpuTranspilable(fd)` tells the viewer whether a field (or anything derived from it) needs no CPU fallback. Derivatives of net fields are core's autodiff programs, transpiled like any net; `ProgramBuilder.gradient(fd)` emits one `vecD` function per field. Loop bounds are opaque (`nb_`) so Metal does not unroll the nests. |

## Semantics preserved

* Row-major grids, `pos` direct reads vs interpolation, NaN outside a field's
  box, `where`-based piecewise derivatives, argument derivatives to any order
  (symbolic → transpiled derivative AST; sampled → core's differenced grids).
* Differences are f32 vs f64 only. Tests use a relative tolerance of 2e-4
  (1e-3 for second derivatives through arguments) plus an absolute slack of
  1e-5 × the value range; NaN must match NaN. Grid coordinates in the op tests
  are chosen so none lands exactly on a discontinuity (`floor` at 1.0, `mod`
  boundaries) where f32 rounding legitimately picks the other side.

## Testing with Dawn in node

`webgpu@0.6` (Dawn bindings, Metal on macOS) runs under vitest. Two hard-won
rules, both encoded in the package:

* **Retain the `GPU` instance and adapter** for the device's lifetime
  (`GpuBackend` keeps them). Dawn's async runner processes events on the
  instance; if it is garbage-collected while buffers are in flight, the process
  segfaults in `InstanceBase::ProcessEvents` (a freed mutex). This looked like
  a random crash after ~15–50 programs until the crash report pointed at it.
* **Use an explicit bind-group layout.** With `layout: "auto"`, a shader that
  never reads the `data` buffer has no binding 1, the bind group fails
  validation (an *uncaptured* error) and the dispatch silently produces zeros.

The build script of `webgpu` must be allowed (`onlyBuiltDependencies` in
`pnpm-workspace.yaml`). Tests skip themselves when no adapter is available.

## Stage 2 (done): the viewer samples on the GPU

`apps/viewer/src/sampler.ts` is a sampling service in front of both backends.
`sampler.request(key, field, grid)` returns cached values, or — on the GPU —
starts the computation and returns `undefined`; the frame skips that layer and
a re-render is scheduled when the result lands. CPU sampling stays
synchronous, so a CPU-only session behaves exactly as before. It feeds the
colorfield raster, the isoline seed values and the streamline vector-field
grid; core still computes statistics (slider ranges), exact isoline
projection and streamline integration on the CPU. NaN masking outside a
field's box is applied uniformly on the viewer side.

The system panel shows the backend (`GPU (apple metal-3)` / `CPU`);
`?backend=cpu|gpu` overrides, `?check=1` computes every GPU sample on the CPU
too and logs the deviation (`agreement <key>: worst 0.12× tolerance …`) —
the test suite's guarantee, live. Measured in Chrome (M-series): at 512² the
gradient of `|∇ mixture|` takes 0.48 s on the GPU vs 2.3 s on the CPU;
`|∇ mixture|` 49 vs 357 ms. For trivial fields the CPU wins (GPU time is
mostly shader compilation + readback, ~50–120 ms).

## Stage 3 (done): geometry kernels

`packages/gpu/src/isolines.ts` and `flow.ts`, driven by a generic
`GpuBackend.runKernel({code, invocations, buffers})` (buffers with roles
`r`/`rw`, explicit layouts per role signature, selective readback).

* **Marching squares** — one invocation per cell, up to two segments written
  to fixed slots plus a per-cell count; the CPU compacts in cell order, so the
  segment list equals core's exactly (tested as a canonical multiset,
  including NaN holes).
* **Projection** — `gpuProjector(field, grid)` transpiles `f`, `∂f/∂x`,
  `∂f/∂y` once; each dispatch projects a batch of `[x, y, maxDist]` vertices
  with the same damped Newton + bisection fallback and box-face locking as
  core, at f32 tolerances.
* **Exact isolines** — `gpuExactIsoContours`: GPU marching squares → CPU
  `joinSegments` (with an f32-scale join tolerance: shared vertices computed
  in two cells differ by ~1e-7) → one projection dispatch for all seed
  vertices → refinement rounds, each projecting the midpoints of every
  still-marked chord in one dispatch and splitting where they deviate. Same
  topology and chords as core within tolerance; residuals ~1e-6 (f32) instead
  of ~1e-11.
* **Streamlines** — `gpuIntegrateFromSeeds`: one invocation per seed, RK4 on
  the unit field both ways (or forward only) into per-seed slots (`packSeeds`: `[x, y, phase,
  back budget, fwd budget, slot]`, slots prefix-summed from the budgets, which
  default to `maxSteps`); seeds and phases come from core's `streamlineSeeds`
  (its LCG now uses `Math.imul`, so it is reproducible bit for bit) or from an
  evenly-spaced / coverage plan (`planStreamlines`, whose budgets the kernel
  honours). Trajectories match core's until f32 drift near separatrices;
  constant fields match exactly; budgeted lines have exactly the planned
  point counts.

Two shader pitfalls found here: **`v != v` is optimized away under Metal's
fast-math**, so NaN tests use bit patterns (`isnan_` / `isfinite_` in the
prelude); and `switch` on `u32` needs `u` suffixes on every case.

In the viewer (`apps/viewer/src/gpuGeometry.ts`) both run asynchronously:
exact isolines for symbolic fields are requested per level and the cheap CPU
marching-squares lines stand in until they land; streamline sets stay on
screen until the new integration arrives. `?geometry=cpu` keeps the CPU
geometry with GPU sampling. Timings in Chrome: `|∇ mixture|` at 128², ~25 ms
per level with all levels concurrent and the main thread free; 1k streamlines
8–50 ms.

## Stage 4 (done): the fused path and the WebGPU renderer

The viewer now has two independent switches (system panel, `?compute=cpu|gpu`,
`?render=canvas|gpu`, remembered in `tensatory.modes`; default gpu/gpu when an
adapter exists):

| compute \ render | canvas (2D) | gpu (WebGPU) |
|---|---|---|
| **cpu** | the original path | CPU values and polylines uploaded (`uploadGrid`, `packPolylines` / `packStreamlines`) and drawn by the GPU renderer |
| **gpu** | sampling + geometry read back asynchronously (`sampler.ts`, `gpuGeometry.ts`); rough lines meanwhile | **fused**: resident grids, fused kernels, `drawIndirect` — nothing crosses back to the CPU |

**One record type for all lines**: `Seg { a, b, ca, cb, arc, len, phase }`
(`segments.ts`). Fused kernels append records through an atomic counter in an
indirect-draw buffer (`allocSegments` / `resetSegments`); CPU lines are packed
into the same layout. **Resident grids** (`resident.ts`): the sampling kernel
keeps its output buffer (`sampleResidentSync`), read by the raster pass and the
fused kernels; `readGrid` only when a CPU consumer needs values.

**Fused kernels** (`fused.ts`): `fusedIsolines` does, per cell, marching
squares → Newton projection of both endpoints → in-thread adaptive midpoint
refinement (up to 16 pieces per seed segment, 4 rounds) → colour-field
evaluation per vertex → append; sampled fields skip projection.
`fusedStreamlines` integrates each seed both ways (within its step budgets)
into scratch slots, then appends segments with arc / length / phase and
colour; scratch and capacity are sized from the budgets (Σ(nb + nf) segments). Both have `run`
(awaited, for tests) and `dispatch` (fire-and-forget) forms; the GPU queue
orders a dispatch before the frame's render pass, so a fused frame is
completely synchronous from the CPU's point of view — no `await`, no readback,
no rough fallback: exact isolines every animation frame.

**Renderer** (`render.ts`, `GpuRenderer`): a raster pipeline (world-space quad
over the grid, bilinear or nearest read of the resident grid, NaN discard,
codomain mapping `lo/hi/log/flip` in the shader, 256×1 LUT texture per
colormap) and a line pipeline (vertex-pulled `Seg` instances, six vertices per
segment extruded to a screen-space width with butt caps, per-fragment particle
window — so tails fade continuously instead of in bins — LUT or solid colour,
premultiplied blending). Points, labels and the box outline stay on a
transparent Canvas 2D overlay (`Renderer2D.render(scene, overlay = true)`)
sharing the same camera (`viewLinear`).

Measured in Chrome (M-series): all four combinations hold ~60 fps with both
animations on `|∇ mixture|` at 128²; in the fused combination the isolines
are exact (projected) in every frame.

## Stage 5 (done): statistics, blur and Taubin passes

`packages/gpu/src/passes.ts` closes the last CPU round trips of the fused path.
Each pass has a test in `test/passes.test.ts` asserting agreement with core.

* `gpuStats(backend, grid, channel)` — a workgroup reduction (`STATS_WG = 256`
  threads, one partial `{min, max, posMin, sum, finite}` per workgroup) and a
  readback of the partials, folded on the CPU. NaN/±∞ samples are skipped by
  bit test (`isfinite_`). Returns the same `Stats` shape as core's
  `computeStats` (`posMin` for log codomains). The viewer's `rangeOf` uses it
  for symbolic fields in gpu compute mode: a provisional range from a 24×24 CPU
  grid is returned synchronously (sliders and legend need a number at once),
  the reduction over the field's default stats grid replaces it when it lands
  and marks the frame dirty. Sampled data keeps core's precomputed stats.
* `blurResidentSync(backend, src, radius)` — separable, edge-truncated box
  blur of a resident grid, NaN propagating, identical to core's `boxBlur`
  (returns `src` unchanged for radius ≤ 0 or multi-channel grids). The viewer
  caches blurred grids per `(grid, radius)` in `FusedGeometry.blur`; blurred
  values are contoured with the non-exact marching-squares kernel
  (`fusedIsolines(…, { exact: false })`), exactly as the CPU path does.
* `smoothedIsolines(backend, values, colour)` — Taubin smoothing of
  marching-squares output on the GPU. Vertices live on grid *edges*
  (`hEdge(i,j) = i*NY+j`, `vEdge(i,j) = HEDGES + i*(NY-1)+j`), each cell
  writes the edge ids of its segments, a neighbour pass gives every edge
  vertex its ≤2 polyline neighbours, then λ = 0.5 / μ = −0.53 passes ping-pong
  positions over the edge array (vertices with < 2 neighbours — open ends —
  stay fixed, as in core), and an emit kernel appends `Seg` records. Same
  `dispatch(segs, level, iterations)` fire-and-forget shape as the fused
  kernels. Matching this exposed a core bug: `joinSegments` closed loops with
  a near-equal (not identical) last point, so `taubinSmooth` treated them as
  open; the seam is now snapped exactly.

With these, gpu/gpu handles every isoline option (`metric`, `line`, exact
projection, I_C colouring) without a readback; the only asynchronous piece is
the stats refinement, and it is invisible unless the coarse range was wrong.

## Stage 6 (done): 3D and adaptive resolution

3D is [3d.md](3d.md) (fused marching tetrahedra, `GpuRenderer3D`). The
append kernels (isolines, smoothed isolines, isosurfaces) now take the set's
real capacity from their params and count every record through the atomic, so
the indirect buffer holds the true count even on overflow; the viewer reads
it back and sizes sets from measured complexity ([resolution.md](resolution.md)).
No kernel bakes its dispatch grid: it travels as a header in the params /
data buffer (`gridWgsl`, `packGrid` in `wgsl.ts`), so one pipeline per field
serves every resolution and crop — see [resolution.md](resolution.md).

## Stage 7 (done): glyphs

`src/glyphs.ts`, `fusedGlyphs` ([glyphs.md](glyphs.md)): a measure kernel
(field evaluated at every lattice point, norm folded into one `atomicMax`
over the f32 bit pattern — positive floats order like u32, non-finite skipped)
and an emit kernel appending three `Seg` / `Seg3` records per point with the
same arrow formulas as core. One kernel per (field, colour); the lattice
(packed coset grids) and the set's capacity travel in the params, so a pan or
zoom re-dispatches without compiling or re-uploading a dense field's data.

## Stage 8 (done): nets

Net-backed fields are transpiled whole (`src/nets.ts`, above), so the raster,
exact isolines, streamlines, glyphs and marching tetrahedra evaluate the net
directly — the iris MLP at 256³ (16.8M points × 30 examples) samples in
milliseconds and the 3D arm runs at the top of its ladder. `test/nets.test.ts`
checks every array op through a small net, `call`, coordinate-expression
inputs and `netv` against the CPU evaluator at 2e-4, the iris fields at
48² / 12³, and — since gradients are autodiff programs on both sides — the
iris gradient and a second derivative at the ordinary f32 tolerance.

Two constraints shape the emitted code, both measured with
`apps/viewer/public/nettiming.html` (dev server; compiles the iris forward /
gradient / projection kernels with and without opaque bounds — Dawn caches
pipelines by generated MSL, so the probe salts the code, not a comment):

**Function-scope memory.** Safari's WGSL compiler rejects any function whose
variables exceed 8192 bytes ("The combined byte size of all variables in this
function exceeds 8192 bytes"); Chrome has no such limit. A bundle must render
in both, so `NET_MAX_FLOATS = 2000` and the emitter works to stay under it:
the program is emitted in A-normal form, dead arrays are reused best-fit by
per-node liveness, an elementwise node whose operand dies there writes in
place, and single-use elementwise producers are fused into their consumer
(`fuseElementwise` + the `elementwise` tree emitter: one loop, one scalar
expression per element). The iris forward went from 2274 floats (rejected by
Safari) to 821, the gradient program from 3576 to 1661; a second derivative
(3943) still falls back to the CPU.

**Compile time.** A kernel inlines every field function at every call site,
and the projection routines call the value and the gradient several times.
WGSL has no roll / unroll attribute ([gpuweb#4110](https://github.com/gpuweb/gpuweb/issues/4110)
is the open request) and Tint's MSL printer emits plain `while(true)` loops,
so the unrolling happens in Apple's Metal compiler where nothing in WGSL can
reach it — an opaque loop bound (`nb_`, a function reading a header word) is
the only lever. Pipeline compile of the iris kernels, ms, median of repeats:

| kernel | Chrome literal | Chrome opaque | Safari literal | Safari opaque |
|---|---|---|---|---|
| forward | 142 | 78 | 1387 | 369 |
| gradient | 236 | 145 | 742 | 700 |
| projection | 675 | 291 | 8743 | 4926 |

Safari is 5–15× slower than Chrome to compile the same WGSL and benefits
even more from the opaque bounds (forward 3.8×), so they stay on in both
browsers (`setOpaqueLoopBounds` exists for the probe). One `vecD` gradient
function per field instead of D component programs was the other big cut
(2.4 s → 0.5 s for the projection kernel in Dawn).

**Shader compiles are asynchronous and never stall a frame.** Neither
browser blocks JS in `createComputePipeline` (0 ms): the compile happens at the
pipeline's first submit and stalls the *frame* — 100–800 ms in Chrome, seconds
in Safari for a transpiled net — so no indicator painted from JS could appear
before it. `GpuBackend` therefore compiles with `createComputePipelineAsync`
and DEFERS dispatches: a `dispatch` whose pipeline is still compiling creates
and returns its buffers as usual but queues the pass; once any dispatch is
queued, every later dispatch and every `write` (the segment-counter resets)
queue behind it, so the GPU sees them in submission order (a seed → smooth →
emit chain must not reorder); the queue drains as compiles land, and
`whenIdle()` gates the readbacks (`readCounter`, `readGrid`, …) and buffer
destroys (`createBuffer`'s wrapper, `release`). The frame that hit a deferred
dispatch does not present (`takeDeferred()` → keep the previous image), and
`onPipelineReady` re-renders: by then everything is cached and the frame is
ordinary. The viewer shows a spinning gear (top right, `#gear`) while
`compiling > 0`. Measured toggling the glyph panel on the iris landscape (two
new kernels): Chrome 11 gear frames, worst frame 17 ms; Safari 43 gear
frames (~700 ms of compile), worst frame 18 ms — where both used to freeze
for the whole compile. `asyncCompile = false` restores synchronous creation.

Open: a space switch restores that space's remembered resolution directly
(256³ for iris: a 16.7 M-evaluation raster, 4 s) instead of ramping when the
caches are cold — a resolution-controller issue, not a compile one.

**Exact projection of nets is latency-bound.** Measured on the iris loss at
192² (Chrome, M-series): marching squares 1 ms per level, exact 47 ms — for
351 segments. Not arithmetic: even with a single evaluation per projection
the kernel takes 12 ms, while the raster does 37k gradient evaluations in
30 ms. Only the ~350 lanes whose cell holds a segment work, one lane per SIMD
group, and a net evaluation in a lone lane is a serial chain of thread-private
memory round trips (~5 ms wall); Newton runs ~8 of them in sequence, so the
cost is per level, not per vertex, and compaction would not help. What did:
`ProgramBuilder.valueGradient` — (value, ∇) in ONE evaluation (a net's
gradient program computes the forward pass anyway), so a Newton step costs
one evaluation, with a step-size stop and geometric acceptance (`|r| / |∇f| <
tolW`) instead of stalling into the bisection at f32's noise floor
(47 → 40 ms); and, decisively, the fused path now shows marching squares
while a net field's level is moving and turns exact when it settles (the CPU
path always did) — 165 → 17 ms per frame during a level drag, one exact
recompute after. Literal loop bounds would make the lone-lane chain 2× faster
(25 ms) at 10× the compile; not taken. The colormap selection is no longer
part of the isoline kernel key: a new key meant a shader compile per drag
event (165 ms per frame in Chrome, seconds in Safari) for a kernel the
selection does not affect.

What stays CPU-side / off: nets over `NET_MAX_FLOATS` fall back to core
sampling on the dispatch grid (`costly` in the viewer); 3D exact projection
is skipped for nets — at 256³ one dispatch of value + gradient per Newton
step per vertex exceeded the GPU watchdog and lost the device (normals still
use the exact gradient; 2D isolines of nets are exact).

## Next stages

1. Interval-arithmetic quadtree seeding for exact isolines (topology still
   comes from the seed grid).
2. Timestamp queries (`timestamp-query` feature) for real GPU frame times in
   the adaptive resolution instead of vsync-quantized rAF intervals.
3. Reuse same-sized resident buffers across frames (a moving crop allocates
   and frees its grids every frame; Dawn zero-fills new buffers).
4. Nets beyond function-scope memory (an MNIST-sized validation set): stream
   the declared dataset axis — every array carrying it is consumed only by
   reductions, so the loop over examples can wrap the per-example body with
   small intermediates. Also: fuse elementwise chains into their consumers
   instead of materializing every node, and call-site batching in the emitter.

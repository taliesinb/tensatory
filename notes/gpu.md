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
| `src/device.ts` | `GpuBackend`: finds `navigator.gpu` in a browser or Dawn's node bindings (`webgpu` package) in node; explicit bind-group layout (out + data); pipeline cache by shader code; validation error scopes; `run(program)` → `Float32Array`. |
| `src/sample.ts` | `gpuSampleOn(backend, field, grid)`: the counterpart of `field.sampleOn(grid)`. |

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

The bundle panel shows the backend (`GPU (apple metal-3)` / `CPU`);
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
  the unit field both ways into fixed slots; seeds and phases come from core's
  `streamlineSeeds` (its LCG now uses `Math.imul`, so it is reproducible bit
  for bit). Trajectories match core's until f32 drift near separatrices;
  constant fields match exactly.

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

## Next stages

1. A WebGPU renderer: raster as a texture sampled with the colormap LUT,
   isolines and streamlines as instanced line geometry with per-vertex colour
   and the particle window evaluated in the fragment shader (as the 3D
   prototype did in GLSL). Removes the raster readback and is the base for 3D.
2. Statistics (min/max for slider ranges) as a reduction pass.
3. Keep grids resident on the GPU between passes (sample → contour → render)
   instead of round-tripping through the CPU.

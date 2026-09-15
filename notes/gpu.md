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

## Next stages

1. Viewer integration: a backend switch (GPU when available) for grid sampling
   and the streamline vector-field sampling, with an agreement check surfaced
   in the log; the render path has to become async-aware for that.
2. Marching squares (then cubes) as compute passes; Newton projection of
   contour vertices on the GPU; streamline integration as a compute pass over
   the sampled vector grid.
3. A WebGPU renderer: raster as a texture sampled with the colormap LUT,
   isolines and streamlines as instanced line geometry with per-vertex colour
   and the particle window evaluated in the fragment shader (as the 3D
   prototype did in GLSL).

# Tensatory

This repo is intended to be a productionized version of the scalar field viewer functionality in the `viewer` folder of `~/projects/loss-landscape`. That project prototyped how to build marching-cube isosurfaces from various metrics (loss, accuracy, etc) collected from training neural networks. The Python-based collection code chose various projects of the full model parameter space and densely sampled these matrics *around* a specific point in the parameter space (e.g. the final trained model).

Productionizing this means decoupling the web viewer / data server code from the ad-hoc choices made by the original LLM in building this end-to-end prototype; and formalizing a definition of a "Tensatory data bundle", which contains any existing raw data, along with metadata explaining how this data corresponds to scalar/vector fields and 1-forms, in *what* spaces, how these spaces and fields and forms may be related.

It also would add a way for a such a bundle to define how to compute *new* scalar/vector fields on user demand. With such definitions, the UI can request a novel field/form from the Tensatory server (e.g. invoking server-side Torch script to sample a new grid of metrics), or to compute this same field client-side from some mathematical expression ("x^2 + y^2"), or via some 3rd party Javascript library (e.g. WebGPU-based neural net training).

## Reusing existing code 

This would involve:
1. choosing an overall framework/architecture for the Tensatory viewer. Lean towards no framework initially, since sliders etc already have a simple shared-code design and `data-` attributes in the original prortype.
2. rewriting the original `viewer/main.js` into viewer-focused TypeScript implementation, spanning multiple files with independent concerns, with well defined interfaces and functionality. Note the existing code evolved over time, and may reflect all kinds of historical details; it is in JS rather than TS; verbatim re-use should be done with caution.
3. sharing relevant subsets of the code between client and server where that makes sense.
4. formalizing a JSON schema to allow describing the various objects (fields, forms, spaces, arrays) that make up a Tensatory data bundle
5. formalizing corresponding TS classes/interfaces/functions; the JSON is parsed and used to create objects; functionality like computing derived fields from other fields; densely sampling field values from mathematical expressions; computed isosurfaces; could live here.

## Phases

To begin with, we could implement only the simplest subset of functionality, avoiding the `.npz`/`.np`/`.zar` data backends and supporting only inline JSON-embedded or symbolic arrays; fields backed by these; and fields defined by mathematical expressions.

## Data Bundle Schema

The bundle schema lives in `schema/` (workspace package `@tensatory/schema`).
It is types only: things ending in `Spec` are the JSON-facing description; they
are parsed (zod, in `@tensatory/core`) into runtime objects with functionality
living on them. The zod schemas are declared as `z.ZodType<XxxSpec>` so they
cannot drift from the types.

It defines manifolds, fields on them (scalar / vector) whose data is
symbolically defined, pointwise-derived from other fields, pulled back
(translate / scale), or backed by arrays that are themselves inline, constant,
one-hot, symbolic, or (later) stored via npz / npy / zarr / bin; and small
neural networks (`nets`, `schema/nets.ts`, `notes/nets.md`) whose outputs can
back fields (`net` / `netv` field data).

## Repository layout

```
schema/            @tensatory/schema  - bundle schema as TS types (+ BUNDLE_VERSION)
packages/core/     @tensatory/core    - runtime: zod parsing, NdArray, symbolic
                                        expressions (normalize / compile / diff),
                                        field data, stats, codomains, nets
                                        (zod + shape inference), Bundle
                                        registry, isolines, streamlines, glyph
                                        lattices. Isomorphic; no DOM. Tests in test/.
packages/gpu/      @tensatory/gpu     - WebGPU backend beside core: WGSL
                                        transpiler, compute-shader sampling,
                                        geometry kernels (marching squares,
                                        projection, fused isolines/streamlines/
                                        glyphs, marching tetrahedra), net
                                        transpiler, WebGPU 2D and 3D
                                        renderers; tests assert CPU/GPU
                                        agreement (Dawn node bindings).
apps/viewer/       @tensatory/viewer  - Vite + vanilla TS viewer: 2D arm (canvas
                                        2D / WebGPU) and 3D arm (WebGPU),
                                        example bundles in public/bundles/.
notes/             architecture notes (start with notes/README.md)
```

`pnpm typecheck` / `pnpm test` / `pnpm dev` (viewer on http://127.0.0.1:5180/,
fixed port so it can be saved to the Dock). `PERF=1 pnpm test` also runs the
profiling test.

## Working conventions

* Read `notes/README.md` for the architecture; the notes are the long-form
  reference, this file is the summary. Keep both current when a decision
  changes.
* Schema changes: edit the types in `schema/` first, then the zod schema in
  `packages/core` (declared `z.ZodType<Spec>`, so a mismatch fails typecheck),
  then tests and the example bundles in `apps/viewer/public/bundles/`.
* Core stays isomorphic (no DOM, no fetch); anything browser-specific lives in
  the viewer.
* Verify UI changes in a browser against the running dev server; uncaught
  errors show in red on the status line and in the L (log) modal.

## Decisions (phase 1)

* Manifolds are R^n with the identity chart; a field's `box` lives in those
  coordinates. Charts / affine frames (`schema/mappings.ts`) come later.
* Dense grids are row-major ("C" order, last axis fastest), matching numpy.
  Grid position (0,…,0) is at box corner `a`, the last at `b`.
* Random arrays (`schema/distribution.ts`, `core/src/arrays/random.ts`):
  `{type:"random", shape, dist, widget?}`; every distribution has
  `seed: null | int | string`; continuous ones are location–scale
  (`uniform` Z∈[-1,1), `gaussian`, `laplace`, `exponential`, `studentT`),
  discrete: `bernoulli`, `discrete`, `integers`. Cell i is a pure function of
  (seed, i) — a counter-based hash stream — so cells are order- and
  chunking-independent. A displaced net's direction is
  `{ arrays, norm?: number | "origin", scale?, name?, widget? }`: ONE vector
  over all its arrays, `norm` its joint length (`"origin"` = that of the
  displaced constants), `scale` a multiplier. `widget` = a row in the
  viewer's `controls` panel; rows hold ADJUSTMENTS (per-bundle options:
  `{ seed?: salt, scale?: multiplier }` by row id) that core's `adjustSpec`
  folds into a spec (salt hashed into every governed random array's seed,
  multiplier into the scale) and the viewer rebuilds the Bundle from
  (`revision++`, `clearFieldCaches()`). Shapes never change, so the WGSL is
  byte-identical and NO shader recompiles (pipelines are cached by code;
  `gpu/test/nets.test.ts` asserts it); rows commit on release and are inert
  while an animation plays; live dragging is a follow-up. Box zoom is the
  same pattern: `-` / `=` (`0` resets) scale the boxes of the current
  space's SYMBOLIC fields by 1.5 around their centres (core `zoomBoxes`;
  per-space option `boxZoom: k`), sampled fields keep their grid, the
  view re-fits.
* Field data has `kind: "symbolic" | "sampled"`. Sampled data has
  `samplePoints`. Pointwise-derived data is sampled iff any argument is; all
  sampled arguments must have IDENTICAL sample points (error otherwise).
  Pullbacks keep the kind and move the support along.
* Symbolic expressions: constants are leaves (`2`, `{op:"const",name}`), not
  parameter slots. Names come from three namespaces (`consts`, `scalars`,
  `vectors`) that must not overlap within a scope; a bare string resolves by
  position. `grad` is symbolic differentiation; derivatives through field
  arguments compose to any order (`derivative(dim)` on field data: exact for
  symbolic data, grid differences for sampled data).
* Nets (`notes/nets.md`): a net is a pure function from named input arrays
  to named output arrays with predeclared per-example shapes (`def` / `bind`
  / `displace` / `grad`); nothing distinguishes weights from data, loss is an
  ordinary `[]` output. The body is the scalar language lifted elementwise
  to arrays plus `matmul` / `einsum`, `reduce`, shape ops and `call`. Dataset
  axes are DECLARED symbolic sizes (`"N"`) and reduced inside the net; extra
  undeclared leading axes are an implicit vmap batch (broadcast across
  inputs, invisible to the body, never reducible) for evaluating one net at
  many parameter points. Symbolic sizes are opaque (never equal to another
  name or a number except via broadcasting with 1); anything unprovable is a
  parse-time error. `bind` fixes inputs (they stay internal arrays);
  `displace` adds a `[K]` coefficient input and moves any named arrays
  (inputs, nodes, bound inputs, constants) along K directions — "around" a
  net; a net whose sole remaining input is `[D]` is a field with just `net`
  + `output` + `box`, so a loss landscape is bind(data, θ*) → displace →
  field. `grad` yields a net; `wrt` may be an input or any internal array; a
  named `seed` becomes a new input (HVPs). `grad` is reverse-mode autodiff
  as a PROGRAM REWRITE (`core/src/nets/autodiff.ts`): forward in A-normal
  form + adjoint nodes in the same op vocabulary (slice / takeAlong adjoints
  via a baked selection matrix + einsum / oneHot + reduce + transpose; `call`
  via the callee's own grad net; einsum may repeat a letter in its OUTPUT —
  `i->ii` writes the diagonal — so diagonal reads and writes are each
  other's adjoints), so the op set is closed under adjoints, every evaluator
  differentiates for free and grad programs compose (bind / displace / call
  / grad again, to any order). Two
  evaluators: the CPU reference (`ops.ts`, `program.ts`: batch-aware strided
  ops, one batched evaluation per sampled grid chunk, Float64) and the
  WebGPU transpiler (`gpu/src/nets.ts`: a net field becomes an ordinary WGSL
  field function — one thread evaluates the whole net for its point in
  function-scope arrays, so raster / exact isolines / streamlines / glyphs /
  marching tetrahedra evaluate nets in place). SAFARI rejects a WGSL
  function with > 8192 bytes of variables, so `NET_MAX_FLOATS = 2000` and
  the emitter minimizes function-scope memory: ANF, best-fit reuse by
  liveness, in-place elementwise, fusion of single-use elementwise producers
  (`fuseElementwise`); `gpuTranspilable(fd)` says whether a field fits; one
  `vecD` gradient function per field (`ProgramBuilder.gradient`); loop bounds
  are opaque (`nb_`) because WGSL has no unroll control and Metal unrolls —
  measured in both browsers with `apps/viewer/public/nettiming.html` (Safari
  compiles 5–15× slower than Chrome; opaque bounds help both). A net
  field is a program with one input (the point; `fieldProgram` folds
  coordinate expressions in); `derivative(dim)` is exact (component of the
  gradient program) to any order. Net fields report `costly`; the viewer's
  `costly(fd)` = the flag AND the GPU cannot take it — then small fixed
  streamline grids (32 / 16), glyph lattices capped at 2k, no exact isoline
  projection, and the resolution controller judges compiled frames by JS
  time (a shader compile never inflates it, CPU sampling does); expression
  fields over costly arguments pre-sample them on the grid in batches. 3D
  exact projection stays off for nets (one 256³ dispatch of value + gradient
  per Newton step exceeded the GPU watchdog); 2D exact projection of a net
  is latency-bound (~40 ms per level: a lone lane's serial chain through
  private memory), so the fused path shows marching squares while the level
  moves and projects on settle; Newton uses `ProgramBuilder.valueGradient`
  (value + ∇ in one evaluation), a step-size stop and geometric acceptance.
  Fused kernel cache keys contain only what changes the CODE (the colormap
  selection once compiled a shader per drag event). Vertex colour in fused
  kernels is a `ColourSource` (`gpu/src/colour.ts`): the field per vertex,
  a resident grid interpolated (`residentReader`; costly fields, capped at
  256² / 64³), or `"level"` (colour field = iso field); resident-coloured
  sets are recoloured EXACTLY over the frames once the level rests
  (`gpu/src/recolour.ts`: interleaved batches written in place, one per
  frame with ≤ 2 in flight, budget following the rAF interval, the image
  refreshed every 4th frame — a re-render costs more than a batch). Reals
  only. `apps/viewer/public/bundles/iris.json` (from
  `tools/iris/train.py`, PyTorch once-off, bit-reproducible) is the example:
  a 4-16-3 MLP (Adam, weight decay 3e-3), 120 training / 30 validation
  examples, θ*, net outputs `loss` / `acc` / `obj` (= loss + wd/2‖θ‖², what
  Adam minimized — θ* is NOT the minimum of the validation loss) /
  `loss0..2` (per class); validation and training binds, each displaced
  along 2 / 3 RANDOM directions (`iris_rnd2/3`, `iris_trn_rnd2/3`: gaussian
  `random` arrays with SHARED seeds, `norm: "origin"`, Controls rows
  d0..d2) for the viewer's 8 fields per space, plus the 3 fixed inline
  orthogonal directions (`iris_rand2/3`, `iris_trn_rand2/3`, no fields) the
  PyTorch reference was computed along. The training-set nets (N = 120)
  exceed `NET_MAX_FLOATS`, so those fields are CPU-sampled (`costly`);
  the CPU evaluator prunes a field's program to its output (`pruneProgram`,
  `evalPoints`); the GPU emitter gets the FULL program — pruning its input
  exposed a latent emitter bug in the gradient (see notes/nets.md).
  `core/test/iris.test.ts` checks every output against PyTorch to 1e-9,
  `core/test/autodiff.test.ts` every op's gradient against finite
  differences, `gpu/test/nets.test.ts` CPU vs GPU per op and for iris (value,
  gradient, second derivative).
* Vector fields are plain vector-valued functions: pullbacks reparametrize the
  domain only (no pushforward). 1-forms are not distinguished yet.
* `exactGradient` on a scalar field names a vector field holding gradients
  sampled during collection (backprop); the viewer uses it for streamlines by
  default.
* Codomains are visualization hints (bounds, log base, flip, wrap, unit,
  marks); they never change values. Numbers are formatted compactly
  (`6.24·10⁻⁵`, unicode superscripts).
* Viewer: one super-stack of panels top-left (`bundle`, `system` — closed by
  default: compute / render, device, `mem cap`, live memory —, `controls`
  when the bundle has rows, `2D space`,
  `colorfield`, `isolines`, `streamlines`, `vector field` — isolines and
  streamlines OFF by default, isosurfaces on at a 3D space's first visit), the `mappings`
  matrix bottom-left, cursor pane + legend bottom-right. The matrix assigns
  fields to slots C, I_V, I_C, S_∇, S_C, V_∇, V_C; rows are every scalar AND
  vector field, as a TREE: names are paths (`train/loss/setosa`), drawn
  flattened with a subtle indent, subtrees collapsed until clicked, a
  locked-selected field always shown with its ancestors but without its
  unselected siblings, pure headings without cells (`notes/viewer.md`);
  selecting a costly field restarts the resolution ladder; a vector slot
  given a scalar uses its gradient (∇ glyph), a scalar slot given a vector uses
  its norm (|·| glyph; log-scaled codomain, since gradient norms span orders
  of magnitude and vanish at critical points). Column headers toggle their
  panel; disabled columns dim. Streamlines have `dir` (ascending /
  descending: whether each path is integrated following the S_∇ field or its
  inverse — which lines get drawn) and `mode` (bi-strat / strat / JL / cover:
  seeding and oversampling — strat = stratified seeds, each line starts at
  its seed and runs in `dir`; bi-strat = the same seeds integrated both ways
  through the seed; JL = Jobard–Lefer evenly spaced, planned on the CPU into
  seeds with per-seed step budgets that the GPU kernels re-integrate; cover =
  strat plus seeds in starved cells; see `notes/isolines.md`).
  ▶/◀ on a panel strip is purely the playback direction of that animation
  (particles forward / backward along the drawn lines); shift-click a ▶ to
  reverse it, plain click only toggles play, space starts both animations
  when none is playing else pauses; loads, space / bundle switches and a
  new I_V / S_∇ / V_∇ field start paused (turning a ▶ on resumes). Colormaps
  are per field: one legend bar per coloured field listing its slots (click
  the name to cycle), black bars for shape-only fields (I_V, S_∇ source), red
  θ* detent, white isoline notches and cursor pip. Every bar is a colormap
  interval selection (`apps/viewer/src/interval.ts`, `cmapInterval.ts`; see
  `notes/viewer.md`): a bracket in codomain parameter space with nullable
  ends; excluded regions clip or mask (barber-pole = not drawn: transparent
  raster, undrawn line stretches, I_V levels not contoured), the included
  region stretches the colormap or keeps it; shape-only bars are fixed-mask.
  Prototypes of new widgets go in `apps/ui-proto` (static single-file pages,
  never imported by production code). Per-bundle options in localStorage
  (`tensatory.opts.<file>`), only user-panned/zoomed views are persisted;
  any control id / slot key can be overridden from the URL. The streamline
  controls opacity / lines / split / tail / length are saved per arm (`ui`
  2D, `ui3` 3D) with their own 3D defaults (`DEFAULTS_3D`: 0.5 / 2k / 4 /
  5 / 100).
* Vector field glyphs (`notes/glyphs.md`; `core/src/flow/glyphs.ts`,
  `gpu/src/glyphs.ts`): the V_∇ field as static arrows on the densest
  lattice — hexagonal in 2D, face-centred cubic in 3D — represented as
  interleaved `DenseGrid` cosets so every sampling path serves it. The
  lattice is FIXED IN SPACE: nested levels anchored at the field's box
  corner, level k = longest side / 2^k (2Λ ⊂ Λ, so refining tessellates);
  the view only picks the level nearest the panel's pixels in log₂ (3D: at
  the camera target, ×2). Only the visible part (view ∩ box; 3D the cropped
  box) is sampled and normalized against, ≤ 100k points. Glyphs
  have length budget `L = 0.9 · spacing · |v| / max |v|` with the maximum
  over the vectors ACTUALLY sampled (the `longest` readout) and three styles
  (`glyph`: arrow = centred shaft + head; head = a chevron of length L
  centred on the point; triangle = solid narrow triangle, base on the point,
  apex at the arrow's tip), all within L/2 of the point so none overlaps;
  glyphs under 4 px on screen are culled (3D cones exactly, by projected size);
  ≤ 3 `Seg` / `Seg3` records each — lines, or one filled-triangle record
  (base in a / b, apex in arc / len / phase) for the renderers' triangle
  pipelines (`kind: "triangles"`; in 3D drawn as ray-cast CONES with true
  depth and a subtle headlight) — coloured by V_C. Fused path:
  one kernel per (field, colour) — measure (`atomicMax` on norm bits) + emit
  — with the lattice as a dispatch parameter. Linear normalization only;
  rescaling for heavy-tailed norms is the planned follow-up. No animation.
* Isolines (`core/src/iso`): `contourField` always uses the best available
  method. Sampled fields: marching squares on their grid. Symbolic fields:
  marching squares only seeds topology; every vertex is then projected onto
  the level set (damped Newton along the exact ∇f, bisection fallback, box
  faces locked) and chords are refined adaptively to a world-space tolerance
  (the viewer passes ¼ pixel). Known limitation: topology still comes from the
  seed grid — planned fix is an interval-arithmetic quadtree seed.
* GPU backend (`packages/gpu`): same semantics as core, f32; every upload
  packed into ONE storage buffer (8-buffer limit); explicit bind-group layout;
  the `GPU` instance must be retained for the device's lifetime under Dawn or
  the process segfaults. See `notes/gpu.md`.
* Resolution is not a control (`notes/resolution.md`, `apps/viewer/src/autores.ts`):
  one controller per arm over a ladder (2D 32 … 2048, 3D 16 … 256) with two
  tiers — `moving` (levels animating / dragged, 2D view dragged) holds 30 fps,
  `settled` is bounded by one recomputation ≤ 200 ms — plus the memory cap
  (`mem cap` 256 / 512 / 1024 / 2048 MB, default 1024; `?memcap=`,
  `tensatory.memcap`). Frame times come from rAF intervals (vsync-quantized,
  so 60 fps windows probe one step up), compiled frames are remeasured, the
  last good pair is remembered per bundle + space, `?res=` / `?res3=` pin.
  Fused kernels count every record even when a set is full (real capacity in
  params), the viewer reads the counts back (`GpuBackend.readCounter`) and
  sizes sets from the measured complexity per family, regrowing on overflow;
  `GpuBackend.createBuffer` accounts resident bytes; caches are byte-aware
  LRUs (`cache.ts`) trimmed to the cap, keyed by grid identity where a kernel
  reads a grid. Dispatches over 65535 workgroups are 2D (`linearize`); the
  device asks for the adapter's buffer limits. No kernel bakes its grid:
  grids travel as a 20-float header in params / data (`gridWgsl`, `packGrid`
  in `gpu/src/wgsl.ts`), so a resolution step, pan or crop drag compiles
  nothing (a dense field's own support stays baked — intrinsic to the field). Streamline grids are FIXED
  (128 in 2D, 64 in 3D) so line lengths do not follow the tier.
* Viewer modes: `compute ∈ {cpu, gpu}` × `render ∈ {canvas, gpu}` (system
  panel, `?compute=`, `?render=`, `tensatory.modes`; default gpu/gpu). gpu/gpu
  is fused: resident grids, fused kernels appending `Seg` records, drawIndirect,
  no readback, exact isolines every frame. gpu/canvas reads back asynchronously
  (pending layers skipped or rough lines shown until results land). cpu/gpu
  uploads CPU values and polylines. `?check=1` logs CPU/GPU agreement.
  Statistics (reduction), box blur and Taubin smoothing are GPU passes too
  (`packages/gpu/src/passes.ts`), so gpu/gpu covers every isoline option;
  symbolic ranges start from a coarse CPU grid and are refined by the GPU
  reduction when it lands.
* Spaces and 3D (`notes/3d.md`): the bundle panel's `space` picker selects one
  of the bundle's 2D/3D manifolds (those with buildable fields; `?space=`);
  fields, point sets, slot selections, view / camera and directions belong to
  a space (options per bundle with a per-space section). `defaultManifold` is
  only required when a field / point set omits `domain`. The `space` and
  `isolines` panels are shared by both arms (titles switch; `.d2` / `.d3`
  rows), so value / split / opacity / ▶ mean the same for isosurfaces. 3D:
  marching tetrahedra (`core/src/iso/marchingTets.ts`, tables shared with the
  fused GPU kernel in `gpu/src/mesh.ts`) into a triangle soup with gradient
  normals (exact for symbolic fields) pointing towards increasing values;
  `GpuRenderer3D` = orbit camera, vertex-pulled meshes via drawIndirect,
  two-sided headlight, weighted-blended OIT for translucent levels, depth-
  tested thick lines (box, trajectories, face outlines, streamlines); points /
  labels on the Canvas 2D overlay. Exact vertices via ∇f projection (flag ∇),
  isolines of I_V on the cropped box faces (`outline`), crop ranges x / y / z
  as interval sliders — the volume grid follows the cropped box, so a small
  crop is a full-resolution close-up and crop drags run in the `moving` tier —,
  adaptive resolution (see above), `value sm` box blur and `surf sm` Taubin
  smoothing (CPU; control ids `metric` / `line`), 3D streamlines with the 2D
  panel's dir / mode (JL planning is dimension-generic). WebGPU only; compute
  cpu / gpu as in 2D. Matrix columns absent in a space are hidden
  (`SlotDef.present`).
* Shader compiles are async (`createComputePipelineAsync`) with DEFERRED
  dispatches (`GpuBackend.dispatch` queues behind a compile, in submission
  order, with `write` for counter resets; `whenIdle()` gates readbacks and
  buffer destroys); a frame that deferred does not present (`takeDeferred()`)
  and `onPipelineReady` re-renders; the viewer spins a gear (`#gear`, top
  right) while `compiling > 0`. Neither browser blocks JS on pipeline
  creation — the stall used to land at first submit, freezing the frame.
* WGSL: NaN tests must use bit patterns (`isnan_`), `v != v` is optimized away
  by Metal's fast-math.
* Performance: the compiler does
  common-subexpression elimination with per-point memoization; streamlines of
  symbolic vector fields are integrated through a sampled copy on the current
  grid; isolines fall back to marching squares while a level is moving and
  become exact ~200 ms after it settles. See `notes/performance.md`.

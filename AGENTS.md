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
one-hot, symbolic, or (later) stored via npz / npy / zarr / bin.

## Repository layout

```
schema/            @tensatory/schema  - bundle schema as TS types (+ BUNDLE_VERSION)
packages/core/     @tensatory/core    - runtime: zod parsing, NdArray, symbolic
                                        expressions (normalize / compile / diff),
                                        field data, stats, codomains, Bundle
                                        registry, isolines, streamlines.
                                        Isomorphic; no DOM. Tests in test/.
packages/gpu/      @tensatory/gpu     - WebGPU backend beside core: WGSL
                                        transpiler + compute-shader sampling;
                                        tests assert CPU/GPU agreement (Dawn
                                        node bindings, `webgpu` package).
apps/viewer/       @tensatory/viewer  - Vite + vanilla TS 2D viewer (canvas 2D),
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
* Vector fields are plain vector-valued functions: pullbacks reparametrize the
  domain only (no pushforward). 1-forms are not distinguished yet.
* `exactGradient` on a scalar field names a vector field holding gradients
  sampled during collection (backprop); the viewer uses it for streamlines by
  default.
* Codomains are visualization hints (bounds, log base, flip, wrap, unit,
  marks); they never change values. Numbers are formatted compactly
  (`6.24·10⁻⁵`, unicode superscripts).
* Viewer: one super-stack of panels top-left (`bundle`, `2D space`,
  `colorfield`, `isolines`, `streamlines`), the `mappings` matrix bottom-left,
  cursor pane + legend bottom-right. The matrix assigns fields to slots C,
  I_V, I_C, S_∇, S_C; rows are every scalar AND vector field; a vector slot
  given a scalar uses its gradient (∇ glyph), a scalar slot given a vector uses
  its norm (|·| glyph). Column headers toggle their panel; disabled columns
  dim. Streamline flow direction is the ▶/◀ play direction (◀ = against the
  field = descent); shift-click a ▶ reverses, plain click only toggles play,
  space starts both animations when none is playing else pauses. Colormaps
  are per field: one legend bar per coloured field listing its slots, black
  bars for shape-only fields (I_V, S_∇ source), red θ* detent, white isoline
  notches and cursor pip. Per-bundle options in localStorage
  (`tensatory.opts.<file>`), only user-panned/zoomed views are persisted;
  any control id / slot key can be overridden from the URL.
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
* The viewer samples fields through `sampler.ts` (GPU when available, else
  CPU; `?backend=cpu|gpu`, `?check=1` logs CPU/GPU agreement). GPU results are
  asynchronous: a layer whose values are pending is skipped for that frame and
  re-rendered when they land.
* Performance (rendering is Canvas 2D; contouring / integration CPU): the compiler does
  common-subexpression elimination with per-point memoization; streamlines of
  symbolic vector fields are integrated through a sampled copy on the current
  grid; isolines fall back to marching squares while a level is moving and
  become exact ~200 ms after it settles. See `notes/performance.md`.

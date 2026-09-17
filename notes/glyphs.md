# Vector field glyphs

The static counterpart of the streamlines: the V_∇ field sampled on a lattice
and drawn as one arrow per sample. No animation, no integration — every glyph
is an independent evaluation of the field, so the picture is exact where it is
drawn and cheap enough to recompute on every pan / zoom. Slots **V_∇** (a
vector field, or the gradient of a scalar) and **V_C** (glyph colour) in the
mappings matrix; the `vector field` panel (`showVec`, `spacing`, `opacity`,
the `longest` readout).

## The lattice

`core/src/flow/glyphs.ts`. Glyphs must not overlap, so with a fixed
nearest-neighbour distance the information density is the packing density of
the lattice: the **hexagonal** lattice in 2D (0.907 vs 0.785 for a square
grid; six neighbours at the spacing), **face-centred cubic** in 3D (0.74 vs
0.52; twelve neighbours). Both are unions of a few interleaved rectangular
grids — the **cosets**: hex = the even and odd rows (period `(s, s√3)`,
offsets `0` and `(s/2, s√3/2)`), FCC = the four sites of the cubic cell
`a = s√2`. So `latticeIn(region, spacing, anchor)` returns a list of
`DenseGrid`s and every existing sampling path (CPU `sampleOn`, the sampler's
GPU read-back, resident sampling) serves it unchanged; the GPU kernel walks
the lattice as packed grid headers in its params. Points are numbered coset
by coset, row-major within each (`latticePoints`).

**Fixed in space.** The lattice does not follow the view continuously: it
is a nested hierarchy of levels anchored at the field's box corner, level
`k` having spacing `longest box side / 2^k`. Since `2Λ ⊂ Λ` for both the
hexagonal and the FCC lattice, every finer level contains the coarser one's
points — refining *tessellates*, nothing shifts. The view only picks the
level: the finest one whose spacing is still at least the control's pixels
(2D: `px · worldPerPixel`; 3D: at the camera's target depth, doubled by
`GLYPH_SPACING_3D` since glyphs at every depth share the screen). Panning,
cropping and zooming within a level move no glyph; zooming across a level
halves the spacing and quadruples (2D) / octuples (3D) the points. The
`lattice` readout shows the level, its spacing and its point count.

**Extent.** The whole field box is sampled while the level fits
`GLYPH_MAX_POINTS` (100k), so the normalizing maximum is view-independent
too; beyond that only the view ∩ box (3D: the cropped box ∩ box) is sampled
(`(in view)` in the readout), coarsening while even that exceeds the cap.

## The glyphs

`arrowGlyphs(points, vectors, D, spacing, { style })`: every glyph has the
length budget `L = fill · spacing · |v| / maxNorm` with `fill = 0.9`, and
every style stays within `L/2` of its point along the vector, so neighbours
never overlap in any of them (the `glyph` control, `?vglyph=`):

* **arrow** — a shaft of length `L` centred on the point, with two barbs at
  its tip at 30° (`head = 0.3` of the shaft, `HEAD_SPREAD` sideways); two
  polylines `[tail, tip, barb]`, `[tip, barb']`, three segments;
* **head** — the arrowhead alone: a chevron of length `L` centred on the
  point, tip at `p + L/2·u`, a 44° opening (`CHEVRON_SPREAD`, narrower than
  the arrowhead so the direction reads); one polyline, two segments;
* **triangle** — a solid, narrow triangle with its base centred on the
  point (half-width `TRIANGLE_HALF_WIDTH` = 0.22 of its length) and its
  apex where the arrow's tip would be, `p + L/2·u`; ONE record, filled. In
  3D the record is drawn as a **cone** — the triangle's solid of revolution
  — so it reads from every direction.

3D barbs / bases lie in the plane of the vector and the axis it is least
aligned with (`glyphNormal`). Zero and non-finite vectors get no glyph.

**Records.** Arrow and head are line segments in `Seg` / `Seg3` records for
the line pipelines. A triangle reuses the same record for a *triangle*
pipeline: `a`, `b` = the base's ends, the apex in the spare floats (`arc`,
`len`[, `phase`]), `ca = cb` = the colour (`packTriangles` /
`packTriangles3`; the kernel's `tri()`). `GpuLineLayer.kind` /
`GpuLineLayer3D.kind` = `"triangles"` selects the pipeline, which draws each
record as two halves (apex–a–mid, apex–mid–b) so the six instance vertices
cover the triangle exactly once — no double blending in 2D. The Canvas 2D
renderer fills `TriangleLayer`s binned by colour like its lines.

**3D cones** (`GpuRenderer3D`, `CONES3`): a flat triangle seen edge-on is a
sliver, so the 3D pipeline draws each record as a cone (apex, base centre =
mid(a, b), radius = |a − b|/2) by ray casting: the vertex shader emits a
camera-facing square in the plane of the cone's bounding sphere's silhouette
(the six instance vertices), the fragment shader intersects the eye ray with
the finite cone (`((X−A)·d)² = cos²α |X−A|²`, `t ∈ [0, h]`) and the base
disc, keeps the nearest hit, writes `frag_depth` from its clip position (so
the translucent shells composite correctly over it) and shades the analytic
normal with a key light between the eye and above-left (0.3 + 0.7 diffuse + a highlight). Glyph layers are `uncropped`: the lattice already lies inside the cropped box, so a cone near a face pokes out by up to L/2 instead of being cut flat.
Exact silhouettes, no tessellation, one instance per glyph.

**Normalization** is against the longest vector *actually sampled* (this
lattice, this view), the `longest` readout: zooming into a flat region
rescales the glyphs to what is in view. Linear only for now; heavy-tailed
norms (gradient norms spanning orders of magnitude) make most glyphs tiny —
a rescaling (log / rank / quantile) is the planned follow-up.

## GPU

`gpu/src/glyphs.ts`, `fusedGlyphs(backend, field, colour?)`: one compiled
kernel pair per (field, colour), the lattice a dispatch parameter
(`packLattice`: counts, fill, head, spacing, the set's capacity, then one
packed grid per coset).

1. **measure** — one thread per lattice point evaluates the field there (the
   program builder's vector function, `pos = -1`), stores the vector in a
   resident buffer (grown on demand) and folds its norm into one
   `atomicMax` on the norm's **bit pattern**: positive f32 bits order like
   u32, so the maximum of the bits is the maximum norm; NaN / ∞ are skipped
   by `isfinite_`, as core's `maxNorm` skips them.
2. **emit** — one thread per point reads its vector and the maximum and
   appends the records of the style in `params[6]` (three segments, two, or
   one triangle) with the same formulas as core; `capacityFor` is 3 per point
   (the most any style appends), the
   counter is the true count. The style is a dispatch parameter like the
   lattice, so switching it compiles nothing.

Nothing is read back for the picture; `readMaxNorm` reads the 4-byte maximum
for the `longest` readout (one in flight per set, the latest lattice read
last). `test/glyphs.test.ts` checks the appended segments against core's as a
multiset (threads append in any order) in 2D and 3D, the colour at the glyph
point and the maximum, for all three styles.

## In the viewer

| compute / render | 2D | 3D |
|---|---|---|
| gpu / gpu | `FusedGeometry.glyphs`: kernel per (V_∇, V_C), set re-dispatched when the lattice key changes | `View3D.glyphLayer`, the same with `Seg3` |
| cpu, or gpu / canvas | `glyphs2d`: cosets through the sampler (synchronous on the CPU, asynchronous read-back on the GPU — the previous set stays until the new one lands), core's arrows, colours per glyph; canvas `LineLayer` or uploaded `Seg`s | cosets sampled by core, uploaded `Seg3`s per lattice |

Legend: V_C gets a colour bar, a scalar behind V_∇ a shape-only (black,
fixed-mask) bar like the S_∇ source; the cursor pane lists the V_∇ vector
when it differs from S_∇. Column headers toggle the panel. The 3D line
pipeline has no alpha, so `opacity` applies in 2D only.

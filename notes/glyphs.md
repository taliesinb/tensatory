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

**Anchoring.** The lattice is anchored at the field's box corner, not at the
view: a pan moves no glyph (new ones enter at the edges), only a zoom changes
the spacing — and then every glyph shows a freshly sampled vector.

**Spacing from the view.** The control is the nearest-neighbour distance in
screen pixels (default 24). 2D: `spacing · worldPerPixel`, over the visible
world rectangle ∩ the field's box. 3D: the world distance is taken at the
camera's target depth (`2 d tan(fov/2) / regionHeight`), doubled
(`GLYPH_SPACING_3D`, since glyphs at every depth share the screen), over the
cropped box ∩ the field's box — a small crop is a close-up, exactly like the
isosurface grid. The lattice is coarsened by 1.5× steps while it would exceed
`GLYPH_MAX_POINTS` (100k).

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
  apex where the arrow's tip would be, `p + L/2·u`; ONE record, filled.

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
renderer fills `TriangleLayer`s binned by colour like its lines. In 3D a flat
triangle seen edge-on is a sliver (the arrow, extruded in screen space, is
not): a fatter 3D glyph (two crossed triangles, or a tetrahedron) is a
possible follow-up.

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

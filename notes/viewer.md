# The 2D viewer

`apps/viewer/` — Vite, vanilla TypeScript, Canvas 2D, no framework. The UI
is a port of the loss-landscape prototype's widgets, adapted to 2D fields.

## Files

| file | role |
|---|---|
| `index.html` | the panels; controls are plain `<div>`s configured by `data-` attributes |
| `src/widgets.ts` | tooltips (0.25 s), collapsible panels (state in `tensatory.collapsed`), tick glyphs over hidden checkboxes, the compact slider (drag scrub, shift-hover preview, nullable click/Backspace, wheel "document", arrow nudges, Escape cancels), the discrete slider (one step per wheel gesture), tab bars |
| `src/metrics.ts` | the mappings matrix: one SVG, hit-testing from pointer coordinates, shift-preview, click-to-lock, wheel/arrows, header clicks toggle panels, dimmed disabled columns, ∇ / \|·\| use glyphs |
| `src/render2d.ts` | camera (centre, scale, flips, quarter turns as one linear map), cached colormap raster drawn through an affine transform, line layers binned by colour/alpha into `Path2D`s, particle tails (alpha fade, butt caps), point sets, box, crop clip; `overlay` mode draws only box + points over the WebGPU canvas |
| `src/sampler.ts` | field values on grids: GPU (async, read back) or CPU, cached, NaN outside the field box, optional agreement check |
| `src/gpuGeometry.ts` | GPU compute + canvas render: exact isolines and streamlines computed on the GPU and read back asynchronously |
| `src/gpuFused.ts` | GPU render: resident grids, fused isoline / streamline kernels appending into resident segment sets, uploaded CPU geometry |
| `src/main.ts` | state, slots and "uses", sampling cache, isolines / streamlines pipelines, legend, cursor pane, persistence, URL overrides, interaction, frame loop |
| `src/colormap.ts` | viridis / plasma / gray / okhue / turbo (polynomial fits), CSS gradients, LUTs |
| `src/log.ts` | console capture, the L (log) modal, `status()` and red error display |

## Compute and render modes

`compute ∈ {cpu, gpu}` × `render ∈ {canvas, gpu}` (bundle panel; `?compute=`,
`?render=`; stored in `tensatory.modes`; default gpu/gpu when WebGPU exists).
GPU/gpu is the fused path: `renderGpu()` in `main.ts` builds a `GpuScene`
from resident grids and segment sets — including the `metric` blur (GPU box
blur of the resident grid, contoured with plain marching squares), `line`
Taubin smoothing (edge-graph kernel) and value ranges (GPU reduction, with a
coarse CPU range shown until it lands); the other combinations reuse the CPU
pipelines and either draw with Canvas 2D or upload their results. See
[gpu.md](gpu.md).

## Layout

* **Left super-stack** (one rounded container; strips are coloured rows):
  `bundle` (picker with wheel/arrow switching, R = wipe storage, L = log,
  ⤒ = open a local JSON), `2D space` (points / box flags; view: fit, flip x,
  flip y, cw, ccw), `colorfield` (strip tick = raster on/off; resolution,
  smooth), `isolines` (value, split, opacity, metric blur, line smoothing,
  ▶ + rate), `streamlines` (lines, length, opacity, tail, split, ▶).
* **mappings** matrix bottom-left, **legend** and **cursor pane** bottom-right,
  status line bottom centre (errors in red).

## Slots and the mappings matrix

Slots: **C** colorfield, **I_V** isoline value, **I_C** isoline colour,
**S_∇** streamline direction (vector), **S_C** streamline colour. Rows are
every scalar *and* vector field of the bundle. A slot resolves its field to a
"use": a vector slot given a scalar uses its gradient (∇ glyph), a scalar
slot given a vector uses its norm (|·| glyph), otherwise the field itself.
Defaults: C = I_V = first scalar field, colour slots none (a colour slot equal
to C would paint lines the raster's own colour and hide them), S_∇ = the
field's `exactGradient` if any, else the field (→ its gradient). Column
headers toggle their panel; disabled columns stay visible, dimmed.

## View box and grid

The view box is the union of the boxes of all selected uses (fields in one
bundle may differ); rasters are transparent outside their own box, streamline
seeds stay inside the vector field's box. The sampling grid is `resolution`
along the longer side, or the fields' native grid when `resolution` is
deselected and every selected sampled field shares one grid filling the view
box (symbolic fields fall back to 128).

## Isolines and streamlines in the viewer

* Levels: `split` levels evenly spaced in the I_V field's *codomain parameter
  space* (so log codomains give log-spaced levels) centred on `value`,
  wrapping past min/max; deselected split = one level. Exact lines for
  symbolic fields, marching squares while the level is moving (see
  [performance.md](performance.md)). Colour from I_C per vertex.
* Streamlines: integrated through a sampled copy of the S_∇ field at the
  current grid, ½ cell per step. Particles are always drawn at the current
  phase; ▶ advances the clock, ◀ (shift-click) reverses the flow direction
  itself (◀ = against the field = descent). Plain click only toggles play.
  Space starts both animations when none is playing, otherwise pauses.

## Legend and cursor

Colormaps are per **field use**, so a field mapped to several slots gets one
bar listing its slots (`C I_C  x² + y²`); clicking cycles its colormap. Fields
that shape the picture without colouring it (I_V, the scalar behind S_∇) get
an all-black bar. Bars carry min/max (swapped for flipped codomains), a red
detent at the value at the bundle's centre point (a single-point set such as
θ*), white notches at the isoline levels (I_V's bar), and a white pip at the
cursor value. The cursor pane lists the space coordinates, the S_∇ vector,
then every legend scalar.

## Persistence and URLs

Per bundle in `localStorage["tensatory.opts.<file>"]`: every control, locked
slot selections, colormaps, animation directions, and the view — but a
*fitted* view is never saved (only flips/rotation), so it always re-fits to
the current layout; only user-panned/zoomed views are restored. Collapsed
panels are global. Query parameters override anything after load:
`?bundle=dense.json&iv=loss&sg=lossGrad&split=3&showScalar=0`.

## Fitting

`fit` (and initial placement) fits the box into the free region right of the
left stack with the same 12 px margin the panels keep from the viewport,
centred in the limiting dimension; resize re-fits unless the view was
customised.

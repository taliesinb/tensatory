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
| `src/main.ts` | state, slots and "uses", sampling cache, isolines / streamlines / glyph pipelines, legend, cursor pane, persistence, URL overrides, interaction, frame loop |
| `src/colormap.ts` | viridis / plasma / gray / okhue / turbo (polynomial fits) |
| `src/interval.ts` | the interval slider: two nullable ends, full / half / none kinds, drag / click / wheel / keyboard gestures, `cmap` variant drawn as a bracket |
| `src/cmapInterval.ts` | colormap interval selection: `Selection` (+ stretch / full, clip / mask modes), `selectParam`, `lutFor` (RGBA LUTs with masked alpha), the legend control `makeCmapInterval` |
| `src/log.ts` | console capture, the L (log) modal, `status()` and red error display |

## Compute and render modes

`compute ∈ {cpu, gpu}` × `render ∈ {canvas, gpu}` (system panel; `?compute=`,
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
  `bundle` (picker and space picker, both with wheel/arrow switching; about
  = the description on one line with the full text as tooltip; R = wipe
  storage, L = log, ⤒ = open a local JSON), `system` (closed by default:
  compute / render modes, device, `mem cap`, live memory with the adaptive
  resolution's last decision, and `iso res` — the current grid and segment /
  triangle count, read-only), `controls` (only when the bundle asks for
  rows, see below), `2D space` (points / box flags; view: fit,
  flip x, flip y, cw, ccw), `colorfield` (strip tick = raster on/off; smooth), `isolines` (value, split, opacity, `value sm` = box blur of the field (control id `metric`), `line sm` / `surf sm` = Taubin smoothing (id `line`),
  ▶ + rate), `streamlines` (dir: ascending / descending; mode: bi-strat /
  strat / JL / cover; lines, length, opacity, tail, split, ▶), `vector field`
  (off by default: spacing in px, glyph: arrow / head / solid triangle, opacity,
  the `longest` readout; see [glyphs.md](glyphs.md)).
* **mappings** matrix bottom-left, **legend** and **cursor pane** bottom-right,
  status line bottom centre (errors in red).

## Controls: reseed and rescale random directions

A bundle's `random` arrays and random directions of displaced nets may ask
for a row (`widget`, schema/distribution.ts; core `controlRows`). The
`controls` panel (`src/controls.ts`) shows one row per id: the label, a
nullable log-spaced scale slider (multiplier; unset = ×1, the bundle's own
scale) and ↻ (reseed: a fresh 32-bit salt; shift-click = the bundle's own
seeds). Rows hold **adjustments**, saved per bundle (`controls` in the
options); they never edit the bundle. Applying one = `state.bundle = new
Bundle(adjustSpec(baseSpec, adjust))`, `revision++`, and `clearFieldCaches()`
(sample / range caches, GPU grids / kernels / sets, recolourers) — the same
clears as a bundle switch minus selections, view, colormaps and intervals,
which are keyed by field id and stay valid. Nothing is recompiled: a
reseed or rescale changes constant VALUES, never shapes, so every program
emits byte-identical WGSL and the device's pipeline cache (keyed by code)
hits; only the packed `data` buffers differ, and those are uploaded per
dispatch anyway (`gpu/test/nets.test.ts` asserts the identity). The rebuild
is ~40 ms of JS for iris (bundle build + coarse CPU statistics for the
ranges); adjustments commit on release, the rows are inert while an
animation plays. Rows are per bundle, not per space (iris shows d₂ in the 2D
space, where it does nothing). Live dragging is the planned follow-up.

## Slots and the mappings matrix

Slots: **C** colorfield, **I_V** isoline value, **I_C** isoline colour,
**S_∇** streamline direction (vector), **S_C** streamline colour, **V_∇**
glyph vector field, **V_C** glyph colour. Rows are
every scalar *and* vector field of the bundle. A slot resolves its field to a
"use": a vector slot given a scalar uses its gradient (∇ glyph), a scalar
slot given a vector uses its norm (|·| glyph), otherwise the field itself.
Derived norms get a log-scaled codomain (`{min: 0, log: "10"}`): gradient
norms are heavy-tailed — thousands in the corners of the Rosenbrock box,
vanishing at the minimum — and a linear range hides everything but the corners.
Defaults: C = I_V = first scalar field, colour slots none (a colour slot equal
to C would paint lines the raster's own colour and hide them), S_∇ = V_∇ = the
field's `exactGradient` if any, else the field (→ its gradient). Column
headers toggle their panel; disabled columns stay visible, dimmed.

## View box and grid

The view box is the union of the boxes of all selected uses (fields in one
bundle may differ, the S_∇ and V_∇ fields included); rasters are transparent
outside their own box, streamline seeds and glyph lattices stay inside their
vector field's box. The sampling grid is the fields'
native grid when every selected sampled field shares one grid filling the
view box, else the adaptive resolution along the longer side
([resolution.md](resolution.md): two tiers, moving ≤ settled, ladder 32 …
2048, frame / latency / memory feedback; `?res=N` pins it).

## Isolines and streamlines in the viewer

* Levels: `split` levels evenly spaced in the I_V field's *codomain parameter
  space* (so log codomains give log-spaced levels) centred on `value`,
  wrapping past min/max; deselected split = one level. Exact lines for
  symbolic fields, marching squares while the level is moving (see
  [performance.md](performance.md)). Colour from I_C per vertex.
* Streamlines: integrated through a sampled copy of the S_∇ field on the
  *streamline grid*, ½ cell per step; `length` counts steps and `tail` counts
  cells of that grid. `dir` (ascending / descending, `?sdir=`, a per-bundle
  UI value) is the integration sign: each path starts at its seed and follows
  the S_∇ field or its inverse (one way, `bidirectional: false`, so only the
  ends of the lines bunch up where the flow converges — except in `bi-strat`,
  which integrates both ways through the seed). It is NOT the ▶/◀
  playback direction (`state.dir.stream`, saved
  per space, shift-click ▶): with a converging field, descending lines drain
  into the sinks — a different set of lines than ascending ones played
  backwards. `mode`
  (`?smode=bi-strat|strat|JL|cover`, a per-bundle UI value) picks core's
  seeding strategy and the `bidirectional` flag (`STREAM_MODES`; see
  [isolines.md](isolines.md)): stratified seeds are generated per frame key
  as before; JL and cover plans are sequential
  CPU work over the integrable (sampled) field, cached in `PLAN_CACHE` per
  (field, grid, mode, options). The CPU / read-back paths draw the plan's
  lines directly; the fused path hands the planned seeds with their step
  budgets to `fusedStreamlines`, which re-integrates them resident (so the
  colour field is still evaluated per vertex on the GPU). The streamline grid
  is the vector field's own grid, else a FIXED 128 over the view box
  (`streamGrid`, `STREAM_N`) — never `currentGrid`, whose adaptive resolution
  (and, before that, its native / 128 switching with the colourfield and
  isolines panels) would rescale the lines' step, `length` and `tail`.
  Particles are always drawn at the current
  phase; ▶ advances the clock, ◀ (shift-click) runs it backwards so the
  particles travel the other way along the same lines. Plain click only
  toggles play.
  Space starts both animations when none is playing, otherwise pauses.

## Legend and cursor

Colormaps are per **field use**, so a field mapped to several slots gets one
bar listing its slots (`C I_C  x² + y²`); clicking the field's *name* cycles
its colormap. Fields that shape the picture without colouring it (I_V, the
scalar behind S_∇ or V_∇) get an all-black bar. Bars carry min/max (swapped for
flipped codomains), a red detent at the value at the bundle's centre point (a
single-point set such as θ*), white notches at the isoline levels (I_V's bar),
and a white pip at the cursor value. The cursor pane lists the space
coordinates, the S_∇ and V_∇ vectors, then every legend scalar.

### Colormap interval selection

Every bar (except one used only as the S_∇ source) is an **interval slider**
(`src/interval.ts`, a two-ended sibling of the compact slider; prototyped in
`apps/ui-proto`) dressed as a colormap control (`src/cmapInterval.ts`). The
selection is drawn as a bracket `|‾‾‾|` over the bar and lives in the field's
codomain *parameter* space (0..1 along the bar), with independently nullable
ends: full `(lo, hi)`, half `(lo, ·)` / `(·, hi)`, none = everything.

Gestures: press a handle and drag to move it (clamped so lo ≤ hi); press
anywhere else and drag to move the whole bracket (a half's only end); click a
handle, or drag it off the bar, to delete it; on an empty bar drag out a new
interval (release past an end for a half) or click to get a small one (10 % of
the bar) at the pointer. Half bars: click the top line to add the missing end
there. Escape cancels, Backspace clears. A *click* is a press released within
150 ms (`CLICK_MS`); a longer press is a drag — only then does the cursor
change and pointer motion count (motion inside the dead zone is ignored), so
a hesitant press-and-release changes nothing. The plain (non-`cmap`) variant
used for the 3D crop ranges additionally clears a full interval when its band
is clicked; clicking outside the band still jumps the interval there.

Three **modes** are part of the selection and toggled by single clicks:
* *included region* (inside the bracket): **stretch** — the colormap is
  compressed to exactly `[lo, hi]` — or **full** — the ordinary colormap, the
  interval only masks / clips;
* *excluded region* (either side, independently): **clip** — saturated to the
  boundary colour — or **mask** — not drawn at all (barber-pole on the bar).

What "not drawn" means per use: **C** the raster is transparent there; **I_C**
/ **S_C** that stretch of a line is not drawn (GPU: per fragment, canvas: per
segment); **I_V** levels inside a masked range are not contoured at all
(notches disappear with them). Bars whose field has no colour use (I_V-only,
S_∇ source) are *fixed-mask*: clip / stretch mean nothing there, excluded
regions always mask, clicks toggle nothing. Colour values outside the colour
field's box (`undefined` / NaN) are likewise not drawn.

Implementation: `selectParam(sel, t)` is the whole semantics (parameter →
colormap parameter, NaN = masked); `lutFor(colormap, sel)` bakes it into a
256-entry RGBA LUT (α = 0 where masked) shared by the canvas raster, the GPU
raster and the GPU lines (shaders discard α < ½); canvas lines apply
`selectParam` per vertex. Selections are stored per use in
`state.intervals` and persisted with the other options; the legend is
rebuilt from that state (widgets are re-created on each `updateInfo`).

## Persistence and URLs

Per bundle in `localStorage["tensatory.opts.<file>"]`: every control, locked
slot selections, colormaps, colormap interval selections, animation
directions, Controls-pane adjustments (`controls`: `{ rowId: { seed?, scale? } }`), and the view — but a
*fitted* view is never saved (only flips/rotation), so it always re-fits to
the current layout; only user-panned/zoomed views are restored. Collapsed
panels are global. Query parameters override anything after load:
`?bundle=dense.json&iv=loss&sg=lossGrad&split=3&showScalar=0`.

## Fitting

`fit` (and initial placement) fits the box into the free region right of the
left stack with the same 12 px margin the panels keep from the viewport,
centred in the limiting dimension; resize re-fits unless the view was
customised.

## Streamline particles

Each line carries one particle window of `tail` cells (transparent tail →
full-colour head) sliding along the flow, or `split` windows spaced `len /
split` apart. Brightness is `(arc − w0) / tail` inside a window `[w0, w0 +
tail]` and nothing else — no fade-in at the start of a path (that dimmed every
line beginning at an inflow edge). With one particle the head runs over `len +
tail`, so it enters the path head-first and its tail slides off the end; the
Canvas renderer clips each segment to the window and shades the clipped piece,
matching the per-fragment GPU shader. `tail` deselected = solid full lines.

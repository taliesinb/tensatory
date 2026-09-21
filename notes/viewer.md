# The 2D viewer

`apps/viewer/` — Vite, vanilla TypeScript, Canvas 2D, no framework. The UI
is a port of the loss-landscape prototype's widgets, adapted to 2D fields.

## Files

| file | role |
|---|---|
| `index.html` | the panels; controls are plain `<div>`s configured by `data-` attributes |
| `src/widgets.ts` | tooltips (0.25 s), collapsible panels (state in `tensatory.collapsed`), tick glyphs over hidden checkboxes, the compact slider (drag scrub, shift-hover preview, nullable click/Backspace, wheel "document", arrow nudges, Escape cancels), the discrete slider (one step per wheel gesture), tab bars |
| `src/info.ts` | summary / details of bundles, spaces and fields: option hover texts, the ⓘ icon binding, the details modal |
| `src/metrics.ts` | the fields matrix: one SVG, hit-testing from pointer coordinates, shift-preview, click-to-lock, wheel/arrows, header clicks toggle panels, dimmed disabled columns, ∇ / \|·\| use glyphs |
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
  `bundle` (picker and space picker, both with wheel/arrow switching; a
  picker with a single alternative is shown as a plain label instead of a
  select — clicking a one-option select shows nothing — its ⓘ stays
  (`syncPickers`); each
  gets a ⓘ to its right when the selected bundle / space has a `summary` or
  `details` — hover = summary else details (the shared 0.25 s tooltip,
  `cursor: help` like the panel keys), click = details else summary in a
  modal; hovering an alternative in the dropdown shows its summary, else the
  first line of its details (the option `title`; for bundles not yet loaded
  that is the `summary` copied into `bundles/index.json`, which the core
  bundle test keeps in sync). R = wipe storage, L = log, ⤒ = open a local
  JSON), `system` (closed by default:
  compute / render modes (choice flippers; unavailable options greyed with
  the reason as tooltip), `mem cap`, device, live memory with the adaptive
  resolution's last decision, and `iso res` — the current grid and segment /
  triangle count, read-only), `controls` (only when the bundle asks for
  rows, see below), `2D space` (points / box flags; view: fit,
  flip x, flip y, cw, ccw), `colorfield` (strip tick = raster on/off; smooth), `isolines` (value, split, opacity, `value sm` = box blur of the field (control id `metric`), `line sm` / `surf sm` = Taubin smoothing (id `line`),
  ▶ + rate), `streamlines` (dir: ascending / descending; mode: bi-strat /
  strat / JL / cover; lines, length, opacity, tail, split, ▶), `vector field`
  (off by default: spacing in px, glyph: arrow / head / solid triangle, opacity,
  the `longest` readout; see [glyphs.md](glyphs.md)).
  Isolines and streamlines are OFF by default (the first frame of a costly
  field is then just the raster; the ticks are saved per bundle); a 3D space
  visited for the first time turns the isosurfaces on, since it has no raster.
  The four GATED panels (colorfield / isolines / streamlines / vector field:
  a `data-gate` ✓ tick on the strip) are `.off` while their tick is off: the
  controls stay as they are under a half-transparent panel-coloured sheet
  (`.body::after`) that takes every pointer event, so nothing in them hovers,
  drags, scrolls or previews; the strip stays live. Their titles' underlined
  first letters `c` / `i` / `s` / `v` are keyboard shortcuts that toggle the
  tick (plain keys only — ⌘C / ⌘V / ⌘S keep their meaning; `c` is inert in 3D
  where the panel is absent) and are the panels' column letters in the
  fields matrix.
  Locking a new I_V / S_∇ / V_∇ field while an animation plays pauses the
  animations first (`GEOMETRY_SLOTS` in `setSel`; space resumes): the new
  field's contours / integrations / lattices are recomputed — on the CPU for a
  costly field — and a re-render every frame on top would pile frames up.
  For the same reason a load or a space / bundle switch starts PAUSED whatever
  the saved ▶ ticks say; space resumes, and turning a ▶ on lifts the pause.
* **Bottom-left column** (`#bottomLeft`; the left stack's max-height leaves
  room for it): the **curves** panel — present when the space has curves —
  above the **fields** matrix. One row per curve: its name (click = show /
  hide; struck through when hidden), the shown range readout in the curve's
  parameter (`param`: name, unit, codomain formatting; "all" when
  unrestricted), a ⓘ for its summary / details, and beneath them an interval
  slider over [t₀, t₁] with the crop-range gestures (drag out a range, drag a
  handle or the band, click a handle to open that end, Backspace clears);
  while a handle is dragged the drawn range follows live (`onPreview`). No
  interval = the whole curve. A curve is drawn as a white polyline over the
  shown range (`CurveData.polyline`: a sampled curve's own samples are the
  vertices, cubic / symbolic / flow data is sampled at 512 points plus the
  samples), small dots at the samples of a sampled curve (≤ 120 of them, with
  their labels) and a red dot at the end of the shown range; in 3D the
  polyline is a depth-tested line layer and the dots live on the overlay.
  Per-bundle option `curves: { id: { on?, lo?, hi? } }`.
* **legend** and **cursor pane** bottom-right, status line bottom centre
  (errors in red).

## Logging and the Dock app

`src/log.ts` wraps `console.*`, `window` errors and unhandled rejections into
`LOG` (the L modal) and, on the dev server, SHIPS every line to
`POST /__tensatory/log` in 500 ms batches (`keepalive`, flushed on
`pagehide`); the Vite plugin `clientLog` (`vite.config.ts`) appends them to
`apps/viewer/.logs/client-<date>.log` with the server's local time and a
per-page-load session tag. The first line of a session says whether it is a
`browser tab` or a `web app (standalone)` (`display-mode: standalone`), the
UA, URL and whether `navigator.gpu` exists. The Dock app
(`~/Applications/Tensatory.app`) is Safari's "Add to Dock" template app
(`com.apple.Safari.WebApp`, start URL `http://127.0.0.1:5180/`): its own
localStorage container, no scriptable console — this file IS its console.

Boot is phased (`bootPhase`): `loading… WebGPU device` →
`loading… bundles/index.json` → `loading… bundle <file>`; after 2 s in a
phase the status appends the seconds elapsed and the log gets a warning at
5 s and every 15 s, so a stall names its phase. Before the main module runs,
an inline script in `index.html` collects early errors (drained into the log)
and, if the module has not started after 5 s, turns the status red with
"the main module did not start" — a failed import would otherwise leave
`loading…` forever with no visible error.

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

## Sweeps: the record rows

When the loaded document is a sweep ([sweeps.md](sweeps.md) §2), the bundle
panel gains rows between the `bundle` picker (the sweep, its ⓘ) and the
`space` picker (`apps/viewer/src/recordPane.ts`): `member` — the member's
name and ⓘ; one flipper per key that varies across the members and is not an
attribute — blue = this member's value, tinted = a DIRECT switch (a member
exists that differs in this key alone), plain = the value exists but reaching
it changes other keys too (the tooltip says which; the status line repeats it
after the jump), disabled = listed in `keys.values` but no member has it; and
`record` — the keys that do not vary plus the member's attributes (`params`,
`test acc`, …) as text. A click loads that member (`Sweep.member`: fetched
once with its sidecars, relative to its own document) and `setBundle`s it in
the same space when it has one. Options are keyed by the sweep file AND the
member's structural signature (`tensatory.opts.<file>#<hash>`), so members
with the same spaces and fields share slots, levels, colormaps, view and
camera; into a signature without saved options the current slots (where the
field exists), the 2D view / 3D camera (same space) or just the camera's
orientation (another 3D space: the distance re-fits to the new box) are
carried. `?member=<id>`; the last member is remembered per sweep
(`tensatory.member.<file>`).

## Slices: N-D spaces

A manifold with 3 < D ≤ 8 dimensions (core `sliceable`) is shown as an
axis-aligned 2D or 3D **slice** through the manifold's `origin` (default 0).
The bundle panel's `slice` row lists the dimensions 1…D as a multi-flipper:
clicking picks / unpicks (blue); the slice in use is tinted; when the pick has
2 or 3 dimensions and differs from the committed one a ✓ appears and commits
it. Committed slices are a per-space option (`slice: {m: [dims]}`, default
`[0, 1, 2]`); a commit rebuilds the bundle and re-enters the space (a 2-
slice gets the 2D arm, a 3-slice the 3D arm, everything else as usual).

The slice is a **spec rewrite** done before the Bundle is built
(`adjustedBundle`: adjustments → slices → box zooms; core
`sliceSpec(spec, manifold, dims)` in `bundle/slice.ts`), so nothing
downstream — zod, field data, the GPU transpiler (WGSL has no vec5), the arms
— ever meets a mixed-dimension object. Semantics are those of the N-D field
on the slice: scalars f(embed(p)); vectors the components along the slice;
expressions are rewritten on the normalized AST with N-D vector
subexpressions UNROLLED into their N components, so `|∇f|`, dot products and
cosine similarities keep their full N-D meaning and `grad` is differentiated
in N-D first. A derivative along a fixed dimension (or a fixed component of
a vector argument) needs the argument's own N-D expression, which is inlined
on demand (symbolic / pointwise / pullback data, following ids); sampled and
net arguments cannot be, and the field is then left out of the slice with a
reason in the `errors` row. Dense grids are sliced at the sample nearest the
origin; scalar net fields get their point input rewritten to `E·p + o`
(vector net fields are not sliced yet); point sets are projected (their
shadow); the manifold keeps its id, gets k dimensions, the sliced `dimNames`
and its `flow` if that field survived. `core/test/slice.test.ts` checks
slices against the N-D fields at embedded points (values, derivatives,
vector projections, dense grids, pullbacks, iris nets); the GPU agreement
test runs every N-D bundle through a 2D and a 3D slice.

## Slots and the fields matrix

Slots: **C** colorfield, **I_V** isoline value, **I_C** isoline colour,
**S_∇** streamline direction (vector), **S_C** streamline colour, **V_∇**
glyph vector field, **V_C** glyph colour. Rows are
every scalar *and* vector field of the bundle; a field with a `summary` /
`details` carries a ⓘ at the right edge of the name column (same hover /
click behaviour as the pickers' ⓘ; the name is shortened to make room). A slot resolves its field to a
"use": a vector slot given a scalar uses its gradient (∇ glyph), a scalar
slot given a vector uses its norm (|·| glyph), otherwise the field itself.
Derived norms get a log-scaled codomain (`{min: 0, log: "10"}`): gradient
norms are heavy-tailed — thousands in the corners of the Rosenbrock box,
vanishing at the minimum — and a linear range hides everything but the corners.
Defaults: C = I_V = first scalar field, colour slots none (a colour slot equal
to C would paint lines the raster's own colour and hide them), S_∇ = V_∇ = the
space's `flow` (a manifold's declared dynamical system ẋ = F(x)) if it has one,
else the field's `exactGradient` if any, else the field (→ its gradient). A
space with a `flow` also starts its streamline `dir` at *ascending* (forward in
time; the descending default is for gradients of losses) unless the bundle has
a saved choice. Column headers toggle their panel; disabled columns stay
visible, dimmed.

Field names are paths (`train/loss/setosa`): the table draws the name tree
flattened with a 9 px indent per level and the last segment as label (full
path in the tooltip). Subtrees are collapsed until their marker is clicked
(`+` / `−`; a heading toggles from anywhere in its name, a field that is also a
parent — `train/loss` — from its marker only, its name selects as usual).
A locked-selected field is never hidden: under a collapsed ancestor the
path to it is shown (dimmer `+`) and its unselected siblings are not, so the
table stays compact while showing what is on screen. Pure headings
(`train`) have no matrix cells; wheel / arrow stepping skips them. The
expansion state lives in the table for the session; `refresh()` redraws
when a selection moved into or out of a collapsed subtree. Selecting a
CPU-sampled (`costly`) field restarts the resolution ladder from the bottom
(`guardResolution`): the remembered settled resolution of a transpiled field
(up to 2048²) would otherwise be sampled on the main thread first — minutes
for the 120-example iris nets — before the controller could react.

## View box and grid

The view box is the union of the boxes of all selected uses (fields in one
bundle may differ, the S_∇ and V_∇ fields included); rasters are transparent
outside their own box, streamline seeds and glyph lattices stay inside their
vector field's box. The sampling grid is the fields'
native grid when every selected sampled field shares one grid filling the
view box, else the adaptive resolution along the longer side
([resolution.md](resolution.md): two tiers, moving ≤ settled, ladder 32 …
2048, frame / latency / memory feedback; `?res=N` pins it).

### Box zoom (`=` in, `-` out, `0` resets)

The wheel zooms the VIEW; `=` / `-` zoom the DOMAIN in / out: every symbolic
field of the current space (`symbolic` / `symbolicv` / `net` / `netv`, inline
arguments of pointwise fields and pullbacks included) gets its box scaled by
1/1.5 (`=`) or 1.5 (`-`) around its centre (core `zoomBoxes`, `bundle/zoom.ts`), sampled
fields keep their grid (a space with none of the former says so on the status
line). It is a spec adjustment like the Controls rows: the exponent is a
per-space option (`boxZoom: { space: k }`), `rebuildBundle()` applies
adjustments + zooms to the base spec and clears the field caches, and the
view / camera re-fits so the new margin is shown. Statistics follow the box
(symbolic ranges are computed on a grid over it), so colormap ranges widen
with the domain — which also means a lone centred blob (a Gaussian) looks
nearly the same at every zoom: the box fills the same pixels, the colours
and the isoline levels re-spread over the new range; the legend numbers and
the status line are the tell. Sliced N-D spaces zoom like any other (the zoom
is applied to the sliced spec, after the slice). A step compiles the kernels that bake the field box
(streamlines bake `A` / `B`): one or two pipelines per step, not per frame.

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

Per bundle in `localStorage["tensatory.opts.<file>"]` (a sweep member:
`tensatory.opts.<sweepFile>#<signature hash>`, see "Sweeps"): every control, locked
slot selections, colormaps, colormap interval selections, animation
directions, Controls-pane adjustments (`controls`: `{ rowId: { seed?, scale? } }`), curve visibility and
ranges (`curves`), and the view — but a
*fitted* view is never saved (only flips/rotation), so it always re-fits to
the current layout; only user-panned/zoomed views are restored. Collapsed
panels are global. Query parameters override anything after load:
`?bundle=dense.json&iv=loss&sg=lossGrad&split=3&showScalar=0`; `?bundle=`
may name a document that is not in `index.json` (`mnist-mlp/bundle.json`,
`loss-landscape/sweep-all.json`) — it joins the picker for the session;
`?member=` picks a sweep member.

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

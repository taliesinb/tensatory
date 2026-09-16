# Isolines and streamlines

`packages/core/src/iso/` and `packages/core/src/flow/`.

## Marching squares (`marchingSquares.ts`)

`marchingSquaresSegments(grid, values, level)` walks every cell, classifies
its four corners against the level, and emits linearly interpolated crossing
segments; saddle cells (codes 5/10) are disambiguated by the cell centre value;
cells touching NaN are skipped (values outside a field's own box are NaN in
the viewer). `joinSegments` chains segments into polylines by matching
endpoints in a bucket grid (searching the 3×3 neighbourhood and dropping
degenerate segments — both needed when a grid vertex lies exactly on the
level). Closed loops repeat their first point.

## Exact isolines for symbolic fields (`exact.ts`)

Marching squares is used **only to seed topology**. Then:

1. `projectToLevel(field, p, level, maxDist)` — damped Newton along ∇f
   (`p ← p − (f−c)∇f/|∇f|²`, halving the step while the residual does not
   drop), with a bracketing + bisection fallback along the gradient line when
   Newton stalls (stiff functions such as Rosenbrock). Coordinates on box faces
   are locked so contours still exit through the boundary exactly. Vanishing
   gradient, leaving the box, or jumping more than `maxDist` (another branch)
   → rejected, the seed vertex is kept.
2. Every seed vertex is projected; then each chord is **refined adaptively**:
   project its midpoint, split while the projected midpoint is more than
   `tolerance` off the chord. Point density follows curvature.

`contourField(field, grid, values, level, {tolerance})` dispatches: symbolic
→ exact, sampled → marching squares. It reports `method`, `vertexCount` and
`maxResidual` (typically 1e-11 for exact). The viewer passes a tolerance of
¼ pixel in world units.

Known limitation — **topology comes from the seed grid**: features thinner
than a cell are missed or fragmented (Rosenbrock's valley at low levels), and
seed vertices that lie on the true curve but are not neighbours along it get
connected. Planned fix: an interval-arithmetic evaluator over the expression
tree driving a quadtree — cells whose enclosure excludes the level are pruned,
the rest are subdivided — which guarantees no missed components and gives the
seed independent of any display grid.

## Smoothing

`fields/smooth.ts` `boxBlur(grid, values, radius)` — separable box blur of
grid values (the isolines panel's `metric`); blurred fields are contoured on
the grid, never projected (they are no longer the symbolic field).
`iso/smoothLine.ts` `taubinSmooth(line, iterations)` — shape-preserving λ|μ
smoothing of polylines (the `line` option), applied only to grid-contoured
lines.

## Streamlines (`flow/streamlines.ts`)

`integrateStreamlines(field, {count, maxSteps, step, sign, seed, box, mode})`:
RK4 on the *unit* field (so vertices are spaced `step` apart), stopping
where the field vanishes or the box is left. `sign = −1` flows against the
field (descent for a gradient). By default a line is integrated both ways
from its seed; with `bidirectional: false` (what the viewer passes) it
*starts* at its seed and runs in the chosen direction only, so line starts
stay as distributed as the seeds and lines only concentrate where the flow
converges — integrating both ways, the backward halves of descending lines
are ascending lines and pile up at the field's sources. Each line carries
its arc length and a random phase for particle animation.

`mode` is how oversampling is handled (`planStreamlines` returns the seeds
*and* the lines for any mode):

* `stratified` (default) — `streamlineSeeds`: one jittered seed per cell of a
  grid of ≈`count` cells filling the box, every line run to `maxSteps`. Cheap
  and embarrassingly parallel, but lines pile up where the field converges
  (the valley of a descent) and leave starved patches elsewhere.
* `evenly-spaced` — Jobard–Lefer (`evenlySpacedStreamlines`, 2D): separation
  d_sep = the seed-cell side for `count` (so `count` becomes a density), a
  line stops within d_test = ½·d_sep of another line (or of its own distant
  past), candidate seeds sit d_sep to either side of every vertex of the
  accepted lines (FIFO front, spatial hash of all vertices), and the jittered
  stratified seeds restart the front when it is exhausted (disconnected
  regions). Sequential; ~10–100 ms in the viewer. The result's seeds carry
  per-seed step **budgets** `[back, fwd]`, which `integrateFromSeeds` and the
  GPU kernels honour — re-integrating the seeds reproduces the plan exactly
  (f64) or to f32 accuracy, with no spatial test, so the fused kernel stays
  resident and its scratch / segment capacity is Σ budgets instead of
  lines × 2 × maxSteps.
* `coverage` (`coverageStreamlines`) — stratified, then up to two rounds of
  fill: vertices are histogrammed on the seed grid and every starved cell
  (fewer than ¼ of the median hit count of the occupied cells) gets one more
  jittered seed. Fixes the dark patches; the extra lines still drain into the
  valleys.

Both isolines (exact projection) and streamline integration also exist as GPU
kernels with agreement tests — see [gpu.md](gpu.md).

The viewer never integrates a symbolic vector field pointwise: it samples the
field once on a grid at the current resolution (over the in-view part of its
box) and integrates through the bilinear interpolant, as the 3D prototype did
with its stored gradient grid — O(grid) evaluations instead of
O(lines × steps × 4). See [performance.md](performance.md).

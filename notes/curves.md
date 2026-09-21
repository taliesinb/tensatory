# Curves

Parametrized paths γ: [t₀, t₁] → M as bundle citizens. Status: **built** —
`curves` in `BundleSpec`, every `CurveDataSpec` (symbolic / sampled / flow /
translate / scale) in `packages/core/src/curves`, the viewer's curves panel
([viewer.md](viewer.md) "Bottom-left column"); the bundles that used
`pointSets.ordered` are migrated. **Not built**: `along` / `velocity` as field
data (declared at the end of `schema/curves.ts`, not in the field-data
unions), slicing of non-sampled curves, adaptive `method.tol`, a 1-D arm.
This note records the reasoning. Ordering against other work:
[roadmap.md](roadmap.md).

## Why

`pointSets: { ordered: true }` is a drawing hint standing in for a geometric
object. The viewer draws a polyline through the points and puts the label on
the last one; that is all it can do, because an ordered list of points knows
nothing about *where between* two samples the path was, what its parameter is,
or how fast it moved. Every use we have wants more:

* the SGD trajectory of the loss-landscape volumes — 48 snapshots; t is the
  snapshot number (or the epoch), and the question is "how did the loss
  evolve along it" and "how does the step direction compare to ∇loss";
* the orbits, limit cycles and separatrices of `dynamical-systems.json` —
  RK4-integrated *at build time* into point sets, so they are frozen: a
  Controls row that changes the field cannot move them, and a zoom cannot
  refine them;
* the parameter-space trajectory itself (48 × 269 322, in the `.pt` files) —
  the same curve seen in ℝⁿ and, through the PCA chart, in ℝ³.

A curve is to a point set what a field is to a table of samples. The design is
therefore the field design again: a `kind` that is symbolic or sampled,
evaluation anywhere in the parameter interval, an exact or finite-difference
derivative, and derived objects.

## The vocabulary

```jsonc
"curves": {
  "sgd": {                                              // sampled: a training trajectory
    "domain": "pca",
    "param": { "name": "step", "unit": "snapshots" },
    "data": { "type": "sampled", "points": "traj.npz/coords", "interp": "cubic",
              "labels": ["init", …, "θ*"] }             // times default to the index
  },
  "circle": {                                           // symbolic: γ(t) written out
    "data": { "type": "symbolic", "interval": [0, 6.2832],
              "expr": { "op": "vec", "vals": [{ "op": "cos", "val": { "op": "coord", "index": 0 } },
                                              { "op": "sin", "val": { "op": "coord", "index": 0 } }] } }
  },
  "vanderpol/cycle": {                                  // flow: the integral curve of a vector field
    "domain": "vanderpol",
    "data": { "type": "flow", "field": "vanderpol/F", "start": [2, 0], "interval": [0, 6.67], "method": { "tol": 1e-8 } }
  },
  "descent": {                                          // flow of a SCALAR field = gradient descent from a point
    "data": { "type": "flow", "field": "loss", "start": [0.8, -0.3], "interval": [0, 10], "dir": "descending" }
  }
}
```

| data type | kind | what it is |
|---|---|---|
| `symbolic` | symbolic | `expr`: ℝ → ℝᴰ in the scalar language with `dimCount = 1` (`coord 0` is t) over `interval`; exact velocity by `grad` |
| `sampled` | sampled | `points [N, D]` at `times [N]` (default: the index); optional `velocities [N, D]` (the `exactGradient` analogue: makes cubic interpolation Hermite), `interp` linear / cubic / step, `labels`, `closed` |
| `flow` | symbolic | ẋ = F(x) from `start` over `interval` (negative t₀ integrates backwards); a scalar `field` means its gradient with `dir` ascending / descending — the slot rule; `method` = integrator, step or tolerance |
| `translate` / `scale` | inherited | curves push **forward**: γ + vec, origin + s·(γ − origin) — the field ops read the other way (fields pull back) |

`param` says what t means for display only (name, unit, a `codomain` for
scaling and formatting — log-spaced steps); it never changes values, like
field codomains.

## Fields along a curve

A curve's parameter interval is a 1-D manifold. Composing a field with the
curve gives a field *on that interval*:

```jsonc
"fields": {
  "sgd/loss":   { "kind": "scalar", "codomain": "celoss", "data": { "type": "along", "curve": "sgd", "field": "loss" } },
  "sgd/speed":  { "kind": "vector", "data": { "type": "velocity", "curve": "sgd" } },
  "sgd/descent": { "kind": "scalar", "data": { "type": "pointwise", "expr": { "op": "dot", "vecs": ["g", "v"] },
                   "vectors": { "g": { "type": "along", "curve": "sgd", "field": "lossGrad" }, "v": "sgd/speed" } } }
}
```

`along` is sampled iff the curve or the field is sampled (sample points: the
curve's times); `velocity` is exact for symbolic / flow / Hermite data and a
difference quotient for linearly interpolated samples. Their `domain` is the
curve's implicit 1-D parameter manifold `<curveId>/t` (box = the interval),
which a field may name explicitly so several along-curve fields of one curve
combine with the identical-sample-points rule. `derivative(0)` of `sgd/loss`
is d(loss)/dt for free; ⟨∇f(γ(t)), γ'(t)⟩ — the tangent quantity the
loss-landscape papers plot — is a `pointwise` over two along-curve vectors, as
above.

## Runtime (as built: `packages/core/src/curves/`)

`CurveData` (`curveData.ts`) mirrors the field data: `kind`, `dimCount`,
`interval`, `sampleTimes` (defined iff sampled), `pointInto(t, out)` /
`velocityInto(t, out)` (false outside the interval), `point(t)` /
`velocity(t)`, `sampleOn(ts)`, `polyline(a, b, n)` (the drawing vertices over
a sub-range: a sampled curve's own samples inside it plus the exact ends,
cubic / symbolic / flow data also `n` uniform points; the step interpolant
gives both ends of every jump) and `arcLength(t)`.

* `SymbolicCurveData`: the expression is normalized in the MANIFOLD's
  dimension (its vectors are D-vectors) with `coord 0` as t; the other
  coordinates, `coordv` and `grad` are rejected (`checkCurveExpr`); the
  velocity is the exact symbolic derivative (`diffVector` wrt coordinate 0).
* `SampledCurveData`: `times` strictly increasing (default the index, or an
  interval sampled uniformly); linear / step / cubic (Hermite with the given
  `velocities`, Catmull–Rom tangents otherwise); `closed` adds one segment
  back to the first point over one more time step; `velocity` of a linear
  curve is the chord slope (or the interpolated given velocities).
* `FlowCurveData`: integrates ẋ = F(x) from `start` (t = 0) once, lazily, by
  RK4 (or Euler) with a fixed `step` (default: the larger |t|-extent / 1000,
  capped at the interval), backwards for t₀ < 0 and forwards for t₁ > 0,
  into a finely sampled linear curve it evaluates through; `velocity` reads F
  at the point. A scalar `field` means its gradient (`dir` descending by
  default) — the slot rule — built with the same `SymbolicVectorFieldData`
  `grad` idiom the viewer uses. `method.tol` is a SpecError until adaptive
  stepping exists.
* `AffineCurveData` (`translateCurve` / `scaleCurve`): origin + s ⊙ (γ − origin)
  + vec, keeping the kind and the sample times; velocities scale.

`Bundle.curve(id)` builds lazily with cycle detection through a
`CurveResolver` (the bundle's fields, curves and arrays); `Bundle.curves`
lists the ones that built; `buildAll()` reports errors under `curves.<id>`.
`collectHandles` walks sampled points / times / velocities, flow starts and
inline fields, pullback vectors, so trajectories may be `.npz` members.
`sliceSpec` projects sampled curves (points and velocities over the kept
dims) and drops the others with a reason under `curves.<id>` in `dropped`.
Tests: `packages/core/test/curves.test.ts` (RK4 against exact solutions,
Hermite reproducing a parabola, the closed cycle, pushforwards, handles,
slices) and the dynamical-systems checks in `bundles.test.ts`.

## Viewer

Built: the **curves panel** above the fields matrix — per curve a name that
toggles drawing, an interval slider over [t₀, t₁] (no interval = the whole
curve; the drawn range follows a drag live), the range readout in the curve's
parameter and a ⓘ; drawn as a white polyline with sample dots (≤ 120, with
labels) and a red end dot, in 2D and 3D ([viewer.md](viewer.md)). Later:

* direction arrows and colouring by a slot (its own `along` value or a colour
  field); a **t scrubber** (the streamline particle animation is the same
  machinery) — by t or by arc length is a scrubber option, not a curve
  property;
* the cursor pane showing the field values *at γ(t)* while scrubbing —
  along-curve fields are useful before any 1-D plot exists;
* a 1-D arm (plots of `along` fields against t), the natural third arm after
  2D and 3D;
* charts (roadmap 4) showing the parameter-space trajectory and its PCA image
  as one object.

## Migration (done)

Every `pointSets` entry with `ordered: true` became a curve; `ordered` is
deprecated (still drawn). `dynamical-systems.json`: the pendulum orbit and the
Lorenz / Rössler attractors are `flow` curves (the build tool lost its
`orbit` RK4; the transients became the interval's t₀), the Hopf cycle is the
exact expression (cos φ, sin φ), the Van der Pol cycle a closed sampled curve
and the competition separatrix a sampled one (both still integrated by the
tool: a limit cycle's period and a stable manifold's two backward branches
are not one flow). The SGD trajectories of `dense.json`, `loss-landscape/mnist-convnet-pca/`
and `mnist-mlp/` are sampled curves with `param.name = "snapshot"`; the
mixture `centres` of the symbolic bundles were never paths and are plain
labelled points now.

## Left out, deliberately

* Reparametrization and arc-length parametrization as spec types — views of
  one curve, not different curves.
* Surfaces (ℝᵏ → ℝᴰ, k > 1) — the same idea, but they belong to charts
  (`mappings.ts`) where they have a customer.
* Random / generated families of curves — many curves with `/` ids; what
  varies across them is the sweep's metadata (sweeps.md §2).
* Curves on curves (a path in the space of paths) — no.

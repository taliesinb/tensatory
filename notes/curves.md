# Curves (design, not yet implemented)

Parametrized paths γ: [t₀, t₁] → M as bundle citizens. The types are in
`schema/curves.ts` (exported, compiled, not yet wired into `BundleSpec` or the
field-data unions); this note records the reasoning. Status against other
work: [roadmap.md](roadmap.md).

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

## Runtime

`CurveData` mirrors `ScalarFieldData` / `VectorFieldData`: `kind`,
`dimCount`, `interval`, `sampleTimes` (defined iff sampled), `point(t)`,
`velocity()` (data on the same interval, composable to acceleration),
`sampleOn(ts)`, `polyline(tolerance)` (adaptive chords — the exact-isoline
refinement applied to a parametric curve), `arcLength(t)`. `Bundle.curve(id)`
is built lazily like fields; `buildAll()` reports errors under `curves.<id>`;
`flow` reuses the streamline integrators (RK4 through a sampled copy of the
field on the grid in the viewer, exact expression evaluation in core) and the
GPU streamline kernels can carry a flow curve as a one-seed job. Slices
(`bundle/slice.ts`) project curves as they project point sets; the `.npz`
handle backend takes the trajectories (`points` is an `ArraySpec`).

## Viewer (later)

* A curve is a polyline with direction arrows and sample markers, coloured by
  a slot (its own `along` value or a colour field), and a **t scrubber**: the
  streamline particle animation is the same machinery; `param.codomain`
  formats the readout ("step 1 200"). By t or by arc length is a scrubber
  option, not a curve property.
* The cursor pane shows the current field values *at γ(t)* while scrubbing —
  along-curve fields are useful before any 1-D plot exists.
* A 1-D arm (plots of `along` fields against t) is the natural third arm after
  2D and 3D; it is not part of this design.
* Curves belong to a space like point sets; charts (roadmap 4) show the
  parameter-space trajectory and its PCA image as one object.

## Migration

Mechanical. Every `pointSets` entry with `ordered: true` becomes
`{ data: { type: "sampled", points } }` (times = the index) and `ordered` is
deprecated; unordered point sets — θ*, equilibria, minima, a sweep's members —
stay point sets. `dynamical-systems.json`'s orbits, cycles and separatrices
become `flow` curves and `tools/dynamical-systems/build.mjs` loses its RK4;
the convnet bundle's trajectory becomes a sampled curve with `param.name =
"step"`, and once the `.pt` checkpoints are exported (roadmap 5) the same
curve appears in parameter space as a `.npz` handle.

## Left out, deliberately

* Reparametrization and arc-length parametrization as spec types — views of
  one curve, not different curves.
* Surfaces (ℝᵏ → ℝᴰ, k > 1) — the same idea, but they belong to charts
  (`mappings.ts`) where they have a customer.
* Random / generated families of curves — many curves with `/` ids; what
  varies across them is the sweep's metadata (sweeps.md §2).
* Curves on curves (a path in the space of paths) — no.

# Symbolic N-D bundle

`node tools/symbolic-nd/build.mjs` writes `apps/viewer/public/bundles/symbolic-nd.json`: symbolic
fields on 4D and 5D manifolds, the test bed for the viewer's **slice** row.

The viewer cannot draw four dimensions. A manifold with 3 < D ≤ 8 dimensions is shown as an
axis-aligned 2D or 3D slice through the manifold's `origin` (default 0): pick 2 or 3 of the dimensions
in the bundle panel's slice row and commit with ✓. The slice is a spec rewrite done before the bundle
is built (core `sliceSpec`, `packages/core/src/bundle/slice.ts`; see `notes/viewer.md` → Slices), so
every field of the sliced space is an ordinary 2D / 3D field for the rest of the pipeline. N-D vector
subexpressions are unrolled, so `|∇f|` on a slice is the norm of the full N-D gradient, and a vector
field shows its components along the slice.

| space | D | what it shows |
|---|---|---|
| `gauss4` | 4 | one Gaussian with a full (rotated) precision matrix: tilted ellipses / ellipsoids in every slice; `p`, `log p`, `∇p`, `\|∇p\|` |
| `quad4` | 4 | an indefinite quadratic form (eigenvalues +, +, −, −): slices are bowls, saddles or hyperbolic sheets depending on the axes |
| `mix5` | 5 | three isotropic Gaussians; `origin` = the second mean, so the default slice passes through *that* peak and the others appear as shadows / tails |
| `rosen5` | 5 | the chained 5D Rosenbrock, `origin` at its minimum (1, …, 1); consecutive axes show the banana, non-consecutive ones a plain trough |
| `waves5` | 5 | Σ sin xᵢ + ½ Σ sin(xᵢ xᵢ₊₁): a periodic landscape with coupled neighbours; `flow` = the descent flow −∇f |

Point sets (peaks, means, the minimum) are projected onto the slice.

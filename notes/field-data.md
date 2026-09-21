# Runtime field data

`packages/core/src/fields/fieldData.ts`. A `ScalarFieldData` /
`VectorFieldData` is the object that *evaluates* a field once its spec has
been built; `ScalarField` / `VectorField` (in `bundle/`) wrap one with id,
name, codomain and domain.

## The two kinds

```
kind: "symbolic"   evaluable anywhere in `box`
kind: "sampled"    has a discrete support `samplePoints` (phase 1: a DenseGrid)
```

Both expose the same evaluator shape, `fn(p, pos)`: `pos` is p's position in
`samplePoints` when p *is* a sample (−1 otherwise). Sampled data returns the
stored value at `pos` and multilinear interpolation elsewhere; symbolic data
ignores `pos`. `sampleOn(grid)` evaluates on any `DenseGrid` (passing `pos`
through when the grid equals the support), `value(p)` returns `undefined`
outside the box, `stats()` is cached (over the support, or a default grid for
symbolic data).

`DenseGrid` (`geometry/grid.ts`) is row-major with the last axis fastest,
position (0,…,0) at box corner `a`; it converts between flat positions, grid
positions and points, and `locate(p)` gives fractional grid coordinates for
interpolation.

## Classes

| class | kind | notes |
|---|---|---|
| `DenseScalarFieldData` / `DenseVectorFieldData` | sampled | `Float64Array` data on a `DenseGrid`; optional precomputed stats |
| `SymbolicScalarFieldData` / `SymbolicVectorFieldData` | derived | a normalized AST plus `FieldArgs` (named scalar/vector data); kind, box and support are derived from the arguments (`deriveSupport`): sampled iff any argument is, all sampled arguments must have identical supports, box = intersection |
| `PulledBackScalarFieldData` / `…Vector…` | inherited | evaluates the inner datum at an axis-aligned affine preimage (`translate`, `scale`); moves box and support along; chain rule for derivatives. Vector *values* are left alone (no pushforward) |
| `ClosureScalarFieldData` | any | data backed by closures — used for numerical derivatives, vector components and value scaling |

`buildScalarFieldData(spec, dimCount, resolver)` turns specs into these,
resolving field-id references through the bundle (lazily, with cycle
detection).

## Derivatives, to any order

`ScalarFieldData.derivative(dim)` returns *field data* (not just a closure), so
derivatives compose:

* symbolic data → symbolic differentiation of the AST with the same
  arguments (cached per dim);
* sampled data → central differences on the grid (one-sided at the edges),
  interpolated off-grid — so a second difference is a difference of the
  differenced grid, not a derivative of the piecewise-linear interpolant;
* pullbacks → chain rule (inner derivative, scaled by 1/factor);
* `VectorFieldData.component(index)` gives a scalar datum (exact for
  symbolic vectors) to differentiate further.

The compile context of a `Symbolic*FieldData` resolves `argd {name, dims}`
by folding `derivative` over `dims`. This is what lets the viewer take
`∇|∇ mixture|` — a second derivative through the argument — exactly.

## Uses (viewer concept)

The viewer's mapping slots consume fields of a fixed rank. `useScalar(id)`
returns a scalar field as itself or a vector field as its **norm**;
`useVector(id)` returns a vector field as itself or a scalar field as its
**gradient**. Both are built as `Symbolic*FieldData` over the field's data,
cached, and carry an id (`norm:<id>`, `grad:<id>`), a display name
(`|∇loss|`, `∇(x² + y²)`) and, for norms, the `norm` codomain.

## Limitations / next

* Sparse supports are declared but not built. External array handles are
  loaded up front by `Bundle.load` (bundle-schema.md, "External arrays").
* Vector fields are plain ℝⁿ-valued functions; 1-forms are not distinguished,
  so `scale` pullbacks do not transform vector values.
* Fields on different manifolds are never combined; the viewer only shows 2D
  fields.

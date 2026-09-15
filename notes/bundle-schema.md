# The bundle schema

Source of truth: `schema/*.ts` (package `@tensatory/schema`, types only).
Runtime validation: zod schemas in `packages/core/src/**/spec.ts` and
`bundle/bundle.ts`, each declared as `z.ZodType<XxxSpec>` so the two cannot
drift without a typecheck failure. Everything ending in `Spec` is the JSON
shape; it is parsed into runtime classes that carry behaviour.

## Root

```jsonc
{
  "tensatory": "0.1",
  "name": "…", "description": "…",
  "manifolds": { "plane": { "numDims": 2, "dimNames": ["x", "y"], "dimWeights": [0.7, 0.3] } },
  "defaultManifold": "plane",          // optional when there is exactly one
  "fields":    { "<fieldId>": FieldSpec, … },
  "pointSets": { "<id>": { "points": [[1, 1]], "labels": ["θ*"], "ordered": false } }
}
```

* **Manifold**: phase 1 treats every manifold as ℝⁿ with the identity chart.
  `dimWeights` is free-form per-dimension metadata (e.g. PCA explained
  variance). Charts / affine frames are sketched in `schema/mappings.ts` and
  not implemented; the loss-landscape PCA volume will become an
  `AffineInjection` of a 3D frame into parameter space.
* If no manifolds are declared, one is inferred from the first field whose
  data makes the dimension evident (a box or a dense shape).
* **Point sets** are labelled points on a manifold; `ordered: true` draws them
  as a path (an optimizer trajectory). A single-point set is treated by the
  viewer as "the centre" (θ*) for legend detents.

## Fields

```jsonc
{ "kind": "scalar", "data": ScalarFieldDataSpec, "codomain": "celoss",
  "exactGradient": "lossGrad", "domain": "plane", "name": "loss", "description": "…" }
{ "kind": "vector", "data": VectorFieldDataSpec, "name": "∇loss" }
```

`exactGradient` names a vector field holding gradients measured during
collection (backprop). `codomain` is a visualization hint only (see below).

## Field data

| type | kind | meaning |
|---|---|---|
| `dense` / `densev` | sampled | values on a regular grid filling `box` (default unit box); the array's shape *is* the grid size (vectors: trailing axis of size D) |
| `sparse` / `sparsev` | sampled | points + values; **not implemented** |
| `symbolic` / `symbolicv` | symbolic | an expression of the coordinates over `box`, with named `consts` |
| `pointwise` / `pointwisev` | inherited | an expression over named `scalars` / `vectors` arguments (inline specs or field ids) |
| `translate` / `scale` | inherited | pull back another datum's domain (p ↦ p − vec; p ↦ origin + (p − origin)/scale) |

Kind rules: pointwise data is *sampled* iff any argument is, and then all
sampled arguments must share **identical** sample points (grid size and box);
its box is the intersection of the arguments' boxes. Pullbacks keep the kind
and move the sample points along. Details in [field-data.md](field-data.md).

## Arrays

`SizedArraySpec` (shape known without loading): `inline` (flat row-major
`data`), `constant`, `oneHot` / `manyHot`, `symbolic` / `symbolicv` (each cell
is an expression of its integer grid position relative to `origin`), and
`handle` (external `path` into npz / npy / zarr / bin — **not implemented**).
Dense grids and arrays are row-major ("C" order, last axis fastest), matching
numpy; grid position (0,…,0) sits at box corner `a`.

## Codomains

The 1-D value space of a scalar field, as hints: `min`/`max` (with open
ends), `log` base, `flip` ("smaller is better": reverse sliders and
colormaps), `wrap`, `unit`, `marks`. Predefined names: `lin`, `p`,
`fraction`, `percent`, `log`, `logp`, `-logp`, `celoss`, `similarity`,
`norm`, `distance`, `angle`. The runtime `Codomain` class maps values to a
slider parameter in [0, 1] (`toParam` / `fromParam`, log-aware and flipped)
and formats values compactly (`6.24·10⁻⁵`). Codomains never change values.

## Symbolic expressions

See [symbolic.md](symbolic.md). In JSON, a scalar expression is a number, a
bare name, or `{ "op": …, … }`; e.g. `x² + y²` is

```json
{ "op": "add", "vals": [{ "op": "square", "val": { "op": "coord", "index": 0 } },
                        { "op": "square", "val": { "op": "coord", "index": 1 } }] }
```

## From spec to runtime

`Bundle.parse(json)` validates, then builds fields **lazily** on first access
(so fields may reference each other by id in any order) with cycle detection;
`buildAll()` reports per-field errors without failing the whole bundle, and
the viewer lists them in the bundle panel. `Bundle.scalarField(id)` /
`vectorField(id)` return `ScalarField` / `VectorField` (id, name, codomain,
domain, `data: ScalarFieldData`).

Example bundles live in `apps/viewer/public/bundles/` and double as
integration tests (`packages/core/test/bundles.test.ts`).

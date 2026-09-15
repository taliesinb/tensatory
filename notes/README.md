# Tensatory architecture notes

Long-form notes on how the pieces fit together. `AGENTS.md` at the repo root is
the short summary; when a decision changes, update both.

| note | what it covers |
|---|---|
| [bundle-schema.md](bundle-schema.md) | the JSON bundle: manifolds, fields, field data, arrays, codomains, point sets; how specs become runtime objects |
| [symbolic.md](symbolic.md) | the expression language: syntax, namespaces, normalization, compilation (CSE), symbolic differentiation |
| [field-data.md](field-data.md) | runtime field data: `symbolic` vs `sampled`, sample grids, derivatives to any order, pullbacks, "uses" (gradient / norm) |
| [isolines.md](isolines.md) | marching squares, exact projected isolines, streamline integration, smoothing |
| [viewer.md](viewer.md) | the 2D viewer: panels, slots and the mappings matrix, legend, rendering, persistence, URL overrides |
| [performance.md](performance.md) | where time goes, what was done about it, and what is left |
| [roadmap.md](roadmap.md) | what comes next (array backends, quadtree seeding, 3D, GPU, server) |

## The one-paragraph version

A **bundle** is a JSON document describing scalar and vector **fields** on
low-dimensional Euclidean **manifolds**. A field's data is either **sampled**
(values on a regular grid, stored inline for now) or **symbolic** (an
expression in a small closed language, evaluable anywhere), and fields can be
**derived** from others pointwise (including exact gradients via symbolic
differentiation) or by pulling back the domain. `@tensatory/schema` holds the
types, `@tensatory/core` parses bundles (zod), builds field objects, samples
them on grids, differentiates them, contours them (exactly, for symbolic
fields) and integrates streamlines; it has no DOM dependency so it can run in
a browser, a worker, or on a server. `@tensatory/viewer` is a framework-free
Canvas 2D viewer whose UI is ported from the loss-landscape prototype.

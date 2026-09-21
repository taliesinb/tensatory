# Tensatory architecture notes

Long-form notes on how the pieces fit together. `AGENTS.md` at the repo root is
the short summary; when a decision changes, update both.

| note | what it covers |
|---|---|
| [bundle-schema.md](bundle-schema.md) | the JSON bundle: manifolds, fields, field data, arrays, codomains, point sets; how specs become runtime objects |
| [symbolic.md](symbolic.md) | the expression language: syntax, namespaces, normalization, compilation (CSE), symbolic differentiation |
| [nets.md](nets.md) | small neural networks in a bundle: `def` / `bind` / `displace` / `grad`, the array expression language, declared symbolic axes vs implicit batch, autodiff as a program rewrite, net-backed fields with exact derivatives, the CPU reference evaluator and the WGSL transpiler, the `costly` flag, the iris example checked against PyTorch |
| [field-data.md](field-data.md) | runtime field data: `symbolic` vs `sampled`, sample grids, derivatives to any order, pullbacks, "uses" (gradient / norm) |
| [isolines.md](isolines.md) | marching squares, exact projected isolines, streamline integration, smoothing |
| [glyphs.md](glyphs.md) | static vector-field glyphs: hex / FCC lattices as coset grids, view-driven spacing, arrows normalized to the longest sampled vector, the fused atomicMax kernel |
| [viewer.md](viewer.md) | the 2D viewer: panels, slots and the mappings matrix, legend, rendering, persistence, URL overrides |
| [performance.md](performance.md) | where time goes, what was done about it, and what is left |
| [gpu.md](gpu.md) | the WebGPU backend: WGSL transpilation, compute-shader sampling, CPU/GPU agreement tests, Dawn-in-node pitfalls |
| [3d.md](3d.md) | spaces and the 3D arm: marching tetrahedra, exact projection, the WebGPU 3D renderer, face outlines, crop, 3D streamlines |
| [resolution.md](resolution.md) | the adaptive resolution (two tiers, frame / latency / memory feedback), sets sized from measured complexity, the memory cap, device limits |
| [curves.md](curves.md) | curves — parametrized paths as bundle citizens: symbolic / sampled / flow data, pushforwards, the runtime and the viewer's curves panel (built); fields along a curve and its velocity on the parameter interval (design) |
| [sweeps.md](sweeps.md) | the design of external arrays (§1, now built — see bundle-schema.md) and of sweeps (§2, not built): members with flat records, `common`, faceted navigation, options keyed by structural signature |
| [roadmap.md](roadmap.md) | what comes next (array backends and the rest of the unimplemented schema, quadtree seeding, 3D round two, GPU, server), and what is done since phase 1 |

## The one-paragraph version

A **bundle** is a JSON document describing scalar and vector **fields** on
low-dimensional Euclidean **manifolds**. A field's data is either **sampled**
(values on a regular grid, stored inline for now) or **symbolic** (an
expression in a small closed language, evaluable anywhere), and fields can be
**derived** from others pointwise (including exact gradients via symbolic
differentiation) or by pulling back the domain. `@tensatory/schema` holds the
types, `@tensatory/core` parses bundles (zod), builds field objects, samples
them on grids, differentiates them, contours them (exactly, for symbolic
fields), integrates streamlines and lays out vector glyphs; it has no DOM dependency so it can run in
a browser, a worker, or on a server. `@tensatory/viewer` is a framework-free
Canvas 2D viewer whose UI is ported from the loss-landscape prototype.

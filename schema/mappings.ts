// Mappings between manifolds. Not implemented in phase 1; kept as the intended
// shape of charts. The loss-landscape prototype's PCA volume is an
// AffineInjection: a 3D frame (origin theta*, 3 direction vectors) into the
// full parameter space.

// import type { ArrayHandleSpec } from "./arrays";
// import type { PointSpec } from "./geometry";
// import type { ManifoldId } from "./manifolds";
// import type { DimIndex } from "./math";

// export type AffineFrame = {
//   type: "frame",
//   space: ManifoldId,
//   origin?: PointSpec,
//   basis?: BasisSpec,
// }

// export type BasisSpec = DenseBasisSpec | SparseBasisSpec;

// export type DenseBasisSpec = {
//   type: "dense";
//   basis: ArrayHandleSpec;  // [k, numDims]: rows are basis vectors
// };

// export type SparseBasisSpec = {
//   type: "sparse";
//   dims: DimIndex[]   // e.g. [0, 3, 4]
// };

// export type AffineInjection = {
//   type: "injection",
//   src: AffineFrame,
//   dst: AffineFrame,
// }

// export type CoordinateInjection = {
//   type: "coordinate",
//   src: ManifoldId,
//   dst: ManifoldId,
//   dstCoords: DimIndex[],
// }

export {};

// The name "Manifold" here refers to a finite-dimensional *differentiable*
// manifold, i.e. one with a finite-dim tangent space at each point.
//
// Phase 1 assumption: every manifold is R^numDims with the identity chart; a
// field's box lives directly in those coordinates. Charts / affine frames come
// with `mappings.ts`.

import type { DimCount, Real, ShowString } from "./math";

export type ManifoldId = string;

export type ManifoldSpec = ManifoldId | ManifoldDefinitionSpec;

export type ManifoldDefinitionSpec = {
  name?: ShowString; // display name for the entire space, e.g. 'random', 'pca'; defaults to its id
  numDims: DimCount;
  dimNames?: ShowString[]; // for low dim spaces, e.g. ['x', 'y', 'z']
  dimWeights?: Real[]; // per-dimension importance, e.g. PCA explained variance
};

//   injects?: [SubspaceDescriptor],
//   projects?: [SubspaceDescriptor],
//   bijects?: [SubspaceDescriptor],
// };

// The name "Manifold" here refers to a finite-dimensional *differentiable*
// manifold, i.e. one with a finite-dim tangent space at each point.
//
// Phase 1 assumption: every manifold is R^numDims with the identity chart; a
// field's box lives directly in those coordinates. Charts / affine frames come
// with `mappings.ts`.

import type { FieldId } from "./fields";
import type { Point } from "./geometry";
import type { DimCount, Real, ShowString } from "./math";

export type ManifoldId = string;

export type ManifoldSpec = ManifoldId | ManifoldDefinitionSpec;

export type ManifoldDefinitionSpec = {
  name?: ShowString; // display name for the entire space, e.g. 'random', 'pca'; defaults to its id
  summary?: ShowString; // one line
  details?: ShowString; // any length
  numDims: DimCount;
  dimNames?: ShowString[]; // for low dim spaces, e.g. ['x', 'y', 'z']
  dimWeights?: Real[]; // per-dimension importance, e.g. PCA explained variance
  origin?: Point; // the point axis-aligned slices pass through (viewer: N-D spaces are shown as 2D / 3D slices); defaults to 0
  flow?: FieldId; // a vector field on this manifold that is the time evolution of a dynamical system, ẋ = flow(x);
                  // a visualization hint: the viewer's default streamline / glyph source for the space
};

//   injects?: [SubspaceDescriptor],
//   projects?: [SubspaceDescriptor],
//   bijects?: [SubspaceDescriptor],
// };

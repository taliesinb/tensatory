// The root object of a Tensatory data bundle.

import type { FieldId, FieldSpec } from "./fields";
import type { Point } from "./geometry";
import type { ManifoldDefinitionSpec, ManifoldId } from "./manifolds";
import type { ShowString } from "./math";
import type { NetId, NetSpec } from "./nets";

export const BUNDLE_VERSION = "0.1";

export type PointSetId = string;

// labelled points on a manifold, e.g. the trained model theta*, or an
// optimizer trajectory (ordered)
export type PointSetSpec = {
  domain?: ManifoldId;
  points: Point[];
  labels?: ShowString[]; // one per point
  ordered?: boolean; // true: draw as a path (trajectory); defaults to false
  name?: ShowString;
};

export type BundleSpec = {
  tensatory: typeof BUNDLE_VERSION;
  name?: ShowString;
  summary?: ShowString; // one line
  details?: ShowString; // any length
  manifolds?: Record<ManifoldId, ManifoldDefinitionSpec>;
  defaultManifold?: ManifoldId; // used by fields without `domain`; defaults to the sole manifold if there is exactly one
  fields: Record<FieldId, FieldSpec>;
  pointSets?: Record<PointSetId, PointSetSpec>;
  nets?: Record<NetId, NetSpec>; // small neural networks (nets.ts); `net` / `netv` field data refers to them
};

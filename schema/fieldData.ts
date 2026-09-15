// A *FieldDataSpec describes how the values of a field are obtained. It is
// parsed into a runtime ScalarFieldData / VectorFieldData object
// (see @tensatory/core `fields/fieldData.ts`), which has
//   kind: "symbolic" | "sampled"
// A "sampled" field has a discrete support (`samplePoints`); a "symbolic" one
// can be evaluated anywhere in its box.
//
// Derivation rules for `kind`:
// * dense/sparse    -> sampled
// * symbolic        -> symbolic
// * pointwise       -> sampled if ANY argument is sampled, else symbolic.
//                      All sampled arguments must have IDENTICAL sample points
//                      (same grid size and box); it is an error otherwise.
// * translate/scale -> same kind as the wrapped data (sample points move with it)

import type { ArraySpec, SizedArraySpec } from "./arrays";
import type { FieldId } from "./fields";
import type { BoxSpec, ScaledSpaceOpSpec, TranslatedSpaceOpSpec } from "./geometry";
import type { Real } from "./math";
import type { ScalarStatistics } from "./statistics";
import type { ConstArgName, ScalarArgName, SymbolicScalar, SymbolicVector, VectorArgName } from "./symbolic";

export type ScalarFieldDataSpec =
  | SampledScalarFieldDataSpec
  | DerivedScalarFieldDataSpec
  | SymbolicScalarFieldDataSpec;

export type VectorFieldDataSpec =
  | SampledVectorFieldDataSpec
  | DerivedVectorFieldDataSpec
  | SymbolicVectorFieldDataSpec;

/*************************************************/
/* sampled: backed by array(s) of sampled values */

export type SampledScalarFieldDataSpec = DenseSampledScalarFieldDataSpec | SparseSampledScalarFieldDataSpec;
export type SampledVectorFieldDataSpec = DenseSampledVectorFieldDataSpec | SparseSampledVectorFieldDataSpec;

// samples on a regular grid filling `box`; array shape [S_0, ..., S_{D-1}] IS the grid size
export type DenseSampledScalarFieldDataSpec = {
  type: "dense";
  samples: SizedArraySpec;
  stats?: ScalarStatistics; // optional, precomputed statistics of sampled values
  box?: BoxSpec; // defaults to the unit box
};

// array shape [S_0, ..., S_{D-1}, D]: last axis holds the vector components
export type DenseSampledVectorFieldDataSpec = {
  type: "densev";
  samples: SizedArraySpec;
  box?: BoxSpec; // defaults to the unit box
};

// not supported in phase 1
export type SparseSampledScalarFieldDataSpec = {
  type: "sparse";
  points: ArraySpec<2>; // [N, D], rows are points
  samples: ArraySpec<1>; // [N], scalar sample at the corresponding point
  stats?: ScalarStatistics;
  box?: BoxSpec; // defaults to bbox of points
};

// not supported in phase 1
export type SparseSampledVectorFieldDataSpec = {
  type: "sparsev";
  points: ArraySpec<2>; // [N, D], rows are points
  samples: ArraySpec<2>; // [N, D], vector sample at the corresponding point
  box?: BoxSpec; // defaults to bbox of points
};

/*********************************************/
/* derived: computed from other field data */

export type DerivedScalarFieldDataSpec = PullbackScalarFieldDataSpec | PointwiseScalarFieldDataSpec;
export type DerivedVectorFieldDataSpec = PullbackVectorFieldDataSpec | PointwiseVectorFieldDataSpec;

// transform the domain of existing field data
export type PullbackScalarFieldDataSpec =
  | TranslatedSpaceOpSpec<ScalarFieldDataSpec | FieldId>
  | ScaledSpaceOpSpec<ScalarFieldDataSpec | FieldId>;
export type PullbackVectorFieldDataSpec =
  | TranslatedSpaceOpSpec<VectorFieldDataSpec | FieldId>
  | ScaledSpaceOpSpec<VectorFieldDataSpec | FieldId>;

// symbolic, derived pointwise from other fields (inline specs or references to
// fields of the bundle by id). Names must be unique across consts/scalars/vectors.
export type PointwiseScalarFieldDataSpec = {
  type: "pointwise";
  expr: SymbolicScalar; // e.g. {op: "add", vals: [{op: "gauss", val: {op: "coord", index: 0}}, "loss"]}
  consts?: Record<ConstArgName, Real>;
  scalars?: Record<ScalarArgName, ScalarFieldDataSpec | FieldId>;
  vectors?: Record<VectorArgName, VectorFieldDataSpec | FieldId>;
};

export type PointwiseVectorFieldDataSpec = {
  type: "pointwisev";
  expr: SymbolicVector;
  consts?: Record<ConstArgName, Real>;
  scalars?: Record<ScalarArgName, ScalarFieldDataSpec | FieldId>;
  vectors?: Record<VectorArgName, VectorFieldDataSpec | FieldId>;
};

/******************************************************/
/* symbolic: not based on other fields, only position */

export type SymbolicScalarFieldDataSpec = {
  type: "symbolic";
  expr: SymbolicScalar; // e.g. {op: "add", vals: [{op: "square", val: {op: "coord", index: 0}}, {op: "square", val: {op: "coord", index: 1}}]}
  box?: BoxSpec; // defaults to the unit box
  consts?: Record<ConstArgName, Real>;
};

export type SymbolicVectorFieldDataSpec = {
  type: "symbolicv";
  expr: SymbolicVector;
  box?: BoxSpec; // defaults to the unit box
  consts?: Record<ConstArgName, Real>;
};

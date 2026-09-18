import type { IntNonNeg, Int, Real } from "./math";
import type { ConstArgName, SymbolicScalar, SymbolicVector } from "./symbolic";
import type { RandomWidgetSpec, ScalarDistributionSpec } from "./distribution";

export type ArrayPath = string;
// "/" has special meaning. "foo/bar", depending on underlying storage, means:
// - Zarr: the array node "bar" within the group node "foo"
// - `.npz` files: the array "foo/bar"
// - `.npy` files: the array "foo/bar.npy"
// - `.bin` files: the raw float32 array "foo/bar.bin"

export type AxisPos = Int; // position along an axis; negative counts from the end (python-style)
export type AxisSize = IntNonNeg; // size of a dimension

export type ArrayShape = AxisSize[]; // e.g. [40, 40, 40]
export type ArrayPart = AxisPos[]; // leading indices to fix, e.g. [0] (first row), [-1] (last row), [0, -1]

/*******************************************************/
// a SizedArraySpec fully determines an array (shape known without loading)

export type SizedArraySpec =
  | InlineArraySpec
  | ConstantArraySpec
  | OneHotArraySpec
  | ManyHotArraySpec
  | SymbolicArraySpec
  | SymbolicRandomArraySpec
  | SizedArrayHandleSpec;

// values embedded directly in the JSON, flattened in row-major order
export type InlineArraySpec = {
  type: "inline";
  shape: ArrayShape;
  data: Real[]; // length must equal the product of `shape`
};

export type ConstantArraySpec = {
  type: "constant";
  shape: ArrayShape;
  value: Real;
};

export type OneHotArraySpec = {
  type: "oneHot";
  shape: [AxisSize] | [AxisSize, AxisSize]; // e.g. [3] for vector, [3,5] for matrix
  pos: AxisPos | AxisPos[]; // for vector: pos of the hot element; for matrix: one pos per row
  hot?: Real; // defaults to 1.0
  cold?: Real; // defaults to 0.0
};

export type ManyHotArraySpec = {
  type: "manyHot";
  shape: [AxisSize] | [AxisSize, AxisSize]; // e.g. [3] for vector, [3,5] for matrix
  pos: AxisPos[] | AxisPos[][]; // for vector: pos list of the hot elements; for matrix: one pos list per row
  hot?: Real; // defaults to 1.0
  cold?: Real; // defaults to 0.0
};

export type SymbolicArraySpec = SymbolicScalarArraySpec | SymbolicVectorArraySpec;

// each cell's value is an expression of its (integer) grid position
export type SymbolicScalarArraySpec = {
  type: "symbolic";
  shape: ArrayShape;
  origin?: AxisPos[]; // which cell is considered (0,0,...); defaults to (0,0,...)
  expr: SymbolicScalar; // `coordv` / `coord` refer to the cell position relative to `origin`
  consts?: Record<ConstArgName, Real>; // constants that can be referenced by name
};

// the last axis holds vector components: its size must equal the number of other axes
export type SymbolicVectorArraySpec = {
  type: "symbolicv";
  shape: ArrayShape;
  origin?: AxisPos[]; // one shorter than `shape`
  expr: SymbolicVector;
  consts?: Record<ConstArgName, Real>;
};

// data stored externally (npz / npy / zarr / bin); not supported in phase 1
export type SizedArrayHandleSpec = {
  type: "handle";
  shape: ArrayShape; // the size of the resulting array, must match what is loaded after `part` is applied
  path: ArrayPath;
  part?: ArrayPart;
};

export type SymbolicRandomArraySpec = SymbolicRandomScalarArraySpec;

// each cell's value is an independent draw from `dist` (see distribution.ts);
// the array is a pure function of the spec and the distribution's seed
export type SymbolicRandomScalarArraySpec = {
  type: "random";
  shape: ArrayShape;
  dist: ScalarDistributionSpec;
  widget?: RandomWidgetSpec | null; // present: a row in the viewer's Controls pane (reseed; scale slider for location–scale dists)
};

/*******************************************************/
// an ArraySpec may leave the shape to be discovered on load.
// The generic parameter documents the expected number of dimensions.

export type ArraySpec<_N extends number = number> = SizedArraySpec | ArrayPath | ArrayHandleSpec;

export type ArrayHandleSpec = {
  type: "handle";
  path: ArrayPath;
  part?: ArrayPart;
};

/*******************************************************/

// a description of a single cell in an array
export type CellSpec = {
  type: "cell";
  path: ArrayPath;
  part: ArrayPart; // must index all axes
};

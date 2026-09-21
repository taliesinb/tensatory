import type { IntNonNeg, Int, Real } from "./math";
import type { ConstArgName, SymbolicScalar, SymbolicVector } from "./symbolic";
import type { RandomWidgetSpec, ScalarDistributionSpec } from "./distribution";

// A path to a stored array, relative to the bundle document (so a bundle with
// external arrays is a directory: the JSON plus its sidecars). The FORMAT
// FOLLOWS THE PATH:
// - "vol.bin"            raw, headerless, row-major; needs `shape` and `dtype` (default float32)
// - "w.npy"              numpy .npy (any version, any dtype below, C or Fortran order, either endianness)
// - "data.npz/foo/bar"   the member "foo/bar" (or "foo/bar.npy") of the zip archive "data.npz"
// - "vol.zarr/foo/bar"   the array node "foo/bar" of the zarr store "vol.zarr" (or the store's root array,
//                        "vol.zarr"); zarr v2 (`.zarray`) and v3 (`zarr.json`); compressors null / zlib / gzip
// Self-describing formats are validated against `shape` / `dtype` when given.
export type ArrayPath = string;

// the element type of a stored array; kept as the matching JS typed array on load (64-bit integers become numbers)
export type Dtype = "float32" | "float64" | "int8" | "uint8" | "int16" | "uint16" | "int32" | "uint32" | "int64" | "uint64" | "bool";

export type AxisPos = Int; // position along an axis; negative counts from the end (python-style)
export type AxisSize = IntNonNeg; // size of a dimension

export type ArrayShape = AxisSize[]; // e.g. [40, 40, 40]
// per axis: an integer FIXES that axis at the position and drops it, `null` KEEPS it; trailing nulls may be omitted.
// [0] = first row; [-1] = last row; [null, null, null, 0] = channel 0 of a channel-interleaved (X, Y, Z, C) volume;
// a part fixing every axis is one cell.
export type ArrayPart = (AxisPos | null)[];

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

// a permutation of the KEPT axes (after `part`), numpy `transpose` semantics: result axis i is kept axis `axes[i]`.
// [2, 1, 0] turns a volume stored (z, y, x) into (x, y, z).
export type ArrayAxes = IntNonNeg[];

// data stored externally (see ArrayPath). `shape` is the STORED shape (what the file holds, before `part`):
// required knowledge for .bin, validation for self-describing formats. The result has the kept axes, permuted by `axes`.
export type SizedArrayHandleSpec = {
  type: "handle";
  shape: ArrayShape;
  path: ArrayPath;
  part?: ArrayPart;
  axes?: ArrayAxes;
  dtype?: Dtype; // .bin: the stored element type (defaults to float32); other formats: validated
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
  axes?: ArrayAxes;
  dtype?: Dtype;
};

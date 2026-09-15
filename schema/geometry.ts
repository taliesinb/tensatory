import type { ArraySpec } from "./arrays";
import type { Count, Index, Real } from "./math";

/*******************************************************/
// e.g. 0.5 for a coordinate along a single dimension
export type Coord = Real;

// e.g. [0, 1] for the range 0 to 1 of a single dimension
export type Interval = [Coord, Coord];

/*******************************************************/
// a box (axis-aligned) in some Euclidean chart

export type BoxSpec = BoxIntervals | BoxCorners;

// e.g. [[0, 1], [0, 1], [0, 1]] for a 3D box
export type BoxIntervals = Interval[];

// e.g. {a: [0,0,0], b: [1,1,1]} for a 3D box
export type BoxCorners = {
  a: Point;
  b: Point;
};

/*******************************************************/
// a point in a Euclidean space
// WHICH space depends on the structure this occurs in

export type Point = Coord[]; // e.g. [0.0, 1.0, 2.0] for a 3D point

export type PointSpec = Point | ArraySpec<1>;

/*******************************************************/
// a displacement between two points in a Euclidean space
// WHICH space depends on the structure this occurs in

export type Vector = Coord[];

export type VectorSpec = Vector | ArraySpec<1>;

/*******************************************************/
/* TRANSFORMATIONS OF SPACES (pullbacks) */

// the wrapped thing is evaluated at (p - vec)
export interface TranslatedSpaceOpSpec<T> {
  type: "translate";
  arg: T;
  vec: VectorSpec;
}

// the wrapped thing is evaluated at ((p - origin) / scale) + origin,
// i.e. the wrapped thing appears scaled by `scale` about `origin`
export interface ScaledSpaceOpSpec<T> {
  type: "scale";
  arg: T;
  origin?: PointSpec; // defaults to the origin
  scale: Real | Vector; // uniform, or one factor per dimension
}

/*******************************************************/
/* SAMPLE POINTS: the discrete support of sampled data */

export type SampleCount = Count;
export type SamplePos = Index; // 0 <= pos < sampleCount
export type GridPos = Index[]; // one index per dimension
export type GridSize = Count[]; // the number of samples in each dimension, e.g. [10, 10]

// Dense grids are always stored in row-major ("C") order: the LAST axis varies
// fastest, matching numpy's default layout, so sample `pos` of a grid with size
// [S_0, ..., S_{D-1}] has grid position (i_0, ..., i_{D-1}) with
//   pos = ((i_0 * S_1 + i_1) * S_2 + ...) + i_{D-1}
// Grid position (0, ..., 0) sits at box corner `a`, (S_0-1, ..., S_{D-1}-1) at corner `b`.

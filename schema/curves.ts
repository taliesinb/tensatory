// Curves: parametrized paths γ: [t0, t1] → M in a manifold. DESIGN, NOT
// IMPLEMENTED: these types compile and are exported, but BundleSpec has no
// `curves` yet, the runtime has no Curve object, and `along` / `velocity` are
// not members of the field-data unions. notes/curves.md is the long form.
//
// Why: `pointSets: { ordered: true }` is a drawing hint standing in for a
// geometric object. A curve deserves what fields got — a `kind` that is
// symbolic or sampled, evaluation anywhere in its parameter interval, an exact
// or finite-difference derivative (its velocity), and derived objects: the
// integral curve of a vector field, and fields composed with a curve
// ("loss along the SGD trajectory"), which live on the curve's 1-D parameter
// interval. Curves push FORWARD under maps of the manifold; fields pull back.
//
// Kind rules (as for field data, schema/fieldData.ts):
// * symbolic -> symbolic          * sampled -> sampled
// * flow     -> symbolic (evaluated by integration to any t; the integrator's
//               tolerance is a method option, not a sample grid)
// * translate / scale -> the kind of the wrapped curve
// * along(curve, field) -> sampled iff the curve OR the field is sampled

import type { ArraySpec } from "./arrays";
import type { CodomainSpec } from "./codomain";
import type { FieldId } from "./fields";
import type { ScalarFieldDataSpec, VectorFieldDataSpec } from "./fieldData";
import type { Interval, PointSpec, ScaledSpaceOpSpec, TranslatedSpaceOpSpec } from "./geometry";
import type { ManifoldId } from "./manifolds";
import type { Real, RealPos, ShowString } from "./math";
import type { ConstArgName, SymbolicVector } from "./symbolic";

// "/"-separated paths like field ids, drawn as a tree: "sgd/seed0", "cycles/vanderpol"
export type CurveId = string;

/*******************************************************/
/* ROOT */
//
// BundleSpec gains
//   curves?: Record<CurveId, CurveSpec>;
// `pointSets` stay for UNORDERED labelled points (θ*, equilibria, minima, the
// members of a sweep); `ordered: true` becomes deprecated sugar for a sampled
// curve whose `times` are the index.

export type CurveSpec = {
  data: CurveDataSpec;
  domain?: ManifoldId; // the manifold the curve lives IN; defaults like fields (the bundle's default manifold)
  param?: ParamSpec; // what t means; defaults to { name: "t" }
  name?: ShowString;
  summary?: ShowString; // one line
  details?: ShowString; // any length
};

// the parameter, for display: a training trajectory's t is a step or an epoch, a flow's is time
export type ParamSpec = {
  name?: ShowString; // "t", "step", "epoch"
  unit?: string | null;
  codomain?: CodomainSpec; // scaling / formatting of t (log-spaced steps, …); hints only, never changes values
};

/*******************************************************/
/* CURVE DATA */

export type CurveDataSpec =
  | SymbolicCurveDataSpec
  | SampledCurveDataSpec
  | FlowCurveDataSpec
  | PushforwardCurveDataSpec;

// γ(t) written out: the symbolic language with dimCount = 1 — `{op: "coord", index: 0}` is t. Exact velocity
// by symbolic differentiation. A circle: { op: "vec", vals: [{ op: "cos", val: coord0 }, { op: "sin", val: coord0 }] }.
export type SymbolicCurveDataSpec = {
  type: "symbolic";
  expr: SymbolicVector; // ℝ → ℝ^D, D the manifold's dimension
  interval: Interval; // [t0, t1]
  consts?: Record<ConstArgName, Real>;
};

// Points at parameter values: a training trajectory, an optimizer path, a digitized orbit.
export type SampledCurveDataSpec = {
  type: "sampled";
  points: ArraySpec<2>; // [N, D]; inline when small, a handle (.npy / .npz member) for real trajectories
  times?: ArraySpec<1> | Interval; // [N] strictly increasing parameter values, or an interval sampled uniformly;
  //                                  default: the index, t ∈ [0, N-1] — a snapshot number IS a natural t
  velocities?: ArraySpec<2>; // [N, D] exact γ'(t_i) when the collector had it (the SGD step direction, the vector
  //                            field at the sample) — the `exactGradient` analogue; makes cubic interpolation Hermite
  interp?: CurveInterp; // default "linear"
  labels?: ShowString[]; // one per sample, optional ("init", "epoch 3", "θ*")
  closed?: boolean; // a cycle: γ(t1) = γ(t0) is implied and interpolation wraps; default false
};

export type CurveInterp =
  | "linear" // chords between samples
  | "cubic" // C¹ spline (Hermite when `velocities` are given, Catmull–Rom otherwise)
  | "step"; // piecewise constant: a discrete process, drawn as a staircase

// The integral curve ẋ = F(x) from a start point — what dynamical-systems.json bakes into point sets with RK4 at
// build time. As a spec it is exact (to the integrator) at any resolution, on the CPU or in the GPU streamline
// kernels, and follows the field when a Controls row changes it.
export type FlowCurveDataSpec = {
  type: "flow";
  field: FieldId | VectorFieldDataSpec | ScalarFieldDataSpec; // a vector field; a SCALAR field means its gradient
  start: PointSpec;
  interval: Interval; // t ∈ [t0, t1]; t0 < 0 integrates backwards from the start (t = 0 is `start`)
  dir?: "ascending" | "descending"; // for a scalar field: gradient ascent / descent; default "descending"
  method?: FlowMethodSpec;
};

export type FlowMethodSpec = {
  integrator?: "rk4" | "euler"; // default rk4
  step?: RealPos; // fixed step in t; default chosen from the field's box and grid
  tol?: RealPos; // when given, adaptive stepping to this local error
};

// Curves push FORWARD (fields pull back): the same translate / scale vocabulary, applied to the points.
// translate: γ'(t) = γ(t) + vec;  scale: γ'(t) = origin + scale · (γ(t) − origin).
export type PushforwardCurveDataSpec =
  | TranslatedSpaceOpSpec<CurveDataSpec | CurveId>
  | ScaledSpaceOpSpec<CurveDataSpec | CurveId>;

/*******************************************************/
/* FIELDS ALONG A CURVE */
//
// A curve's parameter interval is a 1-D manifold. Composing a field with the
// curve gives a field ON THAT INTERVAL — "loss along the SGD trajectory" as a
// first-class 1-D field, with `derivative(0)` = d(loss)/dt for free and
// `pointwise` over it. These become members of ScalarFieldDataSpec /
// VectorFieldDataSpec when curves land; their `domain` is the curve's implicit
// parameter manifold, named "<curveId>/t" (1-D, box = the interval), which a
// field may name explicitly to combine several along-curve fields of ONE curve.

// f∘γ (scalar field → scalar along the curve) or v∘γ (vector field → vector along the curve)
export type AlongCurveFieldDataSpec = {
  type: "along";
  curve: CurveId;
  field: FieldId | ScalarFieldDataSpec | VectorFieldDataSpec;
  // kind: sampled iff the curve or the field is sampled; sample points = the curve's times
};

// γ'(t): a vector along the curve (exact for symbolic / flow / Hermite data, finite differences for linear samples).
// With `along(∇f)` this gives the papers' tangent quantities: ⟨∇f(γ(t)), γ'(t)⟩ is a `pointwise` over the two.
export type CurveVelocityFieldDataSpec = {
  type: "velocity";
  curve: CurveId;
};

/*******************************************************/
/* RUNTIME SHAPE (to implement; mirrors ScalarFieldData / VectorFieldData)
//
// class CurveData {
//   kind: "symbolic" | "sampled";
//   dimCount: number;                              // D of the manifold
//   interval: [number, number];
//   sampleTimes: Float64Array | undefined;         // defined iff sampled (the `samplePoints` analogue)
//   point(t: number): number[] | undefined;        // γ(t); undefined outside the interval
//   velocity(): VectorCurveData;                   // γ' as data on the same interval (composable: acceleration)
//   sampleOn(ts: ArrayLike<number>): Float64Array; // [n, D] flat
//   polyline(tolerance: number): Float64Array;     // adaptive chords for drawing (the exact-isoline refinement)
//   arcLength(t: number): number;                  // for by-distance scrubbing in the UI
// }
//
// Bundle: `curve(id)` built lazily like fields; `buildAll()` reports errors under "curves.<id>"; a curve's
// `domain` must have the points' D; slices (bundle/slice.ts) project curves like point sets; charts (mappings.ts)
// push a parameter-space trajectory into a PCA subspace — the same curve seen in two manifolds.
*/

/*******************************************************/
/* DECISIONS
//
// 1. Reparametrization (t ↦ s(t)) and arc-length parametrization are NOT spec types: they are views of one curve
//    (a viewer scrubber can run by t or by distance). A curve's identity is its image plus its parametrization
//    as collected.
// 2. Surfaces and higher-dimensional embedded manifolds (ℝ^k → ℝ^D) are the same idea with k > 1, but they
//    belong to mappings.ts (charts), where they have a customer.
// 3. Families of curves (many SGD runs) are many curves with "/" ids, grouped in the viewer like fields; the
//    metadata that varies across them is the sweep's business (sweeps.md §2), not a curve attribute.
// 4. A closed curve is a flag, not a type: `flow` finds cycles by integration and cannot know it is closed;
//    `sampled` knows because the collector said so.
// 5. `flow` reuses the streamline machinery (RK4 through a sampled copy on the grid in the viewer, exact
//    expression evaluation in core) and inherits its ascending / descending vocabulary for scalar fields.
*/

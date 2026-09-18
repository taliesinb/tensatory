// Scalar probability distributions: how the cells of a `random` array
// (SymbolicRandomScalarArraySpec, schema/arrays.ts) are drawn.
//
// Design rules:
// * Every distribution carries a `seed`. The array is a pure function of
//   (spec, seed): a counter-based generator keyed by (seed, flat cell index)
//   gives cell i the same value whatever the shape or chunking, and lets the
//   runtime sample a grid of cells in any order (CPU or GPU) with identical
//   results. `null` means "no seed chosen": the runtime draws a fresh one on
//   load (and remembers it in the per-bundle options), so a bundle can ship
//   unseeded randomness and still be reproducible within a session.
// * Continuous distributions are in LOCATION–SCALE form, loc + scale * Z with
//   a documented standard variate Z, so one `scale` control means the same
//   thing across them (the Controls pane's scale slider edits `scale`).
//   Distributions without a `scale` (bernoulli, discrete) get no slider.
// * Parameters are plain numbers. Symbolic parameters (an expression of the
//   cell position, e.g. a per-layer scale) can come later without breaking
//   these types.

import type { Int, IntPos, Real, RealPos, RealUnit, ShowString } from "./math";

/*******************************************************/
/* SEEDS */

// null: unseeded (the runtime picks one); int: used directly; string: hashed.
export type RandomSeed = null | Int | string;

/*******************************************************/
/* DISTRIBUTIONS */

export type ScalarDistributionSpec =
  | UniformDistributionSpec
  | GaussianDistributionSpec
  | LaplaceDistributionSpec
  | ExponentialDistributionSpec
  | StudentTDistributionSpec
  | BernoulliDistributionSpec
  | DiscreteDistributionSpec
  | IntegersDistributionSpec;

// the part shared by every distribution
export type DistributionBase = {
  seed: RandomSeed;
};

// a continuous distribution: loc + scale * Z
export type LocationScaleDistribution = DistributionBase & {
  loc?: Real;      // defaults to 0
  scale?: RealPos; // defaults to 1
};

// Z ~ U[-1, 1): `scale` is the half-width, so the default is symmetric
// around `loc` (the natural choice for a random direction). U[0, 1) is
// loc 0.5, scale 0.5.
export type UniformDistributionSpec = LocationScaleDistribution & {
  type: "uniform";
};

// Z ~ N(0, 1): `loc` is the mean, `scale` the standard deviation
export type GaussianDistributionSpec = LocationScaleDistribution & {
  type: "gaussian";
};

// Z ~ Laplace(0, 1): density exp(-|z|) / 2; `scale` is the diversity b
export type LaplaceDistributionSpec = LocationScaleDistribution & {
  type: "laplace";
};

// Z ~ Exp(1): `scale` is the mean 1/lambda; values are >= loc
export type ExponentialDistributionSpec = LocationScaleDistribution & {
  type: "exponential";
};

// Z ~ t(df): heavy tails; df = 1 is Cauchy, df -> inf is gaussian
export type StudentTDistributionSpec = LocationScaleDistribution & {
  type: "studentT";
  df: RealPos; // degrees of freedom
};

// `hot` with probability p, else `cold`. Rademacher (±1 signs) is
// p 0.5, hot 1, cold -1.
export type BernoulliDistributionSpec = DistributionBase & {
  type: "bernoulli";
  p?: RealUnit;  // defaults to 0.5
  hot?: Real;    // defaults to 1.0
  cold?: Real;   // defaults to 0.0
};

// one of `values`, with the given probabilities (uniform when omitted)
export type DiscreteDistributionSpec = DistributionBase & {
  type: "discrete";
  values: Real[];      // length >= 1
  probs?: RealUnit[];  // same length as `values`; normalized by the runtime
};

// integers in [lo, hi), each equally likely
export type IntegersDistributionSpec = DistributionBase & {
  type: "integers";
  lo?: Int;  // defaults to 0
  hi: Int;   // exclusive; hi > lo
};

/*******************************************************/
/* CONTROLS */
//
// A `random` array with a `widget` — or a whole DIRECTION of a displaced net
// (schema/nets.ts DirectionSpec) — gets a row in the viewer's Controls pane.
// Every row has a `reseed` button; rows of location–scale distributions and
// of directions also have a log-spaced `scale` slider. Rows never edit the
// bundle: they hold ADJUSTMENTS (per-bundle options) that the runtime applies
// to the spec — a row's seed is hashed into the seeds of every random array
// it governs (so the members of a direction stay independent draws), its
// scale multiplies the array's `dist.scale` or the direction's `scale`. Rows
// sharing an `id` are one row.

export type RandomWidgetSpec = {
  id?: ShowString;                 // rows with the same id merge; defaults to the array's / direction's spec path
  label?: ShowString;              // row name; defaults to the direction's `name`, else the id
  scaleRange?: [RealPos, RealPos]; // multiplier bounds of the scale slider (log-spaced); defaults to [0.01, 100]
  scaleSteps?: IntPos;             // slider resolution; defaults to the viewer's
};

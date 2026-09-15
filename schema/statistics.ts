// summarizes statistics of a group of real-valued scalars,
// e.g. those that occur in a discretization of a scalar field

import type { ArraySpec } from "./arrays";
import type { Interval } from "./geometry";
import type { Count, Real } from "./math";

export type ScalarStatistics = {
  extrema?: {
    min: Real; // over finite values
    max: Real;
  };
  counts?: {
    total?: Count;
    pos?: Count;
    neg?: Count;
    zero?: Count;
    nan?: Count;
    posInf?: Count;
    negInf?: Count;
  };
  moments?: {
    mean: Real; // over finite values
    variance?: Real;
    skewness?: Real;
    kurtosis?: Real;
  };
  quantiles?: RealsSpec; // values at evenly spaced quantiles 0..1 (inclusive)
  histogram?: {
    counts: RealsSpec;
    edges: Interval | RealsSpec; // an [lo, hi] for uniform bins, or explicit edges (length counts + 1)
  };
};

export type VectorStatistics = {
  norms?: ScalarStatistics;
};

export type RealsSpec = Real[] | ArraySpec<1>;

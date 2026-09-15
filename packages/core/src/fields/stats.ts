import type { ScalarStatistics } from "@tensatory/schema";

/** Statistics over a collection of reals; NaN/±Inf are counted but excluded from extrema/moments. */
export interface ScalarStats {
  readonly total: number;
  readonly finite: number;
  readonly nan: number;
  readonly posInf: number;
  readonly negInf: number;
  readonly pos: number;
  readonly neg: number;
  readonly zero: number;
  /** over finite values; NaN if none */
  readonly min: number;
  readonly max: number;
  readonly mean: number;
  readonly variance: number;
}

export function computeStats(values: ArrayLike<number>): ScalarStats {
  let nan = 0, posInf = 0, negInf = 0, pos = 0, neg = 0, zero = 0, finite = 0;
  let min = Infinity, max = -Infinity;
  let mean = 0, m2 = 0; // Welford
  for (let i = 0; i < values.length; i++) {
    const v = values[i]!;
    if (Number.isNaN(v)) { nan++; continue; }
    if (v === Infinity) { posInf++; pos++; continue; }
    if (v === -Infinity) { negInf++; neg++; continue; }
    finite++;
    if (v > 0) pos++; else if (v < 0) neg++; else zero++;
    if (v < min) min = v;
    if (v > max) max = v;
    const delta = v - mean;
    mean += delta / finite;
    m2 += delta * (v - mean);
  }
  return {
    total: values.length, finite, nan, posInf, negInf, pos, neg, zero,
    min: finite ? min : NaN,
    max: finite ? max : NaN,
    mean: finite ? mean : NaN,
    variance: finite > 1 ? m2 / finite : finite ? 0 : NaN,
  };
}

/** use precomputed statistics from a spec where available, otherwise compute */
export function statsFromSpec(spec: ScalarStatistics | undefined, values: () => ArrayLike<number>): ScalarStats {
  if (spec?.extrema && spec.moments && spec.counts?.total !== undefined) {
    const c = spec.counts;
    const total = c.total!;
    const nan = c.nan ?? 0, posInf = c.posInf ?? 0, negInf = c.negInf ?? 0;
    return {
      total, nan, posInf, negInf,
      finite: total - nan - posInf - negInf,
      pos: c.pos ?? NaN, neg: c.neg ?? NaN, zero: c.zero ?? NaN,
      min: spec.extrema.min, max: spec.extrema.max,
      mean: spec.moments.mean, variance: spec.moments.variance ?? NaN,
    };
  }
  return computeStats(values());
}

export function statsToSpec(s: ScalarStats): ScalarStatistics {
  return {
    extrema: { min: s.min, max: s.max },
    counts: { total: s.total, pos: s.pos, neg: s.neg, zero: s.zero, nan: s.nan, posInf: s.posInf, negInf: s.negInf },
    moments: { mean: s.mean, variance: s.variance },
  };
}

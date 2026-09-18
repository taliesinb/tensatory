// Random arrays: zod schemas for schema/distribution.ts and a counter-based
// sampler. Cell i of an array is a pure function of (spec, seed, i): the k-th
// 32-bit word of cell i is a hash of (seed, i, k), so cells can be drawn in
// any order or chunking (CPU or GPU) with identical results, and a shape
// change keeps the values of the cells that remain.

import { z } from "zod";
import type { RandomSeed, RandomWidgetSpec, ScalarDistributionSpec, SymbolicRandomScalarArraySpec } from "@tensatory/schema";
import { SpecError } from "../errors";
import { NdArray } from "./ndarray";

/*******************************************************/
/* ZOD */

const positive = z.number().positive();
export const RandomSeedSchema: z.ZodType<RandomSeed> = z.union([z.null(), z.number().int(), z.string()]);

const locScale = { seed: RandomSeedSchema, loc: z.number().optional(), scale: positive.optional() };

export const ScalarDistributionSchema: z.ZodType<ScalarDistributionSpec> = z.discriminatedUnion("type", [
  z.object({ type: z.literal("uniform"), ...locScale }),
  z.object({ type: z.literal("gaussian"), ...locScale }),
  z.object({ type: z.literal("laplace"), ...locScale }),
  z.object({ type: z.literal("exponential"), ...locScale }),
  z.object({ type: z.literal("studentT"), ...locScale, df: positive }),
  z.object({
    type: z.literal("bernoulli"),
    seed: RandomSeedSchema,
    p: z.number().min(0).max(1).optional(),
    hot: z.number().optional(),
    cold: z.number().optional(),
  }),
  z.object({
    type: z.literal("discrete"),
    seed: RandomSeedSchema,
    values: z.array(z.number()).min(1),
    probs: z.array(z.number().nonnegative()).optional(),
  }),
  z.object({ type: z.literal("integers"), seed: RandomSeedSchema, lo: z.number().int().optional(), hi: z.number().int() }),
]);

export const RandomWidgetSchema: z.ZodType<RandomWidgetSpec> = z.object({
  id: z.string().min(1).optional(),
  label: z.string().optional(),
  scaleRange: z.tuple([positive, positive]).optional(),
  scaleSteps: z.number().int().positive().optional(),
});

// a plain ZodObject (so the sized-array discriminated union accepts it), checked against the type below
export const RandomArraySchema = z.object({
  type: z.literal("random"),
  shape: z.array(z.number().int().nonnegative()),
  dist: ScalarDistributionSchema,
  widget: RandomWidgetSchema.nullable().optional(),
});
RandomArraySchema satisfies z.ZodType<SymbolicRandomScalarArraySpec>;

/** Does this distribution have a `scale` (and so a scale slider in the Controls pane)? */
export function hasScale(dist: ScalarDistributionSpec): boolean {
  switch (dist.type) {
    case "uniform": case "gaussian": case "laplace": case "exponential": case "studentT": return true;
    default: return false;
  }
}

/*******************************************************/
/* HASHING */

function fmix32(h: number): number {
  h ^= h >>> 16; h = Math.imul(h, 0x85ebca6b);
  h ^= h >>> 13; h = Math.imul(h, 0xc2b2ae35);
  h ^= h >>> 16;
  return h >>> 0;
}

/** Reduce a seed to 32 bits (strings via FNV-1a over UTF-16 code units). */
export function hashSeed(seed: number | string): number {
  if (typeof seed === "number") {
    const lo = seed >>> 0, hi = Math.floor(seed / 2 ** 32) >>> 0;
    return fmix32(lo ^ fmix32(hi ^ 0x9e3779b9));
  }
  let h = 0x811c9dc5;
  for (let i = 0; i < seed.length; i++) h = Math.imul(h ^ seed.charCodeAt(i), 0x01000193);
  return fmix32(h >>> 0);
}

/** A fresh 32-bit seed (for `null` seeds and the reseed button). */
export function freshSeed(): number {
  return (Math.random() * 2 ** 32) >>> 0;
}

// the seed used for `null` within this process: fixed once so that repeated
// builds of the same spec agree (the viewer replaces it with a remembered one)
let processSeed: number | null = null;
export function resolveSeed(seed: RandomSeed): number {
  if (seed !== null) return hashSeed(seed);
  return (processSeed ??= freshSeed());
}

/** Hash a row's salt into a seed (controls: a reseed re-salts every random array the row governs). */
export function saltSeed(seed: RandomSeed, salt: number): number {
  return fmix32(resolveSeed(seed) ^ Math.imul((salt >>> 0) + 1, 0x9e3779b9));
}

/** Per-cell stream of uniforms: `next()` is the next draw for cell `i`. */
class CellStream {
  private k = 0;
  constructor(private readonly s: number, private i = 0) {}
  cell(i: number) { this.i = i; this.k = 0; }
  private word(): number {
    const h = fmix32(this.s ^ Math.imul(this.i + 1, 0x9e3779b9));
    return fmix32(h ^ Math.imul(this.k++ + 1, 0x85ebca77));
  }
  /** uniform in [0, 1) with 53 random bits */
  next(): number {
    return ((this.word() >>> 5) * 67108864 + (this.word() >>> 6)) / 9007199254740992;
  }
  /** uniform in (0, 1) — safe for logs */
  open(): number {
    return (this.word() + 0.5) / 4294967296;
  }
}

/*******************************************************/
/* STANDARD VARIATES */

const gaussian = (r: CellStream) => Math.sqrt(-2 * Math.log(r.open())) * Math.cos(2 * Math.PI * r.next());

const laplace = (r: CellStream) => { const u = r.open(); return u < 0.5 ? Math.log(2 * u) : -Math.log(2 * (1 - u)); };

const exponential = (r: CellStream) => -Math.log(r.open());

// Gamma(a, 1), Marsaglia–Tsang (a >= 1) with the boost u^(1/a) for a < 1
function gamma(r: CellStream, a: number): number {
  if (a < 1) return gamma(r, a + 1) * Math.pow(r.open(), 1 / a);
  const d = a - 1 / 3, c = 1 / Math.sqrt(9 * d);
  for (;;) {
    let x: number, v: number;
    do { x = gaussian(r); v = 1 + c * x; } while (v <= 0);
    v = v * v * v;
    const u = r.open();
    if (u < 1 - 0.0331 * x * x * x * x) return d * v;
    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
  }
}

const studentT = (r: CellStream, df: number) => gaussian(r) / Math.sqrt(gamma(r, df / 2) * 2 / df);

/*******************************************************/
/* SAMPLERS */

/** Compile a distribution into a per-cell sampler over a stream. */
export function compileDistribution(dist: ScalarDistributionSpec, path: string[] = []): (r: CellStream) => number {
  switch (dist.type) {
    case "uniform": case "gaussian": case "laplace": case "exponential": case "studentT": {
      const loc = dist.loc ?? 0, scale = dist.scale ?? 1;
      let z: (r: CellStream) => number;
      switch (dist.type) {
        case "uniform": z = (r) => 2 * r.next() - 1; break;
        case "gaussian": z = gaussian; break;
        case "laplace": z = laplace; break;
        case "exponential": z = exponential; break;
        case "studentT": { const df = dist.df; z = (r) => studentT(r, df); break; }
      }
      return (r) => loc + scale * z(r);
    }
    case "bernoulli": {
      const p = dist.p ?? 0.5, hot = dist.hot ?? 1, cold = dist.cold ?? 0;
      return (r) => (r.next() < p ? hot : cold);
    }
    case "discrete": {
      const { values } = dist;
      const probs = dist.probs ?? values.map(() => 1);
      if (probs.length !== values.length) throw new SpecError(`discrete has ${values.length} values but ${probs.length} probs`, path);
      const total = probs.reduce((a, b) => a + b, 0);
      if (!(total > 0)) throw new SpecError(`discrete probs must not all be zero`, path);
      const cdf = new Float64Array(values.length);
      for (let i = 0, acc = 0; i < probs.length; i++) { acc += probs[i]! / total; cdf[i] = acc; }
      const last = values[values.length - 1]!;
      return (r) => {
        const u = r.next();
        for (let i = 0; i < cdf.length; i++) if (u < cdf[i]!) return values[i]!;
        return last;
      };
    }
    case "integers": {
      const lo = dist.lo ?? 0, n = dist.hi - lo;
      if (!(n > 0)) throw new SpecError(`integers needs hi > lo, got [${lo}, ${dist.hi})`, path);
      return (r) => lo + Math.floor(r.next() * n);
    }
  }
}

/** Materialize a random array: cell i is draw i of the distribution under its seed. */
export function buildRandomArray(spec: SymbolicRandomScalarArraySpec, path: string[] = []): NdArray {
  const sample = compileDistribution(spec.dist, [...path, "dist"]);
  const r = new CellStream(resolveSeed(spec.dist.seed));
  const arr = new NdArray(spec.shape);
  const n = arr.data.length;
  for (let i = 0; i < n; i++) { r.cell(i); arr.data[i] = sample(r); }
  return arr;
}

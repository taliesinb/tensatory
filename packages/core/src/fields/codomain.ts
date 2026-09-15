import { z } from "zod";
import type { Base, CodomainName, CodomainOptionsSpec, CodomainSpec } from "@tensatory/schema";

const extReal = z.union([z.number(), z.literal("inf"), z.literal("-inf")]);

export const CODOMAIN_NAMES = [
  "lin", "p", "fraction", "percent", "log", "logp", "-logp", "celoss", "similarity", "norm", "distance", "angle",
] as const satisfies readonly CodomainName[];

export const CodomainOptionsSchema: z.ZodType<CodomainOptionsSpec> = z.object({
  min: extReal.nullable().optional(),
  max: extReal.nullable().optional(),
  exclMin: z.boolean().optional(),
  exclMax: z.boolean().optional(),
  log: z.enum(["2", "10", "e"]).nullable().optional(),
  flip: z.boolean().optional(),
  wrap: z.boolean().optional(),
  unit: z.string().nullable().optional(),
  marks: z.array(z.tuple([extReal, z.string()])).optional(),
});

export const CodomainSchema: z.ZodType<CodomainSpec> = z.union([z.enum(CODOMAIN_NAMES), CodomainOptionsSchema]);

const PREDEFINED: Record<CodomainName, CodomainOptionsSpec> = {
  lin: {},
  p: { min: 0, max: 1 },
  fraction: { min: 0, max: 1 },
  percent: { min: 0, max: 100, unit: "%" },
  log: { log: "10" },
  logp: { max: 0, log: "e" },
  "-logp": { min: 0, log: "e", flip: true },
  celoss: { min: 0, log: "e", flip: true, unit: "nats" },
  similarity: { min: -1, max: 1, marks: [[0, "orthogonal"]] },
  norm: { min: 0 },
  distance: { min: 0 },
  angle: { min: 0, max: 2 * Math.PI, wrap: true, unit: "rad" },
};

const SUP = "⁰¹²³⁴⁵⁶⁷⁸⁹";
const superscript = (n: number): string => `${n < 0 ? "⁻" : ""}${String(Math.abs(n)).split("").map((d) => SUP[+d]!).join("")}`;

/**
 * Compact number formatting: `digits` significant digits, scientific notation
 * as "6.24·10⁻⁵" (Unicode superscripts: no extra vertical space) when the
 * magnitude is far from 1.
 */
export function formatReal(v: number, digits = 3): string {
  if (!Number.isFinite(v)) return v > 0 ? "∞" : v < 0 ? "−∞" : "NaN";
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 1e4 || a < 1e-3) {
    const e = Math.floor(Math.log10(a));
    let m = v / Math.pow(10, e);
    let ms = m.toPrecision(digits);
    if (Math.abs(+ms) >= 10) { m /= 10; ms = m.toPrecision(digits); return `${+ms}·10${superscript(e + 1)}`; }
    return `${+ms}·10${superscript(e)}`;
  }
  return String(+v.toPrecision(digits));
}

const toReal = (x: number | "inf" | "-inf" | null | undefined, dflt: number): number =>
  x === undefined || x === null ? dflt : x === "inf" ? Infinity : x === "-inf" ? -Infinity : x;

/** Resolved codomain: the 1-D value space of a scalar field, as visualization hints. */
export class Codomain {
  readonly min: number;
  readonly max: number;
  readonly exclMin: boolean;
  readonly exclMax: boolean;
  readonly log: Base | null;
  readonly flip: boolean;
  readonly wrap: boolean;
  readonly unit: string | null;
  readonly marks: readonly [number, string][];

  constructor(readonly spec: CodomainSpec = "lin") {
    const o = typeof spec === "string" ? PREDEFINED[spec] : spec;
    this.min = toReal(o.min, -Infinity);
    this.max = toReal(o.max, Infinity);
    this.exclMin = o.exclMin ?? this.min === -Infinity;
    this.exclMax = o.exclMax ?? this.max === Infinity;
    this.log = o.log ?? null;
    this.flip = o.flip ?? false;
    this.wrap = o.wrap ?? false;
    this.unit = o.unit ?? null;
    this.marks = (o.marks ?? []).map(([v, s]) => [toReal(v, NaN), s]);
  }

  get name(): string {
    return typeof this.spec === "string" ? this.spec : "custom";
  }

  /** clip a value to the codomain's bounds */
  clip(v: number): number {
    return Math.min(this.max, Math.max(this.min, v));
  }

  /**
   * Map a value in [lo, hi] to a slider parameter t in [0, 1], respecting the
   * log scale (when lo > 0) and flip. Inverse: `fromParam`.
   */
  toParam(v: number, lo: number, hi: number): number {
    const useLog = this.log !== null && lo > 0 && hi > 0;
    let t = useLog ? (Math.log(v) - Math.log(lo)) / (Math.log(hi) - Math.log(lo)) : (v - lo) / (hi - lo);
    if (!Number.isFinite(t)) t = 0;
    return this.flip ? 1 - t : t;
  }

  fromParam(t: number, lo: number, hi: number): number {
    if (this.flip) t = 1 - t;
    const useLog = this.log !== null && lo > 0 && hi > 0;
    return useLog ? Math.exp(Math.log(lo) + t * (Math.log(hi) - Math.log(lo))) : lo + t * (hi - lo);
  }

  format(v: number, digits = 3): string {
    const s = formatReal(v, digits);
    return this.unit ? `${s} ${this.unit}` : s;
  }
}

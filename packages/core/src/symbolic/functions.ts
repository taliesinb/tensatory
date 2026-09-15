// Implementations of the unary scalar functions and their derivatives.

import type { ScalarUnaryOp } from "@tensatory/schema";

const SQRT_2_PI = Math.sqrt(2 * Math.PI);
const SQRT_2 = Math.SQRT2;

/** Abramowitz & Stegun 7.1.26, |error| < 1.5e-7 */
export function erf(x: number): number {
  const s = Math.sign(x);
  const a = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * a);
  const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-a * a);
  return s * y;
}

export const sigmoid = (x: number): number => 1 / (1 + Math.exp(-x));
export const softplus = (x: number): number => (x > 30 ? x : Math.log1p(Math.exp(x)));
export const gelu = (x: number): number => 0.5 * x * (1 + erf(x / SQRT_2));
export const silu = (x: number): number => x * sigmoid(x);
export const elu = (x: number): number => (x > 0 ? x : Math.exp(x) - 1);
export const plogp = (p: number): number => (p === 0 ? 0 : p * Math.log(p));
export const gauss = (x: number): number => Math.exp(-0.5 * x * x);
export const normalPDF = (z: number, sigma: number): number => Math.exp(-0.5 * z * z) / (sigma * SQRT_2_PI);

export const UNARY: Record<ScalarUnaryOp, (x: number) => number> = {
  sin: Math.sin, cos: Math.cos, tan: Math.tan, sinh: Math.sinh, cosh: Math.cosh, tanh: Math.tanh,
  asin: Math.asin, acos: Math.acos, atan: Math.atan, asinh: Math.asinh, acosh: Math.acosh, atanh: Math.atanh,
  relu: (x) => (x > 0 ? x : 0), sigmoid, gelu, silu, softplus, elu, erf,
  floor: Math.floor, ceil: Math.ceil, round: Math.round, sign: Math.sign, abs: Math.abs,
  exp: Math.exp, exp2: (x) => Math.pow(2, x), exp10: (x) => Math.pow(10, x),
  log: Math.log, log2: Math.log2, log10: Math.log10, log1p: Math.log1p, expm1: Math.expm1,
  plogp, sqrt: Math.sqrt, square: (x) => x * x, negate: (x) => -x, reciprocal: (x) => 1 / x, gauss,
};

/** unary ops whose derivative is zero almost everywhere */
export const PIECEWISE_CONSTANT: ReadonlySet<ScalarUnaryOp> = new Set(["floor", "ceil", "round", "sign"]);

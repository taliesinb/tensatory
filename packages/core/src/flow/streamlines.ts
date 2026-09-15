// Streamlines of a vector field: RK4 on the *unit* field so points are spaced
// uniformly (one step = `step` world units); seeds on a jittered grid filling
// the box; integrated both ways from each seed.

import type { VectorFieldData } from "../fields/fieldData";
import type { Box } from "../geometry/box";

export interface StreamlineOptions {
  /** number of seed cells (one jittered seed per cell); actual count is the nearest grid */
  count: number;
  /** max integration steps in each direction */
  maxSteps: number;
  /** step length in world units */
  step: number;
  /** +1 follows the field, -1 flows against it (descent for a gradient) */
  sign?: 1 | -1;
  /** RNG seed for the jitter */
  seed?: number;
  /** restrict seeds / integration to this box (defaults to the field's box) */
  box?: Box;
}

export interface Streamline {
  /** flat [x0, y0, x1, y1, ...] in world coordinates (2D); D-dimensional in general */
  points: Float64Array;
  /** total arc length (steps * step) */
  length: number;
  /** per-line random phase in [0, 1) for particle animation */
  phase: number;
}

/** deterministic LCG in [0, 1) */
export function lcg(seed: number): () => number {
  let s = seed >>> 0;
  return () => ((s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
}

export function integrateStreamlines(field: VectorFieldData, opts: StreamlineOptions): Streamline[] {
  const D = field.dimCount;
  const box = opts.box ?? field.box;
  const sgn = opts.sign ?? 1;
  const h = opts.step;
  const rand = lcg(opts.seed ?? 12345);
  const tmp = new Float64Array(D), q = new Float64Array(D);

  /** unit direction at q (into out); false where the field vanishes / is undefined / outside */
  const dir = (at: Float64Array, out: Float64Array): boolean => {
    if (!box.contains(at, 1e-12)) return false;
    field.fn(at, -1, out);
    let l = 0;
    for (let d = 0; d < D; d++) l += out[d]! * out[d]!;
    if (!(l > 1e-24) || !Number.isFinite(l)) return false;
    const k = sgn / Math.sqrt(l);
    for (let d = 0; d < D; d++) out[d]! *= k;
    return true;
  };
  const k1 = new Float64Array(D), k2 = new Float64Array(D), k3 = new Float64Array(D), k4 = new Float64Array(D);
  const integrate = (seed: Float64Array, s: 1 | -1): number[] => {
    const pts: number[] = [];
    q.set(seed);
    for (let n = 0; n < opts.maxSteps; n++) {
      if (!dir(q, k1)) break;
      for (let d = 0; d < D; d++) tmp[d] = q[d]! + s * 0.5 * h * k1[d]!;
      if (!dir(tmp, k2)) break;
      for (let d = 0; d < D; d++) tmp[d] = q[d]! + s * 0.5 * h * k2[d]!;
      if (!dir(tmp, k3)) break;
      for (let d = 0; d < D; d++) tmp[d] = q[d]! + s * h * k3[d]!;
      if (!dir(tmp, k4)) break;
      for (let d = 0; d < D; d++) q[d]! += (s * h / 6) * (k1[d]! + 2 * k2[d]! + 2 * k3[d]! + k4[d]!);
      if (!box.contains(q, 1e-12)) break;
      for (let d = 0; d < D; d++) pts.push(q[d]!);
    }
    return pts;
  };

  // seed grid: cells as close to cubes as possible, `count` of them overall
  const size = box.size;
  const vol = size.reduce((a, b) => a * (b || 1), 1);
  const side = Math.pow(vol / Math.max(1, opts.count), 1 / D);
  const cnt = size.map((sz) => Math.max(1, Math.round((sz || side) / side)));
  const total = cnt.reduce((a, b) => a * b, 1);
  const idx = new Array<number>(D).fill(0);
  const lines: Streamline[] = [];
  const seed = new Float64Array(D);
  for (let c = 0; c < total; c++) {
    let rem = c;
    for (let d = D - 1; d >= 0; d--) { idx[d] = rem % cnt[d]!; rem = Math.floor(rem / cnt[d]!); }
    for (let d = 0; d < D; d++) seed[d] = box.a[d]! + ((idx[d]! + rand()) / cnt[d]!) * size[d]!;
    const back = integrate(seed, -1), fwd = integrate(seed, 1);
    const n = back.length / D + 1 + fwd.length / D;
    if (n < 2) continue;
    const pts = new Float64Array(n * D);
    // back points come out nearest-first; reverse them
    let o = 0;
    for (let i = back.length / D - 1; i >= 0; i--) for (let d = 0; d < D; d++) pts[o++] = back[i * D + d]!;
    for (let d = 0; d < D; d++) pts[o++] = seed[d]!;
    for (let i = 0; i < fwd.length; i++) pts[o++] = fwd[i]!;
    lines.push({ points: pts, length: (n - 1) * h, phase: rand() });
  }
  return lines;
}

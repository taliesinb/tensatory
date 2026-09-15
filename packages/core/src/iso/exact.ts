// Exact isolines for symbolic (anywhere-evaluable) scalar fields.
//
// Marching squares on a grid gives the topology and a seed polyline whose
// vertices carry linear-interpolation error. Since a symbolic field can be
// evaluated (and differentiated) at any point we
//   1. Newton-project every seed vertex onto the level set along ∇f, and
//   2. adaptively insert projected midpoints until every chord is within a
//      tolerance of the true curve.
// Sampled fields know nothing between their samples, so they keep plain
// marching squares (see `contourField`).

import { EvalError } from "../errors";
import type { DenseGrid } from "../geometry/grid";
import type { ScalarFieldData } from "../fields/fieldData";
import type { ScalarFn } from "../symbolic/compile";
import { isoContours, type Polyline } from "./marchingSquares";

export interface ExactContourOptions {
  /** max allowed distance between a chord and the true curve (world units) */
  tolerance: number;
  /** Newton iterations per projection before falling back to bisection (default 12) */
  newtonIterations?: number;
  /** max subdivision depth per seed segment (default 12) */
  maxDepth?: number;
}

export interface ContourResult {
  lines: Polyline[];
  /** "exact" when projected & refined, "linear" for plain marching squares */
  method: "exact" | "linear";
  vertexCount: number;
  /** max |f(v) - level| over all vertices (NaN when unknown) */
  maxResidual: number;
}

const partialCache = new WeakMap<ScalarFieldData, ScalarFn[]>();

function partials(field: ScalarFieldData): ScalarFn[] {
  let ps = partialCache.get(field);
  if (!ps) {
    ps = Array.from({ length: field.dimCount }, (_, d) => field.partial(d));
    partialCache.set(field, ps);
  }
  return ps;
}

/**
 * Project `p` onto {f = level}: damped Newton along ∇f, falling back to
 * bracketing + bisection along the gradient line. Coordinates that sit on the
 * box boundary stay locked there (the contour leaves the box through that
 * face). Returns undefined when no root is found within `maxDist` of `p`
 * (vanishing gradient, leaving the box, or jumping to another branch).
 */
export function projectToLevel(
  field: ScalarFieldData,
  p: ArrayLike<number>,
  level: number,
  maxDist: number,
  iterations = 12,
): Float64Array | undefined {
  const D = field.dimCount;
  const { a, b } = field.box;
  const eps = 1e-9;
  const grads = partials(field);
  const scale = Math.max(1, Math.abs(level));
  const tolF = 1e-13 * scale;

  const locked = new Uint8Array(D);
  for (let d = 0; d < D; d++) if (Math.abs(p[d]! - a[d]!) < eps || Math.abs(p[d]! - b[d]!) < eps) locked[d] = 1;
  const inBox = (q: Float64Array) => {
    for (let d = 0; d < D; d++) if (q[d]! < a[d]! - eps || q[d]! > b[d]! + eps) return false;
    return true;
  };
  const dist2 = (q: Float64Array) => {
    let s = 0;
    for (let d = 0; d < D; d++) { const dp = q[d]! - p[d]!; s += dp * dp; }
    return s;
  };
  const finish = (q: Float64Array) => {
    for (let d = 0; d < D; d++) if (locked[d]) q[d] = Math.abs(q[d]! - a[d]!) < Math.abs(q[d]! - b[d]!) ? a[d]! : b[d]!;
    return q;
  };
  const residual = (q: Float64Array) => field.fn(q, -1) - level;
  const gradient = (q: Float64Array, g: Float64Array): number => {
    let g2 = 0;
    for (let d = 0; d < D; d++) { g[d] = locked[d] ? 0 : grads[d]!(q, -1); g2 += g[d]! * g[d]!; }
    return g2;
  };

  // 1. damped Newton
  const q = Float64Array.from(p as ArrayLike<number>);
  const trial = new Float64Array(D);
  const g = new Float64Array(D);
  let r = residual(q);
  if (!Number.isFinite(r)) return undefined;
  const r0 = r, g0 = new Float64Array(D);
  const g02 = gradient(q, g0);
  if (!(g02 > 1e-24)) return undefined; // critical point: nothing to project along
  for (let it = 0; it < iterations && Math.abs(r) >= tolF; it++) {
    const g2 = gradient(q, g);
    if (!(g2 > 1e-24)) break;
    let k = r / g2;
    let accepted = false;
    for (let damp = 0; damp < 5; damp++, k *= 0.5) {
      for (let d = 0; d < D; d++) trial[d] = q[d]! - k * g[d]!;
      if (!inBox(trial) || dist2(trial) > maxDist * maxDist) continue;
      const rt = residual(trial);
      if (Number.isFinite(rt) && Math.abs(rt) < Math.abs(r)) { q.set(trial); r = rt; accepted = true; break; }
    }
    if (!accepted) break;
  }
  if (Math.abs(r) < tolF || Math.abs(r) < 1e-9 * scale) return finish(q);

  // 2. fallback: bracket a sign change along the initial gradient line, then bisect
  const gn = Math.sqrt(g02);
  const dir = new Float64Array(D);
  for (let d = 0; d < D; d++) dir[d] = (-Math.sign(r0) * g0[d]!) / gn;
  const at = (t: number, out: Float64Array) => { for (let d = 0; d < D; d++) out[d] = p[d]! + t * dir[d]!; return out; };
  let lo = 0, hi = maxDist / 64, rHi = NaN;
  for (;;) {
    at(hi, trial);
    if (!inBox(trial)) return undefined;
    rHi = residual(trial);
    if (!Number.isFinite(rHi)) return undefined;
    if (Math.sign(rHi) !== Math.sign(r0)) break;
    lo = hi; hi *= 2;
    if (hi > maxDist) return undefined;
  }
  for (let it = 0; it < 60 && hi - lo > 1e-15 * (1 + Math.abs(hi)); it++) {
    const mid = 0.5 * (lo + hi);
    const rm = residual(at(mid, trial));
    if (Math.abs(rm) < tolF) { lo = hi = mid; break; }
    if (Math.sign(rm) === Math.sign(r0)) lo = mid; else hi = mid;
  }
  return finish(at(0.5 * (lo + hi), trial));
}

/** distance from point m to the segment ab (2D) */
function chordDistance(ax: number, ay: number, bx: number, by: number, mx: number, my: number): number {
  const dx = bx - ax, dy = by - ay;
  const l2 = dx * dx + dy * dy;
  if (l2 === 0) return Math.hypot(mx - ax, my - ay);
  const t = Math.min(1, Math.max(0, ((mx - ax) * dx + (my - ay) * dy) / l2));
  return Math.hypot(mx - (ax + t * dx), my - (ay + t * dy));
}

/**
 * Isolines of a symbolic 2D field: marching squares on `grid`/`values` for
 * the seed, then projection and adaptive refinement.
 */
export function exactIsoContours(
  field: ScalarFieldData,
  grid: DenseGrid,
  values: ArrayLike<number>,
  level: number,
  opts: ExactContourOptions,
): ContourResult {
  if (field.dimCount !== 2) throw new EvalError(`exact isolines need a 2D field, got ${field.dimCount}D`);
  const tol = opts.tolerance;
  const maxDepth = opts.maxDepth ?? 12;
  const iters = opts.newtonIterations ?? 12;
  const cell = Math.max(grid.spacing[0]!, grid.spacing[1]!);
  const maxJump = 1.5 * cell;

  const seeds = isoContours(grid, values, level);
  const lines: Polyline[] = [];
  let vertexCount = 0;
  let maxResidual = 0;
  const pt = new Float64Array(2);

  const project = (x: number, y: number, maxDist: number): [number, number] | undefined => {
    pt[0] = x; pt[1] = y;
    const q = projectToLevel(field, pt, level, maxDist, iters);
    return q && ([q[0]!, q[1]!] as [number, number]);
  };

  for (const seed of seeds) {
    const n = seed.length / 2;
    if (n < 2) continue;
    const closed = n > 2 && seed[0] === seed[seed.length - 2] && seed[1] === seed[seed.length - 1];
    // 1. project seed vertices (falling back to the seed position when projection fails)
    const v: [number, number][] = [];
    for (let i = 0; i < (closed ? n - 1 : n); i++) {
      const x = seed[2 * i]!, y = seed[2 * i + 1]!;
      v.push(project(x, y, maxJump) ?? [x, y]);
    }
    if (closed) v.push(v[0]!);

    // 2. adaptive refinement of each chord
    const out: number[] = [v[0]![0], v[0]![1]];
    const refine = (ax: number, ay: number, bx: number, by: number, depth: number): void => {
      const len = Math.hypot(bx - ax, by - ay);
      if (depth < maxDepth && len > 2 * tol) {
        const m = project(0.5 * (ax + bx), 0.5 * (ay + by), len);
        if (m && chordDistance(ax, ay, bx, by, m[0], m[1]) > tol) {
          refine(ax, ay, m[0], m[1], depth + 1);
          refine(m[0], m[1], bx, by, depth + 1);
          return;
        }
      }
      out.push(bx, by);
    };
    for (let i = 0; i + 1 < v.length; i++) refine(v[i]![0], v[i]![1], v[i + 1]![0], v[i + 1]![1], 0);

    const line = Float64Array.from(out);
    lines.push(line);
    vertexCount += line.length / 2;
    for (let i = 0; i < line.length; i += 2) {
      pt[0] = line[i]!; pt[1] = line[i + 1]!;
      const r = Math.abs(field.fn(pt, -1) - level);
      if (r > maxResidual) maxResidual = r;
    }
  }
  return { lines, method: "exact", vertexCount, maxResidual };
}

/**
 * Isolines of a 2D scalar field at `level`, given its values on `grid`:
 * exact (projected + refined) when the field is symbolic, marching squares
 * otherwise.
 */
export function contourField(
  field: ScalarFieldData,
  grid: DenseGrid,
  values: ArrayLike<number>,
  level: number,
  opts: ExactContourOptions,
): ContourResult {
  if (field.kind === "symbolic") return exactIsoContours(field, grid, values, level, opts);
  const lines = isoContours(grid, values, level);
  return { lines, method: "linear", vertexCount: lines.reduce((s, l) => s + l.length / 2, 0), maxResidual: NaN };
}

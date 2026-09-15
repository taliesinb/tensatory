// Marching squares: iso-contours of a scalar sampled on a 2D DenseGrid.

import { EvalError } from "../errors";
import type { DenseGrid } from "../geometry/grid";

/** a polyline in grid-box coordinates: flat [x0, y0, x1, y1, ...] */
export type Polyline = Float64Array;

/**
 * Line segments of the iso-contour `values == level`, as a flat array of
 * segment endpoints in the grid's box coordinates: [x0,y0, x1,y1, ...]
 * (4 numbers per segment). Cells touching NaN are skipped.
 */
export function marchingSquaresSegments(grid: DenseGrid, values: ArrayLike<number>, level: number): Float64Array {
  if (grid.dimCount !== 2) throw new EvalError(`marching squares needs a 2D grid, got ${grid.dimCount}D`);
  if (values.length !== grid.sampleCount) throw new EvalError(`values length ${values.length} != grid sample count ${grid.sampleCount}`);
  const [nx, ny] = grid.size as [number, number];
  const sx = grid.strides[0]!, sy = grid.strides[1]!;
  const ax = grid.box.a[0]!, ay = grid.box.a[1]!;
  const hx = grid.spacing[0]!, hy = grid.spacing[1]!;
  const out: number[] = [];

  // interpolation parameter along an edge between values v0 and v1
  const t = (v0: number, v1: number) => (v0 === v1 ? 0.5 : (level - v0) / (v1 - v0));

  for (let i = 0; i + 1 < nx; i++) {
    for (let j = 0; j + 1 < ny; j++) {
      const v00 = values[i * sx + j * sy]!; // (i, j)
      const v10 = values[(i + 1) * sx + j * sy]!; // (i+1, j)
      const v11 = values[(i + 1) * sx + (j + 1) * sy]!; // (i+1, j+1)
      const v01 = values[i * sx + (j + 1) * sy]!; // (i, j+1)
      if (Number.isNaN(v00) || Number.isNaN(v10) || Number.isNaN(v11) || Number.isNaN(v01)) continue;
      const code = (v00 >= level ? 1 : 0) | (v10 >= level ? 2 : 0) | (v11 >= level ? 4 : 0) | (v01 >= level ? 8 : 0);
      if (code === 0 || code === 15) continue;
      const x0 = ax + i * hx, y0 = ay + j * hy;
      // edge midpoints: bottom (v00-v10), right (v10-v11), top (v01-v11), left (v00-v01)
      const B = () => [x0 + t(v00, v10) * hx, y0];
      const R = () => [x0 + hx, y0 + t(v10, v11) * hy];
      const T = () => [x0 + t(v01, v11) * hx, y0 + hy];
      const L = () => [x0, y0 + t(v00, v01) * hy];
      const seg = (a: number[], b: number[]) => out.push(a[0]!, a[1]!, b[0]!, b[1]!);
      switch (code) {
        case 1: case 14: seg(L(), B()); break;
        case 2: case 13: seg(B(), R()); break;
        case 3: case 12: seg(L(), R()); break;
        case 4: case 11: seg(R(), T()); break;
        case 6: case 9: seg(B(), T()); break;
        case 7: case 8: seg(L(), T()); break;
        case 5: case 10: {
          // saddle: disambiguate with the cell centre value
          const c = 0.25 * (v00 + v10 + v11 + v01);
          const centreHigh = c >= level;
          if ((code === 5) === centreHigh) { seg(L(), T()); seg(B(), R()); } else { seg(L(), B()); seg(R(), T()); }
          break;
        }
      }
    }
  }
  return Float64Array.from(out);
}

/**
 * Join segments into polylines (closed loops or open paths). Endpoints are
 * matched with a tolerance relative to grid spacing.
 */
export function joinSegments(segments: Float64Array, tolerance = 1e-9): Polyline[] {
  // drop degenerate (near-zero-length) segments: they arise when a grid vertex
  // sits exactly on the level and only confuse the matching
  const keep: number[] = [];
  for (let s = 0; s + 3 < segments.length; s += 4) {
    if (Math.hypot(segments[s + 2]! - segments[s]!, segments[s + 3]! - segments[s + 1]!) > tolerance) keep.push(s);
  }
  const n = keep.length;
  if (n === 0) return [];
  const seg = new Float64Array(n * 4);
  keep.forEach((s, i) => seg.set(segments.subarray(s, s + 4), i * 4));

  const bucket = (x: number) => Math.round(x / tolerance);
  const key = (bx: number, by: number) => `${bx},${by}`;
  // bucket -> list of [segment index, end (0|1)]
  const ends = new Map<string, [number, 0 | 1][]>();
  for (let s = 0; s < n; s++) {
    for (const e of [0, 1] as const) {
      const k = key(bucket(seg[s * 4 + e * 2]!), bucket(seg[s * 4 + e * 2 + 1]!));
      let list = ends.get(k);
      if (!list) ends.set(k, (list = []));
      list.push([s, e]);
    }
  }
  const used = new Uint8Array(n);
  const pt = (s: number, e: 0 | 1) => [seg[s * 4 + e * 2]!, seg[s * 4 + e * 2 + 1]!] as const;
  /** unused segment ends within tolerance of (x, y), searching neighbouring buckets */
  const near = (x: number, y: number): [number, 0 | 1] | undefined => {
    const bx = bucket(x), by = bucket(y);
    let best: [number, 0 | 1] | undefined, bestD = Infinity;
    for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) {
      for (const c of ends.get(key(bx + dx, by + dy)) ?? []) {
        if (used[c[0]]) continue;
        const [cx, cy] = pt(c[0], c[1]);
        const d = Math.hypot(cx - x, cy - y);
        if (d <= 2 * tolerance && d < bestD) { best = c; bestD = d; }
      }
    }
    return best;
  };

  const walk = (start: number, startEnd: 0 | 1): number[] => {
    // walk from segment `start`, leaving through its end `startEnd`
    const path: number[] = [];
    let s = start, leave = startEnd;
    path.push(...pt(s, (1 - leave) as 0 | 1));
    for (;;) {
      used[s] = 1;
      const [x, y] = pt(s, leave);
      path.push(x, y);
      const next = near(x, y);
      if (!next) break;
      s = next[0];
      leave = (1 - next[1]) as 0 | 1;
    }
    return path;
  };

  const lines: Polyline[] = [];
  for (let s = 0; s < n; s++) {
    if (used[s]) continue;
    // extend both ways from s: walk backwards first, then reverse, then continue forwards
    const back = walk(s, 0);
    used[s] = 0; // let the forward walk pass through s again
    const fwd = walk(s, 1);
    const rev: number[] = [];
    for (let i = back.length - 2; i >= 4; i -= 2) rev.push(back[i]!, back[i + 1]!); // excludes s's own endpoints (already in fwd)
    const line = Float64Array.from([...rev, ...fwd]);
    // a loop closes with the neighbouring segment's copy of the seam vertex: snap it onto the first point exactly,
    // so consumers can detect closure by equality
    const m = line.length;
    if (m >= 6 && Math.hypot(line[m - 2]! - line[0]!, line[m - 1]! - line[1]!) <= 2 * tolerance) { line[m - 2] = line[0]!; line[m - 1] = line[1]!; }
    lines.push(line);
  }
  return lines;
}

/** convenience: polylines of the iso-contour at `level` */
export function isoContours(grid: DenseGrid, values: ArrayLike<number>, level: number): Polyline[] {
  return joinSegments(marchingSquaresSegments(grid, values, level), Math.min(grid.spacing[0]!, grid.spacing[1]!) * 1e-6 || 1e-9);
}

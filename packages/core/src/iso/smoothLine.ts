import type { Polyline } from "./marchingSquares";

/**
 * Taubin λ|μ smoothing of a polyline (shrink-free Laplacian smoothing).
 * Closed lines (first point == last point) wrap; open lines keep their
 * endpoints fixed.
 */
export function taubinSmooth(line: Polyline, iterations: number, lambda = 0.5, mu = -0.53): Polyline {
  const n = line.length / 2;
  if (iterations <= 0 || n < 3) return line;
  const closed = line[0] === line[line.length - 2] && line[1] === line[line.length - 1];
  const m = closed ? n - 1 : n; // distinct points
  let cur = Float64Array.from(line.subarray(0, m * 2));
  let next = new Float64Array(m * 2);
  const pass = (k: number) => {
    for (let i = 0; i < m; i++) {
      const fixed = !closed && (i === 0 || i === m - 1);
      if (fixed) { next[2 * i] = cur[2 * i]!; next[2 * i + 1] = cur[2 * i + 1]!; continue; }
      const ip = (i - 1 + m) % m, inx = (i + 1) % m;
      for (let c = 0; c < 2; c++) {
        const x = cur[2 * i + c]!;
        const lap = 0.5 * (cur[2 * ip + c]! + cur[2 * inx + c]!) - x;
        next[2 * i + c] = x + k * lap;
      }
    }
    [cur, next] = [next, cur];
  };
  for (let it = 0; it < iterations; it++) { pass(lambda); pass(mu); }
  if (!closed) return cur;
  const out = new Float64Array(n * 2);
  out.set(cur);
  out[2 * m] = cur[0]!; out[2 * m + 1] = cur[1]!;
  return out;
}

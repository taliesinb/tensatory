import type { DenseGrid } from "../geometry/grid";

/**
 * Separable box blur of grid values (radius in cells per axis, truncated at
 * the grid edges: each output is the mean of the in-range neighbours).
 * radius <= 0 returns a copy.
 */
export function boxBlur(grid: DenseGrid, values: ArrayLike<number>, radius: number): Float64Array {
  let src = Float64Array.from(values as ArrayLike<number>);
  if (radius <= 0) return src;
  const r = Math.floor(radius);
  const n = grid.sampleCount;
  let dst = new Float64Array(n);
  for (let axis = 0; axis < grid.dimCount; axis++) {
    const D = grid.size[axis]!, S = grid.strides[axis]!;
    if (D < 2) continue;
    for (let v = 0; v < n; v++) {
      const c = Math.floor(v / S) % D;
      let sum = 0, cnt = 0;
      for (let k = -r; k <= r; k++) {
        const cc = c + k;
        if (cc < 0 || cc >= D) continue;
        sum += src[v + k * S]!;
        cnt++;
      }
      dst[v] = sum / cnt;
    }
    [src, dst] = [dst, src];
  }
  return src;
}

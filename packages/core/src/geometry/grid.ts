import type { GridPos, GridSize, Point, SamplePos } from "@tensatory/schema";
import { SpecError } from "../errors";
import { Box } from "./box";

/** The discrete support of sampled field data. */
export interface SamplePoints {
  readonly type: "dense" | "sparse";
  readonly dimCount: number;
  readonly sampleCount: number;
  /** bounding box of the sample points */
  readonly box: Box;
  /** the pos'th sample point (0 <= pos < sampleCount) */
  point(pos: SamplePos): Point;
  /** same discrete support (same points in the same order) */
  equals(other: SamplePoints, eps?: number): boolean;
}

/**
 * A regular grid of sample points filling a box, stored in row-major ("C")
 * order: the LAST axis varies fastest. Grid position (0,...,0) is at corner
 * `a`, (S_0-1,...,S_{D-1}-1) at corner `b`. Axes of size 1 sit at `a`.
 */
export class DenseGrid implements SamplePoints {
  readonly type = "dense" as const;
  readonly dimCount: number;
  readonly sampleCount: number;
  /** row-major strides: stride[d] = product of sizes of axes after d */
  readonly strides: readonly number[];
  /** spacing between neighbouring samples along each axis (0 for axes of size 1) */
  readonly spacing: readonly number[];

  constructor(
    readonly size: GridSize,
    readonly box: Box,
  ) {
    if (size.length !== box.dimCount)
      throw new SpecError(`grid has ${size.length} axes but its box has ${box.dimCount} dimensions`);
    for (const s of size) if (!Number.isInteger(s) || s < 1) throw new SpecError(`grid sizes must be positive integers, got ${size}`);
    this.dimCount = size.length;
    const strides = new Array<number>(size.length);
    let acc = 1;
    for (let d = size.length - 1; d >= 0; d--) {
      strides[d] = acc;
      acc *= size[d]!;
    }
    this.strides = strides;
    this.sampleCount = acc;
    this.spacing = size.map((n, d) => (n > 1 ? (box.b[d]! - box.a[d]!) / (n - 1) : 0));
  }

  /** flat position of a grid position */
  pos(idx: ArrayLike<number>): SamplePos {
    let p = 0;
    for (let d = 0; d < this.dimCount; d++) p += idx[d]! * this.strides[d]!;
    return p;
  }

  /** grid position of a flat position */
  gridPos(pos: SamplePos): GridPos {
    const idx = new Array<number>(this.dimCount);
    let rem = pos;
    for (let d = 0; d < this.dimCount; d++) {
      const s = this.strides[d]!;
      idx[d] = Math.floor(rem / s);
      rem -= idx[d]! * s;
    }
    return idx;
  }

  gridPoint(idx: ArrayLike<number>): Point {
    const p = new Array<number>(this.dimCount);
    for (let d = 0; d < this.dimCount; d++) p[d] = this.box.a[d]! + idx[d]! * this.spacing[d]!;
    return p;
  }

  point(pos: SamplePos): Point {
    return this.gridPoint(this.gridPos(pos));
  }

  /** write the pos'th point into `out` (avoids allocation in hot loops) */
  pointInto(pos: SamplePos, out: Float64Array): void {
    let rem = pos;
    for (let d = 0; d < this.dimCount; d++) {
      const s = this.strides[d]!;
      const i = Math.floor(rem / s);
      rem -= i * s;
      out[d] = this.box.a[d]! + i * this.spacing[d]!;
    }
  }

  /** the grid position (possibly fractional) of a point; undefined when outside the box */
  locate(p: ArrayLike<number>, eps = 1e-9): number[] | undefined {
    if (!this.box.contains(p, eps)) return undefined;
    return this.size.map((n, d) => (n > 1 ? (p[d]! - this.box.a[d]!) / this.spacing[d]! : 0));
  }

  equals(other: SamplePoints, eps = 0): boolean {
    if (!(other instanceof DenseGrid)) return false;
    if (other.dimCount !== this.dimCount) return false;
    for (let d = 0; d < this.dimCount; d++) if (other.size[d] !== this.size[d]) return false;
    return this.box.equals(other.box, eps);
  }

  translate(v: readonly number[]): DenseGrid {
    return new DenseGrid(this.size, this.box.translate(v));
  }

  scale(origin: readonly number[], factors: readonly number[]): DenseGrid {
    for (const f of factors) if (f < 0) throw new SpecError("cannot scale a grid by a negative factor (would reverse sample order)");
    return new DenseGrid(this.size, this.box.scale(origin, factors));
  }
}

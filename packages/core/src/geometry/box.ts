import type { BoxSpec, Interval, Point } from "@tensatory/schema";
import { SpecError } from "../errors";

/** An axis-aligned box in R^dimCount. Corner `a` is the minimum, `b` the maximum. */
export class Box {
  readonly dimCount: number;

  constructor(
    readonly a: readonly number[],
    readonly b: readonly number[],
  ) {
    if (a.length !== b.length) throw new SpecError(`box corners have different dimensions (${a.length} vs ${b.length})`);
    for (let d = 0; d < a.length; d++) {
      const lo = a[d]!,
        hi = b[d]!;
      if (!Number.isFinite(lo) || !Number.isFinite(hi)) throw new SpecError(`box corner coordinates must be finite`);
      if (lo > hi) throw new SpecError(`box interval ${d} is inverted: [${lo}, ${hi}]`);
    }
    this.dimCount = a.length;
  }

  static unit(dimCount: number): Box {
    return new Box(new Array<number>(dimCount).fill(0), new Array<number>(dimCount).fill(1));
  }

  static fromSpec(spec: BoxSpec, path: string[] = []): Box {
    try {
      if (Array.isArray(spec)) {
        return new Box(
          spec.map((iv) => iv[0]),
          spec.map((iv) => iv[1]),
        );
      }
      return new Box([...spec.a], [...spec.b]);
    } catch (e) {
      if (e instanceof SpecError) throw new SpecError(e.message.replace(/^.*?: /, ""), path);
      throw e;
    }
  }

  get intervals(): Interval[] {
    return this.a.map((lo, d) => [lo, this.b[d]!] as Interval);
  }

  /** side lengths */
  get size(): number[] {
    return this.a.map((lo, d) => this.b[d]! - lo);
  }

  get center(): Point {
    return this.a.map((lo, d) => 0.5 * (lo + this.b[d]!));
  }

  contains(p: ArrayLike<number>, eps = 0): boolean {
    if (p.length !== this.dimCount) return false;
    for (let d = 0; d < this.dimCount; d++) {
      const x = p[d]!;
      if (x < this.a[d]! - eps || x > this.b[d]! + eps) return false;
    }
    return true;
  }

  equals(other: Box, eps = 0): boolean {
    if (other.dimCount !== this.dimCount) return false;
    for (let d = 0; d < this.dimCount; d++) {
      if (Math.abs(this.a[d]! - other.a[d]!) > eps || Math.abs(this.b[d]! - other.b[d]!) > eps) return false;
    }
    return true;
  }

  translate(v: readonly number[]): Box {
    return new Box(
      this.a.map((x, d) => x + v[d]!),
      this.b.map((x, d) => x + v[d]!),
    );
  }

  /** the box scaled about `origin` by per-dimension factors */
  scale(origin: readonly number[], factors: readonly number[]): Box {
    const f = (x: number, d: number) => origin[d]! + (x - origin[d]!) * factors[d]!;
    const a = this.a.map(f),
      b = this.b.map(f);
    // negative factors flip the interval
    return new Box(
      a.map((x, d) => Math.min(x, b[d]!)),
      a.map((x, d) => Math.max(x, b[d]!)),
    );
  }

  /** intersection, or undefined when empty */
  intersect(other: Box): Box | undefined {
    if (other.dimCount !== this.dimCount) return undefined;
    const a = this.a.map((x, d) => Math.max(x, other.a[d]!));
    const b = this.b.map((x, d) => Math.min(x, other.b[d]!));
    for (let d = 0; d < this.dimCount; d++) if (a[d]! > b[d]!) return undefined;
    return new Box(a, b);
  }

  toSpec(): Interval[] {
    return this.intervals;
  }
}

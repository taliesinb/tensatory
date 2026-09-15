import { SpecError } from "../errors";

export type ArrayData = Float64Array | Float32Array;

/** A dense, row-major ("C" order) n-dimensional array of reals. */
export class NdArray {
  readonly shape: readonly number[];
  readonly strides: readonly number[];
  readonly size: number;
  readonly data: ArrayData;

  constructor(shape: readonly number[], data?: ArrayData) {
    for (const s of shape) if (!Number.isInteger(s) || s < 0) throw new SpecError(`array shape must be non-negative integers, got [${shape}]`);
    const strides = new Array<number>(shape.length);
    let acc = 1;
    for (let d = shape.length - 1; d >= 0; d--) {
      strides[d] = acc;
      acc *= shape[d]!;
    }
    this.shape = [...shape];
    this.strides = strides;
    this.size = acc;
    if (data) {
      if (data.length !== acc) throw new SpecError(`array data has ${data.length} elements but shape [${shape}] needs ${acc}`);
      this.data = data;
    } else {
      this.data = new Float64Array(acc);
    }
  }

  get ndim(): number {
    return this.shape.length;
  }

  /** resolve a possibly negative axis position (python-style) */
  axisPos(axis: number, pos: number): number {
    const n = this.shape[axis]!;
    const i = pos < 0 ? n + pos : pos;
    if (!Number.isInteger(i) || i < 0 || i >= n) throw new SpecError(`index ${pos} out of range for axis ${axis} of size ${n}`);
    return i;
  }

  /** flat offset of a multi-index (negative indices allowed) */
  offset(idx: ArrayLike<number>): number {
    if (idx.length !== this.ndim) throw new SpecError(`expected ${this.ndim} indices, got ${idx.length}`);
    let o = 0;
    for (let d = 0; d < this.ndim; d++) o += this.axisPos(d, idx[d]!) * this.strides[d]!;
    return o;
  }

  get(...idx: number[]): number {
    return this.data[this.offset(idx)]!;
  }

  set(idx: ArrayLike<number>, value: number): void {
    this.data[this.offset(idx)] = value;
  }

  /** fix the leading `part.length` indices, returning a view-copy of the remaining axes */
  part(part: readonly number[]): NdArray {
    if (part.length > this.ndim) throw new SpecError(`part [${part}] has more indices than the array has axes (${this.ndim})`);
    let base = 0;
    for (let d = 0; d < part.length; d++) base += this.axisPos(d, part[d]!) * this.strides[d]!;
    const restShape = this.shape.slice(part.length);
    const len = restShape.reduce((a, b) => a * b, 1);
    return new NdArray(restShape, this.data.subarray(base, base + len));
  }

  toNested(): unknown {
    const rec = (base: number, d: number): unknown => {
      if (d === this.ndim) return this.data[base];
      return Array.from({ length: this.shape[d]! }, (_, i) => rec(base + i * this.strides[d]!, d + 1));
    };
    return rec(0, 0);
  }
}

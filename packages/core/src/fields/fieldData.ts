// Runtime field data: the gadgets that evaluate scalar / vector fields.
//
// Every field datum has a `kind`:
//   "sampled"  - has a discrete support `samplePoints` (phase 1: a DenseGrid);
//                values off the support are multilinearly interpolated
//   "symbolic" - can be evaluated anywhere in `box`
// Evaluators take (p, pos): `pos` is p's position in `samplePoints` when p IS
// a sample point (lets sampled data avoid interpolation), otherwise -1.

import type { ArrayData } from "../arrays/ndarray";
import { EvalError, SpecError } from "../errors";
import { Box } from "../geometry/box";
import { DenseGrid } from "../geometry/grid";
import type { SExpr, VExpr } from "../symbolic/ast";
import { compileScalar, compileVector, pureContext, type CompileContext, type ScalarFn, type VectorFn } from "../symbolic/compile";
import { diffScalar } from "../symbolic/diff";
import { computeStats, statsFromSpec, type ScalarStats } from "./stats";
import type { ScalarStatistics } from "@tensatory/schema";

export type FieldKind = "symbolic" | "sampled";

/** default grid used to compute statistics of symbolic fields */
export function defaultStatsGrid(box: Box): DenseGrid {
  const n = box.dimCount <= 1 ? 1024 : box.dimCount === 2 ? 128 : box.dimCount === 3 ? 32 : 8;
  return new DenseGrid(new Array<number>(box.dimCount).fill(n), box);
}

abstract class FieldDataBase {
  abstract readonly dimCount: number;
  abstract readonly box: Box;
  abstract readonly kind: FieldKind;
  /** the discrete support; defined iff kind === "sampled" */
  abstract readonly samplePoints: DenseGrid | undefined;
  /**
   * true when evaluating is expensive (a net evaluated on the CPU, or anything
   * derived from one): consumers should sample sparingly and avoid per-point
   * work such as exact isoline projection. Ordinary symbolic and sampled data
   * is cheap.
   */
  get costly(): boolean { return false; }

  protected posFor(grid: DenseGrid): boolean {
    return this.samplePoints !== undefined && this.samplePoints.equals(grid);
  }
}

export abstract class ScalarFieldData extends FieldDataBase {
  readonly rank = "scalar" as const;
  /** the evaluator (see module comment for `pos`) */
  abstract readonly fn: ScalarFn;
  /** ∂f/∂x_dim as field data (exact for symbolic data, finite differences for sampled); composable to any order */
  abstract derivative(dim: number): ScalarFieldData;
  /** evaluator of ∂f/∂x_dim */
  partial(dim: number): ScalarFn {
    return this.derivative(dim).fn;
  }

  private _stats: ScalarStats | undefined;

  /** value at a point, or undefined outside the box */
  value(p: ArrayLike<number>): number | undefined {
    if (!this.box.contains(p, 1e-12)) return undefined;
    return this.fn(p, -1);
  }

  /** values at every point of `grid` (row-major) */
  sampleOn(grid: DenseGrid): Float64Array {
    if (grid.dimCount !== this.dimCount) throw new EvalError(`grid has ${grid.dimCount} dims, field has ${this.dimCount}`);
    const out = new Float64Array(grid.sampleCount);
    const p = new Float64Array(this.dimCount);
    const usePos = this.posFor(grid);
    const fn = this.fn;
    for (let i = 0; i < out.length; i++) {
      grid.pointInto(i, p);
      out[i] = fn(p, usePos ? i : -1);
    }
    return out;
  }

  /** statistics over the sample points (sampled) or a default grid (symbolic); cached */
  stats(): ScalarStats {
    return (this._stats ??= this.computeStats());
  }

  protected computeStats(): ScalarStats {
    return computeStats(this.sampleOn(this.samplePoints ?? defaultStatsGrid(this.box)));
  }
}

export abstract class VectorFieldData extends FieldDataBase {
  readonly rank = "vector" as const;
  abstract readonly fn: VectorFn;
  /** the index'th component as scalar field data */
  abstract component(index: number): ScalarFieldData;
  /** evaluator of ∂v[index]/∂x_dim */
  partial(index: number, dim: number): ScalarFn {
    return this.component(index).derivative(dim).fn;
  }

  value(p: ArrayLike<number>): number[] | undefined {
    if (!this.box.contains(p, 1e-12)) return undefined;
    return Array.from(this.fn(p, -1, new Float64Array(this.dimCount)));
  }

  /** vectors at every point of `grid`, flattened [sampleCount, dimCount] */
  sampleOn(grid: DenseGrid): Float64Array {
    if (grid.dimCount !== this.dimCount) throw new EvalError(`grid has ${grid.dimCount} dims, field has ${this.dimCount}`);
    const D = this.dimCount;
    const out = new Float64Array(grid.sampleCount * D);
    const p = new Float64Array(D), v = new Float64Array(D);
    const usePos = this.posFor(grid);
    for (let i = 0; i < grid.sampleCount; i++) {
      grid.pointInto(i, p);
      this.fn(p, usePos ? i : -1, v);
      out.set(v, i * D);
    }
    return out;
  }
}

export type FieldData = ScalarFieldData | VectorFieldData;

/*******************************************************/
/* closure-backed scalar data: derivatives / components of other data */

export class ClosureScalarFieldData extends ScalarFieldData {
  constructor(
    readonly kind: FieldKind,
    readonly dimCount: number,
    readonly box: Box,
    readonly samplePoints: DenseGrid | undefined,
    readonly fn: ScalarFn,
    private readonly deriv: (dim: number) => ScalarFieldData,
    private readonly _costly = false,
  ) {
    super();
  }
  override get costly(): boolean { return this._costly; }
  derivative(dim: number): ScalarFieldData {
    return this.deriv(dim);
  }
}

/**
 * Numerical derivatives of arbitrary scalar data. Sampled data: difference the
 * values on the grid and interpolate the differenced field (so higher orders
 * compose). Symbolic data without an exact derivative: central differences.
 */
export function numericalDerivative(f: ScalarFieldData, dim: number): ScalarFieldData {
  const grid = f.samplePoints;
  let fn: ScalarFn;
  if (grid) {
    const vals = f.sampleOn(grid);
    const dvals = new Float64Array(grid.sampleCount);
    for (let i = 0; i < dvals.length; i++) dvals[i] = gridDifference(grid, vals, 1, 0, dim, i);
    fn = (p, pos) => (pos >= 0 ? dvals[pos]! : interpolate(grid, dvals, 1, 0, p));
  } else fn = fdPartial(f.fn, dim, f.box);
  const out: ClosureScalarFieldData = new ClosureScalarFieldData(f.kind, f.dimCount, f.box, grid, fn, (d) => numericalDerivative(out, d), f.costly);
  return out;
}

/** k * f, with derivatives scaled alike */
function scaleValues(f: ScalarFieldData, k: number): ScalarFieldData {
  const fn: ScalarFn = (p, pos) => k * f.fn(p, pos);
  return new ClosureScalarFieldData(f.kind, f.dimCount, f.box, f.samplePoints, fn, (d) => scaleValues(f.derivative(d), k), f.costly);
}

/*******************************************************/
/* sampled */

/** multilinear interpolation of channel `ch` of `[sampleCount, channels]` data at grid coordinates */
function interpolate(grid: DenseGrid, data: ArrayData, channels: number, ch: number, p: ArrayLike<number>): number {
  const loc = grid.locate(p, 1e-9);
  if (!loc) return NaN;
  const D = grid.dimCount;
  let result = 0;
  const corners = 1 << D;
  const i0 = new Array<number>(D), fr = new Array<number>(D);
  for (let d = 0; d < D; d++) {
    const n = grid.size[d]!;
    let i = Math.floor(loc[d]!);
    if (i >= n - 1) i = Math.max(0, n - 2);
    i0[d] = i;
    fr[d] = n > 1 ? loc[d]! - i : 0;
  }
  for (let c = 0; c < corners; c++) {
    let w = 1, pos = 0;
    for (let d = 0; d < D; d++) {
      const hi = (c >> d) & 1;
      const f = fr[d]!;
      w *= hi ? f : 1 - f;
      if (w === 0) break;
      const n = grid.size[d]!;
      pos += Math.min(i0[d]! + hi, n - 1) * grid.strides[d]!;
    }
    if (w !== 0) result += w * data[pos * channels + ch]!;
  }
  return result;
}

/** central (one-sided at edges) difference of channel `ch` along `dim` at sample `pos` */
function gridDifference(grid: DenseGrid, data: ArrayData, channels: number, ch: number, dim: number, pos: number): number {
  const n = grid.size[dim]!;
  if (n < 2) return 0;
  const stride = grid.strides[dim]!;
  const i = Math.floor(pos / stride) % n;
  const h = grid.spacing[dim]!;
  const at = (q: number) => data[q * channels + ch]!;
  if (i === 0) return (at(pos + stride) - at(pos)) / h;
  if (i === n - 1) return (at(pos) - at(pos - stride)) / h;
  return (at(pos + stride) - at(pos - stride)) / (2 * h);
}

/** finite-difference derivative of an evaluator at an arbitrary point */
function fdPartial(fn: ScalarFn, dim: number, box: Box): ScalarFn {
  const h = Math.max(1e-6 * (box.size[dim]! || 1), 1e-9);
  const lo = box.a[dim]!, hi = box.b[dim]!;
  return (p, _pos) => {
    const q = Float64Array.from(p as ArrayLike<number>);
    const x = q[dim]!;
    const xa = Math.min(hi, x + h), xb = Math.max(lo, x - h);
    q[dim] = xa; const fa = fn(q, -1);
    q[dim] = xb; const fb = fn(q, -1);
    return (fa - fb) / (xa - xb);
  };
}

export class DenseScalarFieldData extends ScalarFieldData {
  readonly kind = "sampled" as const;
  readonly dimCount: number;
  readonly box: Box;
  readonly fn: ScalarFn;

  constructor(
    readonly samplePoints: DenseGrid,
    readonly data: ArrayData,
    private readonly statsSpec?: ScalarStatistics,
  ) {
    super();
    if (data.length !== samplePoints.sampleCount)
      throw new SpecError(`dense samples have ${data.length} values but the grid has ${samplePoints.sampleCount} points`);
    this.dimCount = samplePoints.dimCount;
    this.box = samplePoints.box;
    const grid = samplePoints;
    this.fn = (p, pos) => (pos >= 0 ? data[pos]! : interpolate(grid, data, 1, 0, p));
  }

  derivative(dim: number): ScalarFieldData {
    return numericalDerivative(this, dim);
  }

  protected override computeStats(): ScalarStats {
    return statsFromSpec(this.statsSpec, () => this.data);
  }
}

export class DenseVectorFieldData extends VectorFieldData {
  readonly kind = "sampled" as const;
  readonly dimCount: number;
  readonly box: Box;
  readonly fn: VectorFn;

  /** data is flattened [sampleCount, dimCount] */
  constructor(
    readonly samplePoints: DenseGrid,
    readonly data: ArrayData,
  ) {
    super();
    const D = samplePoints.dimCount;
    if (data.length !== samplePoints.sampleCount * D)
      throw new SpecError(`dense vector samples have ${data.length} values but the grid needs ${samplePoints.sampleCount} x ${D}`);
    this.dimCount = D;
    this.box = samplePoints.box;
    const grid = samplePoints;
    this.fn = (p, pos, out) => {
      if (pos >= 0) for (let i = 0; i < D; i++) out[i] = data[pos * D + i]!;
      else for (let i = 0; i < D; i++) out[i] = interpolate(grid, data, D, i, p);
      return out;
    };
  }

  component(index: number): ScalarFieldData {
    const grid = this.samplePoints, data = this.data, D = this.dimCount;
    const comp: ScalarFn = (p, pos) => (pos >= 0 ? data[pos * D + index]! : interpolate(grid, data, D, index, p));
    const out: ClosureScalarFieldData = new ClosureScalarFieldData("sampled", D, this.box, grid, comp, (d) => numericalDerivative(out, d));
    return out;
  }
}

/*******************************************************/
/* symbolic & pointwise */

/** the arguments a pointwise expression may refer to */
export interface FieldArgs {
  readonly scalars: Readonly<Record<string, ScalarFieldData>>;
  readonly vectors: Readonly<Record<string, VectorFieldData>>;
}

export const NO_ARGS: FieldArgs = { scalars: {}, vectors: {} };
const argsCostly = (args: FieldArgs): boolean => Object.values(args.scalars).some((a) => a.costly) || Object.values(args.vectors).some((a) => a.costly);

/**
 * `f` with its values (and, lazily, its derivatives') precomputed on `grid`
 * by ONE batched `sampleOn` each: `fn(p, pos)` answers from the cache when
 * `pos` is a grid position and falls back to `f` elsewhere. This is how an
 * expression over a costly argument is sampled on a grid without a per-point
 * evaluation of the argument (see `SymbolicScalarFieldData.sampleOn`).
 */
function gridCached(f: ScalarFieldData, grid: DenseGrid): ScalarFieldData {
  let vals: Float64Array | undefined;
  const fn: ScalarFn = (p, pos) => (pos >= 0 ? (vals ??= f.sampleOn(grid))[pos]! : f.fn(p, pos));
  const derivs = new Map<number, ScalarFieldData>();
  return new ClosureScalarFieldData(f.kind, f.dimCount, f.box, f.samplePoints, fn, (d) => {
    let g = derivs.get(d);
    if (!g) derivs.set(d, (g = gridCached(f.derivative(d), grid)));
    return g;
  }, f.costly);
}

class GridCachedVectorFieldData extends VectorFieldData {
  readonly kind: FieldKind;
  readonly dimCount: number;
  readonly box: Box;
  readonly samplePoints: DenseGrid | undefined;
  readonly fn: VectorFn;
  override get costly(): boolean { return this.inner.costly; }
  private vals: Float64Array | undefined;
  private readonly comps = new Map<number, ScalarFieldData>();
  constructor(private readonly inner: VectorFieldData, private readonly grid: DenseGrid) {
    super();
    this.kind = inner.kind; this.dimCount = inner.dimCount; this.box = inner.box; this.samplePoints = inner.samplePoints;
    const D = inner.dimCount;
    this.fn = (p, pos, out) => {
      if (pos < 0) return inner.fn(p, pos, out);
      const v = (this.vals ??= inner.sampleOn(grid));
      for (let d = 0; d < D; d++) out[d] = v[pos * D + d]!;
      return out;
    };
  }
  component(index: number): ScalarFieldData {
    let c = this.comps.get(index);
    if (!c) this.comps.set(index, (c = gridCached(this.inner.component(index), this.grid)));
    return c;
  }
}

/** the arguments with every costly one cached on `grid` */
function gridCachedArgs(args: FieldArgs, grid: DenseGrid): FieldArgs {
  return {
    scalars: Object.fromEntries(Object.entries(args.scalars).map(([k, a]) => [k, a.costly ? gridCached(a, grid) : a])),
    vectors: Object.fromEntries(Object.entries(args.vectors).map(([k, a]) => [k, a.costly ? new GridCachedVectorFieldData(a, grid) : a])),
  };
}

function argContext(dimCount: number, args: FieldArgs): CompileContext {
  const need = <T>(table: Readonly<Record<string, T>>, name: string, what: string): T => {
    const a = table[name];
    if (!a) throw new SpecError(`unknown ${what} argument "${name}"`);
    return a;
  };
  return {
    dimCount,
    scalarArg: (name) => need(args.scalars, name, "scalar").fn,
    vectorArg: (name) => need(args.vectors, name, "vector").fn,
    scalarArgPartial: (name, dims) => dims.reduce<ScalarFieldData>((f, d) => f.derivative(d), need(args.scalars, name, "scalar")).fn,
    vectorArgPartial: (name, index, dims) => dims.reduce<ScalarFieldData>((f, d) => f.derivative(d), need(args.vectors, name, "vector").component(index)).fn,
  };
}

/**
 * Determine the kind, box and sample points of an expression-defined field
 * from its arguments: sampled if any argument is (all sampled arguments must
 * share one support); the box is the intersection of the argument boxes.
 */
function deriveSupport(dimCount: number, args: FieldArgs, box: Box | undefined, path: string[]): { kind: FieldKind; box: Box; grid: DenseGrid | undefined } {
  const all: FieldData[] = [...Object.values(args.scalars), ...Object.values(args.vectors)];
  for (const a of all)
    if (a.dimCount !== dimCount) throw new SpecError(`argument has ${a.dimCount} dims, expected ${dimCount}`, path);
  let grid: DenseGrid | undefined;
  for (const a of all) {
    if (a.kind !== "sampled") continue;
    if (!grid) grid = a.samplePoints;
    else if (!grid.equals(a.samplePoints!, 1e-12))
      throw new SpecError(`sampled arguments have different sample points; pointwise combination requires identical discrete supports`, path);
  }
  let b = box ?? (all.length ? undefined : Box.unit(dimCount));
  for (const a of all) {
    if (!b) { b = a.box; continue; }
    const inter = b.intersect(a.box);
    if (!inter) throw new SpecError(`argument boxes do not overlap`, path);
    b = inter;
  }
  if (grid && !grid.box.equals(b!, 1e-12)) throw new SpecError(`box does not match the sampled arguments' box`, path);
  return { kind: grid ? "sampled" : "symbolic", box: b!, grid };
}

export class SymbolicScalarFieldData extends ScalarFieldData {
  readonly kind: FieldKind;
  readonly box: Box;
  readonly samplePoints: DenseGrid | undefined;
  readonly fn: ScalarFn;
  private readonly ctx: CompileContext;

  /**
   * @param ast normalized expression (may refer to `args`)
   * @param box explicit box (symbolic fields without args default to the unit box)
   */
  readonly explicitBox: Box | undefined;

  constructor(
    readonly ast: SExpr,
    readonly dimCount: number,
    readonly args: FieldArgs = NO_ARGS,
    box?: Box,
    path: string[] = [],
  ) {
    super();
    this.explicitBox = box;
    const s = deriveSupport(dimCount, args, box, path);
    this.kind = s.kind;
    this.box = s.box;
    this.samplePoints = s.grid;
    this.ctx = Object.keys(args.scalars).length + Object.keys(args.vectors).length ? argContext(dimCount, args) : pureContext(dimCount);
    this.fn = compileScalar(ast, this.ctx);
  }
  override get costly(): boolean { return argsCostly(this.args); }

  /** costly arguments are sampled on the grid in batches, then the expression runs with `pos` set */
  override sampleOn(grid: DenseGrid): Float64Array {
    if (!this.costly || grid.dimCount !== this.dimCount) return super.sampleOn(grid);
    const fn = compileScalar(this.ast, argContext(this.dimCount, gridCachedArgs(this.args, grid)));
    const out = new Float64Array(grid.sampleCount);
    const p = new Float64Array(this.dimCount);
    for (let i = 0; i < out.length; i++) { grid.pointInto(i, p); out[i] = fn(p, i); }
    return out;
  }

  private readonly derivatives = new Map<number, ScalarFieldData>();
  derivative(dim: number): ScalarFieldData {
    let d = this.derivatives.get(dim);
    if (!d) this.derivatives.set(dim, (d = new SymbolicScalarFieldData(diffScalar(this.ast, dim, this.dimCount), this.dimCount, this.args, this.explicitBox)));
    return d;
  }
}

export class SymbolicVectorFieldData extends VectorFieldData {
  readonly kind: FieldKind;
  readonly box: Box;
  readonly samplePoints: DenseGrid | undefined;
  readonly fn: VectorFn;
  private readonly ctx: CompileContext;
  readonly explicitBox: Box | undefined;

  constructor(
    readonly ast: VExpr,
    readonly dimCount: number,
    readonly args: FieldArgs = NO_ARGS,
    box?: Box,
    path: string[] = [],
  ) {
    super();
    this.explicitBox = box;
    const s = deriveSupport(dimCount, args, box, path);
    this.kind = s.kind;
    this.box = s.box;
    this.samplePoints = s.grid;
    this.ctx = Object.keys(args.scalars).length + Object.keys(args.vectors).length ? argContext(dimCount, args) : pureContext(dimCount);
    this.fn = compileVector(ast, this.ctx);
  }
  override get costly(): boolean { return argsCostly(this.args); }

  /** costly arguments are sampled on the grid in batches, then the expression runs with `pos` set */
  override sampleOn(grid: DenseGrid): Float64Array {
    if (!this.costly || grid.dimCount !== this.dimCount) return super.sampleOn(grid);
    const fn = compileVector(this.ast, argContext(this.dimCount, gridCachedArgs(this.args, grid)));
    const D = this.dimCount;
    const out = new Float64Array(grid.sampleCount * D);
    const p = new Float64Array(D), v = new Float64Array(D);
    for (let i = 0; i < grid.sampleCount; i++) { grid.pointInto(i, p); fn(p, i, v); out.set(v, i * D); }
    return out;
  }

  component(index: number): ScalarFieldData {
    return new SymbolicScalarFieldData({ k: "comp", v: this.ast, index }, this.dimCount, this.args, this.explicitBox);
  }
}

/*******************************************************/
/* pullbacks: translate / scale the domain */

/** an affine, axis-aligned reparametrization q = origin + (p - shift - origin) / factors */
export class AxisMap {
  constructor(
    readonly shift: readonly number[],
    readonly origin: readonly number[],
    readonly factors: readonly number[],
  ) {
    for (const f of factors) if (f === 0) throw new SpecError("scale factor must be non-zero");
  }
  /** the box / grid of the wrapped datum, seen from outside */
  outerBox(inner: Box): Box { return inner.scale(this.origin, this.factors).translate(this.shift); }
  outerGrid(inner: DenseGrid): DenseGrid { return inner.scale(this.origin, this.factors).translate(this.shift); }
  toInner(p: ArrayLike<number>, out: Float64Array): Float64Array {
    for (let d = 0; d < out.length; d++) out[d] = this.origin[d]! + (p[d]! - this.shift[d]! - this.origin[d]!) / this.factors[d]!;
    return out;
  }
  static translate(D: number, vec: readonly number[]) {
    return new AxisMap(vec, new Array<number>(D).fill(0), new Array<number>(D).fill(1));
  }
  static scale(D: number, origin: readonly number[] | undefined, scale: number | readonly number[]) {
    return new AxisMap(new Array<number>(D).fill(0), origin ?? new Array<number>(D).fill(0), typeof scale === "number" ? new Array<number>(D).fill(scale) : scale);
  }
}

export class PulledBackScalarFieldData extends ScalarFieldData {
  readonly kind: FieldKind;
  readonly dimCount: number;
  readonly box: Box;
  readonly samplePoints: DenseGrid | undefined;
  readonly fn: ScalarFn;

  override get costly(): boolean { return this.inner.costly; }

  constructor(readonly inner: ScalarFieldData, readonly map: AxisMap) {
    super();
    this.kind = inner.kind;
    this.dimCount = inner.dimCount;
    this.box = map.outerBox(inner.box);
    this.samplePoints = inner.samplePoints && map.outerGrid(inner.samplePoints);
    const q = new Float64Array(this.dimCount);
    this.fn = (p, pos) => inner.fn(map.toInner(p, q), pos);
  }

  derivative(dim: number): ScalarFieldData {
    // chain rule: d/dp f(q(p)) = f'(q) / factor
    return scaleValues(new PulledBackScalarFieldData(this.inner.derivative(dim), this.map), 1 / this.map.factors[dim]!);
  }

  protected override computeStats(): ScalarStats { return this.inner.stats(); }
}

/** NB: vector VALUES are left untouched (no pushforward); only the domain is reparametrized. */
export class PulledBackVectorFieldData extends VectorFieldData {
  readonly kind: FieldKind;
  readonly dimCount: number;
  readonly box: Box;
  readonly samplePoints: DenseGrid | undefined;
  readonly fn: VectorFn;
  override get costly(): boolean { return this.inner.costly; }

  constructor(readonly inner: VectorFieldData, readonly map: AxisMap) {
    super();
    this.kind = inner.kind;
    this.dimCount = inner.dimCount;
    this.box = map.outerBox(inner.box);
    this.samplePoints = inner.samplePoints && map.outerGrid(inner.samplePoints);
    const q = new Float64Array(this.dimCount);
    this.fn = (p, pos, out) => inner.fn(map.toInner(p, q), pos, out);
  }

  component(index: number): ScalarFieldData {
    return new PulledBackScalarFieldData(this.inner.component(index), this.map);
  }
}

export function translateField<T extends FieldData>(inner: T, vec: readonly number[]): T {
  if (vec.length !== inner.dimCount) throw new SpecError(`translation vector has ${vec.length} components, field has ${inner.dimCount} dims`);
  const map = AxisMap.translate(inner.dimCount, vec);
  return (inner.rank === "scalar" ? new PulledBackScalarFieldData(inner, map) : new PulledBackVectorFieldData(inner, map)) as unknown as T;
}

export function scaleField<T extends FieldData>(inner: T, origin: readonly number[] | undefined, scale: number | readonly number[]): T {
  if (origin && origin.length !== inner.dimCount) throw new SpecError(`scale origin has ${origin.length} components, field has ${inner.dimCount} dims`);
  if (typeof scale !== "number" && scale.length !== inner.dimCount) throw new SpecError(`scale has ${scale.length} factors, field has ${inner.dimCount} dims`);
  const map = AxisMap.scale(inner.dimCount, origin, scale);
  return (inner.rank === "scalar" ? new PulledBackScalarFieldData(inner, map) : new PulledBackVectorFieldData(inner, map)) as unknown as T;
}

/**
 * Restrict a scalar field to the hyperplane `p[axis] = value`: a field of one
 * dimension less on the box with that axis removed. Symbolic data stays
 * symbolic (values and derivatives are the inner field's, evaluated on the
 * plane), so exact contouring works on slices; sampled data stays sampled
 * (contoured linearly on whatever grid the caller samples).
 */
export function sliceScalarField(f: ScalarFieldData, axis: number, value: number): ScalarFieldData {
  const D = f.dimCount;
  if (axis < 0 || axis >= D) throw new EvalError(`slice axis ${axis} out of range for ${D}D data`);
  const keep = Array.from({ length: D }, (_, d) => d).filter((d) => d !== axis);
  const box = new Box(keep.map((d) => f.box.a[d]!), keep.map((d) => f.box.b[d]!));
  const q = new Float64Array(D);
  const fn: ScalarFn = (p) => {
    for (let i = 0; i < keep.length; i++) q[keep[i]!] = p[i]!;
    q[axis] = value;
    return f.fn(q, -1);
  };
  return new ClosureScalarFieldData(f.kind, D - 1, box, undefined, fn, (dim) => sliceScalarField(f.derivative(keep[dim]!), axis, value));
}

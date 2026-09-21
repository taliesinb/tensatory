// Fields backed by nets (schema/nets.ts "FIELDS BACKED BY NETS").
//
// A net field is a PROGRAM WITH ONE INPUT — the point, shape [D] — and one
// chosen output ([] for scalar fields, [D] for vector fields). The field spec's
// `inputs` expressions of the coordinates (and its named `arrays`) are folded
// into that program (`fieldProgram`), so nothing downstream knows about
// coordinates: the CPU evaluator, the WGSL emitter and autodiff all see an
// ordinary program.
//
// kind = "symbolic": evaluable anywhere in the box. Sampling a grid is ONE
// batched evaluation of the program with the G points as the leading axis of
// the point input (chunked to bound intermediate memory). `derivative(dim)`
// is EXACT: the gradient program (autodiff.ts) with respect to the point,
// component `dim` — itself a net field, so derivatives compose to any order.
// Net fields report `costly`, so CPU consumers sample them sparingly.

import type { ArrayExpr } from "@tensatory/schema";
import { NdArray } from "../arrays/ndarray";
import { EvalError, SpecError } from "../errors";
import type { Box } from "../geometry/box";
import { DenseGrid } from "../geometry/grid";
import { ScalarFieldData, VectorFieldData } from "../fields/fieldData";
import { computeStats, type ScalarStats } from "../fields/stats";
import type { ScalarFn, VectorFn } from "../symbolic/compile";
import { gradProgram } from "./autodiff";
import type { Val } from "./ops";
import { evaluate, pruneProgram, type Program, type ProgramResolver } from "./program";
import type { Shape } from "./shapes";

/** points per batched evaluation when sampling a grid */
const CHUNK = 2048;

/** a program with exactly one input (the point, [D]) and the output the field reads */
export interface NetField {
  readonly program: Program;
  readonly output: string;
  readonly nets: ProgramResolver;
}

/** the name of the program's sole input */
export const pointInput = (prog: Program): string => {
  const names = Object.keys(prog.inputs);
  if (names.length !== 1) throw new SpecError(`a net field's program must have exactly one input (the point), this one has ${names.length}`);
  return names[0]!;
};

const POINT = "__p";

/**
 * Fold a field spec's coordinate expressions into the net's program: the
 * result has the single input `__p: [D]`; the net's inputs become nodes
 * computed from it (`coordv` → the point, `coord i` → its i-th component),
 * the spec's named arrays become constants (renamed to avoid the net's names).
 * With `inputs` undefined the net's sole input simply IS the point.
 */
export function fieldProgram(base: Program, inputs: Record<string, ArrayExpr> | undefined, arrays: ReadonlyMap<string, Val>, dimCount: number, path: string[] = ["data"]): Program {
  if (inputs === undefined) { pointInput(base); return base; }
  const taken = new Set([...Object.keys(base.inputs), ...Object.keys(base.consts), ...base.nodes.map((n) => n.name), ...Object.keys(base.shapes)]);
  if (taken.has(POINT)) throw new SpecError(`the net uses the reserved name "${POINT}"`, path);
  const rename = new Map<string, string>();
  const consts: Program["consts"] = { ...base.consts };
  const shapes: Record<string, Shape> = { ...base.shapes, [POINT]: [dimCount] };
  for (const [n, v] of arrays) {
    const nm = taken.has(n) || n === POINT ? `__a_${n}` : n;
    rename.set(n, nm);
    consts[nm] = { arr: v.arr, shape: [...v.arr.shape] };
    shapes[nm] = [...v.arr.shape];
  }
  const fold = (e: ArrayExpr): ArrayExpr => {
    if (typeof e !== "object") return typeof e === "string" ? rename.get(e) ?? e : e;
    if (e.op === "coordv") return POINT;
    if (e.op === "coord") return { op: "reshape", val: { op: "slice", val: POINT, axis: 0, start: e.index, stop: e.index + 1 }, shape: [] };
    if (e.op === "arg") return rename.get(e.name) ?? e.name;
    if (e.op === "call") return { ...e, inputs: Object.fromEntries(Object.entries(e.inputs).map(([k, v]) => [k, fold(v)])) };
    const out = { ...e } as unknown as Record<string, unknown>;
    for (const k of ["val", "vals", "min", "max", "cond", "indices"]) {
      const v = out[k];
      if (v === undefined) continue;
      out[k] = Array.isArray(v) ? (v as ArrayExpr[]).map(fold) : fold(v as ArrayExpr);
    }
    return out as unknown as ArrayExpr;
  };
  const nodes: Program["nodes"] = [];
  for (const [n, e] of Object.entries(inputs)) {
    if (!(n in base.inputs)) throw new SpecError(`"${n}" is not an input of the net`, [...path, "inputs", n]);
    nodes.push({ name: n, expr: fold(e) });
  }
  for (const n of Object.keys(base.inputs)) if (!(n in inputs)) throw new SpecError(`net input "${n}" is not given`, [...path, "inputs"]);
  nodes.push(...base.nodes);
  return { inputs: { [POINT]: [dimCount] }, consts, nodes, outputs: base.outputs, shapes };
}

/** the field's program extended by a node selecting one component of `output` ([D] → []) */
function componentProgram(f: NetField, index: number): NetField {
  const src = f.program.outputs[f.output]!;
  // a fresh name: a component of a gradient of a component (second derivatives) would otherwise define `__c1` twice
  let name = `__c${index}`;
  for (let k = 1; name in f.program.shapes || f.program.nodes.some((n) => n.name === name); k++) name = `__c${index}_${k}`;
  const expr: ArrayExpr = { op: "reshape", val: { op: "slice", val: src, axis: 0, start: index, stop: index + 1 }, shape: [] };
  const program: Program = { ...f.program, nodes: [...f.program.nodes, { name, expr }], outputs: { ...f.program.outputs, [name]: name }, shapes: { ...f.program.shapes, [name]: [] } };
  return { program, output: name, nets: f.nets };
}

/** the gradient of a scalar field with respect to the point: a [D] output; the value stays an output too (`__value`) */
function gradientProgram(f: NetField): NetField {
  const p = pointInput(f.program);
  const program = gradProgram(f.program, [{ name: "__grad", of: f.output, wrt: p }], [f.output], f.nets, ["grad"]);
  const outputs = { ...program.outputs, __value: program.outputs[f.output]! };
  return { program: { ...program, outputs }, output: "__grad", nets: f.nets };
}

/** the output name of the forward value inside a gradient program (see `gradientProgram`) */
export const GRADIENT_VALUE_OUTPUT = "__value";

/** the program cut down to the field's output, for the CPU evaluator (a net may compute several metrics) */
const pruned = new WeakMap<Program, Map<string, Program>>();
function prunedProgram(f: NetField): Program {
  let m = pruned.get(f.program);
  if (!m) pruned.set(f.program, (m = new Map()));
  let p = m.get(f.output);
  if (!p) m.set(f.output, (p = pruneProgram(f.program, [f.output])));
  return p;
}

/** evaluate the chosen output at points [G, D]; returns [G, ...outShape] */
export function evalPoints(f: NetField, coords: NdArray): NdArray {
  const G = coords.shape[0]!;
  const prog = prunedProgram(f);
  const out = evaluate(prog, { [pointInput(prog)]: coords }, { nets: f.nets })[f.output]!;
  const rank = f.program.shapes[f.program.outputs[f.output]!]!.length;
  const batch = out.shape.slice(0, out.ndim - rank);
  if (batch.length === 0) {
    // the output does not depend on the point: the same value everywhere
    const rep = new NdArray([G, ...out.shape]);
    for (let g = 0; g < G; g++) rep.data.set(out.data, g * out.size);
    return rep;
  }
  if (batch.length !== 1 || batch[0] !== G) throw new EvalError(`net output has batch [${batch}], expected [${G}]`);
  return out;
}

function pointsOf(grid: DenseGrid, from: number, to: number): NdArray {
  const D = grid.dimCount;
  const coords = new NdArray([to - from, D]);
  const p = new Float64Array(D);
  for (let i = from; i < to; i++) {
    grid.pointInto(i, p);
    coords.data.set(p, (i - from) * D);
  }
  return coords;
}

/** the grid statistics of a costly field are computed on: far smaller than the default */
function statsGrid(box: Box): DenseGrid {
  const n = box.dimCount <= 1 ? 256 : box.dimCount === 2 ? 48 : box.dimCount === 3 ? 12 : 4;
  return new DenseGrid(new Array<number>(box.dimCount).fill(n), box);
}

export class NetScalarFieldData extends ScalarFieldData {
  readonly kind = "symbolic" as const;
  readonly samplePoints = undefined;
  readonly fn: ScalarFn;
  override get costly(): boolean { return true; }

  constructor(readonly dimCount: number, readonly box: Box, readonly field: NetField) {
    super();
    this.fn = (p) => evalPoints(field, new NdArray([1, dimCount], Float64Array.from(p as ArrayLike<number>))).data[0]!;
  }

  override sampleOn(grid: DenseGrid): Float64Array {
    if (grid.dimCount !== this.dimCount) throw new EvalError(`grid has ${grid.dimCount} dims, field has ${this.dimCount}`);
    const out = new Float64Array(grid.sampleCount);
    for (let from = 0; from < grid.sampleCount; from += CHUNK) {
      const to = Math.min(grid.sampleCount, from + CHUNK);
      out.set(evalPoints(this.field, pointsOf(grid, from, to)).data.subarray(0, to - from), from);
    }
    return out;
  }

  protected override computeStats(): ScalarStats { return computeStats(this.sampleOn(statsGrid(this.box))); }

  private _gradient: NetVectorFieldData | undefined;
  /** ∇f as a net vector field (one backward pass for all D components) */
  gradient(): NetVectorFieldData {
    return (this._gradient ??= new NetVectorFieldData(this.dimCount, this.box, gradientProgram(this.field), this));
  }

  private readonly derivatives = new Map<number, ScalarFieldData>();
  /** ∂f/∂x_dim, exact: component `dim` of the gradient program */
  derivative(dim: number): ScalarFieldData {
    let d = this.derivatives.get(dim);
    if (!d) this.derivatives.set(dim, (d = this.gradient().component(dim)));
    return d;
  }
}

export class NetVectorFieldData extends VectorFieldData {
  readonly kind = "symbolic" as const;
  readonly samplePoints = undefined;
  readonly fn: VectorFn;
  override get costly(): boolean { return true; }

  /**
   * @param gradientOf the scalar net field this is the gradient of (`NetScalarFieldData.gradient()`): a consumer that
   *   cannot evaluate the autodiff program may fall back to differences of the scalar's values
   */
  constructor(readonly dimCount: number, readonly box: Box, readonly field: NetField, readonly gradientOf?: NetScalarFieldData) {
    super();
    this.fn = (p, _pos, out) => {
      const v = evalPoints(field, new NdArray([1, dimCount], Float64Array.from(p as ArrayLike<number>)));
      out.set(v.data.subarray(0, dimCount));
      return out;
    };
  }

  override sampleOn(grid: DenseGrid): Float64Array {
    if (grid.dimCount !== this.dimCount) throw new EvalError(`grid has ${grid.dimCount} dims, field has ${this.dimCount}`);
    const D = this.dimCount;
    const out = new Float64Array(grid.sampleCount * D);
    for (let from = 0; from < grid.sampleCount; from += CHUNK) {
      const to = Math.min(grid.sampleCount, from + CHUNK);
      out.set(evalPoints(this.field, pointsOf(grid, from, to)).data.subarray(0, (to - from) * D), from * D);
    }
    return out;
  }

  private readonly components = new Map<number, NetScalarFieldData>();
  component(index: number): NetScalarFieldData {
    let c = this.components.get(index);
    if (!c) this.components.set(index, (c = new NetScalarFieldData(this.dimCount, this.box, componentProgram(this.field, index))));
    return c;
  }
}

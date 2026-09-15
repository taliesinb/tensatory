// Field data -> a WGSL compute program that samples it on a DenseGrid, with
// exactly the CPU semantics of `field.sampleOn(grid)`:
//   * symbolic data is transpiled (derivatives through arguments included);
//   * dense data lives in storage buffers, read directly when the dispatch
//     grid is its own support and multilinearly interpolated otherwise;
//   * derivatives of sampled arguments are computed by core (grid
//     differences) and uploaded;
//   * pullbacks become coordinate maps;
//   * anything else is sampled by core on the dispatch grid and uploaded.

import {
  DenseGrid,
  DenseScalarFieldData,
  DenseVectorFieldData,
  PulledBackScalarFieldData,
  PulledBackVectorFieldData,
  SymbolicScalarFieldData,
  SymbolicVectorFieldData,
  gradient,
  type AxisMap,
  type ScalarFieldData,
  type VectorFieldData,
} from "@tensatory/core";
import { FunctionEmitter, PRELUDE, expandGrad, f32, vecType, type ArgBindings } from "./wgsl";

export interface GpuProgram {
  code: string;
  /** every uploaded array, packed into one storage buffer (binding 1; binding 0 is the output) */
  data: Float32Array;
  /** number of output values per grid point (1 for scalars, D for vectors) */
  channels: number;
  sampleCount: number;
}

const NAN = "nan_()"; // WGSL rejects NaN constants; the prelude builds one at runtime

export class ProgramBuilder {
  private readonly fns: string[] = [];
  private readonly chunks: Float32Array[] = [];
  private dataLength = 0;
  private readonly emitted = new Map<object, Map<string, string>>();
  private n = 0;
  readonly D: number;

  constructor(readonly grid: DenseGrid) {
    this.D = grid.dimCount;
  }

  private name(prefix: string): string { return `${prefix}_${this.n++}`; }

  private memo(fd: object, key: string, make: () => string): string {
    let m = this.emitted.get(fd);
    if (!m) this.emitted.set(fd, (m = new Map()));
    const have = m.get(key);
    if (have) return have;
    // reserve the name first so recursive references (impossible here, but cheap insurance) terminate
    const nm = make();
    m.set(key, nm);
    return nm;
  }

  /** pack an array into the shared data buffer; returns its element offset */
  private upload(data: ArrayLike<number>): number {
    const chunk = Float32Array.from(data as ArrayLike<number>);
    const offset = this.dataLength;
    this.chunks.push(chunk);
    this.dataLength += chunk.length;
    return offset;
  }

  /** upload a grid-shaped array and emit a reader `fn name(p, pos) -> f32` for channel `ch` */
  private bufferReader(data: ArrayLike<number>, grid: DenseGrid, channels: number, ch: number): string {
    const off = this.upload(data);
    const buf = "data";
    const nm = this.name("rd");
    const direct = grid.equals(this.grid, 1e-12);
    const D = this.D;
    const P = (d: number) => (D === 1 ? "p" : `p[${d}]`);
    const lines: string[] = [];
    // interpolation (unrolled over dimensions and 2^D corners)
    for (let d = 0; d < D; d++) {
      const n = grid.size[d]!, a = grid.box.a[d]!, b = grid.box.b[d]!, sp = grid.spacing[d]!;
      const eps = 1e-6 * (b - a) + 1e-7; // f32-scale containment slack (core uses 1e-9 in f64)
      lines.push(`  if (${P(d)} < ${f32(a - eps)} || ${P(d)} > ${f32(b + eps)}) { return ${NAN}; }`);
      if (n > 1) {
        lines.push(`  let g${d}: f32 = clamp((${P(d)} - ${f32(a)}) / ${f32(sp)}, 0.0, ${f32(n - 1)});`);
        lines.push(`  var i${d}: i32 = i32(floor(g${d})); if (i${d} >= ${n - 1}) { i${d} = ${Math.max(0, n - 2)}; }`);
        lines.push(`  let f${d}: f32 = g${d} - f32(i${d});`);
      } else {
        lines.push(`  let i${d}: i32 = 0; let f${d}: f32 = 0.0;`);
      }
    }
    lines.push(`  var r: f32 = 0.0;`);
    for (let c = 0; c < 1 << D; c++) {
      const w: string[] = [], idx: string[] = [];
      for (let d = 0; d < D; d++) {
        const hi = (c >> d) & 1;
        w.push(hi ? `f${d}` : `(1.0 - f${d})`);
        idx.push(`min(i${d} + ${hi}, ${grid.size[d]! - 1}) * ${grid.strides[d]}`);
      }
      lines.push(`  r = r + ${w.join(" * ")} * ${buf}[${off} + (${idx.join(" + ")}) * ${channels} + ${ch}];`);
    }
    lines.push(`  return r;`);
    const body = direct ? `  if (pos >= 0) { return ${buf}[${off} + pos * ${channels} + ${ch}]; }\n${lines.join("\n")}` : lines.join("\n");
    this.fns.push(`fn ${nm}(p: ${vecType(D)}, pos: i32) -> f32 {\n${body}\n}`);
    return nm;
  }

  private bindings(args: { scalars: Readonly<Record<string, ScalarFieldData>>; vectors: Readonly<Record<string, VectorFieldData>> }): ArgBindings {
    return {
      scalar: (name, dims) => this.scalar(args.scalars[name]!, dims),
      vector: (name) => this.vector(args.vectors[name]!),
      vectorComponent: (name, index, dims) => this.scalar(args.vectors[name]!.component(index), dims),
    };
  }

  private mapCode(map: AxisMap): string {
    const D = this.D;
    const comps = Array.from({ length: D }, (_, d) => `${f32(map.origin[d]!)} + (${D === 1 ? "p" : `p[${d}]`} - ${f32(map.shift[d]! + map.origin[d]!)}) / ${f32(map.factors[d]!)}`);
    return D === 1 ? comps[0]! : `${vecType(D)}(${comps.join(", ")})`;
  }

  /** function name for `dims.reduce(derivative, fd)` */
  scalar(fd: ScalarFieldData, dims: number[] = []): string {
    return this.memo(fd, `s:${dims.join(",")}`, () => {
      if (fd instanceof SymbolicScalarFieldData) {
        const target = dims.reduce<ScalarFieldData>((f, d) => f.derivative(d), fd) as SymbolicScalarFieldData;
        if (dims.length) return this.scalar(target, []);
        const em = new FunctionEmitter(this.D, this.bindings(fd.args));
        const ast = expandGrad(fd.ast, gradient, this.D);
        const nm = this.name("sf");
        this.fns.push(em.functionText(nm, "f32", em.scalar(ast)));
        return nm;
      }
      if (fd instanceof DenseScalarFieldData) {
        const grid = fd.samplePoints;
        const data = dims.length ? dims.reduce<ScalarFieldData>((f, d) => f.derivative(d), fd).sampleOn(grid) : fd.data;
        return this.bufferReader(data, grid, 1, 0);
      }
      if (fd instanceof PulledBackScalarFieldData) {
        const inner = this.scalar(fd.inner, dims);
        const k = dims.reduce((acc, d) => acc / fd.map.factors[d]!, 1);
        const nm = this.name("pb");
        this.fns.push(`fn ${nm}(p: ${vecType(this.D)}, pos: i32) -> f32 {\n  let q = ${this.mapCode(fd.map)};\n  return ${f32(k)} * ${inner}(q, pos);\n}`);
        return nm;
      }
      // fallback: let core evaluate it on the dispatch grid
      const target = dims.reduce<ScalarFieldData>((f, d) => f.derivative(d), fd);
      return this.bufferReader(target.sampleOn(this.grid), this.grid, 1, 0);
    });
  }

  vector(fd: VectorFieldData): string {
    return this.memo(fd, "v", () => {
      const D = this.D, T = vecType(D);
      if (fd instanceof SymbolicVectorFieldData) {
        const em = new FunctionEmitter(D, this.bindings(fd.args));
        const ast = expandGrad(fd.ast, gradient, D);
        const nm = this.name("vf");
        this.fns.push(em.functionText(nm, T, em.vector(ast)));
        return nm;
      }
      if (fd instanceof PulledBackVectorFieldData) {
        const inner = this.vector(fd.inner);
        const nm = this.name("pbv");
        this.fns.push(`fn ${nm}(p: ${T}, pos: i32) -> ${T} {\n  let q = ${this.mapCode(fd.map)};\n  return ${inner}(q, pos);\n}`);
        return nm;
      }
      // dense (or fallback: sampled by core on the dispatch grid), one reader per channel
      const grid = fd instanceof DenseVectorFieldData ? fd.samplePoints : this.grid;
      const data = fd instanceof DenseVectorFieldData ? fd.data : fd.sampleOn(this.grid);
      const readers = Array.from({ length: D }, (_, ch) => this.bufferReader(data, grid, D, ch));
      const nm = this.name("dv");
      const comps = readers.map((r) => `${r}(p, pos)`);
      this.fns.push(`fn ${nm}(p: ${T}, pos: i32) -> ${T} {\n  return ${D === 1 ? comps[0] : `${T}(${comps.join(", ")})`};\n}`);
      return nm;
    });
  }

  /** the complete program sampling `field` on the dispatch grid */
  build(field: ScalarFieldData | VectorFieldData): GpuProgram {
    const D = this.D, grid = this.grid;
    const entry = field.rank === "scalar" ? this.scalar(field) : this.vector(field);
    const direct = field.samplePoints?.equals(grid, 1e-12) ?? false;
    const channels = field.rank === "scalar" ? 1 : D;
    // grid position -> point, row-major (last axis fastest)
    const idx: string[] = ["  var rem: i32 = i;"];
    const comps: string[] = [];
    for (let d = 0; d < D; d++) {
      const s = grid.strides[d]!;
      idx.push(`  let g${d}: i32 = rem / ${s}; rem = rem - g${d} * ${s};`);
      comps.push(`${f32(grid.box.a[d]!)} + f32(g${d}) * ${f32(grid.spacing[d]!)}`);
    }
    const p = D === 1 ? comps[0]! : `${vecType(D)}(${comps.join(", ")})`;
    const write = channels === 1 ? `  out[i] = ${entry}(p, pos);` : `  let v = ${entry}(p, pos);\n${Array.from({ length: D }, (_, d) => `  out[i * ${D} + ${d}] = v[${d}];`).join("\n")}`;
    const main = `@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<storage, read> data: array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i: i32 = i32(id.x);
  if (i >= ${grid.sampleCount}) { return; }
${idx.join("\n")}
  let p = ${p};
  let pos: i32 = ${direct ? "i" : "-1"};
${write}
}`;
    const data = new Float32Array(Math.max(4, this.dataLength));
    let o = 0;
    for (const c of this.chunks) { data.set(c, o); o += c.length; }
    return { code: [PRELUDE, ...this.fns, main].join("\n\n"), data, channels, sampleCount: grid.sampleCount };
  }
}

export function buildSampleProgram(field: ScalarFieldData | VectorFieldData, grid: DenseGrid): GpuProgram {
  return new ProgramBuilder(grid).build(field);
}

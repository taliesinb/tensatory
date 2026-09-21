// Field data -> a WGSL compute program that samples it on a DenseGrid, with
// exactly the CPU semantics of `field.sampleOn(grid)`:
//   * symbolic data is transpiled (derivatives through arguments included);
//   * dense data lives in storage buffers, read directly when the dispatch
//     grid is its own support and multilinearly interpolated otherwise;
//   * derivatives of sampled arguments are computed by core (grid
//     differences) and uploaded;
//   * pullbacks become coordinate maps;
//   * net-backed fields are transpiled whole (nets.ts): one thread evaluates
//     the net for its point; their derivatives are the autodiff programs;
//   * anything else is sampled by core on the dispatch grid and uploaded.

import {
  DenseGrid,
  DenseScalarFieldData,
  DenseVectorFieldData,
  GRADIENT_VALUE_OUTPUT,
  NetScalarFieldData,
  NetVectorFieldData,
  PulledBackScalarFieldData,
  PulledBackVectorFieldData,
  SymbolicScalarFieldData,
  SymbolicVectorFieldData,
  gradient,
  type AxisMap,
  type ScalarFieldData,
  type VectorFieldData,
} from "@tensatory/core";
import { emitNetField, gpuTranspilable } from "./nets";
import { FunctionEmitter, GRID_FLOATS, OPAQUE_BOUND_WGSL, PRELUDE, bakedGrid, expandGrad, f32, gridWgsl, packGrid, vecType, type ArgBindings, type GridRef } from "./wgsl";

export interface GpuProgram {
  code: string;
  /** every uploaded array, packed into one storage buffer (binding 1; binding 0 is the output); the first
   *  GRID_FLOATS describe the dispatch grid, so the code is the same for every grid a field is sampled on */
  data: Float32Array;
  /** number of output values per grid point (1 for scalars, D for vectors) */
  channels: number;
  sampleCount: number;
}

const NAN = "nan_()"; // WGSL rejects NaN constants; the prelude builds one at runtime

export class ProgramBuilder {
  private readonly fns: string[] = [];
  private readonly chunks: Float32Array[] = [];
  private dataLength = GRID_FLOATS; // the dispatch grid's header comes first
  private readonly emitted = new Map<object, Map<string, string>>();
  private n = 0;
  readonly D: number;
  /** the dispatch grid as seen from WGSL (read from the header, never baked) */
  readonly dg: GridRef;
  private readonly dgCode: string;

  constructor(readonly grid: DenseGrid) {
    this.D = grid.dimCount;
    const g = gridWgsl("dg_", "data", 0);
    this.dg = g.ref; this.dgCode = g.code;
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
    const chunk = data instanceof Float32Array ? data : Float32Array.from(data as ArrayLike<number>); // a stored-f32 array is packed as is
    const offset = this.dataLength;
    this.chunks.push(chunk);
    this.dataLength += chunk.length;
    return offset;
  }

  /**
   * Upload a grid-shaped array and emit a reader `fn name(p, pos) -> f32` for channel `ch`. `grid` is baked into
   * the reader when it is the data's own support (intrinsic to the field); data sampled on the dispatch grid
   * (`grid === this.grid`) reads the grid from the header instead, so the code stays grid-independent.
   */
  private bufferReader(data: ArrayLike<number>, grid: DenseGrid, channels: number, ch: number): string {
    const off = this.upload(data);
    const onDispatch = grid === this.grid;
    return this.reader("data", off, onDispatch ? this.dg : bakedGrid(grid), onDispatch || grid.equals(this.grid, 1e-12), channels, ch);
  }

  /**
   * `fn name(p, pos) -> f32`: multilinear interpolation of a RESIDENT grid (a GpuGrid the kernel binds as `buf`).
   * The grid header goes into `data` (so a resolution change changes data, never code); `direct` when the grid is
   * the dispatch grid (then a sample at `pos` is a plain read). This is how a costly (net-backed) colour field is
   * read at every isoline / isosurface / streamline vertex: one GPU sampling on the grid, then lookups.
   */
  residentReader(g: { grid: DenseGrid; channels: number }, buf: string, ch = 0): string {
    const header = new Float32Array(GRID_FLOATS);
    packGrid(g.grid, header, 0);
    const off = this.upload(header);
    const G = gridWgsl(`rg${off}_`, "data", off);
    this.fns.push(G.code);
    return this.reader(buf, 0, G.ref, g.grid.equals(this.grid, 1e-12), g.channels, ch);
  }

  private reader(buf: string, off: number, G: GridRef, direct: boolean, channels: number, ch: number): string {
    const nm = this.name("rd");
    const D = this.D;
    const P = (d: number) => (D === 1 ? "p" : `p[${d}]`);
    const lines: string[] = [];
    // interpolation (unrolled over dimensions and 2^D corners)
    for (let d = 0; d < D; d++) {
      const ds = String(d);
      lines.push(`  let n${d}: i32 = ${G.n(ds)}; let a${d}: f32 = ${G.a(ds)}; let b${d}: f32 = ${G.b(ds)};`);
      lines.push(`  let eps${d}: f32 = 1e-6 * (b${d} - a${d}) + 1e-7;`); // f32-scale containment slack (core uses 1e-9 in f64)
      lines.push(`  if (${P(d)} < a${d} - eps${d} || ${P(d)} > b${d} + eps${d}) { return ${NAN}; }`);
      lines.push(`  var i${d}: i32 = 0; var f${d}: f32 = 0.0;`);
      lines.push(`  if (n${d} > 1) { let g = clamp((${P(d)} - a${d}) / ${G.h(ds)}, 0.0, f32(n${d} - 1)); i${d} = i32(floor(g)); if (i${d} >= n${d} - 1) { i${d} = max(0, n${d} - 2); } f${d} = g - f32(i${d}); }`);
    }
    lines.push(`  var r: f32 = 0.0;`);
    for (let c = 0; c < 1 << D; c++) {
      const w: string[] = [], idx: string[] = [];
      for (let d = 0; d < D; d++) {
        const hi = (c >> d) & 1;
        w.push(hi ? `f${d}` : `(1.0 - f${d})`);
        idx.push(`min(i${d} + ${hi}, n${d} - 1) * ${G.s(String(d))}`);
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
      if (fd instanceof NetScalarFieldData) {
        // derivatives of a net field are net fields (autodiff): transpile the differentiated program
        const target = dims.reduce<ScalarFieldData>((f, d) => f.derivative(d), fd);
        const net = target instanceof NetScalarFieldData ? this.net(target) : undefined;
        if (net) return net;
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
        // `grad(arg f)` of a net field (the viewer's ∇ uses): one backward program, not D component programs
        const ast0 = fd.ast;
        if (ast0.k === "grad" && ast0.s.k === "arg" && fd.args.scalars[ast0.s.name] instanceof NetScalarFieldData) return this.gradient(fd.args.scalars[ast0.s.name]!);
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
      if (fd instanceof NetVectorFieldData) {
        const net = this.net(fd);
        if (net) return net;
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

  /**
   * `fn name(p, pos) -> vecD`: the gradient of a scalar field. A net field's gradient is ONE program (one backward
   * pass for all components) — three component functions would inline the whole backward pass three times into
   * every kernel that projects onto level sets; other data composes its component derivatives.
   */
  gradient(fd: ScalarFieldData): string {
    return this.memo(fd, "grad", () => {
      const D = this.D, T = vecType(D);
      if (fd instanceof NetScalarFieldData) { const v = this.net(fd.gradient()); if (v) return v; }
      const comps = Array.from({ length: D }, (_, d) => `${this.scalar(fd, [d])}(p, pos)`);
      const nm = this.name("gr");
      this.fns.push(`fn ${nm}(p: ${T}, pos: i32) -> ${T} {\n  return ${D === 1 ? comps[0] : `${T}(${comps.join(", ")})`};\n}`);
      return nm;
    });
  }

  /**
   * `fn name(p, pos) -> vec(D+1)`: (value, gradient) of a scalar field in ONE evaluation — what Newton projection
   * needs at every step. For a net the gradient program computes the forward pass anyway, so this halves the
   * evaluations per step; other data composes value and gradient.
   */
  valueGradient(fd: ScalarFieldData): string {
    return this.memo(fd, "vg", () => {
      const D = this.D, T = vecType(D + 1);
      if (fd instanceof NetScalarFieldData && gpuTranspilable(fd.gradient())) {
        const nm = this.name("nvg");
        this.fns.push(emitNetField(nm, fd.gradient().field, { D, upload: (d) => this.upload(d) }, [GRADIENT_VALUE_OUTPUT, fd.gradient().field.output])!.code);
        return nm;
      }
      const f = this.scalar(fd), g = this.gradient(fd);
      const nm = this.name("vg");
      this.fns.push(`fn ${nm}(p: ${vecType(D)}, pos: i32) -> ${T} {\n  return ${T}(${f}(p, pos), ${g}(p, pos));\n}`);
      return nm;
    });
  }

  /** the transpiled net of a net-backed field, or undefined when it does not fit (the caller falls back) */
  private net(fd: NetScalarFieldData | NetVectorFieldData): string | undefined {
    if (!gpuTranspilable(fd)) return undefined;
    return this.memo(fd, "net", () => {
      const nm = this.name("nf");
      this.fns.push(emitNetField(nm, fd.field, { D: this.D, upload: (d) => this.upload(d) })!.code);
      return nm;
    });
  }

  /** everything emitted so far (prelude + field functions) and the packed data, for kernels that add their own entry point */
  library(): { code: string; data: Float32Array } {
    const data = new Float32Array(Math.max(GRID_FLOATS, this.dataLength));
    packGrid(this.grid, data, 0);
    let o = GRID_FLOATS;
    for (const c of this.chunks) { data.set(c, o); o += c.length; }
    return { code: [PRELUDE, "@group(0) @binding(1) var<storage, read> data: array<f32>;", OPAQUE_BOUND_WGSL, this.dgCode, ...this.fns].join("\n\n"), data };
  }

  /** the complete program sampling `field` on the dispatch grid */
  build(field: ScalarFieldData | VectorFieldData): GpuProgram {
    const D = this.D, grid = this.grid;
    const entry = field.rank === "scalar" ? this.scalar(field) : this.vector(field);
    const direct = field.samplePoints?.equals(grid, 1e-12) ?? false;
    const channels = field.rank === "scalar" ? 1 : D;
    // grid position -> point, row-major (last axis fastest); the grid comes from the header
    const G = this.dg;
    const idx: string[] = ["  var rem: i32 = i;"];
    const comps: string[] = [];
    for (let d = 0; d < D; d++) {
      idx.push(`  let s${d}: i32 = ${G.s(String(d))}; let g${d}: i32 = rem / s${d}; rem = rem - g${d} * s${d};`);
      comps.push(`${G.a(String(d))} + f32(g${d}) * ${G.h(String(d))}`);
    }
    const p = D === 1 ? comps[0]! : `${vecType(D)}(${comps.join(", ")})`;
    const write = channels === 1 ? `  out[i] = ${entry}(p, pos);` : `  let v = ${entry}(p, pos);\n${Array.from({ length: D }, (_, d) => `  out[i * ${D} + ${d}] = v[${d}];`).join("\n")}`;
    const main = `@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i: i32 = i32(id.x);
  if (i >= ${G.count}) { return; }
${idx.join("\n")}
  let p = ${p};
  let pos: i32 = ${direct ? "i" : "-1"};
${write}
}`;
    const lib = this.library();
    return { code: `${lib.code}\n\n${main}`, data: lib.data, channels, sampleCount: grid.sampleCount };
  }
}

export function buildSampleProgram(field: ScalarFieldData | VectorFieldData, grid: DenseGrid): GpuProgram {
  return new ProgramBuilder(grid).build(field);
}

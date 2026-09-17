// The 3D arm of the viewer: isosurfaces of the I_V field, coloured by I_C,
// rendered by GpuRenderer3D (WebGPU only) with the box, point sets and labels
// on the Canvas 2D overlay. Compute modes as in 2D: "gpu" samples a resident
// grid and runs the fused marching-tetrahedra kernel into resident meshes
// (nothing read back); "cpu" runs core's marchingTetrahedra and uploads.
// main.ts owns the controls and the shared state and hands them over through
// View3DContext.

import { Box, DenseGrid, DenseScalarFieldData, DenseVectorFieldData, arrowGlyphs, boxBlur, contourField, integrateFromSeeds, isoContours, latticePoints, marchingTetrahedra, projectToLevel, sliceScalarField, smoothIsoMesh, streamlineSeeds, type GlyphStyle, type Lattice, type PointSet, type ScalarFieldData, type StreamlineMode, type StreamlinePlan, type StreamlineSeeds, type VectorFieldData } from "@tensatory/core";
import {
  GpuRenderer3D,
  SEG_FLOATS,
  VERT_FLOATS,
  allocMesh,
  allocSegments,
  allocSegments3,
  blurResidentSync,
  boxEdges,
  fusedGlyphs,
  fusedIsolines,
  fusedIsosurface,
  fusedStreamlines3,
  packMesh,
  packPolylines,
  packPolylines3,
  packStreamlines3,
  project,
  regionAspect,
  resetMesh,
  resetSegments,
  resetSegments3,
  planeSampler,
  planeSlicer,
  sampleResidentSync,
  uploadGrid,
  uploadMesh,
  uploadSegments,
  uploadSegments3,
  type Camera3D,
  type FusedGlyphs,
  type FusedIsolines,
  type FusedIsosurface,
  type FusedStreamlines3,
  type GpuBackend,
  type GpuGrid,
  type GpuLineLayer3D,
  type GpuMesh,
  type GpuMeshLayer,
  type GpuSegments,
  type GpuSegments3,
  type Lut,
  type PlanePass,
  type ValueMap,
} from "@tensatory/gpu";
import { Cache, uidOf, type MemoryUser } from "./cache";

export interface Use3 { id: string; data: ScalarFieldData }
export interface VectorUse3 { id: string; data: VectorFieldData }
export interface StreamOpts3 {
  count: number;
  maxSteps: number;
  sign: 1 | -1;
  mode: StreamlineMode;
  bidirectional: boolean;
  alpha: number;
  /** tail in cells, null = solid lines */
  tail: number | null;
  split: number;
  /** the animation clock (seconds) */
  clock: number;
}
export type CropRange = [number | null, number | null];

export interface View3DContext {
  gpu: GpuBackend;
  /** the WebGPU canvas (#gpu) and the Canvas 2D overlay above it (#gl) */
  canvas: HTMLCanvasElement;
  overlay: HTMLCanvasElement;
  /** the free part of the canvas (css px `[x0, y0, x1, y1]`, right of the panels) the camera frames; the whole canvas when absent */
  region?(): [number, number, number, number];
  isoField(): Use3 | undefined;
  colourField(): Use3 | undefined;
  /** exact gradient data of a symbolic use (for normals), undefined for sampled data */
  gradientOf(u: Use3): VectorFieldData | undefined;
  /** isosurface levels in field units */
  levels(u: Use3): number[];
  alpha(): number;
  /** grid points along the longest box side (the adaptive resolution of the current tier) */
  resolution(): number;
  /** something asynchronous landed (a count readback that resized a set): render again */
  invalidate(): void;
  /** box-blur radius (cells) of the I_V field before contouring, or null */
  blur(): number | null;
  /** Taubin iterations on the welded mesh (non-exact surfaces), 0 = none */
  smoothing(): number;
  compute(): "cpu" | "gpu";
  showIso(): boolean;
  /** project vertices onto the true level set along the exact gradient (symbolic fields) */
  exact(): boolean;
  /** isolines of I_V on the (cropped) box faces */
  showOutline(): boolean;
  showPoints(): boolean;
  showBox(): boolean;
  /** committed crop range per axis as fractions of the box, null = the box's own end */
  crop(): CropRange[];
  /** the range being dragged / shift-previewed (not committed): drawn as a dotted box only */
  cropPreview(): CropRange[] | undefined;
  pointSets(): PointSet[];
  colour(u: Use3): { map: ValueMap; lut: Lut; key: string };
  /** the S∇ field (a vector field, or a scalar's gradient) when streamlines are on and `lines` is set */
  streamVector(): VectorUse3 | undefined;
  streamColour(): Use3 | undefined;
  streamOpts(): StreamOpts3;
  /** planned seeds (JL / coverage), cached by main.ts */
  plan(key: string, field: VectorFieldData, opts: { count: number; maxSteps: number; step: number; sign: 1 | -1; box: Box; mode: StreamlineMode; bidirectional: boolean }): StreamlinePlan;
  /** the V∇ field (arrow glyphs on an FCC lattice) when the vector field panel is on */
  glyphVector(): VectorUse3 | undefined;
  glyphColour(): Use3 | undefined;
  /** glyph spacing in css px at the camera's target depth */
  glyphSpacingPx(): number;
  glyphStyle(): GlyphStyle;
  /** the lattice of `v` inside `region` at world `spacing` (main.ts: anchored, capped), undefined when empty */
  glyphLattice(v: VectorUse3, region: Box, spacing: number): Lattice | undefined;
}

/**
 * A resident set (mesh or segments) appended by a fused kernel, with its record count read back after every
 * dispatch: the kernels count through their atomic even when the set is full, so `count > capacity` is a
 * detected overflow — the set is then reallocated for the true count and dispatched again. The counts also
 * feed `Complexity`, which sizes new sets from what the field actually produced instead of a worst case.
 */
interface Counted<S extends { capacity: number; indirect: GPUBuffer; destroy(): void }> { set: S; stamp: string; pending: boolean; count: number; overflow: boolean }
/** records (triangles / segments) a family of sets (field, colour, options — not the grid) needs, at the resolution it was measured */
interface Complexity { records: number; n: number; t: number }
interface Face { values: GpuGrid; sampler: PlanePass; kernel: FusedIsolines; depth: number; sets: Counted<GpuSegments>[] }
const destroyFace = (f: Face) => { f.values.destroy(); f.sampler.destroy(); f.kernel.destroy(); for (const s of f.sets) s.set.destroy(); };
const faceBytes = (f: Face) => f.values.buffer.size + f.sets.reduce((a, s) => a + s.set.buffer.size, 0);
const gridKey = (u: { id: string }, g: DenseGrid) => `${u.id}|${g.size.join("x")}|${g.box.intervals.flat().join(",")}`;
const bufBytes = (o: { buffer: GPUBuffer }) => o.buffer.size;
/** smallest set worth allocating (records) and the margin over the predicted count */
const MIN_RECORDS = 16384, MARGIN = 1.5;

export interface Frame3DInfo { n: number; grid: number[]; triangles: number; capacity: number; overflow: boolean }

export class View3D implements MemoryUser {
  readonly renderer: GpuRenderer3D;
  camera: Camera3D = { target: [0, 0, 0], distance: 8, yaw: -0.9, pitch: 0.55, fov: 0.7 };
  /** true after the user orbited / zoomed; a fitted camera is re-fitted when the box changes */
  cameraCustom = false;
  /** what the last frame drew: for the resolution row and the adaptive-resolution loop */
  info: Frame3DInfo = { n: 0, grid: [], triangles: 0, capacity: 0, overflow: false };
  /** the normalizing norm of the glyphs drawn (the longest vector sampled); NaN when none */
  glyphMaxNorm = NaN;
  private box = Box.unit(3);
  private boxKey = "";
  // ∝ n³
  private readonly grids = new Cache<GpuGrid>(4, (g) => g.destroy(), bufBytes);
  private readonly cpuValues = new Cache<Float64Array>(16, () => {}, (v) => v.byteLength);
  private readonly vgrids = new Cache<GpuGrid>(4, (g) => g.destroy(), bufBytes);
  private readonly vsampled = new Cache<DenseVectorFieldData>(4, () => {}, (d) => d.samplePoints.sampleCount * d.dimCount * 8);
  // ∝ n²
  private readonly kernels = new Cache<FusedIsosurface>(8, (k) => k.destroy());
  private readonly meshes = new Cache<Counted<GpuMesh>>(24, (m) => m.set.destroy(), (m) => m.set.buffer.size);
  private readonly cpuMeshes = new Cache<GpuMesh>(24, (m) => m.destroy(), bufBytes);
  private readonly complexity = new Map<string, Complexity>();
  // face isolines (GPU): per face its values buffer, plane pass, 2D kernel and segment sets (see faceLines)
  private readonly faces = new Cache<Face>(12, destroyFace, faceBytes);
  private readonly faceCpu = new Cache<GpuSegments>(96, (s) => s.destroy(), bufBytes);
  private readonly lines3 = new Cache<GpuSegments3>(16, (s) => s.destroy(), bufBytes);
  // streamlines: fused kernels and their segment sets
  private readonly streamKernels = new Cache<FusedStreamlines3>(8, (k) => k.destroy());
  private readonly streamSets = new Cache<GpuSegments3>(8, (s) => s.destroy(), bufBytes);
  // glyphs: fused kernel per (field, colour) with one set re-dispatched per lattice; CPU sets per lattice
  private readonly glyphKernels = new Cache<FusedGlyphs>(4, (k) => k.destroy());
  private readonly glyphSets = new Cache<{ segs: GpuSegments3; stamp: string; pending: boolean; read: string }>(4, (e) => e.segs.destroy(), (e) => e.segs.buffer.size);
  private readonly glyphCpu = new Cache<{ segs: GpuSegments3; max: number }>(8, (e) => e.segs.destroy(), (e) => e.segs.buffer.size);
  private readonly ctx2d: CanvasRenderingContext2D;

  constructor(private readonly c: View3DContext) {
    this.renderer = new GpuRenderer3D(c.gpu, c.canvas);
    this.ctx2d = c.overlay.getContext("2d")!;
  }

  private get volumeCaches(): Cache<unknown>[] { return [this.grids, this.cpuValues, this.vgrids, this.vsampled] as Cache<unknown>[]; }
  private get surfaceCaches(): Cache<unknown>[] { return [this.meshes, this.cpuMeshes, this.faces, this.faceCpu, this.lines3, this.streamSets, this.kernels, this.streamKernels, this.glyphSets, this.glyphCpu, this.glyphKernels] as Cache<unknown>[]; }

  clear(): void {
    for (const c of [...this.volumeCaches, ...this.surfaceCaches]) c.clear();
    this.complexity.clear();
    this.boxKey = "";
  }

  /** forget that the meshes and face lines are up to date: the next frame computes them again (timing without compiles) */
  redo(): void {
    for (const m of this.meshes.values()) m.stamp = "";
    for (const f of this.faces.values()) for (const s of f.sets) s.stamp = "";
    if (this.c.compute() === "cpu") this.cpuMeshes.clear();
  }
  /** bytes per cache (debugging) */
  debug(): Record<string, string> {
    const e = { grids: this.grids, cpuValues: this.cpuValues, vgrids: this.vgrids, vsampled: this.vsampled, meshes: this.meshes, cpuMeshes: this.cpuMeshes, faces: this.faces, faceCpu: this.faceCpu, lines3: this.lines3, streamSets: this.streamSets };
    return Object.fromEntries(Object.entries(e).map(([k, c]) => [k, `${c.size} entries, ${(c.bytes / 2 ** 20).toFixed(1)} MB (${(c.liveBytes / 2 ** 20).toFixed(1)} live)`]));
  }
  memory(): { volume: number; surface: number; cpu: number } {
    return { volume: this.volumeCaches.reduce((a, c) => a + c.liveBytes, 0), surface: this.surfaceCaches.reduce((a, c) => a + c.liveBytes, 0), cpu: this.cpuValues.bytes + this.vsampled.bytes };
  }
  trim(bytes: number): number {
    let freed = 0;
    // biggest, least essential first: CPU meshes and face lines are cheap to rebuild, value grids are not
    for (const c of [this.cpuMeshes, this.faceCpu, this.glyphCpu, this.faces, this.meshes, this.streamSets, this.glyphSets, this.lines3, this.vsampled, this.cpuValues, this.vgrids, this.grids]) {
      if (freed >= bytes) break;
      freed += c.trim(bytes - freed);
    }
    return freed;
  }

  /**
   * Read the record count of `set` back once its dispatch has run (one readback in flight per set). On landing:
   * the family's complexity is updated (running max with a slow decay, so a cycle of animated levels keeps its
   * peak), and a set that overflowed is flagged so the next frame reallocates and re-dispatches it.
   */
  private track<S extends { capacity: number; indirect: GPUBuffer; destroy(): void }>(cs: Counted<S>, family: string, n: number, index: number): void {
    if (cs.pending) return;
    cs.pending = true;
    const stamp = cs.stamp;
    void this.c.gpu.readCounter(cs.set.indirect, index).then((raw) => {
      cs.pending = false;
      const count = index === 0 ? raw / 3 : raw; // meshes count vertices
      const prev = this.complexity.get(family), now = performance.now();
      // running max with a slow decay (half-life 60 s): a cycle of animated levels keeps its peak
      const scaled = prev ? prev.records * (n / prev.n) ** 2 * 0.5 ** ((now - prev.t) / 60_000) : 0; // surfaces: records ∝ n²
      this.complexity.set(family, { records: Math.max(count, scaled), n, t: now });
      if (cs.stamp !== stamp) return; // moved on: the count still informed the family
      const changed = cs.count !== count;
      cs.count = count;
      if (count > cs.set.capacity) { cs.overflow = true; cs.stamp = ""; }
      if (changed) this.c.invalidate(); // the resolution row shows the count; an overflow reallocates
    }).catch(() => { cs.pending = false; });
  }
  /** capacity for a new set of `family` at resolution `n`: the measured complexity with a margin, else `fallback` */
  private capacityFor(family: string, n: number, fallback: number, exponent: number, max: number, recordBytes: number): number {
    const cx = this.complexity.get(family);
    const want = cx ? cx.records * (n / cx.n) ** exponent * MARGIN : fallback;
    return Math.max(1, Math.min(max, Math.floor(this.c.gpu.maxBufferBytes / recordBytes), Math.max(MIN_RECORDS, Math.ceil(want))));
  }

  /** the framed part of the canvas as fractions of its size (y down), undefined for all of it */
  private region(): [number, number, number, number] | undefined {
    const r = this.c.region?.(), cv = this.c.overlay;
    if (!r || cv.clientWidth <= 0 || cv.clientHeight <= 0) return undefined;
    const f: [number, number, number, number] = [r[0] / cv.clientWidth, r[1] / cv.clientHeight, r[2] / cv.clientWidth, r[3] / cv.clientHeight];
    return f[2] - f[0] > 0.05 && f[3] - f[1] > 0.05 ? f : undefined; // a degenerate region (tiny window) frames the whole canvas
  }
  /** css px height of the framed region (the vertical fov spans it) */
  private regionHeight(): number {
    const r = this.c.region?.();
    return Math.max(1, r ? r[3] - r[1] : this.c.overlay.clientHeight);
  }
  /** the smaller of the vertical fov and the horizontal one the framed region gives it: what a bounding sphere must fit */
  private fitFov(): number {
    const cv = this.c.overlay, region = this.region();
    const aspect = (cv.clientWidth / Math.max(1, cv.clientHeight) || 1) * regionAspect(region);
    return Math.min(this.camera.fov, 2 * Math.atan(Math.tan(this.camera.fov / 2) * aspect));
  }

  /** frame `box`: look at its centre from a distance that fits it in the framed region */
  fit(box = this.box): void {
    const r = Math.hypot(...box.size) / 2 || 1;
    this.camera = { ...this.camera, target: box.center as [number, number, number], distance: r / Math.sin(this.fitFov() / 2) * 1.05 };
    this.cameraCustom = false;
  }

  /** orbit by a pointer drag (css px) */
  orbit(dx: number, dy: number): void {
    this.camera.yaw -= dx * 0.008;
    this.camera.pitch = Math.max(-1.5, Math.min(1.5, this.camera.pitch + dy * 0.008));
    this.cameraCustom = true;
  }
  /** pan the target in the view plane */
  pan(dx: number, dy: number): void {
    const s = (2 * this.camera.distance * Math.tan(this.camera.fov / 2)) / this.regionHeight();
    const cy = Math.cos(this.camera.yaw), sy = Math.sin(this.camera.yaw), cp = Math.cos(this.camera.pitch), sp = Math.sin(this.camera.pitch);
    const right = [-sy, cy, 0], up = [-sp * cy, -sp * sy, cp];
    const t = this.camera.target;
    this.camera.target = [t[0] - (dx * right[0]! - dy * up[0]!) * s, t[1] - (dx * right[1]! - dy * up[1]!) * s, t[2] - (dx * right[2]! - dy * up[2]!) * s];
    this.cameraCustom = true;
  }
  zoom(factor: number): void {
    this.camera.distance = Math.max(1e-3, this.camera.distance / factor);
    this.cameraCustom = true;
  }

  private grid(box: Box, n = this.c.resolution()): DenseGrid {
    const mx = Math.max(...box.size) || 1;
    return new DenseGrid(box.size.map((s) => Math.max(2, Math.round((n * s) / mx) || 2)), box);
  }
  /** the grid streamlines are measured in (step = ½ cell, `length` counts steps, `tail` cells) and symbolic vectors
   *  are sampled on: FIXED, not the adaptive isosurface resolution, so line lengths do not change with the tier */
  static readonly STREAM_N = 64;

  /** the resident I_V grid, blurred when `metric` is set */
  private volumeGpu(iv: Use3, grid: DenseGrid): { values: GpuGrid; key: string } {
    const gk = gridKey(iv, grid);
    const values = this.grids.getOr(gk, () => sampleResidentSync(this.c.gpu, iv.data, grid));
    const r = this.c.blur();
    if (r === null || r <= 0) return { values, key: gk };
    const bk = `${gk}|blur${r}`;
    return { values: this.grids.getOr(bk, () => blurResidentSync(this.c.gpu, values, r)), key: bk };
  }
  private volumeCpu(iv: Use3, grid: DenseGrid): { values: Float64Array; key: string } {
    const gk = gridKey(iv, grid);
    const vals = this.cpuValues.getOr(gk, () => Float64Array.from(iv.data.sampleOn(grid)));
    const r = this.c.blur();
    if (r === null || r <= 0) return { values: vals, key: gk };
    const bk = `${gk}|blur${r}`;
    return { values: this.cpuValues.getOr(bk, () => boxBlur(grid, vals, r)), key: bk };
  }
  /** exact projection applies to symbolic fields contoured as they are (a blurred field is only known on the grid) */
  private isExact(iv: Use3): boolean { return this.c.exact() && iv.data.kind === "symbolic" && !(this.c.blur()! > 0); }

  private meshesGpu(iv: Use3, ic: Use3 | undefined, grid: DenseGrid, levels: number[]): GpuMesh[] {
    const { values, key: gk } = this.volumeGpu(iv, grid);
    const exact = this.isExact(iv);
    const family = `${iv.id}|${this.c.blur() ?? ""}|${ic?.id ?? ""}|${exact ? "exact" : "lin"}`;
    const kk = `${gk}#${uidOf(values)}|${ic?.id ?? ""}|${exact ? "exact" : "lin"}`; // the kernel reads THIS grid's buffer
    const kernel = this.kernels.getOr(kk, () => fusedIsosurface(this.c.gpu, values, { field: exact ? iv.data : undefined, exact, colour: ic?.data }));
    const n = Math.max(...grid.size);
    // triangle budget from the measured complexity of this field (surfaces touch O(n²) of the n³ cells); before
    // any measurement a modest guess — an overflow is detected by the count readback and the set regrown
    const cells = grid.size.reduce((a, s) => a * (s - 1), 1);
    const guess = Math.max(MIN_RECORDS, Math.min(1 << 18, Math.round(cells * 0.25)));
    let triangles = 0, capacity = 0, overflow = false;
    const out = levels.map((level, k) => {
      const mk = `${kk}|${k}`;
      let cs = this.meshes.get(mk);
      if (cs?.overflow) {
        // the count landed above the capacity: the true count sizes the new set
        this.complexity.set(family, { records: Math.max(cs.count, this.complexity.get(family)?.records ?? 0), n, t: performance.now() });
        this.meshes.delete(mk); cs = undefined; overflow = true;
      }
      const want = this.capacityFor(family, n, guess, 2, kernel.capacity, 3 * VERT_FLOATS * 4);
      if (cs && (cs.set.capacity < want / MARGIN || cs.set.capacity > want * 4)) { this.meshes.delete(mk); cs = undefined; } // too small for what we now expect, or wastefully large
      if (!cs) cs = this.meshes.set(mk, { set: allocMesh(this.c.gpu, want), stamp: "", pending: false, count: 0, overflow: false });
      const stamp = `${kk}|${level}`;
      if (cs.stamp !== stamp) { resetMesh(this.c.gpu, cs.set); kernel.dispatch(cs.set, level); cs.stamp = stamp; this.track(cs, family, n, 0); } // count: the previous level's until the readback lands
      triangles += cs.count; capacity += cs.set.capacity;
      return cs.set;
    });
    this.info = { ...this.info, triangles, capacity, overflow };
    return out;
  }

  private meshesCpu(iv: Use3, ic: Use3 | undefined, grid: DenseGrid, levels: number[]): GpuMesh[] {
    const { values: vals, key: gk } = this.volumeCpu(iv, grid);
    const exact = this.isExact(iv);
    const grad = exact ? this.c.gradientOf(iv) : undefined; // blurred: normals from the blurred grid
    const maxDist = Math.hypot(...grid.spacing);
    const smooth = exact ? 0 : this.c.smoothing();
    let triangles = 0;
    const out = levels.map((level) => {
      const mk = `${gk}|${ic?.id ?? ""}|${exact ? "exact" : `lin|sm${smooth}`}|${level}`;
      const mesh = this.cpuMeshes.getOr(mk, () => {
        let m = marchingTetrahedra(grid, vals, level, {
          gradient: grad ? (p) => grad.value(p) ?? undefined : undefined,
          project: exact ? (p) => projectToLevel(iv.data, p, level, maxDist) : undefined,
          colourAt: ic ? (p) => ic.data.value(p) ?? NaN : undefined,
        });
        if (smooth > 0) m = smoothIsoMesh(m, smooth);
        return uploadMesh(this.c.gpu, packMesh(m));
      });
      triangles += mesh.capacity;
      return mesh;
    });
    this.info = { ...this.info, triangles, capacity: triangles, overflow: false };
    return out;
  }

  /** the box cut to the crop ranges (fractions of the box; an open end is the box's own end) */
  private cropped(box: Box, crop: CropRange[]): Box {
    const f = (v: number | null, dflt: number) => Math.max(0, Math.min(1, v ?? dflt));
    return new Box(box.a.map((a, d) => a + f(crop[d]?.[0] ?? null, 0) * box.size[d]!), box.a.map((a, d) => a + Math.max(f(crop[d]?.[0] ?? null, 0) + 0.01, f(crop[d]?.[1] ?? null, 1)) * box.size[d]!));
  }

  /**
   * Isolines of the I_V field on the six faces of the cropped box (the prototype's "outline"): where each
   * isosurface meets the faces. Every face has ONE grid — the cropped box's face at the volume grid's spacing, so
   * the lines coincide with the mesh boundary — one values buffer, one 2D marching-squares kernel and one segment
   * set per level, all built once per (field, grid); the depth is a dispatch parameter. Since the volume grid
   * follows the cropped box (the resolution is spent inside the crop), a crop change means new grids — crop
   * drags run in the `moving` tier like a 2D pan. Values: the field sampled exactly on the plane (exact case),
   * else the (blurred / sampled) volume grid sliced trilinearly.
   */
  private faceLines(iv: Use3, grid: DenseGrid, cbox: Box, levels: number[], width: number): GpuLineLayer3D[] {
    const out: GpuLineLayer3D[] = [];
    const gpu = this.c.compute() === "gpu";
    const exact = this.isExact(iv);
    const tol = 0.25 * Math.min(...grid.spacing); // world tolerance of the exact chords
    const source = exact ? "exact" : `lin|blur${this.c.blur() ?? 0}`;
    for (let axis = 0; axis < 3; axis++) for (const hi of [false, true]) {
      const depth = hi ? cbox.b[axis]! : cbox.a[axis]!;
      const oa = [0, 1, 2].filter((d) => d !== axis) as [number, number];
      const grid2 = new DenseGrid(oa.map((d) => grid.size[d]!), new Box(oa.map((d) => cbox.a[d]!), oa.map((d) => cbox.b[d]!)));
      const fkey = `${iv.id}|face${axis}|${grid.size.join("x")}|${cbox.intervals.flat().join(",")}|${source}`;
      const side = `${fkey}|${hi ? "+" : "-"}${exact ? "" : `#${uidOf(this.volumeGpu(iv, grid).values)}`}`; // a slicer reads the volume grid's buffer
      const embed = { axis, depth };
      if (gpu) {
        const face = this.faces.getOr(side, () => {
          const values = uploadGrid(this.c.gpu, grid2, new Float32Array(grid2.sampleCount), 1);
          const sampler = exact ? planeSampler(this.c.gpu, iv.data, axis, grid2) : planeSlicer(this.c.gpu, this.volumeGpu(iv, grid).values, axis, grid2);
          const kernel = fusedIsolines(this.c.gpu, undefined, values, undefined, exact ? { slice: { field: iv.data, axis } } : { exact: false });
          return { values, sampler, kernel, depth: NaN, sets: [] };
        });
        if (face.depth !== depth) { face.sampler.dispatch(depth, face.values.buffer); face.depth = depth; for (const st of face.sets) st.stamp = ""; }
        // segments of a face isoline ∝ n: sized from the measured complexity of this face family
        const n = Math.max(...grid2.size), family = `${fkey}|${exact ? "x" : "l"}`;
        const guess = Math.max(MIN_RECORDS, (grid2.size[0]! - 1) * (grid2.size[1]! - 1) * (exact ? 2 : 1));
        levels.forEach((level, k) => {
          let cs = face.sets[k];
          if (cs?.overflow) { this.complexity.set(family, { records: Math.max(cs.count, this.complexity.get(family)?.records ?? 0), n, t: performance.now() }); cs.set.destroy(); cs = undefined; }
          const want = this.capacityFor(family, n, guess, 1, face.kernel.capacity, SEG_FLOATS * 4);
          if (cs && (cs.set.capacity < want / MARGIN || cs.set.capacity > want * 4)) { cs.set.destroy(); cs = undefined; }
          if (!cs) { face.sets[k] = cs = { set: allocSegments(this.c.gpu, want, false), stamp: "", pending: false, count: 0, overflow: false }; this.faces.refresh(side); } // the face's bytes changed
          const stamp = `${level}`;
          if (cs.stamp !== stamp) { resetSegments(this.c.gpu, cs.set); face.kernel.dispatch(cs.set, level, tol, depth); cs.stamp = stamp; this.track(cs, family, n, 1); }
          out.push({ segs: cs.set, embed, width, color: [0.92, 0.92, 0.92] });
        });
      } else {
        // the same points as a degenerate 3D grid (size 1 along the axis): what the field is sampled on
        const size3 = [0, 0, 0], a3 = [0, 0, 0], b3 = [0, 0, 0];
        size3[axis] = 1; a3[axis] = depth; b3[axis] = depth;
        oa.forEach((d, i) => { size3[d] = grid2.size[i]!; a3[d] = grid2.box.a[i]!; b3[d] = grid2.box.b[i]!; });
        const grid3 = new DenseGrid(size3, new Box(a3, b3));
        const vkey = `${side}|${depth}`;
        const vals = this.cpuValues.getOr(vkey, () => (exact ? Float64Array.from(iv.data.sampleOn(grid3)) : Float64Array.from(new DenseScalarFieldData(grid, this.volumeCpu(iv, grid).values).sampleOn(grid3))));
        for (const level of levels) {
          const segs = this.faceCpu.getOr(`${vkey}|${level}`, () => {
            const lines = exact ? contourField(sliceScalarField(iv.data, axis, depth), grid2, vals, level, { tolerance: tol }).lines : isoContours(grid2, vals, level);
            return uploadSegments(this.c.gpu, packPolylines(lines), false);
          });
          out.push({ segs, embed, width, color: [0.92, 0.92, 0.92] });
        }
      }
    }
    return out;
  }

  /** the vector field sampled on the volume grid (what streamlines are integrated through, like the 2D arm) */
  private sampledVector(v: VectorUse3, grid: DenseGrid): DenseVectorFieldData {
    if (v.data.kind === "sampled" && v.data instanceof DenseVectorFieldData) return v.data;
    return this.vsampled.getOr(gridKey(v, grid), () => new DenseVectorFieldData(grid, v.data.sampleOn(grid)));
  }

  /** streamlines of the S∇ field: one segment set per (field, grid, options), particles by the renderer */
  private streamLayer(v: VectorUse3, box: Box, grid: DenseGrid): GpuLineLayer3D | undefined {
    const c = this.c, o = c.streamOpts();
    const vbox = v.data.box.intersect(box) ?? v.data.box;
    const size = [0, 1, 2].map((d) => Math.max(2, Math.round(vbox.size[d]! / (grid.spacing[d]! || 1)) + 1));
    const vgrid = new DenseGrid(size, vbox);
    const cell = Math.min(...vgrid.spacing) || 1e-3, step = 0.5 * cell;
    const sc = c.streamColour();
    const iopts = { maxSteps: o.maxSteps, step, sign: o.sign, box: vbox, bidirectional: o.bidirectional };
    const key = [v.id, gridKey(v, vgrid), o.mode, o.bidirectional, o.count, o.maxSteps, o.sign, step.toExponential(4), sc?.id ?? ""].join("|");
    let segs = this.streamSets.get(key);
    if (!segs) {
      const gpu = c.compute() === "gpu";
      let seeds: StreamlineSeeds | undefined, plan: StreamlinePlan | undefined;
      if (o.mode === "stratified") seeds = streamlineSeeds(vbox, o.count, 12345);
      else { plan = c.plan(key, this.sampledVector(v, vgrid), { count: o.count, mode: o.mode, ...iopts }); seeds = plan.seeds; }
      if (gpu) {
        const vectors = this.vgrids.getOr(`vec:${gridKey(v, vgrid)}`, () => sampleResidentSync(c.gpu, v.data, vgrid));
        const kernel = this.streamKernels.getOr(key, () => fusedStreamlines3(c.gpu, vectors, seeds!, iopts, sc?.data));
        segs = allocSegments3(c.gpu, kernel.capacity, true);
        kernel.dispatch(segs);
      } else {
        const lines = plan?.lines ?? integrateFromSeeds(this.sampledVector(v, vgrid), seeds, iopts);
        const colours = sc ? lines.map((l) => { const out = new Float64Array(l.points.length / 3); for (let i = 0; i < out.length; i++) out[i] = sc.data.value([l.points[3 * i]!, l.points[3 * i + 1]!, l.points[3 * i + 2]!]) ?? NaN; return out; }) : undefined;
        segs = uploadSegments3(c.gpu, packStreamlines3(lines, step, colours), true);
      }
      this.streamSets.set(key, segs);
    }
    const colour = sc ? c.colour(sc) : undefined;
    return {
      segs, width: 1.5, color: [1, 1, 1],
      particles: o.tail === null ? undefined : { tail: o.tail * cell, split: o.split, travel: o.clock * 10 * cell },
      ...(colour ? { map: colour.map, lut: colour.lut } : {}),
    };
  }

  /**
   * Arrow glyphs of the V∇ field on an FCC lattice inside the cropped box, spaced `glyphSpacingPx` px at the camera's
   * target depth (so zooming in refines the lattice). GPU compute: the fused kernel (glyphs.ts) re-dispatched per
   * lattice into a resident Seg3 set; CPU compute: core samples the cosets and builds the arrows, uploaded per lattice.
   */
  private glyphLayer(v: VectorUse3, cbox: Box): GpuLineLayer3D | undefined {
    const c = this.c;
    const worldPerPx = (2 * this.camera.distance * Math.tan(this.camera.fov / 2)) / this.regionHeight();
    const lat = c.glyphLattice(v, cbox, c.glyphSpacingPx() * worldPerPx);
    if (!lat) return undefined;
    const vc = c.glyphColour(), style = c.glyphStyle();
    const latKey = `${lat.spacing.toExponential(6)}|${lat.cosets.map((g) => `${g.size.join("x")}@${g.box.a.map((x) => x.toPrecision(9)).join(",")}`).join(";")}|${style}`;
    const kk = `${v.id}|${vc?.id ?? ""}`;
    let segs: GpuSegments3;
    if (c.compute() === "gpu") {
      const kernel = this.glyphKernels.getOr(kk, () => fusedGlyphs(c.gpu, v.data, vc?.data));
      const need = kernel.capacityFor(lat);
      let e = this.glyphSets.get(kk);
      if (e && (e.segs.capacity < need || e.segs.capacity > need * 4)) { this.glyphSets.delete(kk); e = undefined; }
      if (!e) e = this.glyphSets.set(kk, { segs: allocSegments3(c.gpu, Math.ceil(need * 1.5), false), stamp: "", pending: false, read: "" });
      if (e.stamp !== latKey) { resetSegments3(c.gpu, e.segs); kernel.dispatch(e.segs, lat, style); e.stamp = latKey; this.readGlyphMax(e, kernel); }
      segs = e.segs;
    } else {
      const entry = this.glyphCpu.getOr(`${kk}|${latKey}`, () => {
        const pts = latticePoints(lat);
        const vectors = new Float64Array(lat.pointCount * 3);
        let o = 0;
        for (const g of lat.cosets) { const s = v.data.sampleOn(g); vectors.set(s, o); o += s.length; }
        const g = arrowGlyphs(pts, vectors, 3, lat.spacing, { style });
        const colours = vc ? g.lines.map((l, k) => { const p = g.point[k]!; return new Float64Array(l.length / 3).fill(vc.data.value([pts[3 * p]!, pts[3 * p + 1]!, pts[3 * p + 2]!]) ?? NaN); }) : undefined;
        return { segs: uploadSegments3(c.gpu, packPolylines3(g.lines, colours), false), max: g.maxNorm };
      });
      segs = entry.segs; this.glyphMaxNorm = entry.max;
    }
    const colour = vc ? c.colour(vc) : undefined;
    return { segs, width: 1.5, color: [1, 1, 1], ...(colour ? { map: colour.map, lut: colour.lut } : {}) };
  }
  /** one readback of the normalizing norm in flight per set; a newer dispatch is read after it */
  private readGlyphMax(e: { stamp: string; pending: boolean; read: string }, kernel: FusedGlyphs): void {
    if (e.pending) return;
    e.pending = true;
    const stamp = e.stamp;
    void kernel.readMaxNorm().then((m) => { e.read = stamp; if (m !== this.glyphMaxNorm) { this.glyphMaxNorm = m; this.c.invalidate(); } }).catch(() => {}).finally(() => { e.pending = false; if (e.stamp !== e.read) this.readGlyphMax(e, kernel); });
  }

  /** a cached uploaded 3D segment set */
  private segs3(key: string, make: () => Float32Array): GpuSegments3 {
    return this.lines3.getOr(key, () => uploadSegments3(this.c.gpu, make(), false));
  }

  render(): void {
    const c = this.c;
    const iv = c.showIso() ? c.isoField() : undefined;
    const sv = c.streamVector();
    const gv = c.glyphVector();
    const box = iv?.data.box ?? sv?.data.box ?? gv?.data.box ?? this.box;
    const key = box.intervals.flat().join(",");
    if (key !== this.boxKey) { this.boxKey = key; this.box = box; if (!this.cameraCustom) this.fit(box); }
    const cbox = this.cropped(box, c.crop());
    const preview = c.cropPreview();
    const pbox = preview ? this.cropped(box, preview) : undefined;
    const meshes: GpuMeshLayer[] = [];
    const lines: GpuLineLayer3D[] = [];
    if (c.showBox()) lines.push({ segs: this.segs3(`box|${cbox.intervals.flat().join(",")}`, () => boxEdges(cbox.a, cbox.b)), width: 1.2, color: [0.5, 0.56, 0.72], uncropped: true });
    if (c.showPoints()) {
      for (const ps of c.pointSets()) {
        if (ps.ordered && ps.points.length > 1) lines.push({ segs: this.segs3(`ps|${ps.id}`, () => packPolylines3([ps.points.flat()])), width: 1.5, color: [1, 1, 1] });
      }
    }
    this.info = { n: c.resolution(), grid: this.grid(cbox).size, triangles: 0, capacity: 0, overflow: false };
    if (iv) {
      const ic = c.colourField();
      const grid = this.grid(cbox); // the resolution is spent inside the crop: a small crop is a close-up
      const levels = c.levels(iv);
      // surface smoothing needs welded connectivity: CPU meshing (the fused kernel has none yet)
      const gpuMesh = c.compute() === "gpu" && !(c.smoothing() > 0 && !this.isExact(iv));
      const sets = gpuMesh ? this.meshesGpu(iv, ic, grid, levels) : this.meshesCpu(iv, ic, grid, levels);
      const alpha = c.alpha();
      const colour = ic ? c.colour(ic) : undefined;
      for (const mesh of sets) meshes.push({ mesh, alpha, color: [0.86, 0.87, 0.9], ...(colour ? { map: colour.map, lut: colour.lut } : {}) });
      if (c.showOutline()) lines.push(...this.faceLines(iv, grid, cbox, levels, 2));
    }
    if (sv) { const layer = this.streamLayer(sv, box, this.grid(box, View3D.STREAM_N)); if (layer) lines.push(layer); }
    if (gv) { const layer = this.glyphLayer(gv, cbox); if (layer) lines.push(layer); } else this.glyphMaxNorm = NaN;
    this.renderer.resize();
    this.renderer.render({ camera: this.camera, radius: Math.hypot(...box.size) / 2 || 1, region: this.region(), background: [0x0b / 255, 0x0d / 255, 0x12 / 255], meshes, lines, cropMin: cbox.a as [number, number, number], cropMax: cbox.b as [number, number, number] });
    this.overlay(cbox, pbox && !pbox.equals(cbox, 1e-12) ? pbox : undefined);
  }

  /** world → css px on the overlay */
  project(p: ArrayLike<number>): [number, number, number] | undefined {
    return project(this.renderer.viewProj, p, this.c.overlay.clientWidth, this.c.overlay.clientHeight);
  }

  private overlay(box: Box, preview: Box | undefined): void {
    const cv = this.c.overlay, ctx = this.ctx2d;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.round(cv.clientWidth * dpr)), h = Math.max(1, Math.round(cv.clientHeight * dpr));
    if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cv.clientWidth, cv.clientHeight);
    if (preview) {
      // the box a crop drag / shift-preview would commit: a dotted outline, nothing recomputed
      const [a, b] = [preview.a, preview.b];
      const corner = (m: number) => [m & 1 ? b[0]! : a[0]!, m & 2 ? b[1]! : a[1]!, m & 4 ? b[2]! : a[2]!];
      ctx.strokeStyle = "#93c5fd"; ctx.lineWidth = 1.2; ctx.setLineDash([3, 4]);
      ctx.beginPath();
      for (let m = 0; m < 8; m++) for (const bit of [1, 2, 4]) {
        if (m & bit) continue;
        const p = this.project(corner(m)), q = this.project(corner(m | bit));
        if (p && q) { ctx.moveTo(p[0], p[1]); ctx.lineTo(q[0], q[1]); }
      }
      ctx.stroke(); ctx.setLineDash([]);
    }
    if (this.c.showPoints()) {
      for (const ps of this.c.pointSets()) {
        const inside = (p: ArrayLike<number>) => box.contains(p, 1e-9);
        const pts = ps.points.map((p) => (inside(p) ? this.project(p) : undefined));
        const single = pts.length === 1;
        pts.forEach((p, i) => {
          if (!p) return;
          const head = single || (ps.ordered && i === pts.length - 1);
          ctx.beginPath(); ctx.arc(p[0], p[1], head ? 6 : 2.5, 0, 2 * Math.PI);
          ctx.fillStyle = head ? "#ff4d4d" : "#ffffff"; ctx.fill();
          const label = ps.spec.labels?.[i];
          if (label) { ctx.fillStyle = "#fff"; ctx.font = "11px system-ui, sans-serif"; ctx.fillText(label, p[0] + 8, p[1] - 6); }
        });
      }
    }
  }
}

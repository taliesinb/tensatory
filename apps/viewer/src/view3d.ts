// The 3D arm of the viewer: isosurfaces of the I_V field, coloured by I_C,
// rendered by GpuRenderer3D (WebGPU only) with the box, point sets and labels
// on the Canvas 2D overlay. Compute modes as in 2D: "gpu" samples a resident
// grid and runs the fused marching-tetrahedra kernel into resident meshes
// (nothing read back); "cpu" runs core's marchingTetrahedra and uploads.
// main.ts owns the controls and the shared state and hands them over through
// View3DContext.

import { Box, DenseGrid, DenseScalarFieldData, DenseVectorFieldData, boxBlur, contourField, integrateFromSeeds, isoContours, marchingTetrahedra, projectToLevel, sliceScalarField, smoothIsoMesh, streamlineSeeds, type PointSet, type ScalarFieldData, type StreamlineMode, type StreamlinePlan, type StreamlineSeeds, type VectorFieldData } from "@tensatory/core";
import {
  GpuRenderer3D,
  allocMesh,
  allocSegments,
  allocSegments3,
  blurResidentSync,
  boxEdges,
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
  planeSampler,
  planeSlicer,
  sampleResidentSync,
  uploadGrid,
  uploadMesh,
  uploadSegments,
  uploadSegments3,
  type Camera3D,
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
  /** grid points along the longest box side */
  resolution(): number;
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
}

const lru = <V>(m: Map<string, V>, max: number, drop: (v: V) => void) => { while (m.size > max) { const k = m.keys().next().value as string; drop(m.get(k)!); m.delete(k); } };
interface Face { values: GpuGrid; sampler: PlanePass; kernel: FusedIsolines; depth: number; sets: { segs: GpuSegments; stamp: string }[] }
const destroyFace = (f: Face) => { f.values.destroy(); f.sampler.destroy(); f.kernel.destroy(); for (const s of f.sets) s.segs.destroy(); };
const gridKey = (u: { id: string }, g: DenseGrid) => `${u.id}|${g.size.join("x")}|${g.box.intervals.flat().join(",")}`;

export class View3D {
  readonly renderer: GpuRenderer3D;
  camera: Camera3D = { target: [0, 0, 0], distance: 8, yaw: -0.9, pitch: 0.55, fov: 0.7 };
  /** true after the user orbited / zoomed; a fitted camera is re-fitted when the box changes */
  cameraCustom = false;
  private box = Box.unit(3);
  private boxKey = "";
  private readonly grids = new Map<string, GpuGrid>();
  private readonly kernels = new Map<string, FusedIsosurface>();
  private readonly meshes = new Map<string, { mesh: GpuMesh; stamp: string }>();
  private readonly cpuMeshes = new Map<string, GpuMesh>();
  private readonly cpuValues = new Map<string, Float64Array>();
  // face isolines (GPU): per face its values buffer, plane pass, 2D kernel and segment sets (see faceLines)
  private readonly faces = new Map<string, Face>();
  private readonly faceCpu = new Map<string, GpuSegments>();
  private readonly lines3 = new Map<string, GpuSegments3>();
  // streamlines: resident vector grids, sampled copies for CPU planning, fused kernels and their segment sets
  private readonly vgrids = new Map<string, GpuGrid>();
  private readonly vsampled = new Map<string, DenseVectorFieldData>();
  private readonly streamKernels = new Map<string, FusedStreamlines3>();
  private readonly streamSets = new Map<string, GpuSegments3>();
  private readonly ctx2d: CanvasRenderingContext2D;

  constructor(private readonly c: View3DContext) {
    this.renderer = new GpuRenderer3D(c.gpu, c.canvas);
    this.ctx2d = c.overlay.getContext("2d")!;
  }

  clear(): void {
    for (const g of this.grids.values()) g.destroy();
    for (const k of this.kernels.values()) k.destroy();
    for (const m of this.meshes.values()) m.mesh.destroy();
    for (const m of this.cpuMeshes.values()) m.destroy();
    for (const f of this.faces.values()) destroyFace(f);
    for (const f of this.faceCpu.values()) f.destroy();
    for (const l of this.lines3.values()) l.destroy();
    for (const g of this.vgrids.values()) g.destroy();
    for (const k of this.streamKernels.values()) k.destroy();
    for (const l of this.streamSets.values()) l.destroy();
    this.vgrids.clear(); this.vsampled.clear(); this.streamKernels.clear(); this.streamSets.clear();
    this.grids.clear(); this.kernels.clear(); this.meshes.clear(); this.cpuMeshes.clear(); this.cpuValues.clear();
    this.faces.clear(); this.faceCpu.clear(); this.lines3.clear();
    this.boxKey = "";
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

  private grid(box: Box): DenseGrid {
    const n = this.c.resolution(), mx = Math.max(...box.size) || 1;
    return new DenseGrid(box.size.map((s) => Math.max(2, Math.round((n * s) / mx) || 2)), box);
  }

  /** the resident I_V grid, blurred when `metric` is set */
  private volumeGpu(iv: Use3, grid: DenseGrid): { values: GpuGrid; key: string } {
    const gk = gridKey(iv, grid);
    let values = this.grids.get(gk);
    if (!values) { this.grids.set(gk, (values = sampleResidentSync(this.c.gpu, iv.data, grid))); lru(this.grids, 4, (g) => g.destroy()); }
    const r = this.c.blur();
    if (r === null || r <= 0) return { values, key: gk };
    const bk = `${gk}|blur${r}`;
    let blurred = this.grids.get(bk);
    if (!blurred) { this.grids.set(bk, (blurred = blurResidentSync(this.c.gpu, values, r))); lru(this.grids, 4, (g) => g.destroy()); }
    return { values: blurred, key: bk };
  }
  private volumeCpu(iv: Use3, grid: DenseGrid): { values: Float64Array; key: string } {
    const gk = gridKey(iv, grid);
    let vals = this.cpuValues.get(gk);
    if (!vals) { this.cpuValues.set(gk, (vals = Float64Array.from(iv.data.sampleOn(grid)))); lru(this.cpuValues, 4, () => {}); }
    const r = this.c.blur();
    if (r === null || r <= 0) return { values: vals, key: gk };
    const bk = `${gk}|blur${r}`;
    let blurred = this.cpuValues.get(bk);
    if (!blurred) { this.cpuValues.set(bk, (blurred = boxBlur(grid, vals, r))); lru(this.cpuValues, 4, () => {}); }
    return { values: blurred, key: bk };
  }
  /** exact projection applies to symbolic fields contoured as they are (a blurred field is only known on the grid) */
  private isExact(iv: Use3): boolean { return this.c.exact() && iv.data.kind === "symbolic" && !(this.c.blur()! > 0); }

  private meshesGpu(iv: Use3, ic: Use3 | undefined, grid: DenseGrid, levels: number[]): GpuMesh[] {
    const { values, key: gk } = this.volumeGpu(iv, grid);
    const exact = this.isExact(iv);
    const kk = `${gk}|${ic?.id ?? ""}|${exact ? "exact" : "lin"}`;
    let kernel = this.kernels.get(kk);
    if (!kernel) { this.kernels.set(kk, (kernel = fusedIsosurface(this.c.gpu, values, { field: exact ? iv.data : undefined, exact, colour: ic?.data }))); lru(this.kernels, 8, (k) => k.destroy()); }
    // triangle budget: surfaces touch O(n²) of the n³ cells; overflow drops triangles silently
    const cells = grid.size.reduce((a, s) => a * (s - 1), 1);
    const capacity = Math.min(kernel.capacity, Math.max(65536, Math.min(1 << 20, Math.round(cells * 0.5))));
    return levels.map((level, k) => {
      const mk = `${kk}|${k}`;
      let set = this.meshes.get(mk);
      if (set && set.mesh.capacity !== capacity) { set.mesh.destroy(); this.meshes.delete(mk); set = undefined; }
      if (!set) { this.meshes.set(mk, (set = { mesh: allocMesh(this.c.gpu, capacity), stamp: "" })); lru(this.meshes, 24, (v) => v.mesh.destroy()); }
      const stamp = `${kk}|${level}`;
      if (set.stamp !== stamp) { resetMesh(this.c.gpu, set.mesh); kernel!.dispatch(set.mesh, level); set.stamp = stamp; }
      return set.mesh;
    });
  }

  private meshesCpu(iv: Use3, ic: Use3 | undefined, grid: DenseGrid, levels: number[]): GpuMesh[] {
    const { values: vals, key: gk } = this.volumeCpu(iv, grid);
    const exact = this.isExact(iv);
    const grad = exact ? this.c.gradientOf(iv) : undefined; // blurred: normals from the blurred grid
    const maxDist = Math.hypot(...grid.spacing);
    const smooth = exact ? 0 : this.c.smoothing();
    return levels.map((level) => {
      const mk = `${gk}|${ic?.id ?? ""}|${exact ? "exact" : `lin|sm${smooth}`}|${level}`;
      let mesh = this.cpuMeshes.get(mk);
      if (!mesh) {
        let m = marchingTetrahedra(grid, vals, level, {
          gradient: grad ? (p) => grad.value(p) ?? undefined : undefined,
          project: exact ? (p) => projectToLevel(iv.data, p, level, maxDist) : undefined,
          colourAt: ic ? (p) => ic.data.value(p) ?? NaN : undefined,
        });
        if (smooth > 0) m = smoothIsoMesh(m, smooth);
        this.cpuMeshes.set(mk, (mesh = uploadMesh(this.c.gpu, packMesh(m)))); lru(this.cpuMeshes, 24, (v) => v.destroy());
      }
      return mesh;
    });
  }

  /** the box cut to the crop ranges (fractions of the box; an open end is the box's own end) */
  private cropped(box: Box, crop: CropRange[]): Box {
    const f = (v: number | null, dflt: number) => Math.max(0, Math.min(1, v ?? dflt));
    return new Box(box.a.map((a, d) => a + f(crop[d]?.[0] ?? null, 0) * box.size[d]!), box.a.map((a, d) => a + Math.max(f(crop[d]?.[0] ?? null, 0) + 0.01, f(crop[d]?.[1] ?? null, 1)) * box.size[d]!));
  }

  /**
   * Isolines of the I_V field on the six faces of the cropped box (the prototype's "outline"): where each
   * isosurface meets the faces. Every face has ONE grid — the whole box face at the volume grid's spacing, so on
   * the uncropped faces the lines coincide with the mesh boundary — one values buffer, one 2D marching-squares
   * kernel and one segment set per level, all built once per (field, grid); moving a crop plane only re-dispatches
   * the plane sampler and the kernels with the new depth (no shader compiles, no allocations), and the renderer's
   * crop planes trim the lines to the cropped face. Values: the field sampled exactly on the plane (exact case),
   * else the (blurred / sampled) volume grid sliced trilinearly.
   */
  private faceLines(iv: Use3, grid: DenseGrid, box: Box, cbox: Box, levels: number[], width: number): GpuLineLayer3D[] {
    const out: GpuLineLayer3D[] = [];
    const gpu = this.c.compute() === "gpu";
    const exact = this.isExact(iv);
    const tol = 0.25 * Math.min(...grid.spacing); // world tolerance of the exact chords
    const source = exact ? "exact" : `lin|blur${this.c.blur() ?? 0}`;
    for (let axis = 0; axis < 3; axis++) for (const hi of [false, true]) {
      const depth = hi ? cbox.b[axis]! : cbox.a[axis]!;
      const oa = [0, 1, 2].filter((d) => d !== axis) as [number, number];
      const grid2 = new DenseGrid(oa.map((d) => grid.size[d]!), new Box(oa.map((d) => box.a[d]!), oa.map((d) => box.b[d]!)));
      const fkey = `${iv.id}|face${axis}|${grid.size.join("x")}|${box.intervals.flat().join(",")}|${source}`;
      const side = `${fkey}|${hi ? "+" : "-"}`;
      const embed = { axis, depth };
      if (gpu) {
        let face = this.faces.get(side);
        if (!face) {
          const values = uploadGrid(this.c.gpu, grid2, new Float32Array(grid2.sampleCount), 1);
          const sampler = exact ? planeSampler(this.c.gpu, iv.data, axis, grid2) : planeSlicer(this.c.gpu, this.volumeGpu(iv, grid).values, axis, grid2);
          const kernel = fusedIsolines(this.c.gpu, undefined, values, undefined, exact ? { slice: { field: iv.data, axis } } : { exact: false });
          face = { values, sampler, kernel, depth: NaN, sets: [] };
          this.faces.set(side, face); lru(this.faces, 12, destroyFace);
        }
        if (face.depth !== depth) { face.sampler.dispatch(depth, face.values.buffer); face.depth = depth; for (const st of face.sets) st.stamp = ""; }
        const capacity = Math.min(face.kernel.capacity, Math.max(4096, (grid2.size[0]! - 1) * (grid2.size[1]! - 1) * (exact ? 8 : 2)));
        levels.forEach((level, k) => {
          let set = face!.sets[k];
          if (set && set.segs.capacity !== capacity) { set.segs.destroy(); set = undefined; }
          if (!set) face!.sets[k] = set = { segs: allocSegments(this.c.gpu, capacity, false), stamp: "" };
          const stamp = `${level}`;
          if (set.stamp !== stamp) { resetSegments(this.c.gpu, set.segs); face!.kernel.dispatch(set.segs, level, tol, depth); set.stamp = stamp; }
          out.push({ segs: set.segs, embed, width, color: [0.92, 0.92, 0.92] });
        });
      } else {
        // the same points as a degenerate 3D grid (size 1 along the axis): what the field is sampled on
        const size3 = [0, 0, 0], a3 = [0, 0, 0], b3 = [0, 0, 0];
        size3[axis] = 1; a3[axis] = depth; b3[axis] = depth;
        oa.forEach((d, i) => { size3[d] = grid2.size[i]!; a3[d] = grid2.box.a[i]!; b3[d] = grid2.box.b[i]!; });
        const grid3 = new DenseGrid(size3, new Box(a3, b3));
        const vkey = `${side}|${depth}`;
        let vals = this.cpuValues.get(vkey);
        if (!vals) {
          vals = exact ? Float64Array.from(iv.data.sampleOn(grid3)) : Float64Array.from(new DenseScalarFieldData(grid, this.volumeCpu(iv, grid).values).sampleOn(grid3));
          this.cpuValues.set(vkey, vals); lru(this.cpuValues, 16, () => {});
        }
        for (const level of levels) {
          const sk = `${vkey}|${level}`;
          let segs = this.faceCpu.get(sk);
          if (!segs) {
            const lines = exact ? contourField(sliceScalarField(iv.data, axis, depth), grid2, vals!, level, { tolerance: tol }).lines : isoContours(grid2, vals!, level);
            this.faceCpu.set(sk, (segs = uploadSegments(this.c.gpu, packPolylines(lines), false))); lru(this.faceCpu, 96, (v) => v.destroy());
          }
          out.push({ segs, embed, width, color: [0.92, 0.92, 0.92] });
        }
      }
    }
    return out;
  }

  /** the vector field sampled on the volume grid (what streamlines are integrated through, like the 2D arm) */
  private sampledVector(v: VectorUse3, grid: DenseGrid): DenseVectorFieldData {
    if (v.data.kind === "sampled" && v.data instanceof DenseVectorFieldData) return v.data;
    const key = gridKey(v, grid);
    let d = this.vsampled.get(key);
    if (!d) { this.vsampled.set(key, (d = new DenseVectorFieldData(grid, v.data.sampleOn(grid)))); lru(this.vsampled, 4, () => {}); }
    return d;
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
        const gk = `vec:${gridKey(v, vgrid)}`;
        let vectors = this.vgrids.get(gk);
        if (!vectors) { this.vgrids.set(gk, (vectors = sampleResidentSync(c.gpu, v.data, vgrid))); lru(this.vgrids, 4, (g) => g.destroy()); }
        let kernel = this.streamKernels.get(key);
        if (!kernel) { this.streamKernels.set(key, (kernel = fusedStreamlines3(c.gpu, vectors, seeds, iopts, sc?.data))); lru(this.streamKernels, 8, (k) => k.destroy()); }
        segs = allocSegments3(c.gpu, kernel.capacity, true);
        kernel.dispatch(segs);
      } else {
        const lines = plan?.lines ?? integrateFromSeeds(this.sampledVector(v, vgrid), seeds, iopts);
        const colours = sc ? lines.map((l) => { const out = new Float64Array(l.points.length / 3); for (let i = 0; i < out.length; i++) out[i] = sc.data.value([l.points[3 * i]!, l.points[3 * i + 1]!, l.points[3 * i + 2]!]) ?? NaN; return out; }) : undefined;
        segs = uploadSegments3(c.gpu, packStreamlines3(lines, step, colours), true);
      }
      this.streamSets.set(key, segs); lru(this.streamSets, 8, (s) => s.destroy());
    }
    const colour = sc ? c.colour(sc) : undefined;
    return {
      segs, width: 1.5, color: [1, 1, 1],
      particles: o.tail === null ? undefined : { tail: o.tail * cell, split: o.split, travel: o.clock * 10 * cell },
      ...(colour ? { map: colour.map, lut: colour.lut } : {}),
    };
  }

  /** a cached uploaded 3D segment set */
  private segs3(key: string, make: () => Float32Array): GpuSegments3 {
    let s = this.lines3.get(key);
    if (!s) { this.lines3.set(key, (s = uploadSegments3(this.c.gpu, make(), false))); lru(this.lines3, 16, (v) => v.destroy()); }
    return s;
  }

  render(): void {
    const c = this.c;
    const iv = c.showIso() ? c.isoField() : undefined;
    const sv = c.streamVector();
    const box = iv?.data.box ?? sv?.data.box ?? this.box;
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
    if (iv) {
      const ic = c.colourField();
      const grid = this.grid(box);
      const levels = c.levels(iv);
      // surface smoothing needs welded connectivity: CPU meshing (the fused kernel has none yet)
      const gpuMesh = c.compute() === "gpu" && !(c.smoothing() > 0 && !this.isExact(iv));
      const sets = gpuMesh ? this.meshesGpu(iv, ic, grid, levels) : this.meshesCpu(iv, ic, grid, levels);
      const alpha = c.alpha();
      const colour = ic ? c.colour(ic) : undefined;
      for (const mesh of sets) meshes.push({ mesh, alpha, color: [0.86, 0.87, 0.9], ...(colour ? { map: colour.map, lut: colour.lut } : {}) });
      if (c.showOutline()) lines.push(...this.faceLines(iv, grid, box, cbox, levels, 2));
    }
    if (sv) { const layer = this.streamLayer(sv, box, this.grid(box)); if (layer) lines.push(layer); }
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

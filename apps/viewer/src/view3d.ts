// The 3D arm of the viewer: isosurfaces of the I_V field, coloured by I_C,
// rendered by GpuRenderer3D (WebGPU only) with the box, point sets and labels
// on the Canvas 2D overlay. Compute modes as in 2D: "gpu" samples a resident
// grid and runs the fused marching-tetrahedra kernel into resident meshes
// (nothing read back); "cpu" runs core's marchingTetrahedra and uploads.
// main.ts owns the controls and the shared state and hands them over through
// View3DContext.

import { Box, DenseGrid, contourField, isoContours, marchingTetrahedra, projectToLevel, sliceScalarField, type PointSet, type ScalarFieldData, type VectorFieldData } from "@tensatory/core";
import {
  GpuRenderer3D,
  allocMesh,
  allocSegments,
  boxEdges,
  fusedIsolines,
  fusedIsosurface,
  packMesh,
  packPolylines,
  packPolylines3,
  project,
  resetMesh,
  resetSegments,
  sampleResidentSync,
  uploadMesh,
  uploadSegments,
  uploadSegments3,
  type Camera3D,
  type FusedIsolines,
  type FusedIsosurface,
  type GpuBackend,
  type GpuGrid,
  type GpuLineLayer3D,
  type GpuMesh,
  type GpuMeshLayer,
  type GpuSegments,
  type GpuSegments3,
  type Lut,
  type ValueMap,
} from "@tensatory/gpu";

export interface Use3 { id: string; data: ScalarFieldData }

export interface View3DContext {
  gpu: GpuBackend;
  /** the WebGPU canvas (#gpu) and the Canvas 2D overlay above it (#gl) */
  canvas: HTMLCanvasElement;
  overlay: HTMLCanvasElement;
  isoField(): Use3 | undefined;
  colourField(): Use3 | undefined;
  /** exact gradient data of a symbolic use (for normals), undefined for sampled data */
  gradientOf(u: Use3): VectorFieldData | undefined;
  /** isosurface levels in field units */
  levels(u: Use3): number[];
  alpha(): number;
  /** grid points along the longest box side */
  resolution(): number;
  compute(): "cpu" | "gpu";
  showIso(): boolean;
  /** project vertices onto the true level set along the exact gradient (symbolic fields) */
  exact(): boolean;
  /** isolines of I_V on the (cropped) box faces */
  showOutline(): boolean;
  showPoints(): boolean;
  showBox(): boolean;
  /** crop fractions per axis (0..1]: the box is cut at a + crop · size */
  crop(): [number, number, number];
  pointSets(): PointSet[];
  colour(u: Use3): { map: ValueMap; lut: Lut; key: string };
}

const lru = <V>(m: Map<string, V>, max: number, drop: (v: V) => void) => { while (m.size > max) { const k = m.keys().next().value as string; drop(m.get(k)!); m.delete(k); } };
const gridKey = (u: Use3, g: DenseGrid) => `${u.id}|${g.size.join("x")}|${g.box.intervals.flat().join(",")}`;

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
  // face isolines: per face a sampled face grid (resident or CPU), the 2D kernel and one segment set per level
  private readonly faceGrids = new Map<string, GpuGrid>();
  private readonly faceKernels = new Map<string, FusedIsolines>();
  private readonly faceSets = new Map<string, { segs: GpuSegments; stamp: string }>();
  private readonly faceCpu = new Map<string, GpuSegments>();
  private readonly lines3 = new Map<string, GpuSegments3>();
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
    for (const g of this.faceGrids.values()) g.destroy();
    for (const k of this.faceKernels.values()) k.destroy();
    for (const f of this.faceSets.values()) f.segs.destroy();
    for (const f of this.faceCpu.values()) f.destroy();
    for (const l of this.lines3.values()) l.destroy();
    this.grids.clear(); this.kernels.clear(); this.meshes.clear(); this.cpuMeshes.clear(); this.cpuValues.clear();
    this.faceGrids.clear(); this.faceKernels.clear(); this.faceSets.clear(); this.faceCpu.clear(); this.lines3.clear();
    this.boxKey = "";
  }

  /** frame `box`: look at its centre from a distance that fits it */
  fit(box = this.box): void {
    const r = Math.hypot(...box.size) / 2 || 1;
    this.camera = { ...this.camera, target: box.center as [number, number, number], distance: r / Math.sin(this.camera.fov / 2) * 1.05 };
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
    const s = (2 * this.camera.distance * Math.tan(this.camera.fov / 2)) / this.c.overlay.clientHeight;
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

  private meshesGpu(iv: Use3, ic: Use3 | undefined, grid: DenseGrid, levels: number[]): GpuMesh[] {
    const gk = gridKey(iv, grid);
    let values = this.grids.get(gk);
    if (!values) { this.grids.set(gk, (values = sampleResidentSync(this.c.gpu, iv.data, grid))); lru(this.grids, 4, (g) => g.destroy()); }
    const exact = this.c.exact() && iv.data.kind === "symbolic";
    const kk = `${gk}|${ic?.id ?? ""}|${exact ? "exact" : "lin"}`;
    let kernel = this.kernels.get(kk);
    if (!kernel) { this.kernels.set(kk, (kernel = fusedIsosurface(this.c.gpu, values, { field: iv.data, exact, colour: ic?.data }))); lru(this.kernels, 8, (k) => k.destroy()); }
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
    const gk = gridKey(iv, grid);
    let vals = this.cpuValues.get(gk);
    if (!vals) { this.cpuValues.set(gk, (vals = Float64Array.from(iv.data.sampleOn(grid)))); lru(this.cpuValues, 4, () => {}); }
    const grad = this.c.gradientOf(iv);
    const exact = this.c.exact() && iv.data.kind === "symbolic";
    const maxDist = Math.hypot(...grid.spacing);
    return levels.map((level) => {
      const mk = `${gk}|${ic?.id ?? ""}|${exact ? "exact" : "lin"}|${level}`;
      let mesh = this.cpuMeshes.get(mk);
      if (!mesh) {
        const m = marchingTetrahedra(grid, vals!, level, {
          gradient: grad ? (p) => grad.value(p) ?? undefined : undefined,
          project: exact ? (p) => projectToLevel(iv.data, p, level, maxDist) : undefined,
          colourAt: ic ? (p) => ic.data.value(p) ?? NaN : undefined,
        });
        this.cpuMeshes.set(mk, (mesh = uploadMesh(this.c.gpu, packMesh(m)))); lru(this.cpuMeshes, 24, (v) => v.destroy());
      }
      return mesh;
    });
  }

  /** the cropped box: a … a + crop · size */
  private cropped(box: Box): Box {
    const crop = this.c.crop();
    return new Box([...box.a], box.a.map((a, d) => a + Math.max(0.02, Math.min(1, crop[d]!)) * box.size[d]!));
  }

  /**
   * Isolines of the I_V field on the six faces of the cropped box (the prototype's "outline"): where each
   * isosurface meets the faces. Each face is sampled on a grid with the volume grid's spacing (so on the
   * uncropped faces the lines coincide with the mesh boundary), contoured by marching squares, and drawn
   * as 2D segments embedded on the face plane.
   */
  private faceLines(iv: Use3, grid: DenseGrid, cbox: Box, levels: number[], width: number): GpuLineLayer3D[] {
    const out: GpuLineLayer3D[] = [];
    const gpu = this.c.compute() === "gpu";
    for (let axis = 0; axis < 3; axis++) for (const hi of [false, true]) {
      const depth = hi ? cbox.b[axis]! : cbox.a[axis]!;
      const oa = [0, 1, 2].filter((d) => d !== axis) as [number, number];
      const size2 = oa.map((d) => Math.max(2, Math.round(cbox.size[d]! / (grid.spacing[d]! || 1)) + 1));
      const box2 = new Box(oa.map((d) => cbox.a[d]!), oa.map((d) => cbox.b[d]!));
      const grid2 = new DenseGrid(size2, box2);
      // the same points as a degenerate 3D grid (size 1 along the axis): what the field is sampled on
      const size3 = [0, 0, 0], a3 = [0, 0, 0], b3 = [0, 0, 0];
      size3[axis] = 1; a3[axis] = depth; b3[axis] = depth;
      oa.forEach((d, i) => { size3[d] = size2[i]!; a3[d] = box2.a[i]!; b3[d] = box2.b[i]!; });
      const grid3 = new DenseGrid(size3, new Box(a3, b3));
      const exact = this.c.exact() && iv.data.kind === "symbolic";
      const fkey = `${iv.id}|face${axis}${hi ? "+" : "-"}|${depth}|${size2.join("x")}|${box2.intervals.flat().join(",")}|${exact ? "exact" : "lin"}`;
      const embed = { axis, depth };
      const tol = 0.25 * Math.min(...grid.spacing); // world tolerance of the exact chords
      if (gpu) {
        let values = this.faceGrids.get(fkey);
        if (!values) {
          const v3 = sampleResidentSync(this.c.gpu, iv.data, grid3);
          values = { grid: grid2, channels: 1, buffer: v3.buffer, destroy: () => v3.destroy() };
          this.faceGrids.set(fkey, values); lru(this.faceGrids, 12, (g) => g.destroy());
        }
        let kernel = this.faceKernels.get(fkey);
        if (!kernel) { this.faceKernels.set(fkey, (kernel = fusedIsolines(this.c.gpu, undefined, values, undefined, exact ? { slice: { field: iv.data, axis, depth } } : { exact: false }))); lru(this.faceKernels, 12, (k) => k.destroy()); }
        const capacity = Math.min(kernel.capacity, Math.max(4096, (size2[0]! - 1) * (size2[1]! - 1) * (exact ? 8 : 2)));
        levels.forEach((level, k) => {
          const sk = `${fkey}|${k}`;
          let set = this.faceSets.get(sk);
          if (set && set.segs.capacity !== capacity) { set.segs.destroy(); this.faceSets.delete(sk); set = undefined; }
          if (!set) { this.faceSets.set(sk, (set = { segs: allocSegments(this.c.gpu, capacity, false), stamp: "" })); lru(this.faceSets, 96, (v) => v.segs.destroy()); }
          const stamp = `${fkey}|${level}`;
          if (set.stamp !== stamp) { resetSegments(this.c.gpu, set.segs); kernel!.dispatch(set.segs, level, tol); set.stamp = stamp; }
          out.push({ segs: set.segs, embed, width, color: [0.92, 0.92, 0.92] });
        });
      } else {
        let vals = this.cpuValues.get(fkey);
        if (!vals) { this.cpuValues.set(fkey, (vals = Float64Array.from(iv.data.sampleOn(grid3)))); lru(this.cpuValues, 16, () => {}); }
        for (const level of levels) {
          const sk = `${fkey}|${level}`;
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

  /** a cached uploaded 3D segment set */
  private segs3(key: string, make: () => Float32Array): GpuSegments3 {
    let s = this.lines3.get(key);
    if (!s) { this.lines3.set(key, (s = uploadSegments3(this.c.gpu, make(), false))); lru(this.lines3, 16, (v) => v.destroy()); }
    return s;
  }

  render(): void {
    const c = this.c;
    const iv = c.showIso() ? c.isoField() : undefined;
    const box = iv?.data.box ?? this.box;
    const key = box.intervals.flat().join(",");
    if (key !== this.boxKey) { this.boxKey = key; this.box = box; if (!this.cameraCustom) this.fit(box); }
    const cbox = this.cropped(box);
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
      const sets = c.compute() === "gpu" ? this.meshesGpu(iv, ic, grid, levels) : this.meshesCpu(iv, ic, grid, levels);
      const alpha = c.alpha();
      const colour = ic ? c.colour(ic) : undefined;
      for (const mesh of sets) meshes.push({ mesh, alpha, color: [0.86, 0.87, 0.9], ...(colour ? { map: colour.map, lut: colour.lut } : {}) });
      if (c.showOutline()) lines.push(...this.faceLines(iv, grid, cbox, levels, 2));
    }
    this.renderer.resize();
    this.renderer.render({ camera: this.camera, radius: Math.hypot(...box.size) / 2 || 1, background: [0x0b / 255, 0x0d / 255, 0x12 / 255], meshes, lines, cropMax: cbox.b as [number, number, number] });
    this.overlay(cbox);
  }

  /** world → css px on the overlay */
  project(p: ArrayLike<number>): [number, number, number] | undefined {
    return project(this.renderer.viewProj, p, this.c.overlay.clientWidth, this.c.overlay.clientHeight);
  }

  private overlay(box: Box): void {
    const cv = this.c.overlay, ctx = this.ctx2d;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.round(cv.clientWidth * dpr)), h = Math.max(1, Math.round(cv.clientHeight * dpr));
    if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cv.clientWidth, cv.clientHeight);
    if (this.c.showPoints()) {
      for (const ps of this.c.pointSets()) {
        const inside = (p: ArrayLike<number>) => p[0]! <= box.b[0]! + 1e-9 && p[1]! <= box.b[1]! + 1e-9 && p[2]! <= box.b[2]! + 1e-9;
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

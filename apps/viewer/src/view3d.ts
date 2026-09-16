// The 3D arm of the viewer: isosurfaces of the I_V field, coloured by I_C,
// rendered by GpuRenderer3D (WebGPU only) with the box, point sets and labels
// on the Canvas 2D overlay. Compute modes as in 2D: "gpu" samples a resident
// grid and runs the fused marching-tetrahedra kernel into resident meshes
// (nothing read back); "cpu" runs core's marchingTetrahedra and uploads.
// main.ts owns the controls and the shared state and hands them over through
// View3DContext.

import { Box, DenseGrid, marchingTetrahedra, type PointSet, type ScalarFieldData, type VectorFieldData } from "@tensatory/core";
import {
  GpuRenderer3D,
  allocMesh,
  fusedIsosurface,
  packMesh,
  project,
  resetMesh,
  sampleResidentSync,
  uploadMesh,
  type Camera3D,
  type FusedIsosurface,
  type GpuBackend,
  type GpuGrid,
  type GpuMesh,
  type GpuMeshLayer,
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
  showPoints(): boolean;
  showBox(): boolean;
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
    this.grids.clear(); this.kernels.clear(); this.meshes.clear(); this.cpuMeshes.clear(); this.cpuValues.clear();
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
    const kk = `${gk}|${ic?.id ?? ""}`;
    let kernel = this.kernels.get(kk);
    if (!kernel) { this.kernels.set(kk, (kernel = fusedIsosurface(this.c.gpu, values, this.c.gradientOf(iv), ic?.data))); lru(this.kernels, 8, (k) => k.destroy()); }
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
    return levels.map((level) => {
      const mk = `${gk}|${ic?.id ?? ""}|${level}`;
      let mesh = this.cpuMeshes.get(mk);
      if (!mesh) {
        const m = marchingTetrahedra(grid, vals!, level, {
          gradient: grad ? (p) => grad.value(p) ?? undefined : undefined,
          colourAt: ic ? (p) => ic.data.value(p) ?? NaN : undefined,
        });
        this.cpuMeshes.set(mk, (mesh = uploadMesh(this.c.gpu, packMesh(m)))); lru(this.cpuMeshes, 24, (v) => v.destroy());
      }
      return mesh;
    });
  }

  render(): void {
    const c = this.c;
    const iv = c.showIso() ? c.isoField() : undefined;
    const box = iv?.data.box ?? this.box;
    const key = box.intervals.flat().join(",");
    if (key !== this.boxKey) { this.boxKey = key; this.box = box; if (!this.cameraCustom) this.fit(box); }
    const meshes: GpuMeshLayer[] = [];
    if (iv) {
      const ic = c.colourField();
      const grid = this.grid(box);
      const levels = c.levels(iv);
      const sets = c.compute() === "gpu" ? this.meshesGpu(iv, ic, grid, levels) : this.meshesCpu(iv, ic, grid, levels);
      const alpha = c.alpha();
      const colour = ic ? c.colour(ic) : undefined;
      for (const mesh of sets) meshes.push({ mesh, alpha, color: [0.86, 0.87, 0.9], ...(colour ? { map: colour.map, lut: colour.lut } : {}) });
    }
    this.renderer.resize();
    this.renderer.render({ camera: this.camera, radius: Math.hypot(...box.size) / 2 || 1, background: [0x0b / 255, 0x0d / 255, 0x12 / 255], meshes });
    this.overlay(box);
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
    if (this.c.showBox()) {
      const [a, b] = [box.a, box.b];
      const corner = (m: number) => [m & 1 ? b[0]! : a[0]!, m & 2 ? b[1]! : a[1]!, m & 4 ? b[2]! : a[2]!];
      ctx.strokeStyle = "#7f8fb8"; ctx.lineWidth = 1.2; ctx.globalAlpha = 0.8;
      ctx.beginPath();
      for (let m = 0; m < 8; m++) for (const bit of [1, 2, 4]) {
        if (m & bit) continue;
        const p = this.project(corner(m)), q = this.project(corner(m | bit));
        if (p && q) { ctx.moveTo(p[0], p[1]); ctx.lineTo(q[0], q[1]); }
      }
      ctx.stroke();
      ctx.globalAlpha = 1;
    }
    if (this.c.showPoints()) {
      for (const ps of this.c.pointSets()) {
        const pts = ps.points.map((p) => this.project(p));
        const single = pts.length === 1;
        if (ps.ordered && pts.length > 1) {
          ctx.beginPath(); let started = false;
          for (const p of pts) { if (!p) continue; if (started) ctx.lineTo(p[0], p[1]); else ctx.moveTo(p[0], p[1]); started = true; }
          ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 1.5; ctx.stroke();
        }
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

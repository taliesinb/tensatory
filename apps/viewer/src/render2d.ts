// Canvas-2D renderer for 2D fields: colormapped raster, line layers (isolines,
// streamlines with particle animation), point sets, box outline and crop.
// Axis 0 is horizontal (right), axis 1 vertical (up unless flipped).

import type { Box, DenseGrid, PointSet, Polyline } from "@tensatory/core";
import { toCss, type Colormap, type RGB } from "./colormap";

/** camera: world centre, pixels per world unit, and an orientation (axis flips + quarter turns clockwise) */
export interface View { cx: number; cy: number; scale: number; flipX: boolean; flipY: boolean; rot: 0 | 1 | 2 | 3 }

/** the linear part of world -> screen as [a, b, c, d]: sx = a dx + c dy, sy = b dx + d dy (includes scale) */
export function viewLinear(v: View): [number, number, number, number] {
  // base: x right, y up (screen y grows downwards), with flips
  let a = v.flipX ? -1 : 1, b = 0, c = 0, d = v.flipY ? 1 : -1;
  // quarter turns clockwise on screen: (x, y) -> (-y, x)
  for (let i = 0; i < v.rot; i++) [a, b, c, d] = [-b, a, -d, c];
  return [a * v.scale, b * v.scale, c * v.scale, d * v.scale];
}

export interface RasterLayer {
  key: string; // cache key: re-rasterize only when it changes
  grid: DenseGrid;
  values: ArrayLike<number>;
  /** value -> colormap parameter in [0, 1] (NaN -> transparent) */
  toParam: (v: number) => number;
  lut: Uint8ClampedArray;
  smooth: boolean;
}

export interface Particles {
  /** total arc length of each line (world units) */
  lengths: ArrayLike<number>;
  phases: ArrayLike<number>;
  /** world units per polyline step */
  step: number;
  /** tail length in world units; the tail fades to transparent */
  tail: number;
  split: number;
  /** head travel in world units */
  travel: number;
}

export interface LineLayer {
  lines: Polyline[];
  /** per-line, per-vertex colormap parameters (same count as vertices), or undefined for a solid colour */
  values?: (Float64Array | undefined)[];
  cmap?: Colormap;
  color: RGB;
  width: number;
  alpha: number;
  particles?: Particles;
}

export interface Scene {
  box: Box;
  /** kept fraction along each axis from corner a */
  crop: [number, number];
  showBox: boolean;
  raster?: RasterLayer;
  lines: LineLayer[];
  pointSets: PointSet[];
}

const COLOR_BINS = 32, ALPHA_BINS = 12;

export class Renderer2D {
  private readonly ctx: CanvasRenderingContext2D;
  private readonly off = document.createElement("canvas");
  private rasterKey = "";
  view: View = { cx: 0, cy: 0, scale: 100, flipX: false, flipY: false, rot: 0 };
  private w = 0;
  private h = 0;

  constructor(readonly canvas: HTMLCanvasElement) {
    this.ctx = canvas.getContext("2d")!;
  }

  /** fit the box into a screen region [x0, y0, x1, y1] (css px; defaults to the whole canvas), centred in it */
  fit(box: Box, region?: [number, number, number, number]): void {
    const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
    this.w = w; this.h = h;
    const [x0, y0, x1, y1] = region ?? [0, 0, w, h];
    const rw = Math.max(1, x1 - x0), rh = Math.max(1, y1 - y0);
    const [bw, bh] = box.size as [number, number];
    const [sw, sh] = this.view.rot % 2 ? [bh, bw] : [bw, bh]; // on-screen extents after rotation
    this.view.scale = Math.max(1e-9, Math.min(rw / (sw || 1), rh / (sh || 1)));
    // world point at the region centre = box centre
    const c = box.center;
    const rcx = (x0 + x1) / 2, rcy = (y0 + y1) / 2;
    this.view.cx = c[0]!; this.view.cy = c[1]!;
    const [dx, dy] = this.screenToWorldOffset(rcx - w / 2, rcy - h / 2);
    this.view.cx -= dx; this.view.cy -= dy;
  }

  /** inverse of the linear part: screen offset -> world offset */
  private screenToWorldOffset(sx: number, sy: number): [number, number] {
    const [a, b, c, d] = viewLinear(this.view);
    const det = a * d - b * c;
    return [(d * sx - c * sy) / det, (-b * sx + a * sy) / det];
  }

  toScreen(p: ArrayLike<number>): [number, number] {
    const v = this.view, [a, b, c, d] = viewLinear(v);
    const dx = p[0]! - v.cx, dy = p[1]! - v.cy;
    return [this.w / 2 + a * dx + c * dy, this.h / 2 + b * dx + d * dy];
  }
  toWorld(sx: number, sy: number): [number, number] {
    const [dx, dy] = this.screenToWorldOffset(sx - this.w / 2, sy - this.h / 2);
    return [this.view.cx + dx, this.view.cy + dy];
  }
  get worldPerPixel(): number { return 1 / this.view.scale; }
  /** css-pixel size and the world->screen map, for other renderers sharing this camera */
  get gpuView(): { width: number; height: number; a: number; b: number; c: number; d: number; cx: number; cy: number } {
    const [a, b, c, d] = viewLinear(this.view);
    return { width: this.canvas.clientWidth, height: this.canvas.clientHeight, a, b, c, d, cx: this.view.cx, cy: this.view.cy };
  }

  /** pan by screen pixels */
  pan(dx: number, dy: number): void {
    const [wx, wy] = this.screenToWorldOffset(dx, dy);
    this.view.cx -= wx; this.view.cy -= wy;
  }
  /** zoom by a factor about a screen point */
  zoom(factor: number, sx: number, sy: number): void {
    const [wx, wy] = this.toWorld(sx, sy);
    this.view.scale = Math.max(1e-6, Math.min(1e9, this.view.scale * factor));
    const [nx, ny] = this.toWorld(sx, sy);
    this.view.cx += wx - nx; this.view.cy += wy - ny;
  }

  /** the cropped box as a screen rect [x, y, w, h] */
  private cropRect(s: Scene): [number, number, number, number] {
    const a = s.box.a, sz = s.box.size;
    const b = [a[0]! + sz[0]! * s.crop[0], a[1]! + sz[1]! * s.crop[1]];
    const [x0, y0] = this.toScreen([a[0]!, a[1]!]), [x1, y1] = this.toScreen(b);
    return [Math.min(x0, x1), Math.min(y0, y1), Math.abs(x1 - x0), Math.abs(y1 - y0)]; // quarter turns keep the rect axis-aligned
  }

  /** `overlay`: transparent background, box + points only (raster and lines are drawn by the GPU renderer underneath) */
  render(s: Scene, overlay = false): void {
    const { canvas, ctx } = this;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.w = canvas.clientWidth; this.h = canvas.clientHeight;
    const W = Math.round(this.w * dpr), H = Math.round(this.h * dpr);
    if (canvas.width !== W || canvas.height !== H) { canvas.width = W; canvas.height = H; }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (overlay) ctx.clearRect(0, 0, this.w, this.h);
    else { ctx.fillStyle = "#0b0d12"; ctx.fillRect(0, 0, this.w, this.h); }

    const rect = this.cropRect(s);
    ctx.save();
    ctx.beginPath(); ctx.rect(...rect); ctx.clip();
    if (!overlay) {
      if (s.raster) this.drawRaster(s.raster, s.box);
      for (const l of s.lines) this.drawLines(l);
    }
    this.drawPointSets(s.pointSets);
    ctx.restore();
    if (s.showBox) {
      ctx.strokeStyle = "#7f8fb8"; ctx.lineWidth = 1.5;
      ctx.strokeRect(rect[0] + 0.5, rect[1] + 0.5, rect[2], rect[3]);
    }
  }

  private drawRaster(r: RasterLayer, box: Box): void {
    const [nx, ny] = r.grid.size as [number, number];
    if (r.key !== this.rasterKey) {
      this.rasterKey = r.key;
      this.off.width = nx; this.off.height = ny;
      const octx = this.off.getContext("2d")!;
      const img = octx.createImageData(nx, ny), px = img.data;
      const sx = r.grid.strides[0]!, sy = r.grid.strides[1]!, n = r.lut.length / 3;
      for (let j = 0; j < ny; j++) {
        const row = ny - 1 - j; // grid y grows upwards; image row 0 is the top
        for (let i = 0; i < nx; i++) {
          const v = r.values[i * sx + j * sy]!;
          const o = (row * nx + i) * 4;
          const t = r.toParam(v);
          if (Number.isNaN(t)) { px[o + 3] = 0; continue; }
          const k = Math.max(0, Math.min(n - 1, Math.round(t * (n - 1)))) * 3;
          px[o] = r.lut[k]!; px[o + 1] = r.lut[k + 1]!; px[o + 2] = r.lut[k + 2]!; px[o + 3] = 255;
        }
      }
      octx.putImageData(img, 0, 0);
    }
    // samples sit at box corners, so raster cells extend half a spacing beyond the box; the image's
    // row 0 is the top (max y) of the grid box. Map image pixels -> world -> screen with one affine
    // transform so any orientation (flips, quarter turns) works.
    const [hx, hy] = r.grid.spacing as [number, number];
    const gb = r.grid.box;
    const wx0 = gb.a[0]! - hx / 2, wy1 = gb.b[1]! + hy / 2; // world position of image pixel (0, 0)
    const wpx = (gb.b[0]! + hx / 2 - wx0) / nx, wpy = (wy1 - (gb.a[1]! - hy / 2)) / ny; // world size of an image pixel
    const [a, b, c, d] = viewLinear(this.view);
    const [ox, oy] = this.toScreen([wx0, wy1]);
    const [bx0, by0] = this.toScreen([box.a[0]!, box.a[1]!]), [bx1, by1] = this.toScreen([box.b[0]!, box.b[1]!]);
    const ctx = this.ctx;
    ctx.save();
    ctx.beginPath(); ctx.rect(Math.min(bx0, bx1), Math.min(by0, by1), Math.abs(bx1 - bx0), Math.abs(by1 - by0)); ctx.clip();
    ctx.imageSmoothingEnabled = r.smooth;
    // image (i, j) -> screen: [ox, oy] + i * wpx * (a, b) - j * wpy * (c, d)
    ctx.transform(a * wpx, b * wpx, -c * wpy, -d * wpy, ox, oy);
    ctx.drawImage(this.off, 0, 0, nx, ny);
    ctx.restore();
  }

  private drawLines(l: LineLayer): void {
    const ctx = this.ctx;
    ctx.lineWidth = l.width; ctx.lineCap = "round"; ctx.lineJoin = "round";
    const P = l.particles;
    // Segments in different colour / alpha bins are stroked separately; with round caps the cap of one
    // would overlap the start of the next and translucent colour would composite twice (a bright dot at
    // every joint). Butt caps end each segment exactly at the shared vertex, so nothing overlaps.
    // (Overlaps within one stroke() call never double: a stroke is a single shape.)
    if (l.cmap || P) ctx.lineCap = "butt";
    if (!l.cmap && !P) {
      ctx.strokeStyle = toCss(l.color, l.alpha);
      ctx.beginPath();
      for (const line of l.lines) this.pathOf(line, ctx);
      ctx.stroke();
      return;
    }
    // per-segment colour / brightness: bin into paths
    const paths = new Map<number, Path2D>();
    l.lines.forEach((line, li) => {
      const vals = l.values?.[li];
      const n = line.length / 2;
      const len = P ? P.lengths[li]! : 0, phase = P ? P.phases[li]! : 0;
      const t = P && len > 0 ? (((P.travel + phase * len) % len) + len) % len : 0; // head position along the line
      for (let i = 0; i + 1 < n; i++) {
        let bright = 1;
        if (P) {
          const arc = (i + 0.5) * P.step, k = P.tail;
          if (P.split <= 1) {
            const a = arc - t; if (a < 0 || a > k) continue;
            bright = a / k - Math.max(k - t, 0) / k; if (bright <= 0.02) continue;
          } else {
            const span = len / P.split;
            const d = (((arc - t) % span) + span) % span; if (d > k) continue;
            bright = d / k; if (bright <= 0.02) continue;
          }
        }
        const cbin = vals ? Math.max(0, Math.min(COLOR_BINS - 1, Math.round(0.5 * (vals[i]! + vals[i + 1]!) * (COLOR_BINS - 1)))) : 0;
        const bbin = Math.max(0, Math.min(ALPHA_BINS - 1, Math.round(bright * (ALPHA_BINS - 1))));
        const kk = cbin * ALPHA_BINS + bbin;
        let path = paths.get(kk);
        if (!path) paths.set(kk, (path = new Path2D()));
        const [x0, y0] = this.toScreen([line[2 * i]!, line[2 * i + 1]!]), [x1, y1] = this.toScreen([line[2 * i + 2]!, line[2 * i + 3]!]);
        path.moveTo(x0, y0); path.lineTo(x1, y1);
      }
    });
    for (const [kk, path] of paths) {
      const cbin = Math.floor(kk / ALPHA_BINS), bbin = kk % ALPHA_BINS;
      const base = l.cmap ? l.cmap(cbin / (COLOR_BINS - 1)) : l.color;
      // particle tails fade out through transparency (2D lets us), not to black
      const b = P ? bbin / (ALPHA_BINS - 1) : 1;
      ctx.strokeStyle = toCss(base, l.alpha * b);
      ctx.stroke(path);
    }
  }

  private pathOf(line: Polyline, ctx: CanvasRenderingContext2D | Path2D): void {
    for (let i = 0; i < line.length; i += 2) {
      const [x, y] = this.toScreen([line[i]!, line[i + 1]!]);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
  }

  private drawPointSets(sets: PointSet[]): void {
    const ctx = this.ctx;
    for (const ps of sets) {
      const pts = ps.points.map((p) => this.toScreen(p));
      const single = pts.length === 1;
      if (ps.ordered && pts.length > 1) {
        ctx.beginPath(); pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
        ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 1.5; ctx.stroke();
      }
      pts.forEach(([x, y], i) => {
        const head = single || (ps.ordered && i === pts.length - 1);
        ctx.beginPath(); ctx.arc(x, y, head ? 6 : 2.5, 0, 2 * Math.PI);
        ctx.fillStyle = head ? "#ff4d4d" : "#ffffff"; ctx.fill();
        const label = ps.spec.labels?.[i];
        if (label) { ctx.fillStyle = "#fff"; ctx.font = "11px system-ui, sans-serif"; ctx.fillText(label, x + 8, y - 6); }
      });
    }
  }
}

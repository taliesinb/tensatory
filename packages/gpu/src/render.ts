// WebGPU renderer for the 2D viewer. Draws a colormapped raster from a
// resident grid and line layers from resident Seg sets (drawIndirect), so a
// fully GPU-computed frame never touches the CPU. Points, labels and the box
// outline are left to a transparent Canvas 2D overlay.

import type { Box } from "@tensatory/core";
import type { GpuBackend } from "./device";
import type { GpuGrid } from "./resident";
import { SEG_WGSL, type GpuSegments } from "./segments";

/** world -> screen (css px): screen = (W/2 + a dx + c dy, H/2 + b dx + d dy), dx = x - cx */
export interface GpuView { width: number; height: number; a: number; b: number; c: number; d: number; cx: number; cy: number }
/** codomain mapping of raw values to a colormap parameter in [0, 1] */
export interface ValueMap { lo: number; hi: number; log: boolean; flip: boolean }
/** 256 RGBA entries (0..255) */
export type Lut = Uint8Array;

export interface GpuRasterLayer {
  values: GpuGrid;
  channel?: number;
  /** clip to the field's own box */
  box: Box;
  map: ValueMap;
  lut: Lut;
  smooth: boolean;
  alpha?: number;
}
export interface GpuLineLayer {
  segs: GpuSegments;
  width: number;
  alpha: number;
  color: [number, number, number];
  /** colour by value through `lut` when given, else solid `color` */
  map?: ValueMap;
  lut?: Lut;
  /** particle window (needs segments with len > 0) */
  particles?: { tail: number; split: number; travel: number };
}
export interface GpuScene {
  view: GpuView;
  /** visible world rectangle (crop) */
  clip: Box;
  background: [number, number, number];
  raster?: GpuRasterLayer;
  lines: GpuLineLayer[];
}

const COMMON = `
struct View { lin: vec4<f32>, centre: vec4<f32>, clip: vec4<f32>, map: vec4<f32> }
fn toScreen(p: vec2<f32>, v: View) -> vec2<f32> {
  let d = p - v.centre.xy;
  return vec2<f32>(v.centre.z * 0.5 + v.lin.x * d.x + v.lin.z * d.y, v.centre.w * 0.5 + v.lin.y * d.x + v.lin.w * d.y);
}
fn toClip(s: vec2<f32>, v: View) -> vec4<f32> { return vec4<f32>(2.0 * s.x / v.centre.z - 1.0, 1.0 - 2.0 * s.y / v.centre.w, 0.0, 1.0); }
fn inClip(p: vec2<f32>, v: View) -> bool { return p.x >= v.clip.x && p.x <= v.clip.z && p.y >= v.clip.y && p.y <= v.clip.w; }
fn isnan_(x: f32) -> bool { let b = bitcast<u32>(x); return (b & 0x7f800000u) == 0x7f800000u && (b & 0x007fffffu) != 0u; }
// map: lo, hi, log, flip
fn param(value: f32, m: vec4<f32>) -> f32 {
  let v = clamp(value, min(m.x, m.y), max(m.x, m.y));
  var t: f32;
  if (m.z > 0.5) { t = (log(v) - log(m.x)) / (log(m.y) - log(m.x)); } else { t = (v - m.x) / (m.y - m.x); }
  t = clamp(t, 0.0, 1.0);
  return select(t, 1.0 - t, m.w > 0.5);
}
`;

const RASTER = `${COMMON}
struct RasterU { view: View, gridA: vec4<f32>, gridN: vec4<i32>, box: vec4<f32>, misc: vec4<f32> }
@group(0) @binding(0) var<uniform> u: RasterU;
@group(0) @binding(1) var<storage, read> vals: array<f32>;
@group(0) @binding(2) var lut: texture_2d<f32>;
@group(0) @binding(3) var lutSampler: sampler;
struct VOut { @builtin(position) pos: vec4<f32>, @location(0) world: vec2<f32> }
@vertex fn vs(@builtin(vertex_index) vi: u32) -> VOut {
  // quad over the raster extent: grid box expanded by half a spacing (samples sit at cell centres)
  let a = u.gridA.xy - 0.5 * u.gridA.zw;
  let b = u.gridA.xy + vec2<f32>(f32(u.gridN.x - 1), f32(u.gridN.y - 1)) * u.gridA.zw + 0.5 * u.gridA.zw;
  var corners = array<vec2<f32>, 6>(a, vec2<f32>(b.x, a.y), vec2<f32>(a.x, b.y), vec2<f32>(a.x, b.y), vec2<f32>(b.x, a.y), b);
  let w = corners[vi];
  var o: VOut; o.world = w; o.pos = toClip(toScreen(w, u.view), u.view); return o;
}
fn readVal(i: i32, j: i32) -> f32 { return vals[(i * u.gridN.z + j * u.gridN.w) * i32(u.misc.y) + i32(u.misc.z)]; }
@fragment fn fs(in: VOut) -> @location(0) vec4<f32> {
  let p = in.world;
  if (!inClip(p, u.view) || p.x < u.box.x || p.x > u.box.z || p.y < u.box.y || p.y > u.box.w) { discard; }
  let g = (p - u.gridA.xy) / u.gridA.zw;
  var value: f32;
  if (u.misc.x > 0.5) {
    let gc = clamp(g, vec2<f32>(0.0), vec2<f32>(f32(u.gridN.x - 1), f32(u.gridN.y - 1)));
    var i0 = i32(floor(gc.x)); if (i0 >= u.gridN.x - 1) { i0 = max(0, u.gridN.x - 2); }
    var j0 = i32(floor(gc.y)); if (j0 >= u.gridN.y - 1) { j0 = max(0, u.gridN.y - 2); }
    let f = gc - vec2<f32>(f32(i0), f32(j0));
    let i1 = min(i0 + 1, u.gridN.x - 1); let j1 = min(j0 + 1, u.gridN.y - 1);
    value = (1.0 - f.x) * (1.0 - f.y) * readVal(i0, j0) + f.x * (1.0 - f.y) * readVal(i1, j0) + (1.0 - f.x) * f.y * readVal(i0, j1) + f.x * f.y * readVal(i1, j1);
  } else {
    let i = clamp(i32(round(g.x)), 0, u.gridN.x - 1); let j = clamp(i32(round(g.y)), 0, u.gridN.y - 1);
    value = readVal(i, j);
  }
  if (isnan_(value)) { discard; }
  let t = param(value, u.view.map);
  let c = textureSample(lut, lutSampler, vec2<f32>(t, 0.5));
  return vec4<f32>(c.rgb * u.misc.w, u.misc.w);
}`;

const LINES = `${COMMON}
${SEG_WGSL}
struct LineU { view: View, style: vec4<f32>, color: vec4<f32>, particles: vec4<f32> }
@group(0) @binding(0) var<uniform> u: LineU;
@group(0) @binding(1) var<storage, read> segs: array<Seg>;
@group(0) @binding(2) var lut: texture_2d<f32>;
@group(0) @binding(3) var lutSampler: sampler;
struct VOut { @builtin(position) pos: vec4<f32>, @location(0) world: vec2<f32>, @location(1) value: f32, @location(2) arc: f32, @location(3) len: f32, @location(4) phase: f32 }
@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  var o: VOut;
  if (ii >= arrayLength(&segs)) { o.pos = vec4<f32>(0.0, 0.0, 2.0, 1.0); return o; }
  let s = segs[ii];
  let sa = toScreen(s.a, u.view); let sb = toScreen(s.b, u.view);
  let d = sb - sa; let l = length(d);
  if (!(l > 0.0)) { o.pos = vec4<f32>(0.0, 0.0, 2.0, 1.0); return o; }
  let n = vec2<f32>(-d.y, d.x) / l * (0.5 * u.style.x);
  // 0:(a,-) 1:(b,-) 2:(a,+) 3:(a,+) 4:(b,-) 5:(b,+)
  let atB = (vi == 1u || vi == 4u || vi == 5u);
  let side = select(-1.0, 1.0, vi == 2u || vi == 3u || vi == 5u);
  let sp = select(sa, sb, atB) + n * side;
  o.pos = toClip(sp, u.view);
  o.world = select(s.a, s.b, atB);
  o.value = select(s.ca, s.cb, atB);
  o.arc = select(s.arc, s.arc + distance(s.a, s.b), atB);
  o.len = s.len; o.phase = s.phase;
  return o;
}
@fragment fn fs(in: VOut) -> @location(0) vec4<f32> {
  if (!inClip(in.world, u.view)) { discard; }
  var alpha = u.style.y;
  // particles: tail (world), split, travel (world), enabled
  if (u.particles.w > 0.5 && in.len > 0.0) {
    let k = u.particles.x; let split = u.particles.y;
    var bright: f32;
    if (split <= 1.0) {
      // one particle per line: head at t, tail behind it; t runs over len + k so the particle
      // enters head-first at the start of the path and its tail slides off the end
      let period = in.len + k;
      let t = (u.particles.z + in.phase * period) - floor((u.particles.z + in.phase * period) / period) * period;
      let a = in.arc - (t - k);
      if (a < 0.0 || a > k) { discard; }
      bright = a / k;
    } else {
      let t = (u.particles.z + in.phase * in.len) - floor((u.particles.z + in.phase * in.len) / in.len) * in.len;
      let span = in.len / split;
      let dd = (in.arc - t) - floor((in.arc - t) / span) * span;
      if (dd > k) { discard; }
      bright = dd / k;
    }
    if (bright <= 0.02) { discard; }
    alpha = alpha * bright;
  }
  var rgb = u.color.rgb;
  if (u.style.z > 0.5) { rgb = textureSample(lut, lutSampler, vec2<f32>(param(in.value, u.view.map), 0.5)).rgb; }
  return vec4<f32>(rgb * alpha, alpha);
}`;

export class GpuRenderer {
  private readonly ctx: GPUCanvasContext;
  private readonly format: GPUTextureFormat;
  private readonly rasterPipeline: GPURenderPipeline;
  private readonly linePipeline: GPURenderPipeline;
  private readonly sampler: GPUSampler;
  private readonly luts = new WeakMap<Lut, GPUTexture>();
  private readonly uniformPool: GPUBuffer[] = [];
  private poolIdx = 0;

  constructor(readonly backend: GpuBackend, readonly canvas: HTMLCanvasElement) {
    const dev = backend.device;
    this.ctx = canvas.getContext("webgpu") as GPUCanvasContext;
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.ctx.configure({ device: dev, format: this.format, alphaMode: "opaque" });
    this.sampler = dev.createSampler({ magFilter: "linear", minFilter: "linear", addressModeU: "clamp-to-edge" });
    const blend: GPUBlendState = { color: { srcFactor: "one", dstFactor: "one-minus-src-alpha" }, alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha" } };
    const make = (code: string) => {
      const module = dev.createShaderModule({ code });
      return dev.createRenderPipeline({
        layout: "auto",
        vertex: { module, entryPoint: "vs" },
        fragment: { module, entryPoint: "fs", targets: [{ format: this.format, blend }] },
        primitive: { topology: "triangle-list" },
      });
    };
    this.rasterPipeline = make(RASTER);
    this.linePipeline = make(LINES);
  }

  /** match the backing store to the element size */
  resize(): void {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.round(this.canvas.clientWidth * dpr)), h = Math.max(1, Math.round(this.canvas.clientHeight * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) { this.canvas.width = w; this.canvas.height = h; }
  }

  private lutTexture(lut: Lut): GPUTexture {
    let t = this.luts.get(lut);
    if (!t) {
      t = this.backend.device.createTexture({ size: [256, 1], format: "rgba8unorm", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
      this.backend.device.queue.writeTexture({ texture: t }, lut as unknown as BufferSource, { bytesPerRow: 1024 }, [256, 1]);
      this.luts.set(lut, t);
    }
    return t;
  }

  private uniform(data: Float32Array | ArrayBuffer): GPUBuffer {
    const dev = this.backend.device;
    let b = this.uniformPool[this.poolIdx];
    if (!b) this.uniformPool[this.poolIdx] = b = dev.createBuffer({ size: 256, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.poolIdx++;
    dev.queue.writeBuffer(b, 0, data as unknown as BufferSource);
    return b;
  }

  private viewBlock(view: GpuView, clip: Box, map?: ValueMap): number[] {
    return [
      view.a, view.b, view.c, view.d,
      view.cx, view.cy, view.width, view.height,
      clip.a[0]!, clip.a[1]!, clip.b[0]!, clip.b[1]!,
      map?.lo ?? 0, map?.hi ?? 1, map?.log ? 1 : 0, map?.flip ? 1 : 0,
    ];
  }

  render(scene: GpuScene): void {
    const dev = this.backend.device;
    this.poolIdx = 0;
    const [r, g, b] = scene.background;
    const enc = dev.createCommandEncoder();
    const pass = enc.beginRenderPass({ colorAttachments: [{ view: this.ctx.getCurrentTexture().createView(), clearValue: { r, g, b, a: 1 }, loadOp: "clear", storeOp: "store" }] });
    if (scene.raster) {
      const R = scene.raster, grid = R.values.grid;
      const f = new Float32Array(64);
      f.set(this.viewBlock(scene.view, scene.clip, R.map), 0);
      f.set([grid.box.a[0]!, grid.box.a[1]!, grid.spacing[0]!, grid.spacing[1]!], 16);
      new Int32Array(f.buffer).set([grid.size[0]!, grid.size[1]!, grid.strides[0]!, grid.strides[1]!], 20);
      f.set([R.box.a[0]!, R.box.a[1]!, R.box.b[0]!, R.box.b[1]!], 24);
      f.set([R.smooth ? 1 : 0, R.values.channels, R.channel ?? 0, R.alpha ?? 1], 28);
      const bg = dev.createBindGroup({
        layout: this.rasterPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.uniform(f) } },
          { binding: 1, resource: { buffer: R.values.buffer } },
          { binding: 2, resource: this.lutTexture(R.lut).createView() },
          { binding: 3, resource: this.sampler },
        ],
      });
      pass.setPipeline(this.rasterPipeline);
      pass.setBindGroup(0, bg);
      pass.draw(6);
    }
    for (const L of scene.lines) {
      const f = new Float32Array(64);
      f.set(this.viewBlock(scene.view, scene.clip, L.map), 0);
      f.set([L.width, L.alpha, L.lut && L.map ? 1 : 0, 0], 16);
      f.set([L.color[0], L.color[1], L.color[2], 1], 20);
      const P = L.particles;
      f.set([P?.tail ?? 0, P?.split ?? 1, P?.travel ?? 0, P && L.segs.particles ? 1 : 0], 24);
      const bg = dev.createBindGroup({
        layout: this.linePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.uniform(f) } },
          { binding: 1, resource: { buffer: L.segs.buffer } },
          { binding: 2, resource: this.lutTexture(L.lut ?? WHITE_LUT).createView() },
          { binding: 3, resource: this.sampler },
        ],
      });
      pass.setPipeline(this.linePipeline);
      pass.setBindGroup(0, bg);
      pass.drawIndirect(L.segs.indirect, 0);
    }
    pass.end();
    dev.queue.submit([enc.finish()]);
  }
}

const WHITE_LUT: Lut = new Uint8Array(256 * 4).fill(255);

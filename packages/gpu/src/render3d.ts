// WebGPU renderer for the 3D viewer: isosurface meshes from resident Vert sets
// (drawIndirect), lit two-sided, coloured by a value through a LUT or solid.
// Opaque layers write depth; translucent layers use weighted-blended
// order-independent transparency (McGuire & Bavoil): accumulation +
// revealage targets composited over the opaque image, depth-tested against it.
// The box outline, points and labels are left to a Canvas 2D overlay that
// projects with `Camera3D`.

import type { GpuBackend } from "./device";
import { SEG3_FLOATS, type GpuSegments3 } from "./lines3d";
import { VERT_WGSL, type GpuMesh } from "./mesh";
import type { Lut, ValueMap } from "./render";
import { SEG_FLOATS, type GpuSegments } from "./segments";

/** orbit camera: looks at `target` from `distance` along (yaw, pitch); perspective with vertical `fov` */
export interface Camera3D { target: [number, number, number]; distance: number; yaw: number; pitch: number; fov: number }

export interface GpuMeshLayer {
  mesh: GpuMesh;
  alpha: number;
  color: [number, number, number];
  /** colour by vertex value through `lut` when given */
  map?: ValueMap;
  lut?: Lut;
}
/** thick screen-space lines with depth: Seg3 records, or 2D Seg records embedded on the plane `embed.axis = embed.depth` */
export interface GpuLineLayer3D {
  segs: GpuSegments3 | GpuSegments;
  /** thick lines (default), or filled triangles read from Seg3 records (a, b = base, arc / len / phase = apex) */
  kind?: "lines" | "triangles";
  embed?: { axis: number; depth: number };
  /** css px */
  width: number;
  color: [number, number, number];
  map?: ValueMap;
  lut?: Lut;
  particles?: { tail: number; split: number; travel: number };
  /** ignore the crop planes (the box outline itself) */
  uncropped?: boolean;
}
export interface GpuScene3D {
  camera: Camera3D;
  /** the world box the camera frames (near / far planes are derived from it) */
  radius: number;
  /** the part of the canvas the camera frames, as fractions of its width / height (`[x0, y0, x1, y1]`, y down):
   *  the view is centred in it and its aspect is used, the rest of the canvas shows the periphery. Default: all of it */
  region?: [number, number, number, number];
  background: [number, number, number];
  meshes: GpuMeshLayer[];
  lines?: GpuLineLayer3D[];
  /** fragments outside [cropMin, cropMax] are discarded (crop planes); default: none */
  cropMin?: [number, number, number];
  cropMax?: [number, number, number];
}

/** column-major 4×4 helpers */
export type Mat4 = Float32Array;
export function perspective(fov: number, aspect: number, near: number, far: number): Mat4 {
  const f = 1 / Math.tan(fov / 2), m = new Float32Array(16);
  m[0] = f / aspect; m[5] = f; m[10] = far / (near - far); m[11] = -1; m[14] = (near * far) / (near - far);
  return m;
}
export function lookAt(eye: number[], target: number[], up: number[]): Mat4 {
  const z = norm3(sub3(eye, target)), x = norm3(cross3(up, z)), y = cross3(z, x);
  const m = new Float32Array(16);
  m[0] = x[0]!; m[4] = x[1]!; m[8] = x[2]!; m[12] = -dot3(x, eye);
  m[1] = y[0]!; m[5] = y[1]!; m[9] = y[2]!; m[13] = -dot3(y, eye);
  m[2] = z[0]!; m[6] = z[1]!; m[10] = z[2]!; m[14] = -dot3(z, eye);
  m[15] = 1;
  return m;
}
export function mul4(a: Mat4, b: Mat4): Mat4 {
  const o = new Float32Array(16);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) { let s = 0; for (let k = 0; k < 4; k++) s += a[k * 4 + r]! * b[c * 4 + k]!; o[c * 4 + r] = s; }
  return o;
}
const sub3 = (a: number[], b: number[]) => [a[0]! - b[0]!, a[1]! - b[1]!, a[2]! - b[2]!];
const dot3 = (a: number[], b: number[]) => a[0]! * b[0]! + a[1]! * b[1]! + a[2]! * b[2]!;
const cross3 = (a: number[], b: number[]) => [a[1]! * b[2]! - a[2]! * b[1]!, a[2]! * b[0]! - a[0]! * b[2]!, a[0]! * b[1]! - a[1]! * b[0]!];
const norm3 = (a: number[]) => { const l = Math.hypot(a[0]!, a[1]!, a[2]!) || 1; return [a[0]! / l, a[1]! / l, a[2]! / l]; };

/** eye position of an orbit camera (z up) */
export function cameraEye(c: Camera3D): [number, number, number] {
  const cp = Math.cos(c.pitch);
  return [c.target[0] + c.distance * cp * Math.cos(c.yaw), c.target[1] + c.distance * cp * Math.sin(c.yaw), c.target[2] + c.distance * Math.sin(c.pitch)];
}
/**
 * view / projection of an orbit camera for a canvas of `aspect` (width / height). With `region` (fractions of the
 * canvas, y down) the camera frames that part of the canvas instead: the perspective is built for the region's
 * aspect and then shifted / scaled in clip space so its centre lands on the region's centre (an off-centre frustum —
 * the rest of the canvas simply shows more of the scene).
 */
export function cameraMatrices(c: Camera3D, aspect: number, radius: number, region?: [number, number, number, number]): { view: Mat4; proj: Mat4; viewProj: Mat4; eye: [number, number, number] } {
  const eye = cameraEye(c);
  const near = Math.max(1e-3 * radius, c.distance - 2 * radius), far = c.distance + 2 * radius;
  const view = lookAt(eye, c.target, [0, 0, 1]);
  let proj = perspective(c.fov, aspect * regionAspect(region), near, far);
  if (region) {
    const [x0, y0, x1, y1] = region;
    const m = new Float32Array(16);
    m[0] = x1 - x0; m[5] = y1 - y0; m[10] = 1; m[15] = 1;
    m[12] = x0 + x1 - 1; m[13] = 1 - (y0 + y1); // region centre in NDC (y up)
    proj = mul4(m, proj);
  }
  return { view, proj, viewProj: mul4(proj, view), eye };
}
/** the region's width / height relative to the canvas aspect (1 for the whole canvas) */
export function regionAspect(region?: [number, number, number, number]): number {
  if (!region) return 1;
  const w = region[2] - region[0], h = region[3] - region[1];
  return w > 1e-6 && h > 1e-6 ? w / h : 1;
}
/** world point → css-pixel screen position and depth (NDC z), or undefined behind the camera */
export function project(vp: Mat4, p: ArrayLike<number>, width: number, height: number): [number, number, number] | undefined {
  const x = p[0]!, y = p[1]!, z = p[2]!;
  const cx = vp[0]! * x + vp[4]! * y + vp[8]! * z + vp[12]!, cy = vp[1]! * x + vp[5]! * y + vp[9]! * z + vp[13]!;
  const cz = vp[2]! * x + vp[6]! * y + vp[10]! * z + vp[14]!, cw = vp[3]! * x + vp[7]! * y + vp[11]! * z + vp[15]!;
  if (cw <= 1e-9) return undefined;
  return [((cx / cw) * 0.5 + 0.5) * width, (0.5 - (cy / cw) * 0.5) * height, cz / cw];
}

const MESH_COMMON = `${VERT_WGSL}
struct MeshU { viewProj: mat4x4<f32>, eye: vec4<f32>, style: vec4<f32>, color: vec4<f32>, map: vec4<f32>, crop: vec4<f32>, cropLo: vec4<f32> }
@group(0) @binding(0) var<uniform> u: MeshU;
@group(0) @binding(1) var<storage, read> verts: array<Vert>;
@group(0) @binding(2) var lut: texture_2d<f32>;
@group(0) @binding(3) var lutSampler: sampler;
fn isnan_(x: f32) -> bool { let b = bitcast<u32>(x); return (b & 0x7f800000u) == 0x7f800000u && (b & 0x007fffffu) != 0u; }
fn param(value: f32, m: vec4<f32>) -> f32 {
  let v = clamp(value, min(m.x, m.y), max(m.x, m.y));
  var t: f32;
  if (m.z > 0.5) { t = (log(v) - log(m.x)) / (log(m.y) - log(m.x)); } else { t = (v - m.x) / (m.y - m.x); }
  t = clamp(t, 0.0, 1.0);
  return select(t, 1.0 - t, m.w > 0.5);
}
struct VOut { @builtin(position) pos: vec4<f32>, @location(0) world: vec3<f32>, @location(1) normal: vec3<f32>, @location(2) value: f32 }
@vertex fn vs(@builtin(vertex_index) vi: u32) -> VOut {
  let v = verts[vi];
  var o: VOut;
  o.pos = u.viewProj * vec4<f32>(v.p, 1.0);
  o.world = v.p; o.normal = v.n; o.value = v.c;
  return o;
}
// two-sided headlight shading; style: alpha, useLut, 0, 0
fn shade(in: VOut) -> vec4<f32> {
  if (any(in.world > u.crop.xyz) || any(in.world < u.cropLo.xyz)) { discard; }
  var rgb = u.color.rgb;
  if (u.style.y > 0.5) {
    if (isnan_(in.value)) { discard; }
    let c = textureSample(lut, lutSampler, vec2<f32>(param(in.value, u.map), 0.5));
    if (c.a < 0.5) { discard; }
    rgb = c.rgb;
  }
  let toEye = normalize(u.eye.xyz - in.world);
  var n = normalize(in.normal);
  let facing = dot(n, toEye);
  if (facing < 0.0) { n = -n; }
  let diff = max(dot(n, toEye), 0.0);
  let spec = pow(diff, 24.0) * 0.25;
  // the low side (normal away from the eye) is slightly darker, so nested shells read as inside / outside
  let side = select(1.0, 0.8, facing < 0.0);
  let lit = rgb * (0.28 + 0.72 * diff) * side + vec3<f32>(spec);
  return vec4<f32>(lit, u.style.x);
}`;

// thick lines: 6 vertices per segment, offset perpendicular to the segment's screen direction by half the
// width in pixels; records are read as raw floats so one pipeline serves Seg3 (12 floats) and embedded 2D Seg
// (10 floats + the face plane in u.embed). Particles as in the 2D renderer (head at t, tail behind).
const LINES3 = `
struct LineU { viewProj: mat4x4<f32>, eye: vec4<f32>, style: vec4<f32>, color: vec4<f32>, map: vec4<f32>, crop: vec4<f32>, particles: vec4<f32>, embed: vec4<f32>, viewport: vec4<f32>, cropLo: vec4<f32> }
@group(0) @binding(0) var<uniform> u: LineU;
@group(0) @binding(1) var<storage, read> segs: array<f32>;
@group(0) @binding(2) var lut: texture_2d<f32>;
@group(0) @binding(3) var lutSampler: sampler;
fn isnan_(x: f32) -> bool { let b = bitcast<u32>(x); return (b & 0x7f800000u) == 0x7f800000u && (b & 0x007fffffu) != 0u; }
fn param(value: f32, m: vec4<f32>) -> f32 {
  let v = clamp(value, min(m.x, m.y), max(m.x, m.y));
  var t: f32;
  if (m.z > 0.5) { t = (log(v) - log(m.x)) / (log(m.y) - log(m.x)); } else { t = (v - m.x) / (m.y - m.x); }
  t = clamp(t, 0.0, 1.0);
  return select(t, 1.0 - t, m.w > 0.5);
}
// particle window brightness at arc along a line of length len with phase; -1.0 outside the window.
// particles: tail (world), split, travel (world), enabled
fn particleBright(arc: f32, len: f32, phase: f32, P: vec4<f32>) -> f32 {
  let k = P.x; let split = P.y;
  var bright: f32;
  if (split <= 1.0) {
    // one particle per line: head at t, tail behind it; t runs over len + k so the particle
    // enters head-first at the start of the path and its tail slides off the end
    let period = len + k;
    let t = (P.z + phase * period) - floor((P.z + phase * period) / period) * period;
    let a = arc - (t - k);
    if (a < 0.0 || a > k) { return -1.0; }
    bright = a / k;
  } else {
    let t = (P.z + phase * len) - floor((P.z + phase * len) / len) * len;
    let span = len / split;
    let kk = min(k, span); // a line shorter than split × tail still gets a full-brightness head
    let dd = (arc - t) - floor((arc - t) / span) * span;
    if (dd > kk) { return -1.0; }
    bright = dd / kk;
  }
  return bright;
}
struct Rec { a: vec3<f32>, b: vec3<f32>, ca: f32, cb: f32, arc: f32, len: f32, phase: f32 }
fn lift(q: vec2<f32>) -> vec3<f32> {
  let ax = i32(u.embed.x); let d = u.embed.y;
  if (ax == 0) { return vec3<f32>(d, q.x, q.y); }
  if (ax == 1) { return vec3<f32>(q.x, d, q.y); }
  return vec3<f32>(q.x, q.y, d);
}
fn record(i: u32) -> Rec {
  var r: Rec;
  if (u.embed.z > 0.5) {
    let o = i * ${SEG_FLOATS}u;
    r.a = lift(vec2<f32>(segs[o], segs[o + 1u])); r.b = lift(vec2<f32>(segs[o + 2u], segs[o + 3u]));
    r.ca = segs[o + 4u]; r.cb = segs[o + 5u]; r.arc = segs[o + 6u]; r.len = segs[o + 7u]; r.phase = segs[o + 8u];
  } else {
    let o = i * ${SEG3_FLOATS}u;
    r.a = vec3<f32>(segs[o], segs[o + 1u], segs[o + 2u]); r.ca = segs[o + 3u];
    r.b = vec3<f32>(segs[o + 4u], segs[o + 5u], segs[o + 6u]); r.cb = segs[o + 7u];
    r.arc = segs[o + 8u]; r.len = segs[o + 9u]; r.phase = segs[o + 10u];
  }
  return r;
}
struct VOut { @builtin(position) pos: vec4<f32>, @location(0) world: vec3<f32>, @location(1) value: f32, @location(2) arc: f32, @location(3) len: f32, @location(4) phase: f32 }
@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  var o: VOut;
  let s = record(ii);
  let ca = u.viewProj * vec4<f32>(s.a, 1.0); let cb = u.viewProj * vec4<f32>(s.b, 1.0);
  if (ca.w <= 1e-6 || cb.w <= 1e-6) { o.pos = vec4<f32>(0.0, 0.0, 2.0, 1.0); return o; }
  let vp = u.viewport.xy;
  let sa = ca.xy / ca.w * 0.5 * vp; let sb = cb.xy / cb.w * 0.5 * vp;
  let d = sb - sa; let l = length(d);
  var n = vec2<f32>(1.0, 0.0);
  if (l > 1e-6) { n = vec2<f32>(-d.y, d.x) / l; }
  let atB = (vi == 1u || vi == 4u || vi == 5u);
  let side = select(-1.0, 1.0, vi == 2u || vi == 3u || vi == 5u);
  let c = select(ca, cb, atB);
  let arc = select(s.arc, s.arc + distance(s.a, s.b), atB);
  var w = 0.5 * u.style.x * u.viewport.z; // half width in device px
  // particles: the width tapers with the brightness ramp, to nothing at the tail
  if (u.particles.w > 0.5 && s.len > 0.0) { w = w * max(particleBright(arc, s.len, s.phase, u.particles), 0.0); }
  let off = n * side * w;
  o.pos = vec4<f32>(c.xy + off / (0.5 * vp) * c.w, c.zw);
  o.world = select(s.a, s.b, atB);
  o.value = select(s.ca, s.cb, atB);
  o.arc = arc;
  o.len = s.len; o.phase = s.phase;
  return o;
}
@fragment fn fs(in: VOut) -> @location(0) vec4<f32> {
  if (u.crop.w > 0.5 && (any(in.world > u.crop.xyz) || any(in.world < u.cropLo.xyz))) { discard; }
  var bright = 1.0;
  if (u.particles.w > 0.5 && in.len > 0.0) {
    bright = particleBright(in.arc, in.len, in.phase, u.particles);
    if (bright <= 0.02) { discard; }
  }
  var rgb = u.color.rgb;
  if (u.style.z > 0.5) {
    if (isnan_(in.value)) { discard; }
    let c = textureSample(lut, lutSampler, vec2<f32>(param(in.value, u.map), 0.5));
    if (c.a < 0.5) { discard; }
    rgb = c.rgb;
  }
  return vec4<f32>(rgb * bright, 1.0);
}`;

// filled triangles from Seg3 records (glyphs): a, b = the base, (arc, len, phase) = the apex; two halves per instance
// (apex-a-mid, apex-mid-b) so the six vertices cover the triangle once. Same uniform block as the lines.
const TRIANGLES3 = `
struct LineU { viewProj: mat4x4<f32>, eye: vec4<f32>, style: vec4<f32>, color: vec4<f32>, map: vec4<f32>, crop: vec4<f32>, particles: vec4<f32>, embed: vec4<f32>, viewport: vec4<f32>, cropLo: vec4<f32> }
@group(0) @binding(0) var<uniform> u: LineU;
@group(0) @binding(1) var<storage, read> segs: array<f32>;
@group(0) @binding(2) var lut: texture_2d<f32>;
@group(0) @binding(3) var lutSampler: sampler;
fn isnan_(x: f32) -> bool { let b = bitcast<u32>(x); return (b & 0x7f800000u) == 0x7f800000u && (b & 0x007fffffu) != 0u; }
fn param(value: f32, m: vec4<f32>) -> f32 {
  let v = clamp(value, min(m.x, m.y), max(m.x, m.y));
  var t: f32;
  if (m.z > 0.5) { t = (log(v) - log(m.x)) / (log(m.y) - log(m.x)); } else { t = (v - m.x) / (m.y - m.x); }
  t = clamp(t, 0.0, 1.0);
  return select(t, 1.0 - t, m.w > 0.5);
}
struct VOut { @builtin(position) pos: vec4<f32>, @location(0) world: vec3<f32>, @location(1) value: f32 }
@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  let o = ii * ${SEG3_FLOATS}u;
  let a = vec3<f32>(segs[o], segs[o + 1u], segs[o + 2u]); let b = vec3<f32>(segs[o + 4u], segs[o + 5u], segs[o + 6u]);
  let apex = vec3<f32>(segs[o + 8u], segs[o + 9u], segs[o + 10u]); let m = 0.5 * (a + b);
  var p = apex;
  if (vi == 1u) { p = a; } else if (vi == 2u || vi == 4u) { p = m; } else if (vi == 5u) { p = b; }
  var out: VOut;
  out.pos = u.viewProj * vec4<f32>(p, 1.0);
  out.world = p; out.value = segs[o + 3u];
  return out;
}
@fragment fn fs(in: VOut) -> @location(0) vec4<f32> {
  if (u.crop.w > 0.5 && (any(in.world > u.crop.xyz) || any(in.world < u.cropLo.xyz))) { discard; }
  var rgb = u.color.rgb;
  if (u.style.z > 0.5) {
    if (isnan_(in.value)) { discard; }
    let c = textureSample(lut, lutSampler, vec2<f32>(param(in.value, u.map), 0.5));
    if (c.a < 0.5) { discard; }
    rgb = c.rgb;
  }
  return vec4<f32>(rgb, 1.0);
}`;

const OPAQUE = `${MESH_COMMON}
@fragment fn fs(in: VOut) -> @location(0) vec4<f32> { let c = shade(in); return vec4<f32>(c.rgb, 1.0); }`;

// weighted-blended OIT: accum += (rgb * a, a) * w(z, a); reveal *= (1 - a)
const TRANSPARENT = `${MESH_COMMON}
struct FOut { @location(0) accum: vec4<f32>, @location(1) reveal: f32 }
@fragment fn fs(in: VOut) -> FOut {
  let c = shade(in);
  let a = c.a;
  let z = in.pos.z; // 0..1, near = 0
  let w = clamp(a * 10.0 * (1.0 - z * 0.99) * (1.0 - z * 0.99), 1e-2, 3e3);
  var o: FOut;
  o.accum = vec4<f32>(c.rgb * a, a) * w;
  o.reveal = a;
  return o;
}`;

const COMPOSITE = `
@group(0) @binding(0) var accum: texture_2d<f32>;
@group(0) @binding(1) var reveal: texture_2d<f32>;
struct VOut { @builtin(position) pos: vec4<f32> }
@vertex fn vs(@builtin(vertex_index) vi: u32) -> VOut {
  var o: VOut;
  let x = f32(i32(vi & 1u) * 4 - 1); let y = f32(i32(vi >> 1u) * 4 - 1);
  o.pos = vec4<f32>(x, y, 0.0, 1.0);
  return o;
}
@fragment fn fs(in: VOut) -> @location(0) vec4<f32> {
  let xy = vec2<i32>(in.pos.xy);
  let acc = textureLoad(accum, xy, 0);
  let r = textureLoad(reveal, xy, 0).r; // revealage of the background: product of (1 - a)
  if (acc.a < 1e-5) { return vec4<f32>(0.0, 0.0, 0.0, 0.0); }
  let rgb = acc.rgb / max(acc.a, 1e-5);
  let alpha = 1.0 - r;
  return vec4<f32>(rgb * alpha, alpha); // premultiplied, blended over the opaque image
}`;

export class GpuRenderer3D {
  private readonly ctx: GPUCanvasContext;
  private readonly format: GPUTextureFormat;
  private readonly opaque: GPURenderPipeline;
  private readonly transparent: GPURenderPipeline;
  private readonly composite: GPURenderPipeline;
  private readonly lines: GPURenderPipeline;
  private readonly triangles: GPURenderPipeline;
  private readonly sampler: GPUSampler;
  private readonly luts = new WeakMap<Lut, GPUTexture>();
  private readonly uniformPool: GPUBuffer[] = [];
  private poolIdx = 0;
  private targets: { w: number; h: number; depth: GPUTexture; accum: GPUTexture; reveal: GPUTexture } | undefined;
  /** the view-projection of the last frame, for the overlay */
  viewProj: Mat4 = new Float32Array(16);

  constructor(readonly backend: GpuBackend, readonly canvas: HTMLCanvasElement) {
    const dev = backend.device;
    this.ctx = canvas.getContext("webgpu") as GPUCanvasContext;
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.ctx.configure({ device: dev, format: this.format, alphaMode: "opaque" });
    this.sampler = dev.createSampler({ magFilter: "linear", minFilter: "linear", addressModeU: "clamp-to-edge" });
    const opaqueMod = dev.createShaderModule({ code: OPAQUE });
    this.opaque = dev.createRenderPipeline({
      layout: "auto",
      vertex: { module: opaqueMod, entryPoint: "vs" },
      fragment: { module: opaqueMod, entryPoint: "fs", targets: [{ format: this.format }] },
      primitive: { topology: "triangle-list", cullMode: "none" },
      depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" },
    });
    const transMod = dev.createShaderModule({ code: TRANSPARENT });
    this.transparent = dev.createRenderPipeline({
      layout: "auto",
      vertex: { module: transMod, entryPoint: "vs" },
      fragment: {
        module: transMod, entryPoint: "fs",
        targets: [
          { format: "rgba16float", blend: { color: { srcFactor: "one", dstFactor: "one" }, alpha: { srcFactor: "one", dstFactor: "one" } } },
          { format: "r16float", blend: { color: { srcFactor: "zero", dstFactor: "one-minus-src" }, alpha: { srcFactor: "zero", dstFactor: "one-minus-src" } } },
        ],
      },
      primitive: { topology: "triangle-list", cullMode: "none" },
      depthStencil: { format: "depth24plus", depthWriteEnabled: false, depthCompare: "less" },
    });
    const lineMod = dev.createShaderModule({ code: LINES3 });
    this.lines = dev.createRenderPipeline({
      layout: "auto",
      vertex: { module: lineMod, entryPoint: "vs" },
      fragment: { module: lineMod, entryPoint: "fs", targets: [{ format: this.format }] },
      primitive: { topology: "triangle-list", cullMode: "none" },
      depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" },
    });
    const triMod = dev.createShaderModule({ code: TRIANGLES3 });
    this.triangles = dev.createRenderPipeline({
      layout: "auto",
      vertex: { module: triMod, entryPoint: "vs" },
      fragment: { module: triMod, entryPoint: "fs", targets: [{ format: this.format }] },
      primitive: { topology: "triangle-list", cullMode: "none" },
      depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" },
    });
    const compMod = dev.createShaderModule({ code: COMPOSITE });
    this.composite = dev.createRenderPipeline({
      layout: "auto",
      vertex: { module: compMod, entryPoint: "vs" },
      fragment: { module: compMod, entryPoint: "fs", targets: [{ format: this.format, blend: { color: { srcFactor: "one", dstFactor: "one-minus-src-alpha" }, alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha" } } }] },
      primitive: { topology: "triangle-list" },
    });
  }

  resize(): void {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.round(this.canvas.clientWidth * dpr)), h = Math.max(1, Math.round(this.canvas.clientHeight * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) { this.canvas.width = w; this.canvas.height = h; }
  }

  private ensureTargets(w: number, h: number) {
    if (this.targets && this.targets.w === w && this.targets.h === h) return this.targets;
    if (this.targets) { this.targets.depth.destroy(); this.targets.accum.destroy(); this.targets.reveal.destroy(); }
    const dev = this.backend.device;
    const usage = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING;
    this.targets = {
      w, h,
      depth: dev.createTexture({ size: [w, h], format: "depth24plus", usage: GPUTextureUsage.RENDER_ATTACHMENT }),
      accum: dev.createTexture({ size: [w, h], format: "rgba16float", usage }),
      reveal: dev.createTexture({ size: [w, h], format: "r16float", usage }),
    };
    return this.targets;
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

  private uniform(data: Float32Array): GPUBuffer {
    const dev = this.backend.device;
    let b = this.uniformPool[this.poolIdx];
    if (!b) this.uniformPool[this.poolIdx] = b = dev.createBuffer({ size: 256, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.poolIdx++;
    dev.queue.writeBuffer(b, 0, data as unknown as BufferSource);
    return b;
  }

  private bind(pipeline: GPURenderPipeline, L: GpuMeshLayer, viewProj: Mat4, eye: number[], crop: number[], cropLo: number[]): GPUBindGroup {
    const f = new Float32Array(64);
    f.set(viewProj, 0);
    f.set([eye[0]!, eye[1]!, eye[2]!, 0], 16);
    f.set([L.alpha, L.lut && L.map ? 1 : 0, 0, 0], 20);
    f.set([L.color[0], L.color[1], L.color[2], 1], 24);
    f.set([L.map?.lo ?? 0, L.map?.hi ?? 1, L.map?.log ? 1 : 0, L.map?.flip ? 1 : 0], 28);
    f.set([crop[0]!, crop[1]!, crop[2]!, 1], 32);
    f.set([cropLo[0]!, cropLo[1]!, cropLo[2]!, 0], 36);
    return this.backend.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniform(f) } },
        { binding: 1, resource: { buffer: L.mesh.buffer } },
        { binding: 2, resource: this.lutTexture(L.lut ?? WHITE_LUT).createView() },
        { binding: 3, resource: this.sampler },
      ],
    });
  }

  private bindLines(L: GpuLineLayer3D, viewProj: Mat4, eye: number[], crop: number[], cropLo: number[], w: number, h: number): GPUBindGroup {
    const f = new Float32Array(64);
    f.set(viewProj, 0);
    f.set([eye[0]!, eye[1]!, eye[2]!, 0], 16);
    f.set([L.width, 1, L.lut && L.map ? 1 : 0, 0], 20);
    f.set([L.color[0], L.color[1], L.color[2], 1], 24);
    f.set([L.map?.lo ?? 0, L.map?.hi ?? 1, L.map?.log ? 1 : 0, L.map?.flip ? 1 : 0], 28);
    f.set([crop[0]!, crop[1]!, crop[2]!, L.uncropped ? 0 : 1], 32);
    const P = L.particles;
    f.set([P?.tail ?? 0, P?.split ?? 1, P?.travel ?? 0, P && L.segs.particles ? 1 : 0], 36);
    f.set([L.embed?.axis ?? 0, L.embed?.depth ?? 0, L.embed ? 1 : 0, 0], 40);
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    f.set([w, h, dpr, 0], 44);
    f.set([cropLo[0]!, cropLo[1]!, cropLo[2]!, 0], 48);
    return this.backend.device.createBindGroup({
      layout: (L.kind === "triangles" ? this.triangles : this.lines).getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniform(f) } },
        { binding: 1, resource: { buffer: L.segs.buffer } },
        { binding: 2, resource: this.lutTexture(L.lut ?? WHITE_LUT).createView() },
        { binding: 3, resource: this.sampler },
      ],
    });
  }

  render(scene: GpuScene3D): void {
    const dev = this.backend.device;
    this.poolIdx = 0;
    const w = this.canvas.width, h = this.canvas.height;
    const T = this.ensureTargets(w, h);
    const { viewProj, eye } = cameraMatrices(scene.camera, w / h, scene.radius, scene.region);
    this.viewProj = viewProj;
    const crop = scene.cropMax ?? [Infinity, Infinity, Infinity], cropLo = scene.cropMin ?? [-Infinity, -Infinity, -Infinity];
    const [r, g, b] = scene.background;
    const opaque = scene.meshes.filter((m) => m.alpha >= 0.999), trans = scene.meshes.filter((m) => m.alpha < 0.999);
    const enc = dev.createCommandEncoder();
    const colour = this.ctx.getCurrentTexture().createView();
    const depth = T.depth.createView();
    // opaque pass (also clears colour and depth)
    const p1 = enc.beginRenderPass({
      colorAttachments: [{ view: colour, clearValue: { r, g, b, a: 1 }, loadOp: "clear", storeOp: "store" }],
      depthStencilAttachment: { view: depth, depthClearValue: 1, depthLoadOp: "clear", depthStoreOp: "store" },
    });
    for (const L of scene.lines ?? []) { p1.setPipeline(L.kind === "triangles" ? this.triangles : this.lines); p1.setBindGroup(0, this.bindLines(L, viewProj, eye, crop, cropLo, w, h)); p1.drawIndirect(L.segs.indirect, 0); }
    for (const L of opaque) { p1.setPipeline(this.opaque); p1.setBindGroup(0, this.bind(this.opaque, L, viewProj, eye, crop, cropLo)); p1.drawIndirect(L.mesh.indirect, 0); }
    p1.end();
    if (trans.length) {
      const p2 = enc.beginRenderPass({
        colorAttachments: [
          { view: T.accum.createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: "clear", storeOp: "store" },
          { view: T.reveal.createView(), clearValue: { r: 1, g: 0, b: 0, a: 0 }, loadOp: "clear", storeOp: "store" },
        ],
        depthStencilAttachment: { view: depth, depthLoadOp: "load", depthStoreOp: "store" },
      });
      for (const L of trans) { p2.setPipeline(this.transparent); p2.setBindGroup(0, this.bind(this.transparent, L, viewProj, eye, crop, cropLo)); p2.drawIndirect(L.mesh.indirect, 0); }
      p2.end();
      const p3 = enc.beginRenderPass({ colorAttachments: [{ view: colour, loadOp: "load", storeOp: "store" }] });
      p3.setPipeline(this.composite);
      p3.setBindGroup(0, dev.createBindGroup({ layout: this.composite.getBindGroupLayout(0), entries: [{ binding: 0, resource: T.accum.createView() }, { binding: 1, resource: T.reveal.createView() }] }));
      p3.draw(3);
      p3.end();
    }
    dev.queue.submit([enc.finish()]);
  }
}

const WHITE_LUT: Lut = new Uint8Array(1024).fill(255);

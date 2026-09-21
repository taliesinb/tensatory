// WebGPU renderer for the 3D viewer: isosurface meshes from resident Vert sets
// (drawIndirect), lit two-sided, coloured by a value through a LUT or solid.
// Opaque layers write depth; translucent layers use weighted-blended
// order-independent transparency (McGuire & Bavoil): accumulation +
// revealage targets composited over the opaque image, depth-tested against it.
// The box outline, points and labels are left to a Canvas 2D overlay that
// projects with `Camera3D`.

import type { Box } from "@tensatory/core";
import type { GpuBackend } from "./device";
import { SEG3_FLOATS, type GpuSegments3 } from "./lines3d";
import { VERT_WGSL, type GpuMesh } from "./mesh";
import type { Lut, ValueMap } from "./render";
import type { GpuGrid } from "./resident";
import { SEG_FLOATS, type GpuSegments } from "./segments";

/** unit quaternion [x, y, z, w] */
export type Quat = [number, number, number, number];
/**
 * orbit camera: looks at `target` from `distance`, oriented by `rot` — the rotation taking the camera frame
 * (x right, y up, z towards the eye: the camera looks along −z) to world; perspective with vertical `fov`.
 * A quaternion rather than yaw / pitch so the camera can roll over the poles (no gimbal lock, no clamp).
 */
export interface Camera3D { target: [number, number, number]; distance: number; rot: Quat; fov: number }

/** q · p (apply p first, then q) */
export function quatMul(q: Quat, p: Quat): Quat {
  const [qx, qy, qz, qw] = q, [px, py, pz, pw] = p;
  return [qw * px + qx * pw + qy * pz - qz * py, qw * py - qx * pz + qy * pw + qz * px, qw * pz + qx * py - qy * px + qz * pw, qw * pw - qx * px - qy * py - qz * pz];
}
/** rotation by `angle` (radians, right-handed) about `axis` (any length; a zero axis gives the identity) */
export function quatAxisAngle(axis: ArrayLike<number>, angle: number): Quat {
  const l = Math.hypot(axis[0]!, axis[1]!, axis[2]!);
  if (l < 1e-12) return [0, 0, 0, 1];
  const s = Math.sin(angle / 2) / l;
  return [axis[0]! * s, axis[1]! * s, axis[2]! * s, Math.cos(angle / 2)];
}
export function quatNormalize(q: Quat): Quat {
  const l = Math.hypot(q[0], q[1], q[2], q[3]) || 1;
  return [q[0] / l, q[1] / l, q[2] / l, q[3] / l];
}
/** rotate vector `v` by `q` */
export function quatRotate(q: Quat, v: ArrayLike<number>): [number, number, number] {
  const [x, y, z, w] = q, vx = v[0]!, vy = v[1]!, vz = v[2]!;
  // t = 2 (q_v × v); v' = v + w t + q_v × t
  const tx = 2 * (y * vz - z * vy), ty = 2 * (z * vx - x * vz), tz = 2 * (x * vy - y * vx);
  return [vx + w * tx + y * tz - z * ty, vy + w * ty + z * tx - x * tz, vz + w * tz + x * ty - y * tx];
}
/**
 * the orientation of a camera whose eye lies along `dir` from the target (world) with `up` as near vertical as
 * possible on screen; when `dir` is parallel to `up`, `altUp` (default world y) takes its place.
 */
export function quatLook(dir: ArrayLike<number>, up: ArrayLike<number> = [0, 0, 1], altUp: ArrayLike<number> = [0, 1, 0]): Quat {
  const z = norm3([dir[0]!, dir[1]!, dir[2]!]);
  let x = cross3([up[0]!, up[1]!, up[2]!], z);
  if (Math.hypot(x[0]!, x[1]!, x[2]!) < 1e-6) x = cross3([altUp[0]!, altUp[1]!, altUp[2]!], z);
  x = norm3(x);
  const y = cross3(z, x);
  // rotation matrix with columns x, y, z → quaternion (Shepperd's method)
  const m00 = x[0]!, m10 = x[1]!, m20 = x[2]!, m01 = y[0]!, m11 = y[1]!, m21 = y[2]!, m02 = z[0]!, m12 = z[1]!, m22 = z[2]!;
  const tr = m00 + m11 + m22;
  let q: Quat;
  if (tr > 0) { const s = Math.sqrt(tr + 1) * 2; q = [(m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, s / 4]; }
  else if (m00 > m11 && m00 > m22) { const s = Math.sqrt(1 + m00 - m11 - m22) * 2; q = [s / 4, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s]; }
  else if (m11 > m22) { const s = Math.sqrt(1 + m11 - m00 - m22) * 2; q = [(m01 + m10) / s, s / 4, (m12 + m21) / s, (m02 - m20) / s]; }
  else { const s = Math.sqrt(1 + m22 - m00 - m11) * 2; q = [(m02 + m20) / s, (m12 + m21) / s, s / 4, (m10 - m01) / s]; }
  return quatNormalize(q);
}
/** the camera's world-space axes: right, up, back (towards the eye) */
export function cameraAxes(c: Camera3D): { right: [number, number, number]; up: [number, number, number]; back: [number, number, number] } {
  return { right: quatRotate(c.rot, [1, 0, 0]), up: quatRotate(c.rot, [0, 1, 0]), back: quatRotate(c.rot, [0, 0, 1]) };
}

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
  /** cones (kind "triangles") whose projected size is under this many css px are not drawn (default 0) */
  minPx?: number;
  /** thick lines (default), or "triangle" records (a, b = base, arc / len / phase = apex) drawn as CONES: the
   *  triangle's solid of revolution, ray-cast per fragment with true depth and a subtle headlight */
  kind?: "lines" | "triangles";
  embed?: { axis: number; depth: number };
  /** css px */
  width: number;
  color: [number, number, number];
  /** opacity (default 1). Below 1 the lines join the weighted-blended OIT pass of the translucent meshes (no
   *  sorting), and a particle tail fades in OPACITY from `alpha` at the head to 0 instead of tapering in width. */
  alpha?: number;
  map?: ValueMap;
  lut?: Lut;
  particles?: { tail: number; split: number; travel: number };
  /** ignore the crop planes (the box outline itself) */
  uncropped?: boolean;
  /** pull the line towards the eye by this much NDC depth (default 0): for lines lying ON a surface that is also
   *  drawn (the colorfield planes' intersections), which would otherwise fight it for the depth buffer */
  depthBias?: number;
}
/**
 * a colormapped raster on the axis-aligned plane `axis = depth` (the 3D colorfield): the 2D twin of the 2D
 * renderer's raster — a resident 2D grid over the plane's two other axes (ascending order, as `GpuLineLayer3D.embed`
 * lifts them), bilinear or nearest, through the LUT, clipped to `box` (the field's) and the crop planes. Flat
 * (unlit: its colours must read against the legend). Opaque (the default): depth-tested and written, so shells
 * behind it hide and translucent shells in front composite over it; with `alpha` < 1 it joins the weighted-blended
 * OIT pass of the translucent shells and lines (depth-tested, no write).
 */
export interface GpuPlaneLayer3D {
  /** opacity (default 1) */
  alpha?: number;
  values: GpuGrid;
  channel?: number;
  axis: number;
  depth: number;
  /** the field's box: the quad is clipped to it in the plane's two axes */
  box: Box;
  map: ValueMap;
  lut: Lut;
  smooth: boolean;
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
  /** colorfield planes (opaque) */
  planes?: GpuPlaneLayer3D[];
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

/** eye position of an orbit camera */
export function cameraEye(c: Camera3D): [number, number, number] {
  const b = quatRotate(c.rot, [0, 0, 1]);
  return [c.target[0] + c.distance * b[0], c.target[1] + c.distance * b[1], c.target[2] + c.distance * b[2]];
}
/** world → camera: the inverse of the camera's rigid transform (rows = its axes) */
function cameraView(c: Camera3D, eye: number[]): Mat4 {
  const { right: x, up: y, back: z } = cameraAxes(c);
  const m = new Float32Array(16);
  m[0] = x[0]; m[4] = x[1]; m[8] = x[2]; m[12] = -dot3(x, eye);
  m[1] = y[0]; m[5] = y[1]; m[9] = y[2]; m[13] = -dot3(y, eye);
  m[2] = z[0]; m[6] = z[1]; m[10] = z[2]; m[14] = -dot3(z, eye);
  m[15] = 1;
  return m;
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
  const view = cameraView(c, eye);
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
  // particles (opaque lines): the width tapers with the brightness ramp, to nothing at the tail; translucent lines
  // keep their width and fade in opacity instead (fs)
  if (u.particles.w > 0.5 && s.len > 0.0 && u.style.y < 0.5) { w = w * max(particleBright(arc, s.len, s.phase, u.particles), 0.0); }
  let off = n * side * w;
  o.pos = vec4<f32>(c.xy + off / (0.5 * vp) * c.w, c.z - u.embed.w * c.w, c.w); // embed.w: depth bias (NDC)
  o.world = select(s.a, s.b, atB);
  o.value = select(s.ca, s.cb, atB);
  o.arc = arc;
  o.len = s.len; o.phase = s.phase;
  return o;
}
// colour and particle brightness of a fragment (shared by the opaque and the translucent entry points)
fn lineColour(in: VOut) -> vec4<f32> {
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
  return vec4<f32>(rgb, bright);
}
@fragment fn fs(in: VOut) -> @location(0) vec4<f32> {
  let c = lineColour(in);
  return vec4<f32>(c.rgb * c.a, 1.0); // the tail fades to black
}
// translucent: weighted-blended OIT like the meshes (accum, reveal); the tail fades in opacity, full colour
struct FOut { @location(0) accum: vec4<f32>, @location(1) reveal: f32 }
@fragment fn fsTrans(in: VOut) -> FOut {
  let c = lineColour(in);
  let a = u.style.w * c.a;
  let z = in.pos.z;
  let w = clamp(a * 10.0 * (1.0 - z * 0.99) * (1.0 - z * 0.99), 1e-2, 3e3);
  var o: FOut;
  o.accum = vec4<f32>(c.rgb * a, a) * w;
  o.reveal = a;
  return o;
}`;

// Cones from Seg3 "triangle" records (glyphs): a, b = the base's ends, (arc, len, phase) = the apex; the record's
// triangle is drawn as its solid of revolution about the axis apex → base centre. Ray-cast impostors: the vertex
// shader emits a camera-facing quad covering the cone's bounding sphere (six vertices, one instance per glyph),
// the fragment shader intersects the eye ray with the finite cone and its base disc, writes the true depth and
// shades the analytic normal with a subtle headlight. Exact silhouettes, no tessellation.
const CONES3 = `
// viewport: width, height (device px), dpr, minPx (css px: cones whose projected bounding sphere is smaller are culled)
struct ConeU { viewProj: mat4x4<f32>, eye: vec4<f32>, style: vec4<f32>, color: vec4<f32>, map: vec4<f32>, crop: vec4<f32>, cropLo: vec4<f32>, viewport: vec4<f32> }
@group(0) @binding(0) var<uniform> u: ConeU;
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
struct VOut { @builtin(position) pos: vec4<f32>, @location(0) world: vec3<f32>, @location(1) value: f32, @location(2) apex: vec3<f32>, @location(3) axis: vec3<f32>, @location(4) dims: vec2<f32> }
@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  var out: VOut;
  let o = ii * ${SEG3_FLOATS}u;
  let a = vec3<f32>(segs[o], segs[o + 1u], segs[o + 2u]); let b = vec3<f32>(segs[o + 4u], segs[o + 5u], segs[o + 6u]);
  let apex = vec3<f32>(segs[o + 8u], segs[o + 9u], segs[o + 10u]);
  let base = 0.5 * (a + b); let r = 0.5 * distance(a, b);
  let ax = base - apex; let h = length(ax);
  if (!(h > 0.0) || !(r > 0.0)) { out.pos = vec4<f32>(0.0, 0.0, 2.0, 1.0); return out; }
  // bounding sphere of the cone, and the plane of its silhouette as seen from the eye: a square there covers it
  let centre = 0.5 * (apex + base); let R = 1.02 * sqrt(0.25 * h * h + r * r);
  let toC = centre - u.eye.xyz; let D = length(toC);
  if (D <= R) { out.pos = vec4<f32>(0.0, 0.0, 2.0, 1.0); return out; }
  let v = toC / D;
  let plane = u.eye.xyz + v * (D - R * R / D);
  let rp = R * sqrt(D * D - R * R) / D;
  var upw = vec3<f32>(0.0, 0.0, 1.0);
  if (abs(v.z) > 0.9) { upw = vec3<f32>(1.0, 0.0, 0.0); }
  let right = normalize(cross(v, upw)); let up = cross(right, v);
  // screen size of the silhouette: below the cutoff the glyph is noise, not drawn
  let c0 = u.viewProj * vec4<f32>(plane, 1.0); let c1 = u.viewProj * vec4<f32>(plane + rp * right, 1.0);
  if (c0.w <= 1e-6 || c1.w <= 1e-6) { out.pos = vec4<f32>(0.0, 0.0, 2.0, 1.0); return out; }
  let px = length((c1.xy / c1.w - c0.xy / c0.w) * 0.5 * u.viewport.xy) / u.viewport.z;
  if (2.0 * px < u.viewport.w) { out.pos = vec4<f32>(0.0, 0.0, 2.0, 1.0); return out; }
  // 0:(-,-) 1:(+,-) 2:(-,+) 3:(-,+) 4:(+,-) 5:(+,+)
  let sx = select(-1.0, 1.0, vi == 1u || vi == 4u || vi == 5u);
  let sy = select(-1.0, 1.0, vi == 2u || vi == 3u || vi == 5u);
  let p = plane + rp * (sx * right + sy * up);
  out.pos = u.viewProj * vec4<f32>(p, 1.0);
  out.world = p; out.value = segs[o + 3u];
  out.apex = apex; out.axis = ax / h; out.dims = vec2<f32>(h, r);
  return out;
}
struct FOut { @location(0) color: vec4<f32>, @builtin(frag_depth) depth: f32 }
@fragment fn fs(in: VOut) -> FOut {
  let O = u.eye.xyz; let Dr = normalize(in.world - O);
  let A = in.apex; let d = in.axis; let h = in.dims.x; let r = in.dims.y;
  // finite cone: points X with t = (X − A)·d ∈ [0, h] and |X − A − t d| = t r / h  ⇔  ((X−A)·d)² = cos²α |X−A|²
  let cos2 = h * h / (h * h + r * r);
  let v = O - A;
  let dd = dot(Dr, d); let vd = dot(v, d);
  let qa = dd * dd - cos2; let qb = 2.0 * (dd * vd - cos2 * dot(Dr, v)); let qc = vd * vd - cos2 * dot(v, v);
  var best = 1e30; var n = vec3<f32>(0.0);
  let disc = qb * qb - 4.0 * qa * qc;
  if (disc >= 0.0 && abs(qa) > 1e-12) {
    let sq = sqrt(disc);
    for (var k = 0; k < 2; k++) {
      let s = select((-qb + sq) / (2.0 * qa), (-qb - sq) / (2.0 * qa), k == 0);
      if (s <= 0.0 || s >= best) { continue; }
      let X = O + s * Dr; let q = X - A; let t = dot(q, d);
      if (t < 0.0 || t > h) { continue; }
      let radial = q - t * d; let rl = length(radial);
      if (rl > 0.0) { best = s; n = normalize(radial / rl * h - d * r); }
    }
  }
  // the base disc
  if (abs(dd) > 1e-9) {
    let s = dot(A + h * d - O, d) / dd;
    if (s > 0.0 && s < best) { let X = O + s * Dr; if (distance(X, A + h * d) <= r) { best = s; n = d; } }
  }
  if (best >= 1e30) { discard; }
  let X = O + best * Dr;
  if (u.crop.w > 0.5 && (any(X > u.crop.xyz) || any(X < u.cropLo.xyz))) { discard; }
  var rgb = u.color.rgb;
  if (u.style.z > 0.5) {
    if (isnan_(in.value)) { discard; }
    let c = textureSample(lut, lutSampler, vec2<f32>(param(in.value, u.map), 0.5));
    if (c.a < 0.5) { discard; }
    rgb = c.rgb;
  }
  // a key light between the eye and above-left of the scene, so every cone shows a lit and a shaded side
  let l = normalize(-Dr + vec3<f32>(0.35, 0.25, 0.8));
  let diff = max(dot(n, l), 0.0);
  let spec = pow(max(dot(n, normalize(l - Dr)), 0.0), 24.0);
  let lit = rgb * (0.3 + 0.7 * diff) + vec3<f32>(0.18 * spec);
  let clip = u.viewProj * vec4<f32>(X, 1.0);
  var o: FOut;
  o.color = vec4<f32>(lit, 1.0);
  o.depth = clip.z / clip.w;
  return o;
}`;

// The colorfield plane: a quad over the field's box on the plane `axis = depth`, coloured from a resident 2D grid
// (the plane's two other axes in ascending order) exactly as the 2D renderer's raster — bilinear or nearest, LUT,
// alpha < ½ = masked. Flat: no lighting, so the colours read against the legend. Depth-tested and written.
const PLANE3 = `
struct PlaneU { viewProj: mat4x4<f32>, gridA: vec4<f32>, gridN: vec4<i32>, box: vec4<f32>, map: vec4<f32>, crop: vec4<f32>, cropLo: vec4<f32>, embed: vec4<f32>, misc: vec4<f32> }
@group(0) @binding(0) var<uniform> u: PlaneU;
@group(0) @binding(1) var<storage, read> vals: array<f32>;
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
fn lift(q: vec2<f32>) -> vec3<f32> {
  let ax = i32(u.embed.x); let d = u.embed.y;
  if (ax == 0) { return vec3<f32>(d, q.x, q.y); }
  if (ax == 1) { return vec3<f32>(q.x, d, q.y); }
  return vec3<f32>(q.x, q.y, d);
}
struct VOut { @builtin(position) pos: vec4<f32>, @location(0) plane: vec2<f32>, @location(1) world: vec3<f32> }
@vertex fn vs(@builtin(vertex_index) vi: u32) -> VOut {
  let a = u.box.xy; let b = u.box.zw;
  var corners = array<vec2<f32>, 6>(a, vec2<f32>(b.x, a.y), vec2<f32>(a.x, b.y), vec2<f32>(a.x, b.y), vec2<f32>(b.x, a.y), b);
  let q = corners[vi];
  var o: VOut;
  o.plane = q; o.world = lift(q);
  o.pos = u.viewProj * vec4<f32>(o.world, 1.0);
  return o;
}
fn readVal(i: i32, j: i32) -> f32 { return vals[(i * u.gridN.z + j * u.gridN.w) * i32(u.misc.y) + i32(u.misc.z)]; }
// the colour of a fragment (shared by the opaque and the translucent entry points); discards what is not drawn
fn planeColour(in: VOut) -> vec3<f32> {
  if (u.crop.w > 0.5 && (any(in.world > u.crop.xyz) || any(in.world < u.cropLo.xyz))) { discard; }
  let g = (in.plane - u.gridA.xy) / u.gridA.zw;
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
  let c = textureSample(lut, lutSampler, vec2<f32>(param(value, u.map), 0.5));
  if (c.a < 0.5) { discard; } // masked by the colormap interval selection
  return c.rgb;
}
@fragment fn fs(in: VOut) -> @location(0) vec4<f32> { return vec4<f32>(planeColour(in), 1.0); }
// translucent: weighted-blended OIT like the meshes and lines (accum, reveal); misc.w = alpha
struct FOut { @location(0) accum: vec4<f32>, @location(1) reveal: f32 }
@fragment fn fsTrans(in: VOut) -> FOut {
  let rgb = planeColour(in);
  let a = u.misc.w;
  let z = in.pos.z;
  let w = clamp(a * 10.0 * (1.0 - z * 0.99) * (1.0 - z * 0.99), 1e-2, 3e3);
  var o: FOut;
  o.accum = vec4<f32>(rgb * a, a) * w;
  o.reveal = a;
  return o;
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
  private readonly linesTrans: GPURenderPipeline;
  private readonly transparent: GPURenderPipeline;
  private readonly composite: GPURenderPipeline;
  private readonly lines: GPURenderPipeline;
  private readonly cones: GPURenderPipeline;
  private readonly planes: GPURenderPipeline;
  private readonly planesTrans: GPURenderPipeline;
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
    // translucent lines: the meshes' OIT targets and blend state, depth-tested against the opaque image, no write
    this.linesTrans = dev.createRenderPipeline({
      layout: "auto",
      vertex: { module: lineMod, entryPoint: "vs" },
      fragment: {
        module: lineMod, entryPoint: "fsTrans",
        targets: [
          { format: "rgba16float", blend: { color: { srcFactor: "one", dstFactor: "one" }, alpha: { srcFactor: "one", dstFactor: "one" } } },
          { format: "r16float", blend: { color: { srcFactor: "zero", dstFactor: "one-minus-src" }, alpha: { srcFactor: "zero", dstFactor: "one-minus-src" } } },
        ],
      },
      primitive: { topology: "triangle-list", cullMode: "none" },
      depthStencil: { format: "depth24plus", depthWriteEnabled: false, depthCompare: "less" },
    });
    const coneMod = dev.createShaderModule({ code: CONES3 });
    this.cones = dev.createRenderPipeline({
      layout: "auto",
      vertex: { module: coneMod, entryPoint: "vs" },
      fragment: { module: coneMod, entryPoint: "fs", targets: [{ format: this.format }] },
      primitive: { topology: "triangle-list", cullMode: "none" },
      depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" },
    });
    const planeMod = dev.createShaderModule({ code: PLANE3 });
    this.planes = dev.createRenderPipeline({
      layout: "auto",
      vertex: { module: planeMod, entryPoint: "vs" },
      fragment: { module: planeMod, entryPoint: "fs", targets: [{ format: this.format }] },
      primitive: { topology: "triangle-list", cullMode: "none" },
      depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" },
    });
    this.planesTrans = dev.createRenderPipeline({
      layout: "auto",
      vertex: { module: planeMod, entryPoint: "vs" },
      fragment: {
        module: planeMod, entryPoint: "fsTrans",
        targets: [
          { format: "rgba16float", blend: { color: { srcFactor: "one", dstFactor: "one" }, alpha: { srcFactor: "one", dstFactor: "one" } } },
          { format: "r16float", blend: { color: { srcFactor: "zero", dstFactor: "one-minus-src" }, alpha: { srcFactor: "zero", dstFactor: "one-minus-src" } } },
        ],
      },
      primitive: { topology: "triangle-list", cullMode: "none" },
      depthStencil: { format: "depth24plus", depthWriteEnabled: false, depthCompare: "less" },
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

  private bindLines(L: GpuLineLayer3D, viewProj: Mat4, eye: number[], crop: number[], cropLo: number[], w: number, h: number, pipeline = this.lines): GPUBindGroup {
    const f = new Float32Array(64);
    f.set(viewProj, 0);
    f.set([eye[0]!, eye[1]!, eye[2]!, 0], 16);
    const alpha = L.alpha ?? 1;
    f.set([L.width, alpha < 0.999 ? 1 : 0, L.lut && L.map ? 1 : 0, alpha], 20); // style: width, translucent, useLut, alpha
    f.set([L.color[0], L.color[1], L.color[2], 1], 24);
    f.set([L.map?.lo ?? 0, L.map?.hi ?? 1, L.map?.log ? 1 : 0, L.map?.flip ? 1 : 0], 28);
    f.set([crop[0]!, crop[1]!, crop[2]!, L.uncropped ? 0 : 1], 32);
    const P = L.particles;
    f.set([P?.tail ?? 0, P?.split ?? 1, P?.travel ?? 0, P && L.segs.particles ? 1 : 0], 36);
    f.set([L.embed?.axis ?? 0, L.embed?.depth ?? 0, L.embed ? 1 : 0, L.depthBias ?? 0], 40);
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    f.set([w, h, dpr, 0], 44);
    f.set([cropLo[0]!, cropLo[1]!, cropLo[2]!, 0], 48);
    return this.backend.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniform(f) } },
        { binding: 1, resource: { buffer: L.segs.buffer } },
        { binding: 2, resource: this.lutTexture(L.lut ?? WHITE_LUT).createView() },
        { binding: 3, resource: this.sampler },
      ],
    });
  }

  /** ConeU: viewProj, eye, style (–, –, useLut), color, map, crop, cropLo */
  private bindCones(L: GpuLineLayer3D, viewProj: Mat4, eye: number[], crop: number[], cropLo: number[], w: number, h: number): GPUBindGroup {
    const f = new Float32Array(64);
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    f.set([w, h, dpr, L.minPx ?? 0], 40);
    f.set(viewProj, 0);
    f.set([eye[0]!, eye[1]!, eye[2]!, 0], 16);
    f.set([0, 1, L.lut && L.map ? 1 : 0, 0], 20);
    f.set([L.color[0], L.color[1], L.color[2], 1], 24);
    f.set([L.map?.lo ?? 0, L.map?.hi ?? 1, L.map?.log ? 1 : 0, L.map?.flip ? 1 : 0], 28);
    f.set([crop[0]!, crop[1]!, crop[2]!, L.uncropped ? 0 : 1], 32);
    f.set([cropLo[0]!, cropLo[1]!, cropLo[2]!, 0], 36);
    return this.backend.device.createBindGroup({
      layout: this.cones.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniform(f) } },
        { binding: 1, resource: { buffer: L.segs.buffer } },
        { binding: 2, resource: this.lutTexture(L.lut ?? WHITE_LUT).createView() },
        { binding: 3, resource: this.sampler },
      ],
    });
  }

  /** PlaneU: viewProj, gridA (a0, a1, h0, h1), gridN (n0, n1, s0, s1), box (a0, a1, b0, b1 in the plane's axes), map, crop, cropLo, embed (axis, depth), misc (smooth, channels, channel, alpha) */
  private bindPlane(L: GpuPlaneLayer3D, viewProj: Mat4, crop: number[], cropLo: number[], pipeline = this.planes): GPUBindGroup {
    const f = new Float32Array(64);
    const grid = L.values.grid;
    const keep = [0, 1, 2].filter((d) => d !== L.axis) as [number, number];
    f.set(viewProj, 0);
    f.set([grid.box.a[0]!, grid.box.a[1]!, grid.spacing[0]!, grid.spacing[1]!], 16);
    new Int32Array(f.buffer).set([grid.size[0]!, grid.size[1]!, grid.strides[0]!, grid.strides[1]!], 20);
    f.set([L.box.a[keep[0]]!, L.box.a[keep[1]]!, L.box.b[keep[0]]!, L.box.b[keep[1]]!], 24);
    f.set([L.map.lo, L.map.hi, L.map.log ? 1 : 0, L.map.flip ? 1 : 0], 28);
    f.set([crop[0]!, crop[1]!, crop[2]!, 1], 32);
    f.set([cropLo[0]!, cropLo[1]!, cropLo[2]!, 0], 36);
    f.set([L.axis, L.depth, 0, 0], 40);
    f.set([L.smooth ? 1 : 0, L.values.channels, L.channel ?? 0, L.alpha ?? 1], 44);
    return this.backend.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniform(f) } },
        { binding: 1, resource: { buffer: L.values.buffer } },
        { binding: 2, resource: this.lutTexture(L.lut).createView() },
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
    const linesOpaque = (scene.lines ?? []).filter((L) => (L.alpha ?? 1) >= 0.999 || L.kind === "triangles"), linesTrans = (scene.lines ?? []).filter((L) => (L.alpha ?? 1) < 0.999 && L.kind !== "triangles");
    const planesOpaque = (scene.planes ?? []).filter((L) => (L.alpha ?? 1) >= 0.999), planesTrans = (scene.planes ?? []).filter((L) => (L.alpha ?? 1) < 0.999);
    const enc = dev.createCommandEncoder();
    const colour = this.ctx.getCurrentTexture().createView();
    const depth = T.depth.createView();
    // opaque pass (also clears colour and depth)
    const p1 = enc.beginRenderPass({
      colorAttachments: [{ view: colour, clearValue: { r, g, b, a: 1 }, loadOp: "clear", storeOp: "store" }],
      depthStencilAttachment: { view: depth, depthClearValue: 1, depthLoadOp: "clear", depthStoreOp: "store" },
    });
    for (const L of linesOpaque) {
      if (L.kind === "triangles") { p1.setPipeline(this.cones); p1.setBindGroup(0, this.bindCones(L, viewProj, eye, crop, cropLo, w, h)); }
      else { p1.setPipeline(this.lines); p1.setBindGroup(0, this.bindLines(L, viewProj, eye, crop, cropLo, w, h)); }
      p1.drawIndirect(L.segs.indirect, 0);
    }
    for (const L of opaque) { p1.setPipeline(this.opaque); p1.setBindGroup(0, this.bind(this.opaque, L, viewProj, eye, crop, cropLo)); p1.drawIndirect(L.mesh.indirect, 0); }
    for (const L of planesOpaque) { p1.setPipeline(this.planes); p1.setBindGroup(0, this.bindPlane(L, viewProj, crop, cropLo)); p1.draw(6); }
    p1.end();
    if (trans.length || linesTrans.length || planesTrans.length) {
      const p2 = enc.beginRenderPass({
        colorAttachments: [
          { view: T.accum.createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: "clear", storeOp: "store" },
          { view: T.reveal.createView(), clearValue: { r: 1, g: 0, b: 0, a: 0 }, loadOp: "clear", storeOp: "store" },
        ],
        depthStencilAttachment: { view: depth, depthLoadOp: "load", depthStoreOp: "store" },
      });
      for (const L of trans) { p2.setPipeline(this.transparent); p2.setBindGroup(0, this.bind(this.transparent, L, viewProj, eye, crop, cropLo)); p2.drawIndirect(L.mesh.indirect, 0); }
      for (const L of linesTrans) { p2.setPipeline(this.linesTrans); p2.setBindGroup(0, this.bindLines(L, viewProj, eye, crop, cropLo, w, h, this.linesTrans)); p2.drawIndirect(L.segs.indirect, 0); }
      for (const L of planesTrans) { p2.setPipeline(this.planesTrans); p2.setBindGroup(0, this.bindPlane(L, viewProj, crop, cropLo, this.planesTrans)); p2.draw(6); }
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

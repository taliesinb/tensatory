// STEP 0 of the cooperative kernel (notes/nets.md "the MNIST MLP experiment"): a HAND-WRITTEN WGSL kernel that
// evaluates the MNIST MLP loss with ONE WORKGROUP PER GRID POINT, to validate the performance model before an emitter
// is built. It is the shape the emitter would produce after the two planned rewrites:
//   * the first layer HOISTED: x·W1 and x·D1ₖ are point-independent, so `A = xs·W1` [N, 256] and `B = xs·D1` [K, N, 256]
//     are precomputed here on the CPU and layer 1 becomes relu(A[n] + Σ tₖ B[k, n] + b1(t));
//   * the dataset axis streamed in TILES of E examples: per tile the 256 threads each own one output unit j, read the
//     displaced weight W2(t)[i, j] = W2 + Σ tₖ D2ₖ ONCE per i and reuse it for the E examples (E accumulators in
//     registers), the activation tiles h1 / h2 [E, 256] living in workgroup memory.
// The thing being measured: whether the workgroup mapping turns the ~10 s/dispatch serial lane into tens of ms per
// point-batch, and how much the example tile E buys (E = 1 is the naive "examples outside" mapping, whose weight
// traffic is N × the weights per point).
//
// Agreement with the CPU evaluator at a few points, then timing (PERF=1).

import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { Bundle, DenseGrid, type ByteSource, type NetScalarFieldData, type Program } from "@tensatory/core";
import { GpuBackend } from "../src";

const DIR = join(__dirname, "../../../apps/viewer/public/bundles/mnist-mlp");
const src: ByteSource = {
  bytes: async (path) => {
    try { const b = await readFile(join(DIR, path)); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer; }
    catch (e) { if ((e as NodeJS.ErrnoException).code === "ENOENT") return null; throw e; }
  },
};
const bundleP = readFile(join(DIR, "bundle.json"), "utf8").then((s) => Bundle.load(JSON.parse(s), src));

let gpu: GpuBackend | undefined;
beforeAll(async () => { gpu = await GpuBackend.create(); if (gpu) gpu.asyncCompile = false; });
afterAll(async () => { gpu?.destroy(); await new Promise((r) => setTimeout(r, 50)); });

const J = 256, C = 10;

/** the packed constants of one displaced MNIST program, layer 1 hoisted */
interface Packed { data: Float32Array; off: Record<string, number>; N: number; K: number }

function pack(prog: Program): Packed {
  const c = prog.consts; // the UNHOISTED net program: this prototype hoists the first layer itself
  const arr = (n: string) => c[n]!.arr.data as ArrayLike<number>;
  const x = arr("x"), y = arr("y"), W1 = arr("W1t"), D1 = arr("W1t__dispd");
  const N = y.length, K = c["W1t__dispd"]!.shape[0] as number;
  // xs = (x / 255 - 0.1307) / 0.3081 ; A = xs·W1 [N, J] ; B = xs·D1ₖ [K, N, J]  (f64 on the CPU, stored f32)
  const xs = new Float64Array(N * 784);
  for (let i = 0; i < xs.length; i++) xs[i] = (x[i]! / 255 - 0.1307) / 0.3081;
  const mm = (W: ArrayLike<number>, wOff: number): Float64Array => {
    const out = new Float64Array(N * J);
    for (let n = 0; n < N; n++) for (let i = 0; i < 784; i++) { const xv = xs[n * 784 + i]!; if (xv === 0) continue; const r = n * J, w = wOff + i * J; for (let j = 0; j < J; j++) out[r + j] = out[r + j]! + xv * W[w + j]!; }
    return out;
  };
  const A = mm(W1, 0);
  const B = Array.from({ length: K }, (_, k) => mm(D1, k * 784 * J));
  const parts: [string, ArrayLike<number>][] = [
    ["A", A], ...B.map((b, k) => [`B${k}`, b] as [string, ArrayLike<number>]),
    ["b1", arr("b1")], ["db1", arr("b1__dispd")],
    ["W2", arr("W2t")], ["D2", arr("W2t__dispd")], ["b2", arr("b2")], ["db2", arr("b2__dispd")],
    ["W3", arr("W3t")], ["D3", arr("W3t__dispd")], ["b3", arr("b3")], ["db3", arr("b3__dispd")],
    ["y", y],
  ];
  let total = 0;
  const off: Record<string, number> = {};
  for (const [n, a] of parts) { off[n] = total; total += a.length; }
  const data = new Float32Array(total);
  for (const [n, a] of parts) data.set(Array.from(a as ArrayLike<number>), off[n]!);
  return { data, off, N, K };
}

/** the cooperative kernel: one workgroup of 256 threads per point, N examples in tiles of E */
function kernel(p: Packed, E: number, opaque: boolean): string {
  const { N, K, off } = p;
  const t = (k: number) => `t${k}`;
  const tk = Array.from({ length: K }, (_, k) => k);
  // W(t)[idx] = W[idx] + Σ tₖ D[k, idx]  (D stacked [K, ...shape])
  const disp = (W: string, D: string, size: number, idx: string) => `(data[${off[W]} + ${idx}] + ${tk.map((k) => `${t(k)} * data[${off[D]! + k * size} + ${idx}]`).join(" + ")})`;
  const bound = (n: number) => (opaque ? `nb_(${n})` : String(n));
  return `
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<storage, read> data: array<f32>;
@group(0) @binding(2) var<storage, read> pts: array<f32>;
fn nb_(n: i32) -> i32 { return select(n, 0, bitcast<u32>(pts[0]) == 0x7fc00001u); }
const E: i32 = ${E};
var<workgroup> h1: array<f32, ${E * J}>;
var<workgroup> h2: array<f32, ${E * J}>;
var<workgroup> lg: array<f32, ${E * C}>;
var<workgroup> nl: array<f32, ${E}>;
@compute @workgroup_size(256) fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let count: i32 = i32(pts[1]);
  let point: i32 = i32(wid.x);
  if (point >= count) { return; }
  let j: i32 = i32(lid.x);
  ${tk.map((k) => `let ${t(k)}: f32 = pts[2 + point * ${K} + ${k}];`).join("\n  ")}
  var loss: f32 = 0.0;
  // point-only quantities
  let b1j: f32 = ${disp("b1", "db1", J, "j")};
  let b2j: f32 = ${disp("b2", "db2", J, "j")};
  for (var n0: i32 = 0; n0 < ${bound(N)}; n0 += E) {
    // layer 1, hoisted: h1[e, j] = relu(A[n, j] + Σ tₖ B[k, n, j] + b1(t)[j])
    for (var e: i32 = 0; e < E; e++) {
      let n = n0 + e;
      h1[e * ${J} + j] = max(0.0, data[${off["A"]} + n * ${J} + j] + ${tk.map((k) => `${t(k)} * data[${off[`B${k}`]} + n * ${J} + j]`).join(" + ")} + b1j);
    }
    workgroupBarrier();
    // layer 2: thread j owns unit j; W2(t)[i, j] read once per i, reused over the E examples
    var acc: array<f32, ${E}>;
    for (var e: i32 = 0; e < E; e++) { acc[e] = 0.0; }
    for (var i: i32 = 0; i < ${bound(J)}; i++) {
      let w: f32 = ${disp("W2", "D2", J * J, `i * ${J} + j`)};
      for (var e: i32 = 0; e < E; e++) { acc[e] = acc[e] + h1[e * ${J} + i] * w; }
    }
    for (var e: i32 = 0; e < E; e++) { h2[e * ${J} + j] = max(0.0, acc[e] + b2j); }
    workgroupBarrier();
    // layer 3: logits [E, C] — threads below E·C each take one (e, c)
    if (j < ${E * C}) {
      let e = j / ${C}; let c = j % ${C};
      var s: f32 = ${disp("b3", "db3", C, "c")};
      for (var i: i32 = 0; i < ${bound(J)}; i++) { s = s + h2[e * ${J} + i] * ${disp("W3", "D3", J * C, `i * ${C} + c`)}; }
      lg[j] = s;
    }
    workgroupBarrier();
    // cross-entropy per example — threads below E
    if (j < E) {
      var m: f32 = lg[j * ${C}];
      for (var c: i32 = 1; c < ${C}; c++) { m = max(m, lg[j * ${C} + c]); }
      var s: f32 = 0.0;
      for (var c: i32 = 0; c < ${C}; c++) { s = s + exp(lg[j * ${C} + c] - m); }
      let yc: i32 = i32(round(data[${off["y"]} + n0 + j]));
      nl[j] = (m + log(s)) - lg[j * ${C} + yc];
    }
    workgroupBarrier();
    if (j == 0) { for (var e: i32 = 0; e < E; e++) { loss = loss + nl[e]; } }
    workgroupBarrier();
  }
  if (j == 0) { out[point] = loss / ${N}.0; }
}`;
}

/** evaluate the kernel at `points` ([P, K] flat); returns the losses and the wall time of the dispatch + readback */
async function run(p: Packed, E: number, points: Float32Array, opaque = false): Promise<{ out: Float32Array; ms: number }> {
  const P = points.length / p.K;
  const pts = new Float32Array(2 + points.length);
  pts[0] = 0; pts[1] = P; pts.set(points, 2);
  const t0 = performance.now();
  const { read: [o] } = await gpu!.runKernel({
    code: kernel(p, E, opaque),
    invocations: P * 256,
    workgroups: P,
    buffers: [{ role: "rw", size: P * 4, readback: true }, { role: "r", data: p.data }, { role: "r", data: pts }],
  });
  return { out: new Float32Array(o!, 0, P), ms: performance.now() - t0 };
}

function gridPoints(fd: NetScalarFieldData, n: number): Float32Array {
  const g = new DenseGrid(Array.from({ length: fd.dimCount }, () => n), fd.box);
  const K = fd.dimCount, out = new Float32Array(g.sampleCount * K);
  for (let i = 0; i < g.sampleCount; i++) { const q = g.point(i); for (let k = 0; k < K; k++) out[i * K + k] = q[k]!; }
  return out;
}

describe("cooperative kernel prototype (MNIST MLP, one workgroup per point)", () => {
  it("agrees with the CPU evaluator at the 2×2 grid points (E = 1, 4, 8)", async () => {
    if (!gpu) return;
    const b = await bundleP;
    const fd = b.scalarField("fastLoss2").data as NetScalarFieldData;
    const p = pack(b.net("mlp_pca2_256").program);
    expect(p.N).toBe(256);
    const grid = new DenseGrid([2, 2], fd.box);
    const cpu = fd.sampleOn(grid);
    const pts = gridPoints(fd, 2);
    for (const E of [1, 4, 8]) {
      const { out } = await run(p, E, pts);
      for (let i = 0; i < cpu.length; i++) expect(out[i], `E=${E} point ${i}`).toBeCloseTo(cpu[i]!, 3);
    }
  }, 120_000);

  it.runIf(process.env.PERF)("PERF: ms per dispatch by tile size E and point count", async () => {
    if (!gpu) return;
    const b = await bundleP;
    console.log(`device: ${gpu.adapterInfo}; workgroup storage ${gpu.device.limits.maxComputeWorkgroupStorageSize} B`);
    const lines: string[] = [];
    for (const id of ["fastLoss2", "loss2"]) {
      const fd = b.scalarField(id).data as NetScalarFieldData;
      const t0 = performance.now();
      const p = pack(b.net(id === "loss2" ? "mlp_pca2" : "mlp_pca2_256").program);
      lines.push(`${id}: N = ${p.N}, hoisted layer 1 packed in ${(performance.now() - t0).toFixed(0)} ms (${(p.data.byteLength / 1e6).toFixed(1)} MB)`);
      const macs = p.N * (J * J + J * C); // per point, layers 2–3
      for (const E of id === "loss2" ? [8] : [1, 2, 4, 8]) {
        for (const opaque of id === "loss2" ? [false] : [false, true]) {
          // compile + warm up on a few points, then time
          const tc = performance.now();
          await run(p, E, gridPoints(fd, 2), opaque);
          const compile = performance.now() - tc;
          const row: string[] = [];
          for (const n of id === "loss2" ? [16] : [16, 32]) {
            const pts = gridPoints(fd, n);
            const P = pts.length / p.K;
            const r1 = await run(p, E, pts, opaque);
            const r2 = await run(p, E, pts, opaque);
            const ms = Math.min(r1.ms, r2.ms);
            row.push(`${n}² (${P} pts): ${ms.toFixed(0)} ms = ${(ms / P * 1000).toFixed(0)} µs/pt, ${(2 * macs * P / ms / 1e6).toFixed(0)} GFLOPS`);
          }
          lines.push(`  E = ${E}${opaque ? " opaque bounds" : ""}: compile+warmup ${compile.toFixed(0)} ms; ${row.join("; ")}`);
        }
      }
    }
    console.log(lines.join("\n"));
  }, 600_000);
});

// WebGPU device acquisition (browser `navigator.gpu`, or Dawn's node bindings
// from the optional `webgpu` package) and program execution.

import { setCoopWorkgroupBytes } from "./coop";
import type { GpuProgram } from "./program";

export type BufferRole = "r" | "rw";
export interface KernelBuffer {
  role: BufferRole;
  /** upload (required for "r" unless `buffer` is given) */
  data?: Float32Array | Uint32Array | Int32Array;
  /** byte size for outputs without data */
  size?: number;
  /** bind an existing (resident) buffer instead of creating one; never destroyed here */
  buffer?: GPUBuffer;
  /** read the buffer back after the dispatch */
  readback?: boolean;
  /** keep the created buffer alive and return it (resident output for later passes / rendering) */
  keep?: boolean;
}
export interface Kernel {
  /** complete WGSL with an entry point `main` using @workgroup_size(64) and global_invocation_id.x */
  code: string;
  invocations: number;
  /** COOPERATIVE kernels: dispatch this many workgroups instead of ceil(invocations / 64); the kernel declares its own
   *  @workgroup_size and indexes by workgroup_id / local_invocation_id (one workgroup per grid point) */
  workgroups?: number;
  buffers: KernelBuffer[];
}
export interface KernelResult {
  /** read-back buffers, in binding order of those marked `readback` */
  read: ArrayBuffer[];
  /** kept buffers, in binding order of those marked `keep` */
  kept: GPUBuffer[];
}

/**
 * Multiply-adds per cooperative dispatch: a chunk of points small enough for one dispatch to stay far below the GPU
 * watchdog and for the resolution controller to time fills (~30 ms at the measured ~4·10¹¹ MAC/s).
 */
export const COOP_CHUNK_WORK = 1e10;

/**
 * The dispatches that run a sampling program with `out` and `data` already created (bound at 0 and 1, resident
 * buffers the program reads after): ONE for an ordinary kernel; for a COOPERATIVE program (one workgroup per point)
 * chunks of points bounded by COOP_CHUNK_WORK, each with its own `params` = [first point] at binding 2.
 */
export function programKernels(program: GpuProgram, out: KernelBuffer, data: KernelBuffer): Kernel[] {
  const extra: KernelBuffer[] = (program.bindings ?? []).map((buffer) => ({ role: "r", buffer }));
  const c = program.cooperative;
  if (!c) return [{ code: program.code, invocations: program.sampleCount, buffers: [out, data, ...extra] }];
  const per = Math.max(1, Math.min(program.sampleCount, Math.floor(COOP_CHUNK_WORK / Math.max(1, c.work))));
  const kernels: Kernel[] = [];
  for (let from = 0; from < program.sampleCount; from += per) {
    const n = Math.min(per, program.sampleCount - from);
    kernels.push({ code: program.code, invocations: n * c.workgroupSize, workgroups: n, buffers: [out, data, { role: "r", data: new Uint32Array([from]) }, ...extra] });
  }
  return kernels;
}

/** usage flags for buffers that later passes read (compute storage, render storage/vertex) */
export const RESIDENT_USAGE = 0x80 /* STORAGE */ | 0x4 /* COPY_SRC */ | 0x8 /* COPY_DST */ | 0x20 /* VERTEX */ | 0x100 /* INDIRECT */;

async function findGpu(): Promise<GPU | undefined> {
  const nav = (globalThis as { navigator?: { gpu?: GPU } }).navigator;
  if (nav?.gpu) return nav.gpu;
  try {
    // node: Dawn bindings. The module name is a variable so bundlers (Vite) do not try to resolve it for browsers.
    const name = "webgpu";
    const mod = (await import(/* @vite-ignore */ name)) as { create: (flags: string[]) => GPU; globals: Record<string, unknown> };
    Object.assign(globalThis, mod.globals); // GPUBufferUsage, GPUMapMode, ...
    return mod.create([]);
  } catch {
    return undefined;
  }
}

export class GpuBackend {
  private readonly pipelines = new Map<string, GPUComputePipeline>();
  /** pipelines being compiled (createComputePipelineAsync) */
  private readonly building = new Map<string, { promise: Promise<GPUComputePipeline> }>();
  /** dispatches waiting behind a compile, in submission order: once one is deferred, every later one queues behind
   *  it so the GPU sees them in the order they were issued (a seed → smooth → emit chain must not reorder) */
  private readonly queue: { key: string; run: (p: GPUComputePipeline | undefined) => void }[] = [];
  /** number of pipelines currently compiling: a frame loop shows a busy indicator while > 0 */
  get compiling(): number { return this.building.size + (this.queue.length ? 1 : 0); }
  /** whether a `dispatch` since the last `takeDeferred()` was deferred behind a compile (the frame should not present) */
  private deferredThisFrame = false;
  /** called when a compile finishes and its deferred dispatches have been enqueued (the frame loop re-renders) */
  onPipelineReady: (() => void) | undefined;
  /**
   * Compile asynchronously (default). The compile of a pipeline created synchronously happens at its first
   * submit and stalls the whole frame — hundreds of ms in Chrome, seconds in Safari for a transpiled net — while
   * `createComputePipelineAsync` compiles off the critical path: a `dispatch` that needs a pipeline still compiling
   * is DEFERRED (its buffers are created and returned, the pass is enqueued when the compile lands, in order) and the
   * caller learns from `takeDeferred()` that this frame is incomplete. Tests set this false for determinism.
   */
  asyncCompile = true;
  /** bytes of every buffer created through `createBuffer` and not yet destroyed (resident meshes, segments, grids) */
  bytesAllocated = 0;
  /** number of `dispatch` / `runKernel` calls so far: lets a frame loop tell whether a frame did compute work */
  dispatches = 0;
  /** number of compute pipelines compiled so far: a frame that compiled is not representative for timing */
  pipelinesBuilt = 0;

  /**
   * `gpu` and `adapter` are retained on purpose: with Dawn's node bindings the
   * instance owns the event loop that completes mapAsync etc.; letting it be
   * garbage-collected while the device is in use crashes in
   * InstanceBase::ProcessEvents (a freed mutex).
   */
  private constructor(readonly gpu: GPU, readonly adapter: GPUAdapter, readonly device: GPUDevice, readonly adapterInfo: string) {}

  /** undefined when no WebGPU implementation is available */
  static async create(): Promise<GpuBackend | undefined> {
    const gpu = await findGpu();
    if (!gpu) return undefined;
    const adapter = await gpu.requestAdapter();
    if (!adapter) return undefined;
    // resident meshes / segment sets can exceed the 128 MB default binding size: ask for what the adapter offers (up to 2 GB)
    const lim = adapter.limits, want = (v: number, dflt: number) => Math.min(Math.max(v, dflt), 2 ** 31);
    let device: GPUDevice;
    // cooperative net kernels hold their activation tiles in workgroup memory: 16 KB by default, 32 KB on Apple GPUs
    try { device = await adapter.requestDevice({ requiredLimits: { maxStorageBufferBindingSize: want(lim.maxStorageBufferBindingSize, 134217728), maxBufferSize: want(lim.maxBufferSize, 268435456), maxComputeWorkgroupStorageSize: Math.max(lim.maxComputeWorkgroupStorageSize, 16384) } }); }
    catch { device = await adapter.requestDevice(); }
    setCoopWorkgroupBytes(device.limits.maxComputeWorkgroupStorageSize);
    device.addEventListener?.("uncapturederror", (e) => console.error("WebGPU:", (e as GPUUncapturedErrorEvent).error.message));
    const info = adapter.info;
    return new GpuBackend(gpu, adapter, device, [info?.vendor, info?.architecture, info?.device].filter(Boolean).join(" "));
  }

  private readonly layouts = new Map<string, GPUBindGroupLayout>();

  /**
   * Create a buffer whose size is accounted in `bytesAllocated` until its `destroy()` (a second destroy is a
   * no-op). Every resident allocation (meshes, segment sets, grids, smoothing scratch) goes through here so the
   * viewer can keep the device's memory under a cap; transient kernel buffers do not.
   */
  createBuffer(desc: GPUBufferDescriptor): GPUBuffer {
    const buf = this.device.createBuffer(desc);
    this.bytesAllocated += desc.size;
    const destroy = buf.destroy.bind(buf);
    let live = true;
    // a deferred dispatch (pipeline still compiling) may still target this buffer: destroy after the compiles land
    buf.destroy = () => { if (live) { live = false; this.bytesAllocated -= desc.size; this.destroyed.add(buf); } if (this.building.size || this.queue.length) void this.whenIdle().then(destroy); else destroy(); };
    return buf;
  }
  private readonly destroyed = new WeakSet<GPUBuffer>();
  /** whether `destroy()` was called on a buffer from `createBuffer` (WebGPU itself cannot be asked; a destroyed
   *  buffer in a submit is a validation error) — for work scheduled across frames against buffers that may be gone */
  isDestroyed(buf: GPUBuffer): boolean { return this.destroyed.has(buf); }

  /** destroy a buffer not created through `createBuffer`, after any pending compile's deferred dispatches */
  release(buf: GPUBuffer): void { if (this.building.size || this.queue.length) void this.whenIdle().then(() => buf.destroy()); else buf.destroy(); }

  /**
   * Read one u32 of a small buffer (an indirect draw buffer's counter) once the queue has reached it. The mesh /
   * segment kernels count every record through their atomic even when the set is full, so this is the TRUE
   * record count: `count > capacity` means the set overflowed.
   */
  async readCounter(buffer: GPUBuffer, index: number): Promise<number> {
    await this.whenIdle(); // a deferred dispatch may still have to write it
    const dev = this.device;
    const read = dev.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = dev.createCommandEncoder();
    enc.copyBufferToBuffer(buffer, 0, read, 0, 16);
    dev.queue.submit([enc.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const v = new Uint32Array(read.getMappedRange())[index]!;
    read.unmap(); read.destroy();
    return v;
  }

  /** explicit layout for a buffer-role signature such as "rw,r,r" ("auto" layouts drop unused bindings and then fail validation) */
  private layoutFor(roles: BufferRole[]): GPUBindGroupLayout {
    const key = roles.join(",");
    let l = this.layouts.get(key);
    if (!l) {
      l = this.device.createBindGroupLayout({
        entries: roles.map((role, binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: role === "rw" ? "storage" : "read-only-storage" } })),
      });
      this.layouts.set(key, l);
    }
    return l;
  }

  /** the largest buffer a kernel can bind (bytes): resident sets must stay under it */
  get maxBufferBytes(): number { return Math.min(this.device.limits.maxStorageBufferBindingSize, this.device.limits.maxBufferSize); }

  /** the most workgroups one dispatch dimension takes (WebGPU default limit 65535 → 4.19 M invocations of 64) */
  private get maxGroups(): number { return this.device.limits.maxComputeWorkgroupsPerDimension; }

  /**
   * Kernels are written with a 1D linear index (`id.x`); dispatches larger than one dimension allows (a 2048²
   * grid, any 3D grid past 160³) are laid out over two dimensions, and the entry point is rewritten to
   * reconstruct the linear index from them. Applied to every kernel so a shader has one form.
   */
  private static linearize(code: string): string {
    const sig = "fn main(@builtin(global_invocation_id) id: vec3<u32>) {";
    if (!code.includes(sig)) return code;
    return code.replace(sig, "fn main(@builtin(global_invocation_id) gid_: vec3<u32>, @builtin(num_workgroups) nwg_: vec3<u32>) {\n  let id = vec3<u32>(gid_.x + gid_.y * nwg_.x * 64u, 0u, 0u);");
  }
  private groups(kernel: Kernel): [number, number] {
    const wg = Math.max(1, kernel.workgroups ?? Math.ceil(kernel.invocations / 64)), x = Math.min(wg, this.maxGroups);
    return [x, Math.ceil(wg / x)];
  }

  private pipelineKey(code: string, roles: BufferRole[]): string { return `${roles.join(",")}\n${code}`; }

  /** the ready pipeline, or the compile in flight (async mode) */
  private pipeline(code: string, roles: BufferRole[]): GPUComputePipeline | { promise: Promise<GPUComputePipeline> } {
    const key = this.pipelineKey(code, roles);
    const ready = this.pipelines.get(key);
    if (ready) return ready;
    const inFlight = this.building.get(key);
    if (inFlight) return inFlight;
    this.pipelinesBuilt++;
    const module = this.device.createShaderModule({ code: GpuBackend.linearize(code) });
    const layout = this.device.createPipelineLayout({ bindGroupLayouts: [this.layoutFor(roles)] });
    const desc: GPUComputePipelineDescriptor = { layout, compute: { module, entryPoint: "main" } };
    if (!this.asyncCompile) {
      const p = this.device.createComputePipeline(desc);
      this.pipelines.set(key, p);
      return p;
    }
    const entry = { promise: this.device.createComputePipelineAsync(desc) };
    this.building.set(key, entry);
    entry.promise.then(
      (p) => { this.pipelines.set(key, p); this.building.delete(key); this.drain(); this.onPipelineReady?.(); },
      (e: unknown) => {
        this.building.delete(key);
        this.failed.add(key);
        this.drain();
        this.device.dispatchEvent(new (globalThis as unknown as { CustomEvent: new (t: string, i: object) => Event }).CustomEvent("pipelineerror", { detail: e }));
        this.onPipelineReady?.();
      },
    );
    return entry;
  }
  private readonly failed = new Set<string>();

  /** run queued dispatches from the head while their pipelines are ready (a failed compile drops its dispatches) */
  private drain(): void {
    while (this.queue.length) {
      const head = this.queue[0]!;
      const p = this.pipelines.get(head.key);
      if (head.key === "" || p) { this.queue.shift(); head.run(p); } // "" = an ordered write, needs no pipeline
      else if (this.failed.has(head.key)) this.queue.shift();
      else return;
    }
  }

  /**
   * `queue.writeBuffer` that keeps its place among dispatches: a reset of a segment set's counter must stay before
   * the dispatch that fills it, even when that dispatch is deferred behind a compile.
   */
  write(buffer: GPUBuffer, offset: number, data: BufferSource): void {
    if (this.queue.length) {
      const v = data as ArrayBufferView; // copy: the caller may reuse its array before the write runs
      const copy = ArrayBuffer.isView(v) ? v.buffer.slice(v.byteOffset, v.byteOffset + v.byteLength) : (data as ArrayBuffer).slice(0);
      this.queue.push({ key: "", run: () => this.device.queue.writeBuffer(buffer, offset, copy) });
    }
    else this.device.queue.writeBuffer(buffer, offset, data);
  }

  /** whether a dispatch since the previous call was deferred behind a compile; resets the flag */
  takeDeferred(): boolean { const d = this.deferredThisFrame; this.deferredThisFrame = false; return d; }

  /** resolves when no pipeline is compiling and every deferred dispatch has been enqueued */
  async whenIdle(): Promise<void> {
    while (this.building.size) await Promise.allSettled([...this.building.values()].map((b) => b.promise));
    this.drain();
  }

  /**
   * Run a compute kernel. Buffers are bound at group 0 in order; "r" buffers
   * are uploaded from `data`, "rw" buffers are zero-initialized outputs of
   * `size` bytes (or uploaded when `data` is given) and are read back when
   * `readback` is set. Returns the read-back buffers in binding order.
   */
  async runKernel(kernel: Kernel): Promise<KernelResult> {
    const dev = this.device;
    this.dispatches++;
    const roles = kernel.buffers.map((b) => b.role);
    dev.pushErrorScope("validation");
    const pl = this.pipeline(kernel.code, roles);
    const pipeline = "promise" in pl ? await pl.promise : pl;
    if (this.queue.length) await this.whenIdle(); // stay behind deferred dispatches
    const gpuBuffers = kernel.buffers.map((b) => {
      if (b.buffer) return { buf: b.buffer, size: b.buffer.size, own: false };
      const size = Math.max(16, Math.ceil((b.data ? b.data.byteLength : b.size ?? 0) / 4) * 4);
      const usage = b.keep ? RESIDENT_USAGE : (b.role === "rw" ? GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC : GPUBufferUsage.STORAGE) | GPUBufferUsage.COPY_DST;
      const buf = b.keep ? this.createBuffer({ size, usage }) : dev.createBuffer({ size, usage });
      if (b.data) dev.queue.writeBuffer(buf, 0, b.data as unknown as BufferSource);
      return { buf, size, own: true };
    });
    const bindGroup = dev.createBindGroup({ layout: this.layoutFor(roles), entries: gpuBuffers.map(({ buf }, binding) => ({ binding, resource: { buffer: buf } })) });
    const reads = kernel.buffers.map((b, i) => (b.readback ? { i, buf: dev.createBuffer({ size: gpuBuffers[i]!.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }) } : undefined));
    const enc = dev.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(...this.groups(kernel));
    pass.end();
    for (const r of reads) if (r) enc.copyBufferToBuffer(gpuBuffers[r.i]!.buf, 0, r.buf, 0, gpuBuffers[r.i]!.size);
    dev.queue.submit([enc.finish()]);
    const err = await dev.popErrorScope();
    if (err) throw new Error(`WebGPU validation: ${err.message}`);
    const read: ArrayBuffer[] = [];
    for (const r of reads) {
      if (!r) continue;
      await r.buf.mapAsync(GPUMapMode.READ);
      read.push(r.buf.getMappedRange().slice(0));
      r.buf.unmap(); r.buf.destroy();
    }
    const kept: GPUBuffer[] = [];
    kernel.buffers.forEach((b, i) => { const g = gpuBuffers[i]!; if (b.keep) kept.push(g.buf); else if (g.own) g.buf.destroy(); });
    return { read, kept };
  }

  /**
   * Enqueue a kernel without waiting: the GPU queue orders it before any later
   * render pass, so fused frames need no CPU synchronization. Buffers marked
   * `keep` are returned (filled once the queue reaches them); `readback` is not
   * supported here. Errors surface through the device's uncapturederror event.
   */
  dispatch(kernel: Kernel): GPUBuffer[] {
    const dev = this.device;
    this.dispatches++;
    const roles = kernel.buffers.map((b) => b.role);
    const pl = this.pipeline(kernel.code, roles);
    // buffers are created (and uploaded) NOW, so the caller holds the right handles whether or not the pass runs now
    const gpuBuffers = kernel.buffers.map((b) => {
      if (b.buffer) return { buf: b.buffer, own: false };
      const size = Math.max(16, Math.ceil((b.data ? b.data.byteLength : b.size ?? 0) / 4) * 4);
      const usage = b.keep ? RESIDENT_USAGE : (b.role === "rw" ? GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC : GPUBufferUsage.STORAGE) | GPUBufferUsage.COPY_DST;
      const buf = b.keep ? this.createBuffer({ size, usage }) : dev.createBuffer({ size, usage });
      if (b.data) dev.queue.writeBuffer(buf, 0, b.data as unknown as BufferSource);
      return { buf, own: true };
    });
    const bindGroup = dev.createBindGroup({ layout: this.layoutFor(roles), entries: gpuBuffers.map(({ buf }, binding) => ({ binding, resource: { buffer: buf } })) });
    const groups = this.groups(kernel);
    const run = (pipeline: GPUComputePipeline) => {
      const enc = dev.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(...groups);
      pass.end();
      dev.queue.submit([enc.finish()]);
      kernel.buffers.forEach((b, i) => { const g = gpuBuffers[i]!; if (!b.keep && g.own) g.buf.destroy(); });
    };
    if ("promise" in pl || this.queue.length) {
      // deferred: behind a compile (or behind an earlier deferred dispatch, to keep the order); the frame must not present
      this.deferredThisFrame = true;
      this.queue.push({ key: this.pipelineKey(kernel.code, roles), run: (p) => run(p!) });
    } else run(pl);
    const kept: GPUBuffer[] = [];
    kernel.buffers.forEach((b, i) => { if (b.keep) kept.push(gpuBuffers[i]!.buf); });
    return kept;
  }

  /** run a sampling program (in chunks when cooperative) and read the result back */
  async run(program: GpuProgram): Promise<Float32Array> {
    const dev = this.device;
    const n = program.sampleCount * program.channels;
    const out = dev.createBuffer({ size: Math.max(16, n * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const data = dev.createBuffer({ size: Math.max(16, program.data.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    dev.queue.writeBuffer(data, 0, program.data as unknown as BufferSource);
    try {
      const kernels = programKernels(program, { role: "rw", buffer: out }, { role: "r", buffer: data });
      let read: ArrayBuffer | undefined;
      for (let i = 0; i < kernels.length; i++) {
        const k = kernels[i]!;
        if (i === kernels.length - 1) k.buffers[0] = { ...k.buffers[0]!, readback: true };
        const r = await this.runKernel(k);
        if (r.read.length) read = r.read[0];
      }
      return new Float32Array(read!, 0, n);
    } finally { out.destroy(); data.destroy(); }
  }

  /** release the device (call at the end of a test run; Dawn dislikes being torn down implicitly) */
  destroy(): void {
    this.pipelines.clear();
    this.device.destroy();
  }

  /** compilation diagnostics for a program (useful when a shader fails) */
  async diagnostics(code: string): Promise<string[]> {
    const info = await this.device.createShaderModule({ code }).getCompilationInfo();
    return info.messages.map((m) => `${m.type} ${m.lineNum}:${m.linePos} ${m.message}`);
  }
}

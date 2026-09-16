// WebGPU device acquisition (browser `navigator.gpu`, or Dawn's node bindings
// from the optional `webgpu` package) and program execution.

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
  buffers: KernelBuffer[];
}
export interface KernelResult {
  /** read-back buffers, in binding order of those marked `readback` */
  read: ArrayBuffer[];
  /** kept buffers, in binding order of those marked `keep` */
  kept: GPUBuffer[];
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
    try { device = await adapter.requestDevice({ requiredLimits: { maxStorageBufferBindingSize: want(lim.maxStorageBufferBindingSize, 134217728), maxBufferSize: want(lim.maxBufferSize, 268435456) } }); }
    catch { device = await adapter.requestDevice(); }
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
    buf.destroy = () => { if (live) { live = false; this.bytesAllocated -= desc.size; } destroy(); };
    return buf;
  }

  /**
   * Read one u32 of a small buffer (an indirect draw buffer's counter) once the queue has reached it. The mesh /
   * segment kernels count every record through their atomic even when the set is full, so this is the TRUE
   * record count: `count > capacity` means the set overflowed.
   */
  async readCounter(buffer: GPUBuffer, index: number): Promise<number> {
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
  private groups(invocations: number): [number, number] {
    const wg = Math.max(1, Math.ceil(invocations / 64)), x = Math.min(wg, this.maxGroups);
    return [x, Math.ceil(wg / x)];
  }

  private pipeline(code: string, roles: BufferRole[]): GPUComputePipeline {
    const key = `${roles.join(",")}\n${code}`;
    let p = this.pipelines.get(key);
    if (!p) {
      this.pipelinesBuilt++;
      const module = this.device.createShaderModule({ code: GpuBackend.linearize(code) });
      const layout = this.device.createPipelineLayout({ bindGroupLayouts: [this.layoutFor(roles)] });
      p = this.device.createComputePipeline({ layout, compute: { module, entryPoint: "main" } });
      this.pipelines.set(key, p);
    }
    return p;
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
    const pipeline = this.pipeline(kernel.code, roles);
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
    pass.dispatchWorkgroups(...this.groups(kernel.invocations));
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
    const pipeline = this.pipeline(kernel.code, roles);
    const gpuBuffers = kernel.buffers.map((b) => {
      if (b.buffer) return { buf: b.buffer, own: false };
      const size = Math.max(16, Math.ceil((b.data ? b.data.byteLength : b.size ?? 0) / 4) * 4);
      const usage = b.keep ? RESIDENT_USAGE : (b.role === "rw" ? GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC : GPUBufferUsage.STORAGE) | GPUBufferUsage.COPY_DST;
      const buf = b.keep ? this.createBuffer({ size, usage }) : dev.createBuffer({ size, usage });
      if (b.data) dev.queue.writeBuffer(buf, 0, b.data as unknown as BufferSource);
      return { buf, own: true };
    });
    const bindGroup = dev.createBindGroup({ layout: this.layoutFor(roles), entries: gpuBuffers.map(({ buf }, binding) => ({ binding, resource: { buffer: buf } })) });
    const enc = dev.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(...this.groups(kernel.invocations));
    pass.end();
    dev.queue.submit([enc.finish()]);
    const kept: GPUBuffer[] = [];
    kernel.buffers.forEach((b, i) => { const g = gpuBuffers[i]!; if (b.keep) kept.push(g.buf); else if (g.own) g.buf.destroy(); });
    return kept;
  }

  /** run a sampling program and read the result back */
  async run(program: GpuProgram): Promise<Float32Array> {
    const n = program.sampleCount * program.channels;
    const { read: [out] } = await this.runKernel({
      code: program.code,
      invocations: program.sampleCount,
      buffers: [{ role: "rw", size: n * 4, readback: true }, { role: "r", data: program.data }],
    });
    return new Float32Array(out!, 0, n);
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

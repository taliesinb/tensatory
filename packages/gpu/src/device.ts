// WebGPU device acquisition (browser `navigator.gpu`, or Dawn's node bindings
// from the optional `webgpu` package) and program execution.

import type { GpuProgram } from "./program";

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
    const device = await adapter.requestDevice();
    const info = adapter.info;
    return new GpuBackend(gpu, adapter, device, [info?.vendor, info?.architecture, info?.device].filter(Boolean).join(" "));
  }

  private layout: GPUBindGroupLayout | undefined;
  /** explicit layout: binding 0 = output, binding 1 = packed input data. (An "auto" layout drops
   *  binding 1 when a shader does not read `data`, and the bind group then fails validation.) */
  private bindGroupLayout(): GPUBindGroupLayout {
    return (this.layout ??= this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      ],
    }));
  }

  private pipeline(code: string): GPUComputePipeline {
    let p = this.pipelines.get(code);
    if (!p) {
      const module = this.device.createShaderModule({ code });
      const layout = this.device.createPipelineLayout({ bindGroupLayouts: [this.bindGroupLayout()] });
      p = this.device.createComputePipeline({ layout, compute: { module, entryPoint: "main" } });
      this.pipelines.set(code, p);
    }
    return p;
  }

  /** run a sampling program and read the result back */
  async run(program: GpuProgram): Promise<Float32Array> {
    const dev = this.device;
    const n = program.sampleCount * program.channels;
    const outSize = Math.max(16, n * 4);
    const out = dev.createBuffer({ size: outSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const read = dev.createBuffer({ size: outSize, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const input = dev.createBuffer({ size: program.data.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    dev.queue.writeBuffer(input, 0, program.data as unknown as BufferSource);
    dev.pushErrorScope("validation");
    const pipeline = this.pipeline(program.code);
    const bindGroup = dev.createBindGroup({
      layout: this.bindGroupLayout(),
      entries: [{ binding: 0, resource: { buffer: out } }, { binding: 1, resource: { buffer: input } }],
    });
    const enc = dev.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(program.sampleCount / 64));
    pass.end();
    enc.copyBufferToBuffer(out, 0, read, 0, outSize);
    dev.queue.submit([enc.finish()]);
    const err = await dev.popErrorScope();
    if (err) throw new Error(`WebGPU validation: ${err.message}`);
    await read.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(read.getMappedRange().slice(0, n * 4));
    read.unmap();
    out.destroy(); read.destroy(); input.destroy();
    return result;
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

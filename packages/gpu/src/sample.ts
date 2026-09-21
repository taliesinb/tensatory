import type { DenseGrid, ScalarFieldData, VectorFieldData } from "@tensatory/core";
import type { GpuBackend } from "./device";
import { buildSampleProgram, type ResidentProvider } from "./program";
import { sampleResidentSync, type GpuGrid } from "./resident";

/**
 * GPU counterpart of `field.sampleOn(grid)`: scalar fields give one value per
 * grid point, vector fields D values per point (row-major, like core). Nets a
 * lane cannot evaluate are sampled by the cooperative kernel — as the field
 * itself, or as an argument of it (`resident`, a temporary resident grid per
 * call unless the caller supplies a cache).
 */
export async function gpuSampleOn(backend: GpuBackend, field: ScalarFieldData | VectorFieldData, grid: DenseGrid, resident?: ResidentProvider): Promise<Float32Array> {
  const temps: GpuGrid[] = [];
  const provider: ResidentProvider = resident ?? ((fd, g) => { const r = sampleResidentSync(backend, fd, g); temps.push(r); return r.buffer; });
  const program = buildSampleProgram(field, grid, provider);
  try {
    return await backend.run(program);
  } catch (e) {
    const diag = await backend.diagnostics(program.code).catch(() => []);
    throw new Error(`GPU sampling failed: ${e instanceof Error ? e.message : String(e)}${diag.length ? `\n${diag.join("\n")}` : ""}`);
  } finally {
    if (temps.length) void backend.whenIdle().then(() => temps.forEach((t) => t.destroy()));
  }
}

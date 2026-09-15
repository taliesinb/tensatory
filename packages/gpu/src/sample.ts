import type { DenseGrid, ScalarFieldData, VectorFieldData } from "@tensatory/core";
import type { GpuBackend } from "./device";
import { buildSampleProgram } from "./program";

/**
 * GPU counterpart of `field.sampleOn(grid)`: scalar fields give one value per
 * grid point, vector fields D values per point (row-major, like core).
 */
export async function gpuSampleOn(backend: GpuBackend, field: ScalarFieldData | VectorFieldData, grid: DenseGrid): Promise<Float32Array> {
  const program = buildSampleProgram(field, grid);
  try {
    return await backend.run(program);
  } catch (e) {
    const diag = await backend.diagnostics(program.code).catch(() => []);
    throw new Error(`GPU sampling failed: ${e instanceof Error ? e.message : String(e)}${diag.length ? `\n${diag.join("\n")}` : ""}`);
  }
}

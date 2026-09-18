// The colour of a fused kernel's vertices (isolines, isosurfaces, streamlines, glyphs): where `colour_(p)` comes from.
//
//   * a ScalarFieldData: the field's program evaluated at every vertex — exact, and fine for cheap fields;
//   * a resident GpuGrid: multilinear interpolation of a grid sampled ONCE on the GPU — for costly (net-backed)
//     colour fields, where an evaluation per vertex was millions of net evaluations per level;
//   * "level": the colour IS the dispatch level (colour field = iso field); nothing is evaluated;
//   * undefined: 0.

import type { ScalarFieldData } from "@tensatory/core";
import type { KernelBuffer } from "./device";
import type { ProgramBuilder } from "./program";
import type { GpuGrid } from "./resident";
import { vecType } from "./wgsl";

export type ColourSource = ScalarFieldData | GpuGrid | "level" | undefined;

export const isResidentGrid = (c: ColourSource): c is GpuGrid => typeof c === "object" && c !== null && "buffer" in c && "grid" in c;

export interface ColourCode {
  /** the `fn colour_(p) -> f32` definition (and the resident buffer's binding declaration when there is one) */
  code: string;
  /** buffers to append to the kernel's list (the resident grid, bound at `binding`) */
  buffers: KernelBuffer[];
}

/**
 * Emit `colour_` for `colour`. `levelExpr` is the WGSL expression holding the dispatch level ("level" source);
 * `binding` is the index the resident grid is bound at (= its position in the kernel's buffer list).
 */
export function colourCode(b: ProgramBuilder, colour: ColourSource, D: number, levelExpr: string, binding: number): ColourCode {
  const P = vecType(D);
  if (colour === undefined) return { code: `fn colour_(p: ${P}) -> f32 { return 0.0; }`, buffers: [] };
  if (colour === "level") return { code: `fn colour_(p: ${P}) -> f32 { return ${levelExpr}; }`, buffers: [] };
  if (isResidentGrid(colour)) {
    const rd = b.residentReader(colour, "cgrid", 0);
    return {
      code: `@group(0) @binding(${binding}) var<storage, read> cgrid: array<f32>;\nfn colour_(p: ${P}) -> f32 { return ${rd}(p, -1); }`,
      buffers: [{ role: "r", buffer: colour.buffer }],
    };
  }
  const fn = b.scalar(colour);
  return { code: `fn colour_(p: ${P}) -> f32 { return ${fn}(p, -1); }`, buffers: [] };
}

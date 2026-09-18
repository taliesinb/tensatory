// Progressive EXACT recolouring of geometry that was coloured from a resident grid.
//
// A fused kernel colours its vertices from a resident (interpolated) grid when the colour field is costly
// (colour.ts) — fast, but visibly aliased on a surface. Once the geometry is at rest, this kernel evaluates the
// colour field's own program at a BATCH of records per frame and writes the exact colour back into the record
// buffer in place. Records are visited INTERLEAVED: with G frames to go, frame f recolours records f, f + G,
// f + 2G, … — every G-th record, so a batch is spread over the whole geometry (consecutive records are spatial
// neighbours) and the image sharpens uniformly, dither-like, instead of wiping; after G frames every record is
// exact. The kernel reads the set's live count from its indirect buffer, so the caller only needs the record
// capacity (or the last read-back count) to size the work.

import type { ScalarFieldData } from "@tensatory/core";
import { DenseGrid } from "@tensatory/core";
import type { GpuBackend } from "./device";
import { ProgramBuilder } from "./program";

/** where a record keeps its points and their colours (float offsets within the record) */
export interface RecordLayout {
  /** floats per record */
  floats: number;
  /** each coloured point: position offset (D floats) and colour offset (1 float) */
  points: { pos: number; col: number }[];
  /** index of the record count in the set's indirect buffer (u32) */
  countIndex: number;
  /** points are 2D or 3D */
  D: number;
}

export const VERT_LAYOUT: RecordLayout = { floats: 8, points: [{ pos: 0, col: 6 }], countIndex: 0, D: 3 }; // mesh.ts Vert (count = vertexCount)
export const SEG_LAYOUT: RecordLayout = { floats: 10, points: [{ pos: 0, col: 4 }, { pos: 2, col: 5 }], countIndex: 1, D: 2 }; // segments.ts Seg (count = instanceCount)
export const SEG3_LAYOUT: RecordLayout = { floats: 12, points: [{ pos: 0, col: 3 }, { pos: 4, col: 7 }], countIndex: 1, D: 3 }; // lines3d.ts Seg3

export interface Recolourer {
  /** recolour `phases` consecutive interleave phases starting at `phase`: records phase + i + j·stride for i < phases,
   *  j < perPhase (`perPhase · phases` threads); records beyond the live count are skipped */
  dispatch(buffer: GPUBuffer, indirect: GPUBuffer, phase: number, stride: number, perPhase: number, phases: number): void;
}

export function recolourer(backend: GpuBackend, field: ScalarFieldData, layout: RecordLayout): Recolourer {
  const D = layout.D;
  // the dispatch grid is irrelevant here (the field function is called with pos = -1); a 2-point grid on the box
  const b = new ProgramBuilder(new DenseGrid(new Array<number>(D).fill(2), field.box));
  const fn = b.scalar(field);
  const lib = b.library();
  const point = (pos: number) => (D === 2 ? `vec2<f32>(recs[o + ${pos}u], recs[o + ${pos + 1}u])` : `vec3<f32>(recs[o + ${pos}u], recs[o + ${pos + 1}u], recs[o + ${pos + 2}u])`);
  const code = `${lib.code}
@group(0) @binding(0) var<storage, read_write> recs: array<f32>;
@group(0) @binding(2) var<storage, read> ind: array<u32>;
@group(0) @binding(3) var<storage, read> params: array<u32>;
// params: [phase, stride, perPhase, phases]
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let t = id.x;
  if (t >= params[2] * params[3]) { return; }
  let k = params[0] + (t % params[3]) + (t / params[3]) * params[1];
  if (k >= ind[${layout.countIndex}u]) { return; }
  let o = k * ${layout.floats}u;
${layout.points.map((pt) => `  recs[o + ${pt.col}u] = ${fn}(${point(pt.pos)}, -1);`).join("\n")}
}`;
  return {
    dispatch(buffer, indirect, phase, stride, perPhase, phases) {
      backend.dispatch({
        code,
        invocations: perPhase * phases,
        buffers: [
          { role: "rw", buffer },
          { role: "r", data: lib.data },
          { role: "r", buffer: indirect },
          { role: "r", data: new Uint32Array([phase, stride, perPhase, phases]) },
        ],
      });
    },
  };
}

/** progress of one set's recolouring: frames done of `frames` (the interleave stride), and whether every record is exact */
export interface RecolourProgress { next: number; frames: number; done: boolean }
export const freshProgress = (): RecolourProgress => ({ next: 0, frames: 0, done: false });

/**
 * Round-robin scheduler: spend `budget` records over the pending jobs this frame (each job has `total` records,
 * the count if known else the capacity), updating their progress. The interleave stride G is fixed when a job
 * starts (⌈total / share⌉ with a fine share, so the lattice of records stays the same however the budget moves);
 * a step then covers as many consecutive phases as the share allows. Returns whether anything is left.
 */
export function recolourStep(jobs: { r: Recolourer; buffer: GPUBuffer; indirect: GPUBuffer; total: number; progress: RecolourProgress }[], budget: number): boolean {
  const pending = jobs.filter((j) => !j.progress.done && j.total > 0);
  if (!pending.length) return false;
  const share = Math.max(64, Math.floor(budget / pending.length));
  for (const j of pending) {
    if (j.progress.frames === 0) j.progress.frames = Math.max(1, Math.ceil(j.total / Math.min(share, RECOLOUR_LATTICE_SHARE)));
    const G = j.progress.frames, perPhase = Math.ceil(j.total / G);
    const phases = Math.max(1, Math.min(G - j.progress.next, Math.floor(share / perPhase)));
    j.r.dispatch(j.buffer, j.indirect, j.progress.next, G, perPhase, phases);
    j.progress.next += phases;
    if (j.progress.next >= G) j.progress.done = true;
  }
  return pending.some((j) => !j.progress.done);
}
/** the share that fixes a job's interleave lattice: fine enough that the smallest budget still spreads a batch over the geometry */
const RECOLOUR_LATTICE_SHARE = 4096;

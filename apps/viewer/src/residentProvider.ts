// Resident grids of nets one GPU lane cannot evaluate (the cooperative kernel, gpu/coop.ts): a `ResidentProvider`
// over a resident-grid cache, so a program that reads such a net — an expression over it, the central differences
// standing in for its gradient — finds its values on the dispatch grid already enqueued, and every program on the
// same grid shares them.

import type { DenseGrid } from "@tensatory/core";
import { sampleResidentSync, type GpuBackend, type GpuGrid, type ResidentProvider } from "@tensatory/gpu";
import type { Cache } from "./cache";

const ids = new WeakMap<object, number>();
let next = 0;
/** a stable key for a field DATA object (fields have ids, their data objects do not) */
export const dataKey = (fd: object): string => { let n = ids.get(fd); if (n === undefined) ids.set(fd, (n = next++)); return `data#${n}`; };
export const gridSig = (g: DenseGrid): string => `${g.size.join("x")}|${g.box.intervals.flat().join(",")}`;

/** a provider whose grids live in `grids` (the caller's LRU: trimmed by the memory cap, cleared with the bundle) */
export function residentProvider(gpu: GpuBackend, grids: Cache<GpuGrid>): ResidentProvider {
  const provider: ResidentProvider = (fd, grid) => grids.getOr(`${dataKey(fd)}|${gridSig(grid)}`, () => sampleResidentSync(gpu, fd, grid, provider)).buffer;
  return provider;
}

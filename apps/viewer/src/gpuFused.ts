// Fully GPU-resident geometry for the fused (GPU compute + GPU render) path.
// Everything here is synchronous from the CPU's point of view: kernels are
// enqueued and the GPU queue orders them before the render pass, so a frame
// needs no readback and no await. Results are cached by key; a level change
// re-dispatches into the same segment set.

import type { DenseGrid, Lattice, ScalarFieldData, StreamlineSeeds, VectorFieldData } from "@tensatory/core";
import {
  SEG_FLOATS,
  allocSegments,
  blurResidentSync,
  fusedGlyphs,
  fusedIsolines,
  fusedStreamlines,
  resetSegments,
  sampleResidentSync,
  smoothedIsolines,
  uploadGrid,
  uploadSegments,
  type FusedGlyphs,
  type FusedIsolines,
  type GlyphDispatch,
  type FusedStreamlineOptions,
  type FusedStreamlines,
  type GpuBackend,
  type GpuGrid,
  type GpuSegments,
  type SmoothedIsolines,
  type ColourSource,
  type RecolourProgress,
  freshProgress,
} from "@tensatory/gpu";
import { Cache, uidOf, type MemoryUser } from "./cache";

/** a segment set with its count read back after every dispatch (see `track`) */
interface Counted { segs: GpuSegments; stamp: string; pending: boolean; count: number; overflow: boolean; recolour?: RecolourProgress }
/** segments a family of isoline sets (field, colour, options — not the grid) produced, at the resolution measured */
interface Complexity { records: number; n: number; t: number }
const bufBytes = (o: { buffer: GPUBuffer }) => o.buffer.size;
const MIN_RECORDS = 16384, MARGIN = 1.5;

export interface FusedInfo { segments: number; capacity: number; overflow: boolean }

export class FusedGeometry implements MemoryUser {
  // ∝ n²
  private readonly grids = new Cache<GpuGrid>(24, (g) => g.destroy(), bufBytes);
  private readonly blurred = new Cache<GpuGrid>(8, (g) => g.destroy(), bufBytes);
  private readonly smoothKernels = new Cache<SmoothedIsolines>(8, (k) => k.destroy()); // own n²-sized scratch (not counted: destroyed with the kernel)
  // ∝ n
  private readonly isoKernels = new Cache<FusedIsolines>(16, () => {});
  private readonly isoSets = new Cache<Counted>(32, (s) => s.segs.destroy(), (s) => s.segs.buffer.size);
  private readonly streamKernels = new Cache<{ kernel: FusedStreamlines; segs: GpuSegments; recolour?: RecolourProgress }>(8, (e) => e.segs.destroy(), (e) => e.segs.buffer.size);
  private readonly uploaded = new Cache<GpuSegments>(32, (s) => s.destroy(), bufBytes);
  // glyphs: one kernel per (field, colour) — the lattice is a dispatch parameter — and one segment set per kernel
  private readonly glyphKernels = new Cache<FusedGlyphs>(4, (k) => k.destroy());
  private readonly glyphSets = new Cache<{ segs: GpuSegments; stamp: string; pending: boolean; read: string }>(4, (e) => e.segs.destroy(), (e) => e.segs.buffer.size);
  private readonly complexity = new Map<string, Complexity>();
  /** what the last frame's isoline sets held (summed over the sets used) */
  info: FusedInfo = { segments: 0, capacity: 0, overflow: false };

  /** @param invalidate called when a count readback resized a set: render again */
  constructor(readonly gpu: GpuBackend, private readonly invalidate: () => void) {}

  clear(): void {
    for (const c of [this.grids, this.blurred, this.smoothKernels, this.isoKernels, this.isoSets, this.streamKernels, this.uploaded, this.glyphSets, this.glyphKernels] as Cache<unknown>[]) c.clear();
    this.complexity.clear();
  }
  /** call at the start of a frame: the info accumulates over the frame's isoline sets */
  beginFrame(): void { this.info = { segments: 0, capacity: 0, overflow: false }; }
  /** forget that the isoline sets are up to date: the next frame dispatches them again (timing without compiles) */
  redo(): void { for (const s of this.isoSets.values()) s.stamp = ""; }

  debug(): Record<string, string> {
    const e = { grids: this.grids, blurred: this.blurred, isoSets: this.isoSets, streamKernels: this.streamKernels, uploaded: this.uploaded };
    return Object.fromEntries(Object.entries(e).map(([k, c]) => [k, `${c.size} entries, ${(c.bytes / 2 ** 20).toFixed(1)} MB (${(c.liveBytes / 2 ** 20).toFixed(1)} live)`]));
  }
  memory(): { volume: number; surface: number; cpu: number } {
    return { volume: this.grids.liveBytes + this.blurred.liveBytes, surface: this.isoSets.liveBytes + this.streamKernels.liveBytes + this.uploaded.liveBytes + this.glyphSets.liveBytes, cpu: 0 };
  }
  trim(bytes: number): number {
    let freed = 0;
    for (const c of [this.uploaded, this.glyphSets, this.isoSets, this.streamKernels, this.smoothKernels, this.blurred, this.grids] as Cache<unknown>[]) {
      if (freed >= bytes) break;
      freed += c.trim(bytes - freed);
    }
    return freed;
  }

  /** resident samples of `field` on `grid` (enqueued on first use) */
  grid(key: string, field: ScalarFieldData | VectorFieldData, grid: DenseGrid): GpuGrid {
    return this.grids.getOr(key, () => sampleResidentSync(this.gpu, field, grid));
  }
  /** an already-resident grid, or undefined (no sampling) */
  gridIfResident(key: string): GpuGrid | undefined { return this.grids.get(key); }

  /** CPU values as a resident grid (CPU compute + GPU render) */
  uploadGrid(key: string, grid: DenseGrid, values: ArrayLike<number>, channels: number): GpuGrid {
    return this.grids.getOr(key, () => uploadGrid(this.gpu, grid, values, channels));
  }

  /** box-blurred copy of a resident grid (core's `boxBlur` semantics) */
  blur(key: string, values: GpuGrid, radius: number): GpuGrid {
    if (radius <= 0) return values;
    return this.blurred.getOr(`${key}|blur${radius}`, () => blurResidentSync(this.gpu, values, radius));
  }

  /**
   * Read the segment count of `set` back once its dispatch ran (one readback in flight per set); the family's
   * complexity becomes a running max with a slow decay (half-life 60 s, so a cycle of animated levels keeps
   * its peak), and an overflowed set is flagged for reallocation at the true count.
   */
  private track(cs: Counted, family: string, n: number): void {
    if (cs.pending) return;
    cs.pending = true;
    const stamp = cs.stamp;
    void this.gpu.readCounter(cs.segs.indirect, 1).then((count) => {
      cs.pending = false;
      const prev = this.complexity.get(family), now = performance.now();
      const scaled = prev ? prev.records * (n / prev.n) * 0.5 ** ((now - prev.t) / 60_000) : 0; // isolines: segments ∝ n
      this.complexity.set(family, { records: Math.max(count, scaled), n, t: now });
      if (cs.stamp !== stamp) return;
      const changed = cs.count !== count;
      cs.count = count;
      if (count > cs.segs.capacity) { cs.overflow = true; cs.stamp = ""; }
      if (changed) this.invalidate(); // the resolution row shows the count; an overflow reallocates
    }).catch(() => { cs.pending = false; });
  }

  /** the segment set for `setKey`, sized from the family's measured complexity (a modest guess before any) */
  private isoSet(setKey: string, family: string, values: GpuGrid, max: number): Counted {
    const n = Math.max(...values.grid.size);
    const cells = (values.grid.size[0]! - 1) * (values.grid.size[1]! - 1);
    let cs = this.isoSets.get(setKey);
    if (cs?.overflow) { this.complexity.set(family, { records: Math.max(cs.count, this.complexity.get(family)?.records ?? 0), n, t: performance.now() }); this.isoSets.delete(setKey); cs = undefined; this.info.overflow = true; }
    const cx = this.complexity.get(family);
    const want = Math.max(1, Math.min(max, Math.floor(this.gpu.maxBufferBytes / (SEG_FLOATS * 4)), Math.max(MIN_RECORDS, Math.ceil(cx ? cx.records * (n / cx.n) * MARGIN : Math.min(cells, 1 << 18)))));
    if (cs && (cs.segs.capacity < want / MARGIN || cs.segs.capacity > want * 4)) { this.isoSets.delete(setKey); cs = undefined; }
    if (!cs) cs = this.isoSets.set(setKey, { segs: allocSegments(this.gpu, want, false), stamp: "", pending: false, count: 0, overflow: false });
    return cs;
  }

  /** marching-squares isolines of a resident grid with Taubin smoothing on the edge graph (non-exact path with `line` > 0) */
  smoothedIsolines(kernelKey: string, setKey: string, values: GpuGrid, colour: ColourSource, level: number, iterations: number): GpuSegments {
    const kernel = this.smoothKernels.getOr(`${kernelKey}#${uidOf(values)}`, () => smoothedIsolines(this.gpu, values, colour)); // the kernel reads THIS grid's buffer
    const family = `${kernelKey.replace(/\|\d+x\d+\|[^|]*/, "")}|smooth`;
    const cs = this.isoSet(setKey, family, values, kernel.capacity);
    const stamp = `${kernelKey}|${level}|it${iterations}`;
    if (cs.stamp !== stamp) { resetSegments(this.gpu, cs.segs); kernel.dispatch(cs.segs, level, iterations); cs.stamp = stamp; cs.recolour = freshProgress(); this.track(cs, family, Math.max(...values.grid.size)); }
    this.info.segments += cs.count; this.info.capacity += cs.segs.capacity;
    return cs.segs;
  }

  /**
   * Segments of the isoline of `field` at `level`. `kernelKey` identifies (field, grid, colour field);
   * `setKey` identifies the slot (kernel + level index) whose segment set is reused across level changes.
   */
  isolines(kernelKey: string, setKey: string, field: ScalarFieldData, values: GpuGrid, colour: ColourSource, level: number, tol: number, exact?: boolean): GpuSegments {
    const kernel = this.isoKernels.getOr(`${kernelKey}#${uidOf(values)}`, () => fusedIsolines(this.gpu, field, values, colour, { exact }));
    const family = `${kernelKey.replace(/\|\d+x\d+\|[^|]*/, "")}|${exact ? "exact" : "ms"}`; // the kernel key without its grid part
    const cs = this.isoSet(setKey, family, values, kernel.capacity);
    const stamp = `${kernelKey}|${level}|${tol.toExponential(3)}`;
    if (cs.stamp !== stamp) { resetSegments(this.gpu, cs.segs); kernel.dispatch(cs.segs, level, tol); cs.stamp = stamp; cs.recolour = freshProgress(); this.track(cs, family, Math.max(...values.grid.size)); }
    this.info.segments += cs.count; this.info.capacity += cs.segs.capacity;
    return cs.segs;
  }

  /** recolouring progress of an isoline set (by its set key) or a streamline set (by its key): total = count if known, else capacity */
  recolourProgress(key: string): { total: number; progress: RecolourProgress } | undefined {
    const cs = this.isoSets.get(key);
    if (cs) return { total: cs.count || cs.segs.capacity, progress: (cs.recolour ??= freshProgress()) };
    const st = this.streamKernels.get(key);
    if (st) return { total: st.segs.capacity, progress: (st.recolour ??= freshProgress()) };
    return undefined;
  }

  /** segments of the streamlines through resident `vectors` from `seeds` (key covers everything that affects them) */
  streamlines(key: string, vectors: GpuGrid, seeds: StreamlineSeeds, opts: FusedStreamlineOptions, colour: ColourSource): GpuSegments {
    return this.streamKernels.getOr(key, () => {
      const kernel = fusedStreamlines(this.gpu, vectors, seeds, opts, colour);
      const segs = allocSegments(this.gpu, kernel.capacity, true);
      kernel.dispatch(segs);
      return { kernel, segs };
    }).segs;
  }

  /**
   * Arrow glyphs of `field` on `lattice` (three segments per point), coloured by `colour`. The kernel is built once
   * per `kernelKey` (field + colour) and re-dispatched whenever `latticeKey` or the dispatch parameters (style,
   * cutoff) change; the set is regrown when the lattice outgrows it. `onMax` receives the normalizing norm once the
   * readback lands.
   */
  glyphs(kernelKey: string, field: VectorFieldData, lattice: Lattice, latticeKey: string, d: GlyphDispatch, colour: ColourSource, onMax?: (max: number) => void): GpuSegments {
    latticeKey = `${latticeKey}|${d.style ?? ""}|${(d.minLength ?? 0).toPrecision(4)}`;
    const kernel = this.glyphKernels.getOr(kernelKey, () => fusedGlyphs(this.gpu, field, colour));
    const need = kernel.capacityFor(lattice);
    let e = this.glyphSets.get(kernelKey);
    if (e && (e.segs.capacity < need || e.segs.capacity > need * 4)) { this.glyphSets.delete(kernelKey); e = undefined; }
    if (!e) e = this.glyphSets.set(kernelKey, { segs: allocSegments(this.gpu, Math.ceil(need * 1.5), false), stamp: "", pending: false, read: "" });
    if (e.stamp !== latticeKey) {
      resetSegments(this.gpu, e.segs);
      kernel.dispatch(e.segs, lattice, d);
      e.stamp = latticeKey;
      if (onMax) this.readGlyphMax(e, kernel, onMax);
    }
    return e.segs;
  }
  /** one maximum readback in flight per set; a dispatch during the wait is read after it (the label follows the last lattice) */
  private readGlyphMax(e: { stamp: string; pending: boolean; read: string }, kernel: FusedGlyphs, onMax: (max: number) => void): void {
    if (e.pending) return;
    e.pending = true;
    const stamp = e.stamp;
    void kernel.readMaxNorm().then((m) => { e.read = stamp; onMax(m); }).catch(() => {}).finally(() => { e.pending = false; if (e.stamp !== e.read) this.readGlyphMax(e, kernel, onMax); });
  }

  /** CPU-computed lines packed as Seg records (GPU render of CPU geometry) */
  uploadedSegments(key: string, pack: () => Float32Array, particles: boolean): GpuSegments {
    return this.uploaded.getOr(key, () => uploadSegments(this.gpu, pack(), particles));
  }
}

// A small LRU cache with byte accounting, shared by the resident-geometry caches of both arms (view3d.ts,
// gpuFused.ts, sampler.ts). Entries are stamped with the frame that last used them so `trim` can free memory
// without touching what the current frame is drawing.

export class Cache<V> {
  /** the current frame number; main.ts advances it once per rendered frame */
  static frame = 0;

  private readonly map = new Map<string, { v: V; bytes: number; frame: number }>();
  private total = 0;

  /**
   * @param max      entry-count bound (oldest evicted first)
   * @param dispose  called for every evicted / deleted / cleared value
   * @param bytesOf  memory attributed to a value (device buffers, typed arrays); 0 when unknown
   */
  constructor(private readonly max: number, private readonly dispose: (v: V) => void, private readonly bytesOf: (v: V) => number = () => 0) {}

  get size(): number { return this.map.size; }
  /** bytes of every entry */
  get bytes(): number { return this.total; }
  /** bytes of the entries the current frame used (the working set; the rest is trimmable) */
  get liveBytes(): number { let b = 0; for (const e of this.map.values()) if (e.frame === Cache.frame) b += e.bytes; return b; }

  /** the value under `key`, marked as used by the current frame and moved to the most-recent end */
  get(key: string): V | undefined {
    const e = this.map.get(key);
    if (!e) return undefined;
    this.map.delete(key); e.frame = Cache.frame; this.map.set(key, e);
    return e.v;
  }
  has(key: string): boolean { return this.map.has(key); }

  /** store `v` (disposing a previous value under the same key), then evict beyond `max` */
  set(key: string, v: V): V {
    this.delete(key);
    const bytes = this.bytesOf(v);
    this.map.set(key, { v, bytes, frame: Cache.frame }); this.total += bytes;
    while (this.map.size > this.max) this.delete(this.map.keys().next().value!);
    return v;
  }

  /** `get`, else `set(make())` */
  getOr(key: string, make: () => V): V {
    const have = this.get(key);
    return have !== undefined ? have : this.set(key, make());
  }

  /** re-measure an entry whose value grew or shrank in place */
  refresh(key: string): void {
    const e = this.map.get(key);
    if (!e) return;
    this.total -= e.bytes; e.bytes = this.bytesOf(e.v); this.total += e.bytes;
  }

  delete(key: string): boolean {
    const e = this.map.get(key);
    if (!e) return false;
    this.map.delete(key); this.total -= e.bytes; this.dispose(e.v);
    return true;
  }

  clear(): void { for (const k of [...this.map.keys()]) this.delete(k); }

  /** remove every entry WITHOUT disposing; the caller owns the values (to destroy them once the GPU queue is past
   *  dispatches that still read them) */
  takeAll(): V[] { const vs = [...this.map.values()].map((e) => e.v); this.map.clear(); this.total = 0; return vs; }

  values(): IterableIterator<V> { return [...this.map.values()].map((e) => e.v).values(); }

  /**
   * Free least-recently-used entries (never those used by the current frame) until at least `bytes` bytes are
   * released or nothing more can go; returns the bytes released.
   */
  trim(bytes: number): number {
    let freed = 0;
    for (const [k, e] of this.map) {
      if (freed >= bytes) break;
      if (e.frame === Cache.frame) continue;
      freed += e.bytes; this.delete(k);
    }
    return freed;
  }
}

/**
 * A stable id per object, for cache keys that must follow an object's IDENTITY: a kernel built over a resident
 * grid holds that grid's buffer, so when the grid is evicted and rebuilt the kernel must be rebuilt too.
 */
export const uidOf = ((): ((o: object) => number) => { const ids = new WeakMap<object, number>(); let next = 0; return (o) => { let id = ids.get(o); if (id === undefined) ids.set(o, (id = ++next)); return id; }; })();

/** a memory pool: the caches of one arm, summed and trimmed together */
export interface MemoryUser {
  /** bytes the current frame uses (its working set — everything else is trimmable), split by how they scale with
   *  the resolution n: `volume` ∝ nᴰ (grids), `surface` ∝ nᴰ⁻¹ (meshes, lines); `cpu` = bytes held in JS arrays
   *  (all entries, not only live ones) rather than device buffers, which the backend counts itself */
  memory(): { volume: number; surface: number; cpu: number };
  /** release up to `bytes` of entries the current frame does not use; returns the bytes released */
  trim(bytes: number): number;
}

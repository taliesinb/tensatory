// Controls: the rows a bundle asks for (schema/distribution.ts RandomWidgetSpec
// on `random` arrays and on displaced-net directions) and the ADJUSTMENTS a
// user makes in them (reseed, scale).
//
// Adjustments never touch live objects: `adjustSpec` returns a new BundleSpec
// with every governed random array re-salted and every governed scale
// multiplied, and the caller builds a fresh Bundle from it. Nothing about
// shapes changes, so every program compiled from the adjusted bundle has the
// same structure — the GPU code is byte-identical and only the packed
// constant data differs (no shader recompiles; the viewer keys its caches by
// the bundle revision so old results simply age out).
//
// Discovery is a walk over the spec: a `{ type: "displace" }` net's
// directions with a `widget` are direction rows (reseed re-salts every random
// array inside `arrays`, scale multiplies the direction's `scale`); a
// `{ type: "random" }` array with a `widget` is an array row (reseed re-salts
// it, scale multiplies `dist.scale` when the distribution has one). A random
// array inside a widgeted direction may also carry its own widget: both rows
// govern it. Rows sharing an `id` merge; the default id is the spec path.

import type { BundleSpec, DirectionSpec, RandomSeed, RandomWidgetSpec, SymbolicRandomScalarArraySpec } from "@tensatory/schema";
import { hasScale, saltSeed } from "../arrays/random";

export interface ControlRow {
  readonly id: string;
  readonly label: string;
  readonly kind: "direction" | "array";
  /** spec paths of the random arrays this row re-salts (none: no reseed button) */
  readonly members: readonly string[];
  /** whether the row has a scale slider */
  readonly hasScale: boolean;
  readonly scaleRange: readonly [number, number];
  readonly scaleSteps: number | undefined;
}

/** one row's adjustment: a 32-bit salt hashed into its members' seeds, a multiplier on its scale */
export interface RowAdjustment {
  seed?: number;
  scale?: number;
}
export type Adjustments = Readonly<Record<string, RowAdjustment>>;

const DEFAULT_SCALE_RANGE: readonly [number, number] = [0.01, 100];

type Json = unknown;
const sub = (path: string, k: string | number) => (path ? `${path}.${k}` : String(k));
const isObj = (v: Json): v is Record<string, Json> => typeof v === "object" && v !== null && !Array.isArray(v);
const isRandom = (v: Json): v is SymbolicRandomScalarArraySpec & Record<string, Json> => isObj(v) && v.type === "random" && isObj(v.dist);
const isDisplace = (v: Json): v is Record<string, Json> & { directions: Json[] } => isObj(v) && v.type === "displace" && Array.isArray(v.directions);

interface Visitor {
  /** a direction with a widget, before its arrays are visited */
  enterDirection?(dir: DirectionSpec, path: string, rowId: string): void;
  /** a direction with a widget, after its arrays were visited (`dir` has the rebuilt arrays); return value: replacement */
  direction?(dir: DirectionSpec, path: string, rowId: string): DirectionSpec;
  /** a random array; `dirs` are the ids of the widgeted directions enclosing it, innermost last */
  random?(arr: SymbolicRandomScalarArraySpec, path: string, dirs: readonly string[]): SymbolicRandomScalarArraySpec;
}

/** walk the spec, rebuilding it through the visitor (identity when the visitor returns its input) */
function walk(v: Json, path: string, dirs: readonly string[], vis: Visitor): Json {
  if (Array.isArray(v)) return v.map((x, i) => walk(x, sub(path, i), dirs, vis));
  if (!isObj(v)) return v;
  if (isRandom(v)) return vis.random?.(v, path, dirs) ?? v;
  if (isDisplace(v)) {
    const out: Record<string, Json> = { ...v };
    out.directions = v.directions.map((d, k) => {
      const p = sub(path, `directions.${k}`);
      if (!isObj(d)) return d;
      const dir = d as unknown as DirectionSpec;
      const widget = dir.widget ?? undefined;
      const rowId = widget ? widget.id ?? p : undefined;
      const inner = rowId === undefined ? dirs : [...dirs, rowId];
      if (rowId !== undefined) vis.enterDirection?.(dir, p, rowId);
      const arrays = walk(dir.arrays, sub(p, "arrays"), inner, vis) as DirectionSpec["arrays"];
      const rebuilt: DirectionSpec = arrays === dir.arrays ? dir : { ...dir, arrays };
      return rowId === undefined ? rebuilt : vis.direction?.(rebuilt, p, rowId) ?? rebuilt;
    });
    if ("net" in out) out.net = walk(out.net, sub(path, "net"), dirs, vis);
    return out;
  }
  const out: Record<string, Json> = {};
  for (const [k, x] of Object.entries(v)) out[k] = walk(x, sub(path, k), dirs, vis);
  return out;
}

/** the Controls-pane rows of a bundle, in spec order (merged by id) */
export function controlRows(spec: BundleSpec): ControlRow[] {
  const rows = new Map<string, { label: string; kind: ControlRow["kind"]; members: string[]; hasScale: boolean; widget: RandomWidgetSpec }>();
  const add = (id: string, label: string, kind: ControlRow["kind"], scale: boolean, widget: RandomWidgetSpec, members: string[]) => {
    const r = rows.get(id);
    if (r) { r.members.push(...members); r.hasScale ||= scale; return; }
    rows.set(id, { label, kind, members: [...members], hasScale: scale, widget });
  };
  walk(spec, "", [], {
    enterDirection(dir, path, rowId) {
      const members = randomPaths(dir.arrays, sub(path, "arrays"));
      add(rowId, dir.widget!.label ?? dir.name ?? rowId, "direction", true, dir.widget!, members);
    },
    random(arr, path) {
      const w = arr.widget ?? undefined;
      if (w) add(w.id ?? path, w.label ?? w.id ?? path, "array", hasScale(arr.dist), w, [path]);
      return arr;
    },
  });
  return [...rows.entries()].map(([id, r]) => ({
    id, label: r.label, kind: r.kind, members: r.members, hasScale: r.hasScale,
    scaleRange: r.widget.scaleRange ?? DEFAULT_SCALE_RANGE, scaleSteps: r.widget.scaleSteps,
  }));
}

/** spec paths of the random arrays under `v` */
function randomPaths(v: Json, path: string): string[] {
  const out: string[] = [];
  walk(v, path, [], { random(arr, p) { out.push(p); return arr; } });
  return out;
}

/**
 * Apply adjustments: a new spec in which every random array governed by an
 * adjusted row has its seed salted by the row's `seed` (all governing rows, in
 * nesting order) and its `dist.scale` multiplied by array rows' `scale`; every
 * adjusted direction has its `scale` multiplied. Rows without an adjustment
 * (or an empty one) leave their members alone, so `adjustSpec(spec, {})` is a
 * structural copy.
 */
export function adjustSpec(spec: BundleSpec, adj: Adjustments): BundleSpec {
  const out = walk(spec, "", [], {
    direction(dir, _path, rowId) {
      const a = adj[rowId];
      if (a?.scale === undefined || a.scale === 1) return dir;
      return { ...dir, scale: (dir.scale ?? 1) * a.scale };
    },
    random(arr, path, dirs) {
      const own = arr.widget ? arr.widget.id ?? path : undefined;
      const governing = own === undefined ? dirs : [...dirs, own];
      let seed: RandomSeed = arr.dist.seed;
      let salted = false;
      for (const id of governing) {
        const s = adj[id]?.seed;
        if (s === undefined) continue;
        seed = saltSeed(seed, s);
        salted = true;
      }
      const scale = own === undefined ? undefined : adj[own]?.scale;
      const rescale = scale !== undefined && scale !== 1 && hasScale(arr.dist);
      if (!salted && !rescale) return arr;
      const dist = { ...arr.dist, seed } as SymbolicRandomScalarArraySpec["dist"];
      if (rescale) (dist as { scale?: number }).scale = ((dist as { scale?: number }).scale ?? 1) * scale;
      return { ...arr, dist };
    },
  });
  return out as BundleSpec;
}

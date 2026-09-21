// The INVENTORY of a Tensatory document that may not be loaded: what the bundle table (picker.ts) shows for every
// alternative — its own summary, and its spaces / fields / curves counted from the spec (core `inventoryOf`). The
// document is fetched (JSON only, never its sidecar arrays: `bundles/index.json` carries no structure, structure is
// discovered — AGENTS.md); a sweep's inventory is that of every member, `common` merged in, members by path fetched
// too (relative to the sweep document). Results are cached per URL for the session; a failure is a result as well
// (the table shows "?" with the reason), so nothing is refetched in a loop.

import { Bundle, Sweep, inventoryOf, mergeCommon, rootKind } from "@tensatory/core";
import type { Inventory } from "@tensatory/core";

export interface DocInventory {
  /** the document's own one-line summary (a not-yet-loaded bundle's index entry may lag behind it) */
  summary: string | undefined;
  /** a sweep's member count; undefined for a lone bundle */
  members: number | undefined;
  /** the bundle's inventory, or one per member of a sweep */
  inventories: Inventory[];
  /** a sweep's members by id: their (merged) bundle's name and summary, for describing members not loaded yet */
  memberInfo: Record<string, { name?: string | undefined; summary?: string | undefined }> | undefined;
}
export type DocInventoryResult = { ok: true; inv: DocInventory } | { ok: false; error: string };

const cache = new Map<string, Promise<DocInventoryResult>>();

/** the inventory of the document at `url`, fetched once; `onSettle` runs when a fresh fetch lands (to redraw a table) */
export function docInventory(url: URL, onSettle?: () => void): Promise<DocInventoryResult> {
  const key = url.href;
  let p = cache.get(key);
  if (!p) {
    p = compute(url)
      .catch((e): DocInventoryResult => ({ ok: false, error: e instanceof Error ? e.message : String(e) }))
      .then((r) => { settled.set(key, r); onSettle?.(); return r; });
    cache.set(key, p);
  }
  return p;
}

/** the settled result, if any (the table is drawn synchronously; a pending document shows "…") */
export function docInventoryNow(url: URL): DocInventoryResult | undefined { return settled.get(url.href); }
const settled = new Map<string, DocInventoryResult>();

async function fetchJson(url: URL): Promise<unknown> {
  const res = await fetch(url, { cache: "no-cache" });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

async function compute(url: URL): Promise<DocInventoryResult> {
  const json = await fetchJson(url);
  let out: DocInventory;
  if (rootKind(json) === "sweep") {
    const spec = Sweep.validate(json);
    const merged = await Promise.all(Object.entries(spec.members).map(async ([id, m]) => {
      const member = typeof m.bundle === "string" ? Bundle.validate(await fetchJson(new URL(m.bundle, url))) : m.bundle;
      return [id, mergeCommon(spec.common, member)] as const;
    }));
    out = {
      summary: spec.summary, members: merged.length, inventories: merged.map(([, m]) => inventoryOf(m)),
      memberInfo: Object.fromEntries(merged.map(([id, m]) => [id, { name: m.name, summary: m.summary }])),
    };
  } else {
    const spec = Bundle.validate(json);
    out = { summary: spec.summary, members: undefined, inventories: [inventoryOf(spec)], memberInfo: undefined };
  }
  return { ok: true, inv: out };
}

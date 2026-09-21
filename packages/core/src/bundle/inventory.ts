// What a bundle HOLDS, counted: its manifolds by dimension, its fields and curves by kind — the columns of the
// viewer's bundle / space tables (`apps/viewer/src/picker.ts`). Like `signatureOf`, this reads the SPEC only, so it
// works for a document that has not been loaded (no sidecars, nothing built); a field that would fail to build
// still counts, since the table describes what the document declares.

import type { BundleSpec } from "@tensatory/schema";
import { inferDims } from "./bundle";

export interface Inventory {
  /** manifolds per dimension, ascending by dimension; a bundle without `manifolds` has its one implicit space (dimension inferred from the fields, 0 when nothing says) */
  spaces: { dims: number; count: number }[];
  scalars: number;
  vectors: number;
  curves: number;
}

/**
 * Count a bundle's spaces, fields and curves; with `manifold`, only the fields and curves whose domain is that manifold
 * (and `spaces` is that one manifold, or empty when the bundle has no such manifold).
 */
export function inventoryOf(spec: BundleSpec, manifold?: string): Inventory {
  let manifolds = spec.manifolds ?? {};
  if (!Object.keys(manifolds).length) { let d = 0; try { d = inferDims(spec); } catch { /* nothing says */ } manifolds = { default: { numDims: d } }; }
  const ids = Object.keys(manifolds);
  const dflt = spec.defaultManifold ?? (ids.length === 1 ? ids[0] : undefined);
  const domainOf = (id: string | undefined): string | undefined => id ?? dflt;
  const counts = (id: string | undefined) => manifold === undefined || domainOf(id) === manifold;
  const byDims = new Map<number, number>();
  for (const id of manifold === undefined ? ids : ids.filter((i) => i === manifold)) { const d = manifolds[id]!.numDims; byDims.set(d, (byDims.get(d) ?? 0) + 1); }
  const inv: Inventory = { spaces: [...byDims].sort((a, b) => a[0] - b[0]).map(([dims, count]) => ({ dims, count })), scalars: 0, vectors: 0, curves: 0 };
  for (const f of Object.values(spec.fields)) if (counts(f.domain)) { if (f.kind === "scalar") inv.scalars++; else inv.vectors++; }
  for (const c of Object.values(spec.curves ?? {})) if (counts(c.domain)) inv.curves++;
  return inv;
}

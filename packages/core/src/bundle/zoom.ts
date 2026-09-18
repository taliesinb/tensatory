// Box zoom: scale the domain of the SYMBOLIC fields of one manifold around
// their box centres. Symbolic data (`symbolic` / `symbolicv` / `net` / `netv`)
// can be evaluated anywhere, so its box is a choice, not a fact; sampled data
// (`dense*` / `sparse*`) is left alone — its box is its support. Pointwise
// fields take their box from their arguments, pullbacks from their inner
// field (a translation or linear map sends the centre to the centre), so
// recursing into inline argument specs is enough; arguments referenced by id
// are zoomed as fields of the manifold themselves.
//
// Like `adjustSpec` this returns a new spec for the caller to rebuild a
// Bundle from; the field programs keep their code (a box is never baked into
// symbolic WGSL), only the statistics and grids change.

import type { BundleSpec, FieldSpec, ScalarFieldDataSpec, VectorFieldDataSpec } from "@tensatory/schema";
import { Box } from "../geometry/box";
import { inferDims } from "./bundle";

/** the manifold id a field / point set spec belongs to */
export function domainOf(spec: BundleSpec, domain: string | undefined): string | undefined {
  if (domain !== undefined) return domain;
  const ids = Object.keys(spec.manifolds ?? {});
  return spec.defaultManifold ?? (ids.length <= 1 ? ids[0] ?? "default" : undefined);
}

type DataSpec = ScalarFieldDataSpec | VectorFieldDataSpec;

function zoomData(d: DataSpec, dimCount: number, factor: number): DataSpec {
  switch (d.type) {
    case "symbolic": case "symbolicv": case "net": case "netv": {
      const box = d.box ? Box.fromSpec(d.box) : Box.unit(dimCount);
      const zoomed = box.scale(box.center, new Array<number>(dimCount).fill(factor));
      return { ...d, box: zoomed.intervals.map(([a, b]) => [a, b] as [number, number]) };
    }
    case "pointwise": case "pointwisev": {
      const map = <T extends DataSpec>(r: Record<string, T | string> | undefined) =>
        r && Object.fromEntries(Object.entries(r).map(([k, v]) => [k, typeof v === "string" ? v : zoomData(v, dimCount, factor)]));
      return { ...d, ...(d.scalars ? { scalars: map(d.scalars as Record<string, ScalarFieldDataSpec | string>) } : {}), ...(d.vectors ? { vectors: map(d.vectors as Record<string, VectorFieldDataSpec | string>) } : {}) } as DataSpec;
    }
    case "translate": case "scale":
      return typeof d.arg === "string" ? d : ({ ...d, arg: zoomData(d.arg, dimCount, factor) } as DataSpec);
    default:
      return d;
  }
}

/** whether the manifold has a field whose box a zoom would change */
export function zoomable(spec: BundleSpec, manifold: string): boolean {
  const before = JSON.stringify(Object.values(spec.fields).filter((f) => domainOf(spec, f.domain) === manifold));
  return JSON.stringify(Object.values(zoomBoxes(spec, manifold, 2).fields).filter((f) => domainOf(spec, f.domain) === manifold)) !== before;
}

/** a new spec with the symbolic fields of `manifold` zoomed by `factor` (> 1 widens) around their box centres */
export function zoomBoxes(spec: BundleSpec, manifold: string, factor: number): BundleSpec {
  if (factor === 1) return spec;
  const dimCount = spec.manifolds?.[manifold]?.numDims ?? (Object.keys(spec.manifolds ?? {}).length === 0 && manifold === "default" ? inferDims(spec) : undefined);
  const fields: Record<string, FieldSpec> = {};
  for (const [id, f] of Object.entries(spec.fields)) {
    if (domainOf(spec, f.domain) !== manifold || dimCount === undefined) { fields[id] = f; continue; }
    fields[id] = { ...f, data: zoomData(f.data as DataSpec, dimCount, factor) } as FieldSpec;
  }
  return { ...spec, fields };
}

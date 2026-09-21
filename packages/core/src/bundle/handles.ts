// External arrays of a bundle: find every `handle` (and bare-path) array spec
// in a BundleSpec, load them all through a ByteSource, and hand the decoded
// arrays to the synchronous build as an ArrayResolver.
//
// The walk is typed against the schema (a switch per spec type) so a new
// field-data or net type fails typecheck here until it says where its arrays
// are. Bare strings are array paths only where the schema says ArraySpec; a
// `grad` output's `seed` string is an ArrayName, so only its object form is
// an array.

import type {
  ArrayExpr,
  ArraySpec,
  BundleSpec,
  CurveDataSpec,
  NetSpec,
  PointSpec,
  ScalarFieldDataSpec,
  ScalarStatistics,
  SizedArraySpec,
  VectorFieldDataSpec,
} from "@tensatory/schema";
import { SpecError, TensatoryError } from "../errors";
import { loadArray, type ArrayHint, type ByteSource, type RawArray } from "../arrays/load";
import { handleHint, mapResolver, type ArrayResolver } from "../arrays/spec";

/** every stored-array path a bundle refers to, with what the bundle claims about it (merged over its uses) */
export function collectHandles(spec: BundleSpec): Map<string, ArrayHint> {
  const out = new Map<string, ArrayHint>();
  const add = (path: string, hint: ArrayHint, at: string[]) => {
    const prev = out.get(path);
    if (!prev) { out.set(path, { ...hint }); return; }
    if (hint.shape) {
      if (prev.shape && (prev.shape.length !== hint.shape.length || prev.shape.some((s, i) => s !== hint.shape![i])))
        throw new SpecError(`array "${path}" is declared with shape [${hint.shape}] here and [${prev.shape}] elsewhere`, at);
      prev.shape = hint.shape;
    }
    if (hint.dtype) {
      if (prev.dtype && prev.dtype !== hint.dtype) throw new SpecError(`array "${path}" is declared ${hint.dtype} here and ${prev.dtype} elsewhere`, at);
      prev.dtype = hint.dtype;
    }
  };
  const any = (a: ArraySpec | undefined, at: string[]) => {
    if (a === undefined) return;
    if (typeof a === "string") add(a, {}, at);
    else if (a.type === "handle") add(a.path, handleHint(a), at);
  };
  const sized = (a: SizedArraySpec | undefined, at: string[]) => { if (a?.type === "handle") add(a.path, handleHint(a), at); };
  const point = (p: PointSpec | undefined, at: string[]) => { if (p !== undefined && !(Array.isArray(p) && p.every((x) => typeof x === "number"))) any(p as ArraySpec, at); };
  const reals = (r: number[] | ArraySpec | undefined, at: string[]) => { if (r !== undefined && !(Array.isArray(r) && r.every((x) => typeof x === "number"))) any(r as ArraySpec, at); };
  const stats = (s: ScalarStatistics | undefined, at: string[]) => {
    if (!s) return;
    reals(s.quantiles, [...at, "quantiles"]);
    if (s.histogram) { reals(s.histogram.counts, [...at, "histogram", "counts"]); reals(s.histogram.edges as number[] | ArraySpec, [...at, "histogram", "edges"]); }
  };
  const expr = (e: ArrayExpr, at: string[]) => {
    if (typeof e !== "object" || e === null) return;
    if ("op" in e && e.op === "call") { if (typeof e.net === "object") net(e.net, [...at, "net"]); for (const [k, v] of Object.entries(e.inputs)) expr(v, [...at, "inputs", k]); return; }
    for (const [k, v] of Object.entries(e)) {
      if (Array.isArray(v)) v.forEach((x, i) => expr(x as ArrayExpr, [...at, k, String(i)]));
      else if (typeof v === "object" && v !== null) expr(v as ArrayExpr, [...at, k]);
    }
  };
  const net = (n: NetSpec, at: string[]) => {
    switch (n.type) {
      case "def":
        for (const [k, a] of Object.entries(n.arrays ?? {})) sized(a, [...at, "arrays", k]);
        for (const [k, e] of Object.entries(n.nodes ?? {})) expr(e, [...at, "nodes", k]);
        return;
      case "bind":
        if (typeof n.net === "object") net(n.net, [...at, "net"]);
        for (const [k, a] of Object.entries(n.bind)) any(a, [...at, "bind", k]);
        return;
      case "displace":
        if (typeof n.net === "object") net(n.net, [...at, "net"]);
        n.directions.forEach((d, i) => { for (const [k, a] of Object.entries(d.arrays)) any(a, [...at, "directions", String(i), "arrays", k]); });
        return;
      case "grad":
        if (typeof n.net === "object") net(n.net, [...at, "net"]);
        for (const [k, a] of Object.entries(n.bind ?? {})) any(a, [...at, "bind", k]);
        for (const [k, g] of Object.entries(n.outputs)) if (g.seed !== undefined && typeof g.seed !== "string") any(g.seed, [...at, "outputs", k, "seed"]);
        return;
    }
  };
  const data = (d: ScalarFieldDataSpec | VectorFieldDataSpec, at: string[]) => {
    switch (d.type) {
      case "dense": sized(d.samples, [...at, "samples"]); stats(d.stats, [...at, "stats"]); return;
      case "densev": sized(d.samples, [...at, "samples"]); return;
      case "sparse": any(d.points, [...at, "points"]); any(d.samples, [...at, "samples"]); stats(d.stats, [...at, "stats"]); return;
      case "sparsev": any(d.points, [...at, "points"]); any(d.samples, [...at, "samples"]); return;
      case "translate": if (typeof d.arg === "object") data(d.arg, [...at, "arg"]); point(d.vec, [...at, "vec"]); return;
      case "scale": if (typeof d.arg === "object") data(d.arg, [...at, "arg"]); point(d.origin, [...at, "origin"]); return;
      case "pointwise": case "pointwisev":
        for (const [k, s] of Object.entries(d.scalars ?? {})) if (typeof s === "object") data(s, [...at, "scalars", k]);
        for (const [k, v] of Object.entries(d.vectors ?? {})) if (typeof v === "object") data(v, [...at, "vectors", k]);
        return;
      case "symbolic": case "symbolicv": return;
      case "net": case "netv":
        if (typeof d.net === "object") net(d.net, [...at, "net"]);
        for (const [k, a] of Object.entries(d.arrays ?? {})) any(a, [...at, "arrays", k]);
        for (const [k, e] of Object.entries(d.inputs ?? {})) expr(e, [...at, "inputs", k]);
        return;
    }
  };
  const curve = (c: CurveDataSpec, at: string[]) => {
    switch (c.type) {
      case "symbolic": return;
      case "sampled":
        any(c.points, [...at, "points"]);
        if (c.times !== undefined && !Array.isArray(c.times)) any(c.times, [...at, "times"]);
        any(c.velocities, [...at, "velocities"]);
        return;
      case "flow":
        if (typeof c.field === "object") data(c.field, [...at, "field"]);
        point(c.start, [...at, "start"]);
        return;
      case "translate": if (typeof c.arg === "object") curve(c.arg, [...at, "arg"]); point(c.vec, [...at, "vec"]); return;
      case "scale": if (typeof c.arg === "object") curve(c.arg, [...at, "arg"]); point(c.origin, [...at, "origin"]); return;
    }
  };
  for (const [id, f] of Object.entries(spec.fields)) data(f.data, ["fields", id, "data"]);
  for (const [id, n] of Object.entries(spec.nets ?? {})) net(n, ["nets", id]);
  for (const [id, c] of Object.entries(spec.curves ?? {})) curve(c.data, ["curves", id, "data"]);
  return out;
}

/** progress of a load: `done` of `total` arrays, the one just finished */
export type LoadProgress = { done: number; total: number; path: string };

/**
 * Load every external array of a bundle through `src` (≤ `concurrency` at a time) into a resolver. An array that fails
 * to load does not fail the bundle: the resolver rethrows its error when a field asks for it, so `buildAll()` reports
 * it against that field.
 */
export async function loadArrays(spec: BundleSpec, src: ByteSource, opts: { concurrency?: number; onProgress?: (p: LoadProgress) => void } = {}): Promise<ArrayResolver> {
  const wanted = [...collectHandles(spec)];
  const loaded = new Map<string, RawArray>();
  const failed = new Map<string, TensatoryError>();
  let next = 0, done = 0;
  const worker = async () => {
    for (;;) {
      const i = next++;
      if (i >= wanted.length) return;
      const [path, hint] = wanted[i]!;
      try { loaded.set(path, await loadArray(src, path, hint, ["arrays", path])); }
      catch (e) { failed.set(path, e instanceof TensatoryError ? e : new SpecError(`loading "${path}": ${e instanceof Error ? e.message : String(e)}`, ["arrays", path])); }
      opts.onProgress?.({ done: ++done, total: wanted.length, path });
    }
  };
  await Promise.all(Array.from({ length: Math.min(opts.concurrency ?? 6, wanted.length) }, worker));
  const ok = mapResolver(loaded);
  return {
    raw: (path, at) => {
      const err = failed.get(path);
      if (err) throw new SpecError(`external array "${path}" failed to load: ${err.message}`, at);
      return ok.raw(path, at);
    },
  };
}

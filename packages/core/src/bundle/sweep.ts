// Sweeps (schema/sweep.ts, notes/sweeps.md §2): a root document holding MEMBERS
// — each a BundleSpec, inline or by path — with flat metadata RECORDS and a
// `common` partial merged into every member. Nothing here knows what the keys
// mean: structure is DISCOVERED from the records (`facets`), and a member's
// "type" is the structural signature of the bundle it turns out to be
// (`signatureOf`), so options saved under one signature apply to every member
// that shares it.

import { z } from "zod";
import type { BundleSpec, CommonSpec, KeySpec, MemberSpec, RecordSpec, RecordValue, SweepSpec } from "@tensatory/schema";
import { BUNDLE_VERSION, SWEEP_VERSION } from "@tensatory/schema";
import { SpecError, specErrorOf } from "../errors";
import { rebaseSource, type ByteSource } from "../arrays/load";
import { CodomainSchema } from "../fields/codomain";
import { CurveSchema } from "../curves/spec";
import { FieldSchema } from "../fields/spec";
import { NetSchema } from "../nets/spec";
import { Bundle, BundleSchema, ManifoldDefinitionSchema, PointSetSchema, infoOf, inferDims, type Info } from "./bundle";
import { collectHandles, loadArrays, type LoadProgress } from "./handles";

/*******************************************************/
/* schema */

const recordValue: z.ZodType<RecordValue> = z.union([z.string(), z.number(), z.boolean()]);
const RecordSchema: z.ZodType<RecordSpec> = z.record(z.string(), recordValue);

export const KeySchema: z.ZodType<KeySpec> = z.object({
  name: z.string().optional(),
  summary: z.string().optional(),
  kind: z.enum(["nominal", "ordinal"]).optional(),
  values: z.array(recordValue).optional(),
  codomain: CodomainSchema.optional(),
  attribute: z.boolean().optional(),
});

export const CommonSchema: z.ZodType<CommonSpec> = z.object({
  name: z.string().optional(),
  summary: z.string().optional(),
  details: z.string().optional(),
  manifolds: z.record(z.string(), ManifoldDefinitionSchema).optional(),
  defaultManifold: z.string().optional(),
  fields: z.record(z.string(), FieldSchema).optional(),
  pointSets: z.record(z.string(), PointSetSchema).optional(),
  curves: z.record(z.string(), CurveSchema).optional(),
  nets: z.record(z.string(), NetSchema).optional(),
});

export const MemberSchema: z.ZodType<MemberSpec> = z.object({
  record: RecordSchema,
  bundle: z.union([z.string(), BundleSchema]),
});

export const SweepSchema: z.ZodType<SweepSpec> = z.object({
  tensatory: z.literal(SWEEP_VERSION),
  name: z.string().optional(),
  summary: z.string().optional(),
  details: z.string().optional(),
  keys: z.record(z.string(), KeySchema).optional(),
  common: CommonSchema.optional(),
  members: z.record(z.string(), MemberSchema),
});

/** the root kind of a document, by its `tensatory` version; undefined for anything else */
export function rootKind(json: unknown): "bundle" | "sweep" | undefined {
  const v = typeof json === "object" && json !== null ? (json as { tensatory?: unknown }).tensatory : undefined;
  return v === BUNDLE_VERSION ? "bundle" : v === SWEEP_VERSION ? "sweep" : undefined;
}

/*******************************************************/
/* merging */

/** `common` merged into a member: per top-level record, per id, the member wins; scalars (name, summary, …) too */
export function mergeCommon(common: CommonSpec | undefined, member: BundleSpec): BundleSpec {
  if (!common) return member;
  const out: BundleSpec = { ...member };
  const scalar = <K extends "name" | "summary" | "details" | "defaultManifold">(k: K) => { const v = member[k] ?? common[k]; if (v !== undefined) out[k] = v; };
  scalar("name"); scalar("summary"); scalar("details"); scalar("defaultManifold");
  const records = <K extends "manifolds" | "fields" | "pointSets" | "curves" | "nets">(k: K) => {
    const c = common[k], m = member[k];
    if (c === undefined && m === undefined) return;
    out[k] = { ...(m ?? {}), ...Object.fromEntries(Object.entries(c ?? {}).filter(([id]) => !(m && id in m))) } as BundleSpec[K];
  };
  records("manifolds"); records("fields"); records("pointSets"); records("curves"); records("nets");
  return out;
}

/** `common` as a BundleSpec of its own (to walk it with the bundle tools) */
const commonAsBundle = (c: CommonSpec): BundleSpec => ({ tensatory: BUNDLE_VERSION, ...c, fields: c.fields ?? {} });

/*******************************************************/
/* facets: structure discovered from the records */

export interface FacetValue {
  value: RecordValue;
  /** the members carrying this value (in sweep order); empty for a value listed in `keys.values` that no member has */
  members: string[];
  /** whether some member with this value agrees with the current member on every other key they share — a switch that changes nothing else */
  direct: boolean;
}

export interface Facet {
  key: string;
  spec: KeySpec | undefined;
  name: string;
  kind: "nominal" | "ordinal";
  values: FacetValue[];
  /** more than one value occurs across the members */
  varying: boolean;
  /** a per-member measurement (`keys.<k>.attribute`), not a coordinate: shown with the record, never a control */
  attribute: boolean;
}

/** the keys that do not count when comparing two records */
const attributeKeys = (spec: SweepSpec): Set<string> => new Set(Object.entries(spec.keys ?? {}).filter(([, k]) => k.attribute).map(([k]) => k));

const sameValue = (a: RecordValue, b: RecordValue): boolean => a === b;

/** keys in declaration order (`keys`), then as discovered in the records */
export function keyOrder(spec: SweepSpec): string[] {
  const seen = new Set<string>(Object.keys(spec.keys ?? {}));
  const out = [...seen];
  for (const m of Object.values(spec.members)) for (const k of Object.keys(m.record)) if (!seen.has(k)) { seen.add(k); out.push(k); }
  return out;
}

/** the number of shared (non-attribute) keys on which two records DIFFER, and the number they share */
function distance(a: RecordSpec, b: RecordSpec, skip: Set<string>, except?: string): { differ: number; shared: number } {
  let differ = 0, shared = 0;
  for (const k of Object.keys(a)) {
    if (k === except || skip.has(k) || !(k in b)) continue;
    shared++;
    if (!sameValue(a[k]!, b[k]!)) differ++;
  }
  return { differ, shared };
}

/**
 * One facet per key the CURRENT member's record has (every key when there is no current member), with the values
 * found across all members, which members carry each, and whether switching to it leaves the other keys alone.
 */
export function facets(spec: SweepSpec, current?: string): Facet[] {
  const cur = current !== undefined ? spec.members[current]?.record : undefined;
  const skip = attributeKeys(spec);
  const out: Facet[] = [];
  for (const key of keyOrder(spec)) {
    if (cur && !(key in cur)) continue;
    const ks = spec.keys?.[key];
    const found: FacetValue[] = [];
    const lookup = (v: RecordValue) => found.find((f) => sameValue(f.value, v));
    for (const v of ks?.values ?? []) if (!lookup(v)) found.push({ value: v, members: [], direct: false });
    const listed = found.length;
    for (const [id, m] of Object.entries(spec.members)) {
      const v = m.record[key];
      if (v === undefined) continue;
      let f = lookup(v);
      if (!f) { f = { value: v, members: [], direct: false }; found.push(f); }
      f.members.push(id);
      if (!cur || distance(cur, m.record, skip, key).differ === 0) f.direct = true;
    }
    const kind = ks?.kind ?? (found.every((f) => typeof f.value === "number") ? "ordinal" : "nominal");
    if (kind === "ordinal") {
      // listed values keep their order; the discovered rest is sorted (numbers numerically, everything else as text)
      const rest = found.splice(listed).sort((a, b) => typeof a.value === "number" && typeof b.value === "number" ? a.value - b.value : String(a.value).localeCompare(String(b.value)));
      found.push(...rest);
    }
    out.push({ key, spec: ks, name: ks?.name ?? key, kind, values: found, varying: found.filter((f) => f.members.length).length > 1, attribute: !!ks?.attribute });
  }
  return out;
}

/**
 * The member to switch to when `key` is set to `value` from `current`: among the members with that value, the one whose
 * record differs from the current one on the fewest other shared keys (ties: more shared keys, then sweep order).
 */
export function nearestMember(spec: SweepSpec, current: string | undefined, key: string, value: RecordValue): string | undefined {
  const cur = current !== undefined ? spec.members[current]?.record : undefined;
  const skip = attributeKeys(spec);
  let best: { id: string; differ: number; shared: number } | undefined;
  for (const [id, m] of Object.entries(spec.members)) {
    const v = m.record[key];
    if (v === undefined || !sameValue(v, value)) continue;
    const d = cur ? distance(cur, m.record, skip, key) : { differ: 0, shared: 0 };
    if (!best || d.differ < best.differ || (d.differ === best.differ && d.shared > best.shared)) best = { id, ...d };
  }
  return best?.id;
}

/** the members whose records carry every key of `selection` with that value */
export function membersWhere(spec: SweepSpec, selection: RecordSpec): string[] {
  return Object.entries(spec.members).filter(([, m]) => Object.entries(selection).every(([k, v]) => k in m.record && sameValue(m.record[k]!, v))).map(([id]) => id);
}

/*******************************************************/
/* structural signature */

/**
 * What a member IS, for the purpose of remembering how it was viewed: its spaces (id, dimension) and its fields (id,
 * kind, dimension), sorted — two seeds of one architecture share it, so slots, levels, camera and colormaps carry
 * over. Curves and point sets are deliberately left out (a trajectory that exists in one member and not another
 * should not reset the camera). Read off the spec, so it needs no build and does not depend on adjustments.
 */
export function signatureOf(spec: BundleSpec): string {
  let manifolds = spec.manifolds ?? {};
  if (!Object.keys(manifolds).length) { let d: number | string; try { d = inferDims(spec); } catch { d = "?"; } manifolds = { default: { numDims: d as number } }; }
  const dflt = spec.defaultManifold ?? (Object.keys(manifolds).length === 1 ? Object.keys(manifolds)[0] : undefined);
  const dims = (id: string | undefined): number | string => (id !== undefined && manifolds[id] ? manifolds[id]!.numDims : "?");
  const parts: string[] = [];
  for (const [id, m] of Object.entries(manifolds)) parts.push(`space ${id}:${m.numDims}`);
  for (const [id, f] of Object.entries(spec.fields)) parts.push(`${f.kind} ${id}:${dims(f.domain ?? dflt)}`);
  return parts.sort().join(";");
}

/** a short, stable hash of a string (FNV-1a, base 36) — the localStorage-friendly form of a signature */
export function shortHash(s: string): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 0x01000193) >>> 0; }
  return h.toString(36);
}

/*******************************************************/
/* runtime */

const dirname = (path: string): string => { const i = path.lastIndexOf("/"); return i < 0 ? "" : path.slice(0, i); };

/** the id a lone bundle gets when wrapped as a one-member sweep */
export const SINGLE_MEMBER = "bundle";

export class Sweep {
  readonly name: string;
  private readonly loading = new Map<string, Promise<Bundle>>();
  private readonly done = new Map<string, Bundle>();

  constructor(readonly spec: SweepSpec, private readonly src: ByteSource) {
    this.name = spec.name ?? "untitled sweep";
  }

  /** validate a JSON value against the sweep schema; `common` may not refer to external arrays */
  static validate(json: unknown): SweepSpec {
    const r = SweepSchema.safeParse(json);
    if (!r.success) throw specErrorOf(r.error.issues);
    if (r.data.common) {
      const handles = collectHandles(commonAsBundle(r.data.common));
      if (handles.size) throw new SpecError(`\`common\` refers to external arrays (${[...handles.keys()].join(", ")}); handles belong to the members`, ["common"]);
    }
    return r.data;
  }

  /** parse a sweep document whose member documents and sidecars come from `src` (paths relative to the sweep document) */
  static parse(json: unknown, src: ByteSource): Sweep {
    return new Sweep(Sweep.validate(json), src);
  }

  /** a lone bundle as a sweep with one member (`SINGLE_MEMBER`) and an empty record */
  static single(bundle: Bundle): Sweep {
    const spec: SweepSpec = { tensatory: SWEEP_VERSION, members: { [SINGLE_MEMBER]: { record: {}, bundle: bundle.spec } } };
    if (bundle.spec.name !== undefined) spec.name = bundle.spec.name;
    if (bundle.spec.summary !== undefined) spec.summary = bundle.spec.summary;
    if (bundle.spec.details !== undefined) spec.details = bundle.spec.details;
    const s = new Sweep(spec, { bytes: async () => null });
    s.loading.set(SINGLE_MEMBER, Promise.resolve(bundle));
    s.done.set(SINGLE_MEMBER, bundle);
    return s;
  }

  get info(): Info | undefined { return infoOf(this.name, this.spec); }
  get memberIds(): string[] { return Object.keys(this.spec.members); }
  get keys(): string[] { return keyOrder(this.spec); }
  /** whether any coordinate key varies across the members (a lone bundle, or several identical records, has none) */
  get hasFacets(): boolean { return facets(this.spec).some((f) => f.varying && !f.attribute); }

  record(id: string): RecordSpec {
    const m = this.spec.members[id];
    if (!m) throw new SpecError(`unknown member "${id}"`, ["members"]);
    return m.record;
  }

  facets(current?: string): Facet[] { return facets(this.spec, current); }
  nearest(current: string | undefined, key: string, value: RecordValue): string | undefined { return nearestMember(this.spec, current, key, value); }

  /**
   * A member's bundle: `common` merged in, its document fetched when given by path, its external arrays loaded
   * (relative to the member document). Loaded once; later calls share the promise.
   */
  member(id: string, opts: { onProgress?: (p: LoadProgress) => void } = {}): Promise<Bundle> {
    let p = this.loading.get(id);
    if (!p) {
      p = this.loadMember(id, opts);
      this.loading.set(id, p);
      p.then((b) => this.done.set(id, b), () => this.loading.delete(id)); // a failed load may be retried
    }
    return p;
  }

  /** a member's bundle if it has been loaded already (synchronous; for labels of members seen before) */
  loaded(id: string): Bundle | undefined { return this.done.get(id); }

  private async loadMember(id: string, opts: { onProgress?: (p: LoadProgress) => void }): Promise<Bundle> {
    const m = this.spec.members[id];
    if (!m) throw new SpecError(`unknown member "${id}"`, ["members"]);
    const at = ["members", id, "bundle"];
    let spec: BundleSpec, src = this.src;
    if (typeof m.bundle === "string") {
      const bytes = await this.src.bytes(m.bundle);
      if (bytes === null) throw new SpecError(`member document "${m.bundle}" not found`, at);
      let json: unknown;
      try { json = JSON.parse(new TextDecoder().decode(bytes)); }
      catch (e) { throw new SpecError(`member document "${m.bundle}" is not JSON: ${(e as Error).message}`, at); }
      try { spec = Bundle.validate(json); }
      catch (e) { if (e instanceof SpecError) throw new SpecError(`member document "${m.bundle}": ${e.message}`, at); throw e; }
      src = rebaseSource(this.src, dirname(m.bundle));
    } else {
      spec = m.bundle;
    }
    const merged = mergeCommon(this.spec.common, spec);
    return new Bundle(merged, await loadArrays(merged, src, opts));
  }
}

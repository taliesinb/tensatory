// A SWEEP: many bundles sharing a metadata vocabulary — a hyperparameter sweep,
// independently seeded runs, one θ* seen along several kinds of directions.
//
// A sweep is a set of MEMBERS, each a whole BundleSpec (inline, or a path to a
// document relative to the sweep document) with a flat metadata RECORD, plus a
// `common` partial bundle merged into every member. The format encodes terms;
// structure (which keys vary, which values exist together) is discovered by the
// viewer from the records — see notes/sweeps.md §2. A lone bundle is a sweep
// with one member and an empty record.

import type { ArrayPath } from "./arrays";
import type { BundleSpec } from "./bundle";
import type { CodomainSpec } from "./codomain";
import type { ShowString } from "./math";

export const SWEEP_VERSION = "0.2";

export type MemberId = string;
export type KeyName = string;
/** a record cell; keys absent from a member's record simply do not apply to it (a convnet has no `num_layers`) */
export type RecordValue = string | number | boolean;
export type RecordSpec = Record<KeyName, RecordValue>;

/** how to read / display one record column; every key a record uses may (need not) be described */
export type KeySpec = {
  name?: ShowString; // display name; defaults to the key
  summary?: ShowString; // one line, shown behind the key's ⓘ
  kind?: "nominal" | "ordinal"; // defaults to ordinal when every value is a number, else nominal
  values?: RecordValue[]; // display order (values not listed follow, in the order they are found); a listed value no member has is shown disabled
  codomain?: CodomainSpec; // ordinal keys: scaling (log-spaced learning rates) and formatting of the values
  attribute?: boolean; // a per-member MEASUREMENT (test accuracy, parameter count, wall time) rather than a coordinate of the sweep: shown with the record, never a control, ignored when comparing records
};

/** the part of a bundle every member shares; merged per top-level record, per id, the member winning. No `handle` arrays here. */
export type CommonSpec = Omit<Partial<BundleSpec>, "tensatory">;

export type MemberSpec = {
  record: RecordSpec;
  bundle: ArrayPath | BundleSpec; // a path (relative to the sweep document; its own sidecars relative to IT) or an inline bundle
};

export type SweepSpec = {
  tensatory: typeof SWEEP_VERSION;
  name?: ShowString;
  summary?: ShowString; // one line
  details?: ShowString; // any length
  keys?: Record<KeyName, KeySpec>;
  common?: CommonSpec;
  members: Record<MemberId, MemberSpec>;
};

/** what a Tensatory document at its root may be; `tensatory` tells which */
export type RootSpec = BundleSpec | SweepSpec;

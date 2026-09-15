// These describe 1-dimensional spaces, used as the fiber (value space) of
// scalar fields. They do not change the *values*; they inform visualization:
// slider scaling, colormap choice/direction, marks, units.
//
// Examples:
// * "norm":          "norm"       | {min: 0}
// * "accuracy":      "fraction"   | {min: 0, max: 1}
// * "loss":          "log"        | {log: "10"}
// * "cross_entropy": "celoss"     | {min: 0, log: "e", flip: true, unit: "nats"}
// * "cosine_sim":    "similarity" | {min: -1, max: 1, marks: [[0, "orthogonal"]]}
// * "percent":       "percent"    | {min: 0, max: 100, unit: "%"}

import type { Base, Real, ShowString } from "./math";

export type UnitString = string;

export type ExtReal = Real | "inf" | "-inf";

export type CodomainOptionsSpec = {
  min?: ExtReal | null;  // defaults to null, meaning "-inf"
  max?: ExtReal | null;  // defaults to null, meaning "inf"
  exclMin?: boolean;     // open at min; defaults to false (true if min is "-inf")
  exclMax?: boolean;     // open at max; defaults to false (true if max is "inf")
  log?: Base | null;     // defaults to null; otherwise the base: log-scaled sliders / colormaps
  flip?: boolean;        // defaults to false; true means "smaller is better": reverse sliders / colormaps
  wrap?: boolean;        // defaults to false; true means sliders / animations / colormaps wrap min-to-max
  unit?: UnitString | null; // defaults to null, meaning unitless
  marks?: [ExtReal, ShowString][]; // special values to highlight on sliders / colormaps
};

export type CodomainName =
  | "lin"        // {}
  | "p"          // {min: 0, max: 1}
  | "fraction"   // {min: 0, max: 1}
  | "percent"    // {min: 0, max: 100, unit: "%"}
  | "log"        // {log: "10"}
  | "logp"       // {max: 0, log: "e"}
  | "-logp"      // {min: 0, log: "e", flip: true}
  | "celoss"     // {min: 0, log: "e", flip: true, unit: "nats"}
  | "similarity" // {min: -1, max: 1}
  | "norm"       // {min: 0}
  | "distance"   // {min: 0}
  | "angle";     // {min: 0, max: 2*pi, wrap: true, unit: "rad"}

export type CodomainSpec = CodomainName | CodomainOptionsSpec;

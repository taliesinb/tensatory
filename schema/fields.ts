import type { CodomainSpec } from "./codomain";
import type { ScalarFieldDataSpec, VectorFieldDataSpec } from "./fieldData";
import type { ManifoldId } from "./manifolds";
import type { ShowString } from "./math";

export type FieldId = string;

export type FieldSpec = ScalarFieldSpec | VectorFieldSpec;

export type ScalarFieldSpec = {
  kind: "scalar";
  data: ScalarFieldDataSpec; // the values taken by the scalar field at different positions
  domain?: ManifoldId; // the logical domain of the field; defaults to the bundle's default manifold
  codomain?: CodomainSpec; // hints for visualization; defaults to "lin"
  exactGradient?: FieldId; // a vector field holding the EXACT gradient (e.g. from backprop during collection)
  name?: ShowString; // displayed name; defaults to the field id
  summary?: ShowString; // one line
  details?: ShowString; // any length
};

export type VectorFieldSpec = {
  kind: "vector";
  data: VectorFieldDataSpec;
  domain?: ManifoldId;
  name?: ShowString;
  summary?: ShowString; // one line
  details?: ShowString; // any length
};

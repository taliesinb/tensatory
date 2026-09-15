// @tensatory/schema: the JSON-facing description of a Tensatory data bundle.
//
// Everything here is a type (plus one version constant). Runtime validation
// and behaviour live in @tensatory/core, whose zod schemas are checked
// against these types.

export * from "./math";
export * from "./geometry";
export * from "./arrays";
export * from "./symbolic";
export * from "./codomain";
export * from "./statistics";
export * from "./manifolds";
export * from "./mappings";
export * from "./fieldData";
export * from "./fields";
export * from "./bundle";

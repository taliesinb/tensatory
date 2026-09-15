export { SymbolicScalarSchema, SymbolicVectorSchema, SCALAR_UNARY_OPS, SCALAR_NARY_OPS, SCALAR_BINARY_OPS } from "./spec";
export type { SExpr, VExpr } from "./ast";
export { hasArgs, argNames } from "./ast";
export { normalizeScalar, normalizeVector, emptyEnv, checkNamespaces } from "./normalize";
export type { NameEnv } from "./normalize";
export { diffScalar, diffVector, gradient } from "./diff";
export { compileScalar, compileVector, pureContext } from "./compile";
export type { ScalarFn, VectorFn, CompileContext } from "./compile";

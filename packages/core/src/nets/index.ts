export {
  ArrayExprSchema,
  NetSchema,
  NetScalarFieldDataSchema,
  NetVectorFieldDataSchema,
  ShapeSchema,
  ARRAY_REDUCE_FNS,
  ARRAY_COMPARE_OPS,
} from "./spec";
export { inferNet, inferExpr, inferNetField, broadcastShapes, fmtShape, noNetResolver } from "./shapes";
export type { NetSignature, NetResolver, ArrayEnv, Dim, Shape } from "./shapes";
export { compileNet, evaluate } from "./program";
export { exprNames } from "./shapes";
export type { Program, ProgramResolver, EvalContext } from "./program";
export { NetScalarFieldData, NetVectorFieldData, fieldProgram, pointInput, GRADIENT_VALUE_OUTPUT } from "./fieldData";
export type { NetField } from "./fieldData";
export { gradProgram, anfProgram } from "./autodiff";
export type { GradRequest } from "./autodiff";
export type { Val } from "./ops";

// Primitive numeric aliases. These carry documentation only: TypeScript cannot
// enforce positivity etc. Runtime validation lives in @tensatory/core.

export type Base = "10" | "2" | "e";

export type Real = number;
export type RealPos = number;     // > 0
export type RealNonNeg = number;  // >= 0
export type RealUnit = number;    // in [0, 1]

export type Int = number;
export type IntPos = number;      // > 0
export type IntNonNeg = number;   // >= 0

export type Index = IntNonNeg;
export type DimIndex = Index;     // index of a dimension of a space / axis of an array
export type ArgIndex = Index;

export type Count = IntNonNeg;
export type DimCount = Count;

// a string intended to be shown to a human (labels, names)
export type ShowString = string;

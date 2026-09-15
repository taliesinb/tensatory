// A small closed expression language over scalar and vector fields.
//
// Design rules:
// * Every node has an `op`. Scalar-valued and vector-valued nodes are distinct
//   sets; the expected type of a subexpression is always known from its
//   position, so a bare string (a name) or bare number is unambiguous.
// * Constants are LEAVES, not parameter slots: `x + 2` is
//   {op: "add", vals: [{op: "coord", index: 0}, 2]}. The only non-expression
//   parameters are ones that must be static (dimension indices).
// * Names live in three namespaces, bound by the enclosing spec:
//     ConstArgName  -> Real                (consts)
//     ScalarArgName -> scalar field/array  (scalars)
//     VectorArgName -> vector field/array  (vectors)
//   A name must be unique across ALL THREE namespaces of a given scope, so a
//   bare string resolves without ambiguity.
// * Being closed, the same tree can be evaluated in JS, differentiated
//   symbolically, or transpiled (GLSL/WGSL/Torch).

import type { DimIndex, Real } from "./math";
import type { Vector } from "./geometry";

export type ConstArgName = string;
export type ScalarArgName = string;
export type VectorArgName = string;

/*******************************************************/
/* SCALAR-VALUED NODES */

// unary scalar functions
export type ScalarUnaryOp =
  | "sin" | "cos" | "tan" | "sinh" | "cosh" | "tanh"                         // trig
  | "asin" | "acos" | "atan" | "asinh" | "acosh" | "atanh"                   // inv trig
  | "relu" | "sigmoid" | "gelu" | "silu" | "softplus" | "elu" | "erf"        // nn activations
  | "floor" | "ceil" | "round" | "sign" | "abs"                              // non-differentiable
  | "exp" | "exp2" | "exp10" | "log" | "log2" | "log10" | "log1p" | "expm1"  // exponentiation
  | "plogp"                                                                  // p*log(p), with 0 at 0
  | "sqrt" | "square"                                                        // more accurate than pow
  | "negate" | "reciprocal"                                                  // field operations
  | "gauss";                                                                 // exp(-x^2/2)

// n-ary scalar functions (associative/commutative; `vals` may have any length >= 1)
export type ScalarNaryOp = "add" | "mul" | "min" | "max" | "mean" | "rms";

// binary scalar functions (order matters)
export type ScalarBinaryOp =
  | "sub" | "div"        // a - b, a / b
  | "pow"                // a ^ b
  | "logBase"            // log_b(a)
  | "atan2"              // atan2(a, b)
  | "mod";               // a mod b (result has sign of b)

export interface ScalarUnary<S>   { op: ScalarUnaryOp;  val: S }
export interface ScalarNary<S>    { op: ScalarNaryOp;   vals: S[] }
export interface ScalarBinary<S>  { op: ScalarBinaryOp; vals: [S, S] }
export interface ScalarClamp<S>   { op: "clamp";        val: S; min: S; max: S }
export interface ScalarGaussKernel<S> { op: "gaussKernel"; val: S; mu: S; sigma: S } // exp(-z^2/2), z = (val - mu) / sigma; peak 1
export interface ScalarNormalPDF<S>   { op: "normalPDF";   val: S; mu: S; sigma: S } // exp(-z^2/2) / (sigma * sqrt(2 pi)); integrates to 1
export interface ScalarDot<V>     { op: "dot" | "cosineSim"; vecs: [V, V] }
export interface ScalarNorm<V>    { op: "norm";  vec: V }              // Euclidean norm
export interface ScalarComp<V>    { op: "comp";  vec: V; index: DimIndex } // index'th component

// leaf nodes that produce scalars
export type SymbolicScalarLeaf =
  | Real                                    // literal constant
  | { op: "const"; name: ConstArgName }     // named constant
  | { op: "coord"; index: DimIndex }        // the coordinate function p -> p[index]
  | { op: "arg"; name: ScalarArgName }      // named scalar argument
  | { op: "argvi"; name: VectorArgName; index: DimIndex } // component of a named vector argument
  | ScalarArgName | ConstArgName;           // bare name: a scalar arg or a const (namespaces are disjoint)

export type SymbolicScalarNode =
  | ScalarUnary<SymbolicScalar>
  | ScalarNary<SymbolicScalar>
  | ScalarBinary<SymbolicScalar>
  | ScalarClamp<SymbolicScalar>
  | ScalarGaussKernel<SymbolicScalar>
  | ScalarNormalPDF<SymbolicScalar>
  | ScalarDot<SymbolicVector>
  | ScalarNorm<SymbolicVector>
  | ScalarComp<SymbolicVector>;

export type SymbolicScalar = SymbolicScalarNode | SymbolicScalarLeaf;

/*******************************************************/
/* VECTOR-VALUED NODES */

export interface VectorScale<V, S>   { op: "scalev";   vec: V; by: S }        // scalar * vector
export interface VectorNary<V>       { op: "addv" | "meanv"; vecs: V[] }      // sum / mean of vectors
export interface VectorSub<V>        { op: "subv";     vecs: [V, V] }         // a - b
export interface VectorLinComb<V, S> { op: "sumv";     vecs: V[]; coeffs: S[] } // scalar-weighted sum
export interface VectorFromComps<S>  { op: "compv";    coeffs: S[] }          // vector from its components
export interface VectorNormalize<V>  { op: "normalize"; vec: V }              // v / |v| (0 stays 0)
export interface VectorGrad<S>       { op: "grad";     val: S }               // gradient of a scalar wrt the coordinates

// leaf nodes that produce vectors
export type SymbolicVectorLeaf =
  | { op: "constv"; value: Vector }        // constant vector
  | { op: "basisv"; index: DimIndex }      // standard basis vector e_index
  | { op: "coordv" }                       // the identity p -> p
  | { op: "argv"; name: VectorArgName }    // named vector argument
  | VectorArgName;                         // bare name: a vector arg

export type SymbolicVectorNode =
  | VectorScale<SymbolicVector, SymbolicScalar>
  | VectorNary<SymbolicVector>
  | VectorSub<SymbolicVector>
  | VectorLinComb<SymbolicVector, SymbolicScalar>
  | VectorFromComps<SymbolicScalar>
  | VectorNormalize<SymbolicVector>
  | VectorGrad<SymbolicScalar>;

export type SymbolicVector = SymbolicVectorNode | SymbolicVectorLeaf;

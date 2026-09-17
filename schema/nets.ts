// Small neural networks as bundle citizens.
//
// Status: implemented end to end. Shape inference, a CPU reference
// evaluator (packages/core/src/nets) and a WebGPU transpiler
// (packages/gpu/src/nets.ts) evaluate `def` / `bind` / `displace` / `call`;
// `grad` is reverse-mode autodiff as a program rewrite (autodiff.ts), so
// gradient nets, Hessian-vector products and exact derivatives of net fields
// run on both backends. The iris example bundle
// (apps/viewer/public/bundles/iris.json, from tools/iris/train.py) matches
// PyTorch to 1e-9; CPU and GPU agree to f32. Long form: notes/nets.md.
//
// Design rules (continuing schema/symbolic.ts):
// * A NET is a pure function from named input ARRAYS to named output ARRAYS,
//   with the shapes of both predeclared. Scalars are 0-d arrays.
// * NOTHING distinguishes weights from data: `x`, `y`, `W1` are all inputs.
//   Loss is not special: it is an output of shape [] like any other.
// * The body is a DAG of named array expressions. The expression language is
//   the scalar language of symbolic.ts lifted elementwise to arrays (numpy
//   broadcasting), plus contractions, reductions, shape ops, and calling other
//   nets. On 0-d arrays it IS the scalar symbolic language (minus the
//   coordinate leaves), so `normalize` / `compile` / `diff` generalize rather
//   than fork.
// * BATCHING is implicit and structural: a declared shape is the shape of ONE
//   example. Any array passed in with EXTRA LEADING axes beyond its declared
//   rank is batched over those axes (vmap semantics). Batch prefixes of the
//   different inputs broadcast together numpy-style (right-aligned, size 1
//   stretches); every output gets the broadcast prefix in front of its declared
//   shape. A net body never sees batch axes. Weights are typically unbatched;
//   evaluating a loss landscape at G parameter points means passing weights
//   with a leading [G] — one vmapped evaluation, not G calls.
// * SYMBOLIC AXES ("N") give explicit control over a dataset axis INSIDE a net
//   (mean over the validation set is a reduction over "N"), so a bound net can
//   collapse the dataset to a scalar and leave only the weights as inputs.
//   Batching is for the axes nobody declared; symbolic axes are for the ones
//   the net wants to reduce over. There is deliberately NO way to reduce over
//   batch axes: if a mean over the dataset is wanted, declare the axis.
// * `grad` is an operator on nets producing another net (see GradNetSpec) —
//   implemented as a REWRITE into the same vocabulary (forward in A-normal
//   form + adjoint nodes), so gradients of gradients, Hessian-vector products
//   and per-example gradients all fall out, every evaluator gets them for
//   free, and a net-backed field's `derivative(dim)` is exact.
//
// What this needs from the rest of the roadmap: the `.bin` / npz array backend
// (roadmap §1) — a 10k×784 validation set does not want to be inline JSON.
// Everything else here works with inline arrays for toy sizes.

import type { ArraySpec, AxisSize, SizedArraySpec } from "./arrays";
import type { BoxSpec } from "./geometry";
import type { DimIndex, Int, Real, ShowString } from "./math";
import type { ScalarBinaryOp, ScalarNaryOp, ScalarUnaryOp } from "./symbolic";

export type NetId = string;

// One namespace per net for inputs, nodes and constant arrays (like the
// consts / scalars / vectors rule of symbolic.ts: a bare name must resolve
// without ambiguity).
export type ArrayName = string;

// A symbolic axis size, e.g. "N". Bound when arrays are supplied; every
// occurrence within one net evaluation must agree.
export type AxisName = string;

// Axis of the DECLARED (unbatched) shape; negative counts from the end.
export type AxisIndex = Int;

// A declared shape. `[]` is a scalar, `["N", 784]` a dataset of vectors.
export type ShapeSpec = (AxisSize | AxisName)[];

/*******************************************************/
/* ARRAY EXPRESSIONS */

// leaves
export type ArrayExprLeaf =
  | Real                                 // literal, a 0-d array
  | { op: "arg"; name: ArrayName }       // an input, node, or constant array of the enclosing net
  | ArrayName                            // bare name: same as `arg`
  | { op: "coord"; index: DimIndex }     // the field coordinate p[index] as a 0-d array — ONLY in a
  | { op: "coordv" };                    //   field-data context (NetScalarFieldDataSpec.inputs), see below

// elementwise, numpy broadcasting; op names shared with the scalar language
export interface ArrayUnary<A>   { op: ScalarUnaryOp;  val: A }
export interface ArrayNary<A>    { op: ScalarNaryOp;   vals: A[] }        // add mul min max mean rms, elementwise over vals
export interface ArrayBinary<A>  { op: ScalarBinaryOp; vals: [A, A] }
export interface ArrayClamp<A>   { op: "clamp"; val: A; min: A; max: A }
export interface ArrayCompare<A> { op: "lt" | "le" | "gt" | "ge" | "eq" | "ne"; vals: [A, A] } // 0 / 1; derivative 0
export interface ArrayWhere<A>   { op: "where"; cond: A; vals: [A, A] }  // cond != 0 ? a : b

// contractions, over declared axes only (the batch prefix is implicit)
export interface ArrayMatmul<A>  { op: "matmul"; vals: [A, A] }          // numpy semantics: 1-d · 1-d, 2-d · 1-d, 2-d · 2-d, stacked
// e.g. "ij,jk->ik"; no "..." (batch is implicit). A letter repeated within an operand reads its diagonal ("ii->i",
// "ii->" the trace); a letter repeated in the OUTPUT writes the diagonal and leaves the rest zero ("i->ii") — the two
// are each other's adjoints, which keeps the op set closed under differentiation.
export interface ArrayEinsum<A>  { op: "einsum"; subscripts: string; vals: A[] }

// reductions over declared axes; `axes` omitted = all declared axes -> 0-d
export type ArrayReduceFn = "sum" | "mean" | "prod" | "max" | "min" | "logsumexp";
export interface ArrayReduce<A>  { op: "reduce"; fn: ArrayReduceFn; val: A; axes?: AxisIndex[]; keepDims?: boolean }
export interface ArrayArgReduce<A> { op: "argmax" | "argmin"; val: A; axis?: AxisIndex } // default -1; integer-valued floats; derivative 0

// normalizations along one axis (default -1); fused for stability
export interface ArraySoftmax<A> { op: "softmax" | "logSoftmax"; val: A; axis?: AxisIndex }

// shape operations
export interface ArrayReshape<A>   { op: "reshape";   val: A; shape: (AxisSize | AxisName | -1)[] } // at most one -1
export interface ArrayTranspose<A> { op: "transpose"; val: A; perm?: DimIndex[] }  // default: reverse all declared axes
export interface ArrayConcat<A>    { op: "concat";    vals: A[]; axis: AxisIndex }
export interface ArraySlice<A>     { op: "slice";     val: A; axis: AxisIndex; start?: Int; stop?: Int; step?: Int } // python semantics
export interface ArrayOneHot<A>    { op: "oneHot";    val: A; size: AxisSize | AxisName }  // appends an axis; derivative 0
export interface ArrayTakeAlong<A> { op: "takeAlong"; val: A; indices: A; axis: AxisIndex } // torch.gather: out[..i..] = val[..indices[..i..]..]
export interface ArrayStopGradient<A> { op: "stopGradient"; val: A }

// call another net: the callee's declared shapes vs the passed arrays follow
// the same rule as top-level evaluation (extra leading axes batch). Nets may
// not call themselves, directly or indirectly.
export interface ArrayCall<A> {
  op: "call";
  net: NetId | NetSpec;
  inputs: Record<ArrayName, A>;  // every input of the callee that is not bound in the callee itself
  output: ArrayName;             // which output to take
}

export type ArrayExprNode =
  | ArrayUnary<ArrayExpr>
  | ArrayNary<ArrayExpr>
  | ArrayBinary<ArrayExpr>
  | ArrayClamp<ArrayExpr>
  | ArrayCompare<ArrayExpr>
  | ArrayWhere<ArrayExpr>
  | ArrayMatmul<ArrayExpr>
  | ArrayEinsum<ArrayExpr>
  | ArrayReduce<ArrayExpr>
  | ArrayArgReduce<ArrayExpr>
  | ArraySoftmax<ArrayExpr>
  | ArrayReshape<ArrayExpr>
  | ArrayTranspose<ArrayExpr>
  | ArrayConcat<ArrayExpr>
  | ArraySlice<ArrayExpr>
  | ArrayOneHot<ArrayExpr>
  | ArrayTakeAlong<ArrayExpr>
  | ArrayStopGradient<ArrayExpr>
  | ArrayCall<ArrayExpr>;

export type ArrayExpr = ArrayExprNode | ArrayExprLeaf;

/*******************************************************/
/* NETS */

export type NetSpec = NetDefinitionSpec | BoundNetSpec | DisplacedNetSpec | GradNetSpec;

// A net written out: inputs with shapes, a DAG of nodes, outputs with shapes.
export type NetDefinitionSpec = {
  type: "def";
  inputs: Record<ArrayName, ShapeSpec>;      // declared per-example shapes; symbolic axes allowed
  nodes?: Record<ArrayName, ArrayExpr>;      // internal arrays; any order, must be acyclic
  arrays?: Record<ArrayName, SizedArraySpec>; // constants baked into the net (masks, class weights, ...)
  outputs: Record<ArrayName, ShapeSpec>;     // names of inputs or nodes; the declared shape is CHECKED
                                             // against the inferred one (a typo becomes an error, not NaNs)
  name?: ShowString;
  description?: ShowString;
};

// Partial application: fix some inputs to arrays. The result is a net whose
// inputs are the remaining ones; the bound inputs become internal arrays (so
// `grad` may still differentiate with respect to them — the gradient at fixed
// weights). Bound arrays may carry extra leading axes (batch) — they broadcast
// with whatever is passed later. Binding the validation set to the MLP gives
// `loss(weights)`; binding weights at G points gives a [G]-batched loss.
export type BoundNetSpec = {
  type: "bind";
  net: NetId | NetSpec;
  bind: Record<ArrayName, ArraySpec>;
  name?: ShowString;
  description?: ShowString;
};

// Displacement: "around" a net. Takes a net and K displacement records; each
// key names an array of the net — an input, an internal node (an activation),
// a bound-away input, or a baked constant — and each value is an array of
// that array's per-example shape. The result is a net with one ADDITIONAL
// input `coeffs` of shape [K] such that every named array A is replaced,
// everywhere it is used, by
//   A + sum_k coeffs[k] * directions[k][A]      (a missing key = zero direction)
// while everything else (inputs, outputs, other nodes) keeps its name and
// shape. `bind` supplies the origin (theta*), `displace` the directions, and a
// net with `coeffs` as its sole remaining input is a field of the coordinates
// (see the default rule under FIELDS BACKED BY NETS). Because any array can be
// displaced, "around" is flexible: parameter subspaces (random / PCA /
// Hessian directions), but equally perturbations of an activation or of the
// data. Displacing several arrays with one record moves them together.
export type DisplacedNetSpec = {
  type: "displace";
  net: NetId | NetSpec;
  coeffs?: ArrayName;                          // the new [K] input; defaults to "t"
  directions: Record<ArrayName, ArraySpec>[];  // K records; K >= 1
  name?: ShowString;
  description?: ShowString;
};

// The gradient operator. Takes a net, optionally binds some inputs, and
// returns a NET whose inputs are the remaining inputs of `net` (same batch
// rule) and whose outputs are the requested gradients, plus any forward
// outputs listed in `keep` (so loss AND its gradient come from one pass).
export type GradNetSpec = {
  type: "grad";
  net: NetId | NetSpec;
  bind?: Record<ArrayName, ArraySpec>;
  outputs: Record<ArrayName, GradOutputSpec>; // new output name -> which gradient
  keep?: ArrayName[];                          // forward outputs to pass through unchanged
  name?: ShowString;
  description?: ShowString;
};

// One gradient output: d(of)/d(wrt).
// * `of` is an output of `net`. Without `seed` its declared shape must be []
//   (per batch element). With `seed` it is a vector–Jacobian product,
//   seed · d(of)/d(wrt), for any shape of `of`.
// * `wrt` is an input OR an internal array of `net`: a node (activations,
//   logits...) or an input bound away by `bind`.
// * The output's shape is the (batch prefix) + declared shape of `wrt`.
//   Gradients are per batch element (a batched `of` is a family of scalars,
//   each differentiated on its own): a loss evaluated at G weight points gives
//   G gradients.
// * `seed` as an ArraySpec is a fixed cotangent. As an ArrayName that is not
//   already an input of `net`, it becomes a NEW INPUT of the gradient net with
//   the shape of `of` — this is how a Hessian-vector product is written:
//   grad of a grad net, seeding with `v`.
export type GradOutputSpec = {
  of: ArrayName;
  wrt: ArrayName;
  seed?: ArraySpec | ArrayName;
};

/*******************************************************/
/* FIELDS BACKED BY NETS */
//
// Members of ScalarFieldDataSpec / VectorFieldDataSpec (schema/fieldData.ts).
// kind = "symbolic": the field can be evaluated at any
// point of its box, and `derivative(dim)` is EXACT via the chain rule
//   d out / d p_k = sum over inputs W of  < d out / d W ,  d inputs[W] / d p_k >
// (the runtime builds the grad net wrt every coordinate-dependent input and
// contracts with the derivative of the input expressions). Sampling a grid of
// G points is ONE batched evaluation with a leading [G] on each input.
//
// The remaining inputs of `net` are given as ArrayExprs of the coordinates:
// `coord` / `coordv` are allowed here (and only here), `arg` refers to
// `arrays` of this spec.
//
// DEFAULT: when `inputs` is omitted and the net has exactly one remaining
// input, that input receives the point itself (`coordv`, shape [D]). So a net
// written as a map R^D -> R (or R^D -> R^D) is a field with nothing but `net`,
// `output` and `box` — and so is the loss landscape: bind the data and theta*,
// `displace` along D directions, and the `coeffs` input is the point. When
// charts land (schema/mappings.ts), the displacement moves to the manifold
// mapping and the net field lives on the parameter manifold.

export type NetScalarFieldDataSpec = {
  type: "net";
  net: NetId | NetSpec;
  output: ArrayName;                       // declared shape []; the bound arrays must not add batch axes
                                           // (reduce over a declared axis such as "N" inside the net instead)
  inputs?: Record<ArrayName, ArrayExpr>;   // per remaining input, an expression of the coordinates; see DEFAULT
  arrays?: Record<ArrayName, ArraySpec>;   // named arrays usable from `inputs`
  box?: BoxSpec;                           // the field's domain; defaults to the unit box
};

export type NetVectorFieldDataSpec = {
  type: "netv";
  net: NetId | NetSpec;
  output: ArrayName;                       // declared shape [D], D = manifold dimension
  inputs?: Record<ArrayName, ArrayExpr>;   // omitted: the sole input gets the point (see DEFAULT)
  arrays?: Record<ArrayName, ArraySpec>;
  box?: BoxSpec;                           // the field's domain; defaults to the unit box
};

/*******************************************************/
/* ROOT */
//
// BundleSpec has
//   nets?: Record<NetId, NetSpec>;
// Nets are built lazily like fields (they reference each other by id in any
// order; cycles are errors).

/*******************************************************/
/* WORKED EXAMPLE: MLP on MNIST, loss landscape around θ*  (JSONC)
//
// {
//   "tensatory": "0.1",
//   "manifolds": { "pca": { "numDims": 2, "dimNames": ["pc1", "pc2"], "dimWeights": [0.61, 0.22] } },
//   "nets": {
//     "mlp": {
//       "type": "def",
//       "inputs": { "x": ["N", 784], "y": ["N"], "W1": [784, 128], "b1": [128], "W2": [128, 10], "b2": [10] },
//       "nodes": {
//         "h":      { "op": "relu", "val": { "op": "add", "vals": [{ "op": "matmul", "vals": ["x", "W1"] }, "b1"] } },
//         "logits": { "op": "add", "vals": [{ "op": "matmul", "vals": ["h", "W2"] }, "b2"] },
//         "logp":   { "op": "logSoftmax", "val": "logits" },
//         "nll":    { "op": "negate", "val": { "op": "takeAlong", "val": "logp", "axis": -1,
//                       "indices": { "op": "reshape", "val": "y", "shape": ["N", 1] } } },
//         "loss":   { "op": "reduce", "fn": "mean", "val": "nll" },
//         "pred":   { "op": "argmax", "val": "logits" },
//         "acc":    { "op": "reduce", "fn": "mean", "val": { "op": "eq", "vals": ["pred", "y"] } }
//       },
//       "outputs": { "loss": [], "acc": [], "logits": ["N", 10] }
//     },
//     // loss(weights) on the validation set: the dataset axis is reduced INSIDE the net
//     "mlp_val": { "type": "bind", "net": "mlp", "bind": { "x": "val/x", "y": "val/y" } },
//     // loss, accuracy and dloss/dW in one pass; inputs W1 b1 W2 b2 (batched if passed with a leading axis)
//     "mlp_val_grad": {
//       "type": "grad", "net": "mlp_val", "keep": ["loss", "acc"],
//       "outputs": { "gW1": { "of": "loss", "wrt": "W1" }, "gb1": { "of": "loss", "wrt": "b1" },
//                    "gW2": { "of": "loss", "wrt": "W2" }, "gb2": { "of": "loss", "wrt": "b2" } }
//     },
//     // Hessian-vector product: v is a new input (shape of gW1, i.e. [784, 128])
//     "mlp_val_hvp_W1": {
//       "type": "grad", "net": "mlp_val_grad",
//       "outputs": { "HvW1": { "of": "gW1", "wrt": "W1", "seed": "v" } }
//     },
//     // the trained model: no inputs left, loss / acc are plain numbers
//     "mlp_star": { "type": "bind", "net": "mlp_val", "bind": { "W1": "theta/W1", "b1": "theta/b1", "W2": "theta/W2", "b2": "theta/b2" } },
//     // ... and the 2D subspace around it: one input t: [2]
//     "mlp_pca": {
//       "type": "displace", "net": "mlp_star",
//       "directions": [{ "W1": "pca/d0/W1", "b1": "pca/d0/b1", "W2": "pca/d0/W2", "b2": "pca/d0/b2" },
//                      { "W1": "pca/d1/W1", "b1": "pca/d1/b1", "W2": "pca/d1/W2", "b2": "pca/d1/b2" }]
//     }
//   },
//   "fields": {
//     // `t` is the sole remaining input, so it receives the point: loss(t) on the pca plane
//     "loss": { "kind": "scalar", "codomain": "celoss", "domain": "pca",
//               "data": { "type": "net", "net": "mlp_pca", "output": "loss", "box": [[-1, 1], [-1, 1]] } },
//     "acc":  { "kind": "scalar", "codomain": "fraction", "domain": "pca",
//               "data": { "type": "net", "net": "mlp_pca", "output": "acc", "box": [[-1, 1], [-1, 1]] } }
//     // no `exactGradient` needed: the net field is symbolic, ∇loss is exact by autodiff,
//     // so S_∇ / V_∇ slots and `grad` in pointwise expressions just work.
//   },
//   "pointSets": { "theta": { "domain": "pca", "points": [[0, 0]], "labels": ["θ*"] } }
// }
//
// The same `mlp` def also expresses training: an SGD step is a def with a
// `call` of `mlp_val_grad` and `sub`/`mul` nodes (W - lr * gW); a training
// trajectory is that net iterated — outside this schema, but nothing in it
// needs a new vocabulary.

/*******************************************************/
/* DECISIONS
//
// 1. Reals only. Arrays are Real; labels / indices are rounded on use
//    (takeAlong, oneHot, eq). No dtype in the schema for now (storage may
//    still carry one, e.g. uint8 images, decoded to f32 on load).
// 2. Dataset axes are DECLARED, never implicit. A net that reduces over a
//    dataset declares the axis as a symbolic size ("N") and `reduce`s over it,
//    so a loss is a true [] output. Implicit batching (extra leading axes) is
//    only for evaluating one net at many parameter points — a [G] of weights
//    — and there is no operator that reduces over batch axes. The alternative
//    (per-example nets whose dataset axis is an implicit batch, with an
//    external `batchReduce`) was considered and rejected as a second way to
//    say the same thing.
// 3. `einsum` stays as the general contraction (one node, one autodiff rule:
//    swap the differentiated operand with the output subscripts); `matmul` is
//    kept as sugar for the common case.
// 4. Execution target is WebGPU (packages/gpu): the WGSL transpiler already
//    covers the elementwise layer; matmul / einsum / reductions become
//    compute kernels, and a grid of G points is one batched dispatch. The CPU
//    NdArray evaluator exists for correctness only — tests assert CPU / GPU
//    agreement within a tolerance, as the GPU package already does for
//    fields. (The MLP above on 10k examples at a 64×64 grid is ~2·10^11
//    MACs — not a CPU workload.)
//
// STILL OPEN
//
// * Randomness (dropout, data order) is deliberately absent: nets are pure.
// * Storage: weights and datasets want the handle backend (roadmap §1) — the
//   first real driver for it.
*/

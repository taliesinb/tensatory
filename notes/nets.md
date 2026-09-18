# Nets: small neural networks in a bundle

Types in `schema/nets.ts`; zod, shape inference, the CPU reference evaluator
and autodiff in `packages/core/src/nets/`; the WebGPU transpiler in
`packages/gpu/src/nets.ts`. Status: everything in the schema is implemented.
`def` / `bind` / `displace` / `call` / `grad` evaluate on the **CPU**
(reference, Float64) and on the **GPU** (a net field becomes an ordinary WGSL
field function; see [gpu.md](gpu.md) stage 8); `grad` is reverse-mode
autodiff as a program rewrite, so net fields have exact derivatives to any
order on both backends. The iris example bundle agrees with PyTorch to 1e-9
and CPU / GPU agree to f32.

## Why

The loss-landscape prototype sampled loss / accuracy over a 2D or 3D slice of
parameter space in Python and shipped the grids. With nets in the bundle, the
bundle carries the *model* instead: the MLP, its trained weights, the
validation set, the loss as an output — and the viewer samples the landscape
itself, at any resolution, along any slice, with exact gradients by autodiff.
The slice / grid / resolution machinery already exists for symbolic fields;
a net-backed field is just another `kind: "symbolic"` datum.

## The pieces

**Net** = pure function, named input arrays → named output arrays, both with
predeclared per-example shapes. Nothing distinguishes weights from data; loss
is an output of shape `[]` like any other. `BundleSpec.nets` maps `NetId` →
`NetSpec`, one of:

| type | meaning |
|---|---|
| `def` | `inputs: {name: shape}`, `nodes: {name: ArrayExpr}` (a DAG, any order), `arrays` (baked constants), `outputs: {name: shape}` — the declared output shapes are *checked* against the inferred ones |
| `bind` | fix some inputs to `ArraySpec`s → a net over the remaining inputs; bound inputs become internal arrays |
| `displace` | K directions (`{ arrays, norm?, scale?, name?, widget? }`) over any arrays of the net → the same net plus a `[K]` input `coeffs` (default `t`); every named array A becomes `A + Σₖ tₖ·directions[k].arrays[A]` wherever it is used (each direction scaled to `norm` × `scale`) |
| `grad` | net (+ optional `bind`) → a net whose outputs are gradients `d(of)/d(wrt)` plus `keep`-ed forward outputs |

One namespace per net for inputs, nodes and constant arrays.

**ArrayExpr** — the scalar language of `symbolic.md` lifted to arrays:

* leaves: number (0-d), bare name / `arg`, and — only in a net-backed field's
  `inputs` — `coord {index}` (0-d) and `coordv` (`[D]`);
* elementwise with numpy broadcasting, reusing `ScalarUnaryOp` /
  `ScalarNaryOp` / `ScalarBinaryOp` and `clamp`; comparisons `lt le gt ge eq
  ne` (0/1) and `where {cond, vals}`;
* contractions `matmul` (numpy semantics) and `einsum {subscripts}` (letters
  only, no `...`; a letter repeated in an operand reads its diagonal, a
  letter repeated in the output writes one; `matmul` is sugar for the common
  case);
* `reduce {fn: sum|mean|prod|max|min|logsumexp, axes?, keepDims?}`,
  `argmax` / `argmin {axis}`, `softmax` / `logSoftmax {axis}`;
* shape ops `reshape` (one `-1`, symbolic names allowed), `transpose {perm}`,
  `concat {axis}`, `slice {axis, start, stop, step}` (python semantics),
  `oneHot {size}`, `takeAlong {indices, axis}` (torch.gather), `stopGradient`;
* `call {net, inputs, output}` — invoke another net; the composition
  mechanism (an SGD step is a `def` calling a `grad` net). No recursion.

Everything is `Real`; indices (`takeAlong`, `oneHot`, `eq` against labels)
are rounded on use.

## Two kinds of extra axes

This is the one real design decision; get it right and the rest follows.

* **Symbolic axes** (`"N"`) are *declared*: `x: ["N", 784]`, `y: ["N"]`. The
  size is bound when arrays are supplied and must agree everywhere. Because
  the axis is declared, the body sees it and can `reduce` over it — that is
  how a loss becomes a true `[]` output (mean over the validation set inside
  the net). PyTorch style.
* **Batch axes** are *undeclared*: any array passed with extra leading axes
  beyond its declared rank is vmapped over them. Batch prefixes of the
  different inputs broadcast numpy-style (right-aligned, 1 stretches) and
  every output gets the broadcast prefix in front of its declared shape. The
  body never sees them and there is deliberately **no operator that reduces
  over batch axes**. Batching is for evaluating one net at many parameter
  points: weights passed as `[G, 784, 128]` give `loss: [G]` — one dispatch,
  not G calls — and per-point gradients.

The alternative (per-example nets with the dataset as an implicit batch and an
external "reduce the batch" knob) was considered and rejected as a second way
to say the same thing.

Symbolic sizes are opaque in inference: `"N"` is never assumed equal to `"M"`
or to a number, except through broadcasting with 1. Whatever cannot be proven
is an error at parse time — nets are checked once, not discovered to be wrong
by NaNs at sample time. Consequences: `concat` / `slice` along a symbolic
axis and `reshape` with `-1` absorbing a symbolic size are errors.

## displace: "around" a net

```jsonc
"iris_star":  { "type": "bind", "net": "iris_val", "bind": { "W1": …θ*… } },          // no inputs left
"iris_rnd2":  { "type": "displace", "net": "iris_star",
                "directions": [{ "arrays": { "W1": …, "b1": …, "W2": …, "b2": … }, "norm": "origin", "widget": { "id": "d0" } }, …] }  // one input t: [2]
```

A direction (`DirectionSpec`) has `arrays`: array names → arrays of matching
per-example shape; the names may be inputs, internal nodes (activations),
inputs bound away by `bind`, or baked constants — a missing name is a zero
direction, several names in one direction move together. `bind` supplies the
origin, `displace` the directions, and since the displaced net's sole
remaining input is `t: [K]`, the field needs nothing but `net` + `output` +
`box` (see below). "Around" is therefore flexible: parameter subspaces
(random / PCA / Hessian directions), but equally a perturbation of a hidden
activation or of the data. This replaced an earlier `frame` sugar on the
field spec: one mechanism, and it composes (displace a grad net, displace
along the output of another net…).

A direction's arrays are ONE vector in the joint space of the named arrays:
`norm` fixes its Euclidean length over all of them together — a number, or
`"origin"` for the joint norm of the displaced arrays' own values (the
loss-landscape convention "each direction as long as θ*"; the arrays must
then be constants: bound inputs or baked arrays, a node has no value before
evaluation) — and `scale` multiplies it afterwards. Random directions are
directions whose arrays are `random` (bundle-schema.md): the iris bundle's
are gaussian with seeds `"d0/W1"` etc. and `norm: "origin"`, so a direction
is a fresh draw per seed, normalized like the fixed ones were. Random draws
are not orthogonalized (in 131 dimensions the cosines are ~0.09).

`widget` on a direction asks the viewer for a Controls-pane row for the whole
direction (reseed + scale); see viewer.md "Controls". Rows never edit the
bundle: core's `adjustSpec` (bundle/controls.ts) returns a spec with the
row's salt hashed into every random array of the direction and its
multiplier folded into `scale`.

Implementation (`program.ts`): each direction's arrays are built, its factor
(`scale` × `norm` / joint norm) computed, and the inner program gains a
constant `A__dispd = [K, …shape(A)]` per target (direction k scaled by its
factor) and a node `A__disp = A + einsum("k,k…->…", t, A__dispd)` right after
`A` is available; every later use of `A` — in nodes, `call` inputs and the
output map — is renamed to `A__disp`. So displacement is an ordinary graph
rewrite and the evaluator knows nothing about it.

## grad

```jsonc
{ "type": "grad", "net": "mlp_val", "keep": ["loss", "acc"],
  "outputs": { "gW1": { "of": "loss", "wrt": "W1" }, "gb1": { "of": "loss", "wrt": "b1" } } }
```

* `of` is an output of the net; without `seed` it must be `[]` per example.
  With `seed` it is a vector–Jacobian product for any output shape.
* `wrt` is an input or any internal array — a node (activations, logits) or
  an input bound away by `bind` (the gradient at fixed weights, which is the
  loss-landscape case).
* The gradient output has `wrt`'s per-example shape; the batch prefix goes in
  front at evaluation: a `[G]`-batched loss gives `[G, 784, 128]`.
* `seed` as an `ArraySpec` is a fixed cotangent; as a name not already an
  input it becomes a **new input** of the gradient net with the shape of
  `of`. Grad of a grad net seeded with `v` is a Hessian–vector product.
* `keep` passes forward outputs through, so loss and ∇loss come from one
  pass. Forward outputs become internal arrays of the gradient net.

## Autodiff: `grad` as a program rewrite (`autodiff.ts`)

A gradient net compiles to an ordinary `Program`: the forward program in
**A-normal form** — every subexpression becomes a named node, so every
intermediate a backward rule needs (the pre-activation of a `relu`, the
output of a `softmax`, the max of a `logsumexp`) is addressable — followed by
**adjoint nodes** emitted while walking the forward nodes in reverse. Each
node's adjoint is the sum of its consumers' contributions; a contribution
that broadcast up from a smaller operand is summed back down (`unbroadcast`:
leading axes and size-1 axes). Every adjoint is written with the existing
ops, so both evaluators differentiate without knowing it, and the result
composes: a grad program can be bound, displaced, called and differentiated
again — a Hessian-vector product is a seeded rewrite of a rewrite, and
`derivative(dim)` of a net field is component `dim` of the gradient program
wrt the point.

**The op set is closed under adjoints**: every adjoint is written with ops
that are themselves differentiable, so `grad` never fails and higher orders
compose. The one extension this needed is in `einsum`: a letter repeated in
the *output* (`i->ii`) writes the diagonal and leaves the rest zero — the
adjoint of reading a diagonal (`ii->i`, `ii->` the trace), and vice versa.

Rules worth knowing: `matmul` is lowered to `einsum` first, so one contraction
rule serves both (adjoint of operand k = einsum of the output gradient with
the other operands, output letters = k's letters — repeats included; letters
only k carries come back through a ones operand). `reduce`: sum/mean broadcast the gradient,
max/min mask the arg (ties share), prod uses `y/x`, logsumexp `exp(x − y)`.
`softmax` / `logSoftmax` use their output. `slice`'s adjoint scatters through
a **baked selection matrix** contracted by `einsum`; `takeAlong`'s through
`oneHot` · gradient summed over the gathered axis and transposed back.
`concat` slices; `reshape` / `transpose` invert. `call` differentiates by
calling the callee's own gradient net with the seed (an inline `grad` spec —
the rewrite recurses through nets). Piecewise-constant ops (`argmax`,
`oneHot`, comparisons, `floor`…) contribute nothing; `stopGradient` cuts.
Symbolic axes are fine throughout (reductions and reshapes carry names); the
implicit batch never appears, so gradients come out per batch element.
`test/autodiff.test.ts` checks every op against central differences, batched
weights, VJP / HVP, and second derivatives of the iris field.

Call-site batching (passing `[2,3]` to a callee declaring `["N"]`) is part of
the *caller's* declared shape; the CPU evaluator accounts for it, the WGSL
emitter does not support it yet.

## Evaluation (CPU reference)

`ops.ts` — a value is an `NdArray` plus its DECLARED rank; the extra leading
axes are the batch. Every op acts on the declared axes and broadcasts the
batch prefixes numpy-style by aligning strided views (`aligned`: pad the
batch to B and the declared part to R with size-1 axes, stride 0), so vmap
semantics fall out and no op knows whether it is batched. `einsum` is the
one contraction engine (private letters are prepended for the batch, the
largest letter goes innermost, a 2-operand fast path); `matmul` is spelled
in it. Reductions, `argmax`, `softmax`, shape ops (`reshape` keeps the batch,
`transpose` / `concat` / `slice` / `takeAlong` shift axes by the batch rank).
Hot loops run per row along the last axis (`forEachRow`), not per element.
Float64, no blocking: ~27 µs per grid point for the iris MLP (30 examples,
4-16-3) batched, ~70 µs for a single point.

`program.ts` — `compileNet` flattens a spec into a `Program` (inputs with
declared shapes, constant arrays incl. bound inputs, nodes in dependency
order, output map); `evaluate(program, inputs)` binds symbolic sizes from the
actual arrays, runs the nodes, returns the outputs with their batch. `call`
evaluates the callee recursively — it sees the caller's batch as extra
leading axes, so vmap composes by itself. `Bundle.net(id).program` compiles
lazily; `Net.evaluate(inputs)` is the public entry.

## Net-backed fields

```jsonc
{ "type": "net", "net": "iris_rand2", "output": "loss", "box": [[-1, 1], [-1, 1]] }
```

The net's remaining inputs are given as `inputs: {name: ArrayExpr}` —
expressions of the coordinates (`coord` / `coordv`) over named `arrays`. When
`inputs` is omitted, a net with exactly one remaining input receives the
point itself (`coordv`, `[D]`), so a net written as a map ℝᴰ → ℝ or ℝᴰ → ℝᴰ
is a field with just `net`, `output` and `box` — and so is a loss landscape
(bind → displace → field, above). `box` is the field's domain like every
other field datum (default the unit box). Checks: every input given, shapes
unify with the declared ones, no batch axes from the net's bindings or from
the given arrays (a field needs one value per point), `output` is `[]` for
`net` and `[D]` for `netv`.

`nets/fieldData.ts`: a net field is a **program with one input — the point,
`[D]`** — and one output. `fieldProgram` folds the spec's coordinate
expressions and named arrays into the net's program (`coordv` → the point,
`coord i` → its i-th component), so the evaluators and autodiff never see
coordinates. `NetScalarFieldData` / `NetVectorFieldData` are
`kind: "symbolic"`; `sampleOn(grid)` is one batched evaluation per chunk of
2048 points with a leading `[G]` on the point; `fn` evaluates a batch of one.
`gradient()` is the gradient program wrt the point as a net vector field
(one backward pass for all D components); `derivative(dim)` is its
component, so derivatives are exact and compose to any order. Statistics use
a small grid (48², 12³).

### On the GPU

`packages/gpu/src/nets.ts` turns a net field into a WGSL function
`fn nf(p, pos) -> f32 | vecD` like any symbolic field: one thread evaluates
the whole program for its point, so there is no batch inside the function —
every array has its declared per-example shape (symbolic sizes resolved from
the bound arrays) in a function-scope `array<f32, N>`, constants sit in the
shared storage buffer, ops are nested loops (`einsum` / `reduce` with an inner
accumulation), `call` is inlined, `reshape` aliases. Because it is a field
function, every kernel — raster, exact isolines, streamlines, glyphs,
marching tetrahedra — evaluates the net in place; the iris MLP samples 256³
in milliseconds. Function-scope memory is the limit (`NET_MAX_FLOATS`, 16k
floats): larger nets fall back to CPU sampling on the dispatch grid.
`gpuTranspilable(fd)` reports whether a field (or anything derived from it)
stays on the GPU. Derivatives are the autodiff programs, transpiled like any
net; `ProgramBuilder.gradient(fd)` emits ONE `vecD` function per field (the
projection routines and the ∇ uses call it), never D component programs.
The limit is real: **Safari rejects a WGSL function whose variables exceed
8192 bytes**, so `NET_MAX_FLOATS` is 2000 and the emitter minimizes
function-scope memory — A-normal form, best-fit reuse of dead arrays by
per-node liveness, in-place elementwise updates, and fusion of single-use
elementwise producers into their consumer (one loop per elementwise tree;
the autodiff idioms `mul(x, 0)` / `add(…, 1)` fold away). Iris: forward 821
floats, gradient 1661, second derivative 3943 (CPU fallback). Compile time
is the other constraint; see [gpu.md](gpu.md) stage 8 for the Chrome / Safari
measurements and why loop bounds are opaque. The 3D arm still skips exact
projection for nets: at 256³ a single dispatch doing value + gradient per
Newton step per vertex exceeded the GPU watchdog and lost the device —
normals do use the exact gradient, and 2D isolines of nets are exact.

### `costly`

Net fields — and everything derived from them: derivatives, expression fields
over them, pullbacks — report `costly: true` (`FieldData.costly`): sampling
them on the CPU is expensive. The viewer's `costly(fd)` helper combines this
with the compute mode: in GPU mode a transpilable net is as cheap as any
symbolic field. When a field IS costly (CPU compute, or a net over the size
limit) the viewer uses small fixed streamline grids (32 in 2D, 16 in 3D
instead of 128 / 64), caps glyph lattices at 2k points, skips exact isoline
projection, and lets the resolution controller judge a *compiled* frame by
its JS time when that alone exceeds the budget — a shader compile stalls the
GPU, not the main thread, whereas CPU sampling does; without this the
controller discarded every CPU-slow sample as a compile hiccup (the fallback
buffer reader bakes its grid, so each resolution step compiles), remeasured
from the cache, and climbed to 128³. Expression fields with costly arguments
(`∇loss` as a `SymbolicVectorFieldData` over the net field) pre-sample each
costly argument — and, lazily, its derivatives — on the grid in batched calls
(`gridCached`) and evaluate the expression with `pos` set, instead of
per-point closures.

## The iris example

`tools/iris/train.py` (PyTorch, once-off, deterministic; data in
`tools/iris/iris.data`) trains a 4-16-3 ReLU MLP on 120 iris examples (Adam,
weight decay) and writes `apps/viewer/public/bundles/iris.json`: the net
(`iris`, with standardization `mu` / `sigma` as baked arrays), the validation
set bound (`iris_val`), θ* bound (`iris_star`), and `displace`d along 2 and 3
orthogonal Gaussian directions each as long as θ* (`iris_rand2`,
`iris_rand3`), with `loss` / `acc` fields on manifolds `rand2` / `rand3` and
θ* as a point set. It also writes
`packages/core/test/fixtures/iris-reference.json`: PyTorch's float64 loss and
accuracy at θ* and at displaced points; `packages/core/test/iris.test.ts`
checks Tensatory's evaluation — per point through the field, and all points
in one batched call — against it to 1e-9, and that grid sampling equals
per-point evaluation. Weights are the exact float32 values as decimals, so
both sides compute the same function in double precision.

## Implementation (`packages/core/src/nets/`)

* `spec.ts` — zod: `ArrayExprSchema`, `NetSchema` (mutually recursive via
  `call`), `NetScalarFieldDataObject` / `NetVectorFieldDataObject` spliced
  into the field-data discriminated unions in `fields/spec.ts`.
* `shapes.ts` — `inferNet(spec, resolver) → NetSignature { inputs, outputs,
  nodes, batch }` for `def` / `bind` / `displace` / `grad`, `inferExpr` over
  an `ArrayEnv` (names in scope, declared axis names, `coordDims` when
  coordinates are allowed), `inferNetField`. Unification of declared vs
  actual shapes goes through a `Subst` (symbolic name → Dim) shared across
  one bind / call. Constant `arrays` are internal arrays of the signature
  (displaceable, differentiable) like nodes and bound inputs.
* `ops.ts`, `program.ts`, `autodiff.ts`, `fieldData.ts` — the evaluator, the
  gradient rewrite and the field classes (above); `packages/gpu/src/nets.ts`
  — the WGSL transpiler, `packages/gpu/test/nets.test.ts` — CPU / GPU
  agreement per op, for iris and its gradient and second derivative.
* `Bundle.net(id)` builds signatures lazily with cycle detection (a net may
  not call or bind itself, directly or indirectly) and compiles programs on
  demand; `buildAll()` reports net errors under `nets.<id>`. Field builders
  take an optional `nets` resolver (`ProgramResolver`).

Tests: `packages/core/test/nets.test.ts` — the MLP of `schema/nets.ts` as the
running example (signature, bind fixing N, batched weights, grad / HVP, call
batching, displace, every op's evaluation incl. implicit-batch alignment, the
MLP forward pass by hand, net-backed field checks and values);
`iris.test.ts` — against PyTorch.

## Next

1. **Larger nets on the GPU**: the transpiler keeps every per-example
   intermediate in function-scope memory, which caps the dataset size (iris:
   ~1k floats; MNIST's 10k × 128 hidden would not fit). The way out is to
   stream the declared dataset axis: every array carrying `"N"` is consumed
   only by reductions over it, so a loop over examples can wrap the
   per-example body with small intermediates and accumulate the reductions.
   With autodiff, the same transformation gives the gradient in one pass.
2. Storage: real weights and datasets (MNIST) need the handle backend
   (roadmap §1) — the first real driver for it.
3. Emitter: call-site batching (a loop over the extra axes around the inlined
   callee); fusing multi-use elementwise producers (recompute vs store);
   second derivatives of nets within Safari's limit (stream the dataset axis).

Open: randomness (dropout, data order) is deliberately absent — nets are pure.

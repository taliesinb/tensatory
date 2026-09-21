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

`tools/iris/train.py` (PyTorch, once-off, deterministic — a rerun reproduces
θ* bit for bit; data in `tools/iris/iris.data`) splits the 150 examples
stratified into 120 training / 30 validation (10 per class, no test set),
standardizes with the training mean / std, trains a 4-16-3 ReLU MLP with Adam
(lr 0.01, 400 epochs, **weight decay 3e-3**) and writes
`apps/viewer/public/bundles/iris.json`. The net `iris` (standardization `mu`
/ `sigma` as baked arrays) has outputs `loss` (mean cross-entropy), `acc`,
`obj` = loss + wd/2·‖θ‖² — the objective Adam actually minimized, since
PyTorch's `weight_decay` adds wd·θ to the gradient — and `loss0..2`, the mean
loss of each class (`eq(y, c)` masks; `nll` is `[N, 1]` from `takeAlong`, so
it is reshaped to `[N]` first). Two binds of the data — `iris_val` (30) and
`iris_trn` (120) — each bound to θ* and displaced two ways: `iris_rnd2/3` and
`iris_trn_rnd2/3` along RANDOM gaussian directions (`random` arrays with
seeds `d0/W1`…, `norm: "origin"`, Controls rows `d0..d2` shared by both
binds, so a reseed moves every field together), and `iris_rand2/3` /
`iris_trn_rand2/3` along the three fixed inline orthogonal directions the
PyTorch reference was computed along (no fields; the tests attach their
own). Fields per space (`rand2` / `rand3`): validation `loss` / `accuracy`,
`train loss` / `train acc`, `objective`, and `loss: setosa / versicolor /
virginica` (training set), θ* as a point set.

Why θ* is not the minimum of the *validation* loss picture: it minimizes
`obj` on the training set — the validation minimum along a random direction
is generically elsewhere, and the weight-decay term pulls θ* off even the
training-loss minimum. `objective` is the field θ* is the (Adam-approximate)
minimum of; the test checks its gradient at θ* is smaller than the loss's and
that every displaced reference point is higher.

GPU: the validation nets (N = 30) transpile (821 / 1661 floats for loss /
its gradient); the training-set nets (N = 120: `h` alone is 1920 floats)
exceed it BATCHED, which is where STREAMING comes in (below). Dead-code
elimination to the field's output (`pruneProgram`: `loss` needs none of
`acc` / `loss0..2` / `obj`) is applied for the CPU evaluator (`evalPoints`,
memoized per program — the 12³ gradient grid went from 1.9 s back to
0.35 s). The GPU emitter's batched path still gets the full program; the
streamed path prunes to the wanted outputs. (A "latent emitter bug" once
seen with pruned input — ∇loss off by 10³ at some points — is not
reproducible: value, gradient, components and second derivatives of pruned
programs agree with the CPU in 2D and 3D. It was almost certainly the fixed
`componentProgram` bug: it named its node `__c<i>` without checking, so a
component of a gradient of a component defined `__c1` twice; evaluators
resolved it by last-definition-wins, pruning by name did not.)

### Streaming the dataset axis

`NetEmitter.streamed` (`gpu/src/nets.ts`) emits the same batched program
with its symbolic axis S ("N") in TIME instead of SPACE: the nodes whose
shape carries S are computed one example at a time inside `for n < N`
(reading the batched constants through a one-example VIEW), and the nodes
that consume S — a `reduce` over it, an `einsum` / `matmul` contracting it:
the loss's mean, the weight gradients' `ij,ik->kj` — ACCUMULATE across the
loop (sum / mean / prod / max / min, logsumexp online). Per op nothing
changes but the axis bookkeeping (`perExample`: axis arguments shift past
S, einsum letters lose it, reshape must name S); the N loop moves outside
the whole per-example chain, so `[N, 16]` becomes `[16]`. It is legal
exactly when S only ever ends in a reduction — an op that couples examples
(softmax / slice / concat / argmax along S, an einsum where S meets an
unbatched operand, a call with batched inputs) throws `StreamError` and the
batched emission stands. Classification is by DEPENDENCE, not shape:
autodiff's `1/N` seed `add(mul(nll, 0), 1)` has shape [N, 1] but reads
nothing (the emitter folds `mul(x, 0)`), so it is UNIFORM — `elementwise`
folds an index-free tree to one scalar `let` — and lives outside the loop,
which keeps the gradient at one pass (a batched node that needs an
accumulated result belongs to a later loop that recomputes its per-example
chain: rematerialization, supported but not needed here). Values defined
outside the loop are never released or written in place inside it (the
first version put the hidden layer into the bias array). `emitNetField`
tries the batched emission first and streams when it does not fit
(`stream: "auto"`; `setEmitStream("always")` forces it, for tests). iris:
`train/loss` 3731 floats batched → 159 streamed (its gradient 424,
`objective` 555); the 12 training fields are on the GPU (`gpuTranspilable`
true, no longer `costly` for the viewer), in Safari too; `gpu/test/nets.test.ts`
checks values, gradients and second derivatives against the CPU and the
streamed against the batched emission of the validation fields.

`packages/core/test/fixtures/iris-reference.json` holds PyTorch's float64
validation loss / accuracy and training loss / accuracy / objective /
per-class loss at θ* and at displaced points along the fixed directions;
`packages/core/test/iris.test.ts` checks Tensatory's evaluation — per point
through the field, and all points in one batched call — against it to 1e-9,
and that grid sampling equals per-point evaluation. Weights are the exact
float32 values as decimals, so both sides compute the same function in
double precision.

### Lazy arrays and the work budget: the MNIST MLP experiment

`apps/viewer/public/bundles/mnist-mlp/` (`tools/mnist/export.py`) is the
loss-landscape prototype's trained 784-256-256-10 MLP (269 322 parameters)
as a live bundle: θ*, the three PCA directions of its SGD trajectory and the
1 024-example eval set (uint8 pixels, normalized in the net) are `.npz`
members; `bind(data, θ*) → displace → field`, exactly the iris pattern at
real size; the prototype's own 64³ PyTorch volume rides along (32³, `vol.npy`)
in the same box as a sampled reference. `core/test/mnist.test.ts` checks the
CPU evaluator against PyTorch at displaced points to 1e-6 (~0.1 s per point
for 256 examples). Two things it forced:

* **Lazy arrays** (`Arr` kind `"expr"` in `gpu/src/nets.ts`). A displaced
  weight is a node `W + einsum("k,k…->…", t, D)`; materialized, `W1t__disp`
  is 200 960 function-scope floats. Now an elementwise tree or a short
  contraction (≤ `LAZY_MAX_SUMMED` summed elements — the K directions) whose
  result is ≥ `LAZY_MIN_FLOATS` and whose operands are all STABLE (storage
  data, literals, the point, other lazy arrays — never a function-scope
  array, whose buffer is reused, nor a streamed view, whose loop variable is
  local) is not emitted at all: it becomes an index → expression function
  that every `read` inlines, so the matmul reads
  `data[W + i] + (p[0] * data[D0 + i] + …)` straight from the storage buffer.
  The MNIST fields then fit: 1 584 floats, N streamed, 3 kB of WGSL, and the
  GPU agrees with the CPU (`gpu/test/mnist.test.ts`, `PERF=1`).
* **The work budget** (`NET_MAX_WORK`). Fitting is not enough: ONE lane
  evaluates the whole net for its point as a serial chain of storage reads,
  latency-bound at roughly 10⁷ multiply-adds per second, and more points
  only run alongside. Measured (Apple GPU, Dawn): 256 examples × 269k
  weights = 7·10⁷ MAC per point → **10 s per dispatch at any grid size**
  (2×2 and 32² alike); 1 024 examples → 40 s; without the displacement reads
  in layers 1–2 still 8.5 s, so the lazy reads cost ~17% and the base matmul
  chain is the problem — hoisting the point-independent first layer (75% of
  the MACs, a `t`-linear precomputation) would buy at most 4×, not the 100×
  needed. Such a dispatch would trip the GPU watchdog, so the emitter now
  estimates the per-point work (fills plus contraction multiply-adds, times
  the streamed N) and `gpuTranspilable` refuses fields beyond 4·10⁶
  (`setNetMaxWork` overrides for benchmarks); iris is ~10⁵. The MNIST fields
  are therefore `costly` in the viewer, i.e. CPU-sampled — ~110 ms per point,
  two minutes for the first 32² rung — which is why the bundle is NOT in
  `bundles/index.json` (it loads via `?bundle=mnist-mlp/bundle.json`; expect
  that freeze).

What would make it interactive is not a better per-thread program but a
different mapping: a **cooperative kernel** — one workgroup per grid point (or
per few points), its threads splitting each matmul's output units with the
activations in workgroup memory, the example loop outside — writing a
resident values grid that the fused kernels then read, the way `costly`
fields already are consumed. That is a second emitter over the same
`Program` (the ops are the same; only the loop-to-thread mapping changes) and
the natural first customer of the resident-grid path for nets; the
per-thread function stays for small nets, where it is optimal (no barriers,
usable at isoline vertices). Expected: 7·10⁷ MAC spread over 256 lanes ≈
3·10⁵ per lane ≈ tens of ms per point-batch, so a 64² grid in well under a
second even before hoisting the first layer. Roadmap 5.

**Step 0, measured** (`gpu/test/coop-proto.test.ts`, `PERF=1`; Apple M4, Dawn):
a HAND-WRITTEN cooperative kernel for the MNIST loss — one workgroup of 256
threads per point, layer 1 hoisted, the dataset axis streamed in TILES of E
examples with the activation tiles `[E, 256]` in workgroup memory, thread j
owning output unit j and reading the displaced weight `W2(t)[i, j]` ONCE per
i for the E examples — agrees with the CPU evaluator to 1e-3 and takes, per
point at N = 256: **245 µs (E = 1), 131 (2), 77 (4), 45 µs (E = 8)**, i.e. a
32² grid in 46 ms (780 GFLOPS effective); N = 1024 at E = 8: 271 µs/point.
Versus 10 s per dispatch on the serial lane and 110 ms/point on the CPU. The
linear gain in E says the cooperative kernel is **memory-bound on the weight
stream** (the naive mapping re-reads the point's weights per example; the
tile divides that by E), so example tiling is the core of the design; opaque
loop bounds (`nb_`) cost 2–3× here (unrolling the 256-long contraction lets
the compiler batch the loads).

### As built: hoisting and the cooperative kernel

Three pieces, all on by default, nothing in the bundle format changed:

1. **Hoisting** (`core/src/nets/hoist.ts`, `hoistProgram`), applied lazily
   to every net FIELD's program (`fields/spec.ts` `netField`: the field's
   `program` getter hoists on first use and memoizes per program object, so
   fields sharing a net pay once — ~0.2 s for N = 256, ~0.9 s for 1024). A
   displaced constant is the node `W__disp = W + einsum("k,k…->…", t, Dd)`
   (`matchDisp`); a two-operand contraction (matmul or einsum) of it with a
   CONSTANT distributes, wherever it sits in an expression tree —
   `relu(add(matmul(xs, W1__disp), b))` becomes `relu(add(A + einsum("k,knj
   ->nj", t, XD), b))` with `A = xs·W1` and `XD = stack_k xs·D1_k` folded
   constants (`<node>__h`, `<node>__hd`, declared shapes with the symbolic
   axis kept: `XD: [K, "N", 256]`). Every other node whose operands are all
   constants folds too (up to `FOLD_MAX_ELEMENTS`, 4 M), a fully bound
   program folds down to its outputs, and dead nodes / constants are pruned
   (`x`, `W1t`, `W1t__dispd`, `xs` disappear from the MNIST program). Only
   field programs are hoisted: their sole input is the point, so nobody asks
   for a gradient with respect to a folded array; `grad` of the field
   differentiates the hoisted program through the einsum with t. The CPU
   evaluator gains the same 4× (per-point work 7·10⁷ → 1.8·10⁷ MAC).
   `core/test/hoist.test.ts`: structure, values and gradients against the
   unhoisted program (synthetic net and MNIST); the iris and MNIST PyTorch
   comparisons run through hoisted programs.

2. **The cooperative emitter** (`gpu/src/coop.ts`, `CoopEmitter extends
   NetEmitter`): the same op vocabulary with a different loop-to-thread
   mapping. Arrays are `var<workgroup>` at module scope (`declare` hook)
   with the base emitter's liveness reuse; every element loop is STRIDED
   over the workgroup (`for (f = lid; f < n; f += WG)`) and followed by a
   `workgroupBarrier()` — only ever in uniform control flow (bounds are
   literals or `nb_`, which reads read-only storage; the chunk guard
   `if (point >= count) return` is uniform per workgroup). `tiled()` is
   `streamed()` with the axis KEPT at size E instead of dropped: the plan
   (`streamPlan`, factored out of `streamed`: axis, shapes, batched /
   boundary / plain classification, loop levels) is shared; arrays carrying
   the axis are tile views `sVar + e`; `perExample` is consulted only for
   legality and the accumulation function; a `mean` over the axis is summed
   per tile and divided by the full count at the end; `sizeOf(S)` inside the
   loop is E, so a `reshape` naming N works per tile. E is the largest of
   8 / 4 / 2 / 1 dividing N whose emission fits the workgroup budget
   (`setCoopWorkgroupBytes` from the device limit: 32 KB on Apple → 7872
   floats → E = 8 for MNIST at 4787 floats; the 16 KB default gives E = 4).
   Contractions have three mappings: OUTPUT-PARALLEL WITH A REGISTER TILE
   when a storage-resident operand (the weight, kind `data` / `expr` /
   data view) lacks some output letters (the tile axis) and the rest give
   ≥ WG/2 threads work — thread ↔ its letters, E private accumulators, the
   weight read once per summed index; plain output-parallel otherwise (the
   `[E, 10]` logits); CONTRACTION-PARALLEL with a tree reduction through
   `red_[WG]` when the output is ≤ 32 elements and the contraction ≥ 2·WG
   (the same for `reduce`). The displacement term `einsum(t, XD)` stays
   LAZY (a tile view of storage data counts as stable inside the loop) so
   the hoisted layer is one in-place elementwise fill. Thread 0 writes the
   wanted outputs' elements to `out[point * channels + c]`; the point comes
   from the dispatch grid header offset by `params[0]` (the chunk).
   `emitCoopKernel` returns the `main` + declarations; `ProgramBuilder.
   buildCooperative(fd, outputs)` prepends its `library()`;
   `coopCapable(fd)` is a cached dry emission. Two CODE-SHAPE knobs
   (`CoopEmitContext.unroll` / `exampleBound`, defaults via
   `setCoopCodeShape`) were settled with `apps/viewer/public/cooptiming.html`
   (dev server; builds the MNIST fast/loss kernel in every shape and times a
   32² grid, GPU completion timed; `?field=loss2`, `?n=16`): the register
   tile's small per-thread loops are emitted UNROLLED with one scalar
   accumulator per tile element — Chrome is indifferent (50 µs/point either
   way) but Safari goes from 163 to 73 µs/point and compiles in half the
   time; the example loop's `nb_` bound costs nothing in either browser and
   stays (the body is huge). The remaining Safari gap (73 vs 50) is in the
   per-tile weight stream itself, not the barriers — both browsers pay the
   same for halving the tile (E = 8 / 4 / 2: Chrome 50 / 77 / 143, Safari
   73 / 103 / 188 µs) — most likely WebKit's bounds-checked storage reads,
   out of WGSL's reach. Measured: the EMITTED MNIST kernel runs at 50
   µs/point (N = 256; 32² in 51 ms) and 200 µs/point (N = 1024), within 15 %
   of the hand-written kernel; a 3×3 grid including the compile takes 98 ms
   against 0.5 s on the CPU. `gpu/test/coop.test.ts`
   checks iris training nets (values and the autodiff GRADIENT as a vector
   output, N = 120 → E = 8), MNIST loss / accuracy in 2D and 3D, the
   integration below, and times (`PERF=1`).

3. **Integration**: `buildSampleProgram(field, grid, resident?)` returns a
   cooperative `GpuProgram` (`cooperative: { workgroupSize, work, tile,
   floats }`) for a net field that is not `gpuTranspilable` but
   `coopCapable`; `programKernels` turns a program into its dispatches —
   one for an ordinary kernel, CHUNKS of points bounded by
   `COOP_CHUNK_WORK` (10¹⁰ MAC ≈ 30 ms) with their own `params` binding for
   a cooperative one, `Kernel.workgroups` dispatching one workgroup per
   point — and `GpuBackend.run`, `sampleResident(Sync)` and `gpuSampleOn`
   all go through it, so the viewer's fills (`F.grid`, `View3D` grids, the
   `Sampler`) are cooperative without knowing. A `ResidentProvider`
   supplies a program with the RESIDENT values of such a net on the
   dispatch grid: `ProgramBuilder.scalar` / `vector` ask it where they used
   to sample on the CPU (extra read-only bindings 2, 3, … carried in
   `GpuProgram.bindings`), so an expression over a cooperative net
   (`log10 loss`) reads the net's grid, and the net's GRADIENT — whose
   autodiff program has weight-shaped adjoints (145 k floats) and fits no
   workgroup — is taken as CENTRAL DIFFERENCES of the resident values over
   one grid spacing, one-sided at the box edges (`difference`; a
   `NetVectorFieldData` remembers the scalar it is the gradient of,
   `gradientOf`, so `∇` reaches it either way; within ~7 % of the exact
   gradient on an 8² grid, fine for streamlines and glyphs). The viewer
   wires the provider to its resident-grid caches
   (`apps/viewer/src/residentProvider.ts`: keys by data-object identity;
   `FusedGeometry.grids`, `View3D.netGrids`); the range statistics of a
   costly field sample 2 points per axis on the CPU before the GPU
   reduction (24² used to be 15 s); the cursor pane shows costly fields as
   `…` while the pointer moves and evaluates them 250 ms after it rests
   (25 ms–1 s per CPU evaluation). Verified in Chrome and Safari: the
   1024-example loss as colour field with isolines at 48² (fill 0.36 s in
   Chrome, 1.15 s in Safari — its WGSL compiler is ~3× slower on this
   kernel, to be looked at), descending streamlines and glyphs from the
   differenced gradient, `log10 loss` over the resident grid, 3D
   isosurfaces of fast/loss coloured by fast/accuracy at 16³; the ladder
   holds where the next fill would exceed `FILL_BUDGET_MS`. The MNIST
   bundle is in `bundles/index.json`. A REMEASURE (`AutoRes.onRemeasure`)
   now evicts the resident value grids too (`FusedGeometry.redo`,
   `View3D.redo`: `Cache.takeAll`, the old buffers destroyed after
   `whenIdle`) and clears the sampler: a frame drawing only the colour
   field used to find everything cached, never recompute, and leave the
   ladder at its first rung for any bundle (symbolic2d held at 32² with
   isolines off; it now reaches 2048², and fast/loss alone 64²).

Known limits: the exact cooperative gradient (below); Safari at 1.5× Chrome's
per-point cost on this kernel.

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

1. ~~Big nets on the GPU: the cooperative kernel~~ — built ("As built"
   above). Left: the Safari gap (1.5× per point after unrolling the tile
   loops; what remains looks like bounds-checked storage reads), a GPU point
   evaluation for the cursor pane (a one-point cooperative dispatch read
   back asynchronously), hoisting's constant folding as a GPU pre-pass for
   large datasets (0.9 s on the CPU for N = 1024).
2. **The exact cooperative gradient.** The MNIST gradient's adjoints are
   weight-shaped (`adjW2 = einsum("ni,nj->ij", h1, g)`, 145 k floats), so
   the viewer differences the resident values instead. Either fuse the
   chain `einsum("ij,kij->k", adjW, Dd)` ∘ `einsum("ni,nj->ij", h1, g)`
   into one 3-operand contraction (`ni,nj,kij->k`: never stores `adjW`,
   ~3× the forward cost of the layer — the cooperative emitter's
   contraction-parallel mapping takes a `[K]` output over a large
   contraction), or apply the distributivity rewrite to every displaced
   matmul, after which autodiff produces no weight-shaped node (at (1 + K)×
   the forward MACs).
3. Emitter: call-site batching (a loop over the extra axes around the inlined
   callee); fusing multi-use elementwise producers (recompute vs store);
   second derivatives of nets within Safari's limit (stream the dataset axis).

Open: randomness (dropout, data order) is deliberately absent — nets are pure.

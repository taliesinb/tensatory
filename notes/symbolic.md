# The symbolic expression language

`packages/core/src/symbolic/`. A small, closed language over scalar and vector
fields, chosen so the same tree can be evaluated in JS, differentiated
symbolically, and — later — transpiled to WGSL/GLSL or a Torch script.

## Syntax (`spec.ts`, types in `schema/symbolic.ts`)

Scalar-valued nodes:

* leaves: a number literal; `{op:"const", name}`; `{op:"coord", index}` (the
  coordinate function p ↦ p[index]); `{op:"arg", name}` (a scalar field
  argument); `{op:"argvi", name, index}` (a component of a vector argument);
  a bare string (a scalar argument or a constant).
* unary: `sin cos tan sinh cosh tanh asin acos atan asinh acosh atanh relu
  sigmoid gelu silu softplus elu erf floor ceil round sign abs exp exp2 exp10
  log log2 log10 log1p expm1 plogp sqrt square negate reciprocal gauss`
* n-ary (`vals: [...]`): `add mul min max mean rms`
* binary (`vals: [a, b]`): `sub div pow logBase atan2 mod`
* `clamp {val,min,max}`, `gaussKernel {val,mu,sigma}`, `normalPDF {val,mu,sigma}`
* from vectors: `dot`/`cosineSim {vecs}`, `norm {vec}`, `comp {vec,index}`

Vector-valued nodes: `constv {value}`, `basisv {index}`, `coordv` (p ↦ p),
`argv {name}` / bare string, `scalev {vec, by}`, `addv`/`meanv {vecs}`,
`subv {vecs:[a,b]}`, `sumv {vecs, coeffs}`, `compv {coeffs}`, `normalize {vec}`,
`grad {val}`.

Design rules:

* **Constants are leaves**, never parameter slots (`x + 2` is
  `add [x, 2]`). Only dimension indices are static parameters.
* Scalar and vector positions are distinct, so a bare name is unambiguous
  from its position; the three namespaces (`consts`, `scalars`, `vectors`) of
  a scope must not overlap (`checkNamespaces`).

## Pipeline

1. **zod** validates syntax (`SymbolicScalarSchema` / `SymbolicVectorSchema`).
2. **normalize** (`normalize.ts`): resolves names against a `NameEnv`
   (dimCount, consts, scalar/vector argument names), checks dimension indices,
   inlines named constants, and builds the internal AST (`ast.ts`) through
   smart constructors that fold constants and drop identities (`x·1`, `x+0`,
   `pow(x,2) → square`). The AST uses `k` for the node kind; the spec uses
   `op`.
3. **compile** (`compile.ts`): AST → closures `ScalarFn = (p, pos) => number`
   and `VectorFn = (p, pos, out) => out`. `pos` is the sample position when
   `p` is a point of the enclosing grid (lets sampled arguments skip
   interpolation), else −1. A `CompileContext` supplies argument evaluators
   and argument derivatives.
4. **diff** (`diff.ts`): `diffScalar(e, dim)` / `diffVector(v, dim)` /
   `gradient(e)`. Every op has an exact rule; piecewise ops (`relu`, `abs`,
   `min/max`, `clamp`, `elu`) use an internal `where` node (compare-and-select),
   non-differentiable ones (`floor`, `sign`, …) differentiate to 0.
   Derivatives of field arguments become `argd {name, dims}` / `argvid` nodes
   carrying the list of differentiated coordinates, so **derivatives through
   arguments compose to any order**; the compile context resolves them via
   `FieldData.derivative(dim)` (see [field-data.md](field-data.md)).

Tests compare every derivative rule against finite differences
(`test/symbolic.test.ts`).

## Common-subexpression elimination

Differentiation copies subtrees generously: for the 3-Gaussian mixture in the
example bundle, `|∇m|` is 31 nodes, its first derivative 2 616, a Hessian
entry 26 334 — almost entirely repetition. `compile.ts` therefore wraps each
compilation in a `Compilation` that

* hash-conses nodes by a structural key (`keyOf`, memoized per object),
  compiling each distinct subtree once;
* wraps subtrees referenced more than once in a **per-point memo**: the
  closure caches its last (coordinates, pos) and result and returns the cache
  on a repeat query. Hits are exact, so this is safe across nested argument
  evaluation (which passes the same point down) and gives no wrong answers for
  finite-difference probes (different coordinates → miss).

Gains measured: 3× on gradients of derived fields (see
[performance.md](performance.md)). Remaining headroom: make `diff` emit shared
references instead of copies, and compile value + gradient together.

## Symbolic arrays

`SymbolicArraySpec` reuses the language with the cell's integer grid position
as the coordinates (`coord` / `coordv`), relative to an optional `origin`, to
define sample arrays inline (checkerboards, ramps, synthetic data).

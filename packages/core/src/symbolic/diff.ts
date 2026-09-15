// Symbolic differentiation of normalized expressions with respect to a coordinate.
// Derivatives of field arguments become `argd` / `argvid` nodes, which the
// compiler evaluates by finite differences on the argument's field data.

import { C, ONE, ZERO, add, div, isConst, isConstVal, mul, neg, pow, sub, un, type SExpr, type VExpr } from "./ast";

const LN2 = Math.LN2;
const LN10 = Math.LN10;
const TWO_OVER_SQRT_PI = 2 / Math.sqrt(Math.PI);
const INV_SQRT_2PI = 1 / Math.sqrt(2 * Math.PI);

function where(test: SExpr, ref: SExpr, lt: SExpr, eq: SExpr, gt: SExpr): SExpr {
  if (isConst(test) && isConst(ref)) return test.value < ref.value ? lt : test.value > ref.value ? gt : eq;
  if (isConst(lt) && isConst(eq) && isConst(gt) && lt.value === eq.value && eq.value === gt.value) return lt;
  return { k: "where", test, ref, lt, eq, gt };
}

/** vector helpers */
const zerosV = (dimCount: number): VExpr => ({ k: "constv", value: new Array<number>(dimCount).fill(0) });
const dotV = (a: VExpr, b: VExpr): SExpr => ({ k: "dot", a, b });
const normV = (v: VExpr): SExpr => ({ k: "norm", v });
const scaleV = (v: VExpr, s: SExpr): VExpr => (isConstVal(s, 1) ? v : { k: "scalev", v, s });
const addV = (...args: VExpr[]): VExpr => (args.length === 1 ? args[0]! : { k: "naryv", op: "addv", args });
const subV = (a: VExpr, b: VExpr): VExpr => ({ k: "subv", a, b });

/** d(unary op)(x) / dx, expressed in terms of x */
function dUnary(op: SExpr & { k: "un" }, x: SExpr): SExpr {
  switch (op.op) {
    case "sin": return un("cos", x);
    case "cos": return neg(un("sin", x));
    case "tan": return add(ONE, un("square", un("tan", x)));
    case "sinh": return un("cosh", x);
    case "cosh": return un("sinh", x);
    case "tanh": return sub(ONE, un("square", un("tanh", x)));
    case "asin": return un("reciprocal", un("sqrt", sub(ONE, un("square", x))));
    case "acos": return neg(un("reciprocal", un("sqrt", sub(ONE, un("square", x)))));
    case "atan": return un("reciprocal", add(ONE, un("square", x)));
    case "asinh": return un("reciprocal", un("sqrt", add(un("square", x), ONE)));
    case "acosh": return un("reciprocal", un("sqrt", sub(un("square", x), ONE)));
    case "atanh": return un("reciprocal", sub(ONE, un("square", x)));
    case "relu": return where(x, ZERO, ZERO, ZERO, ONE);
    case "sigmoid": { const s = un("sigmoid", x); return mul(s, sub(ONE, s)); }
    case "gelu": {
      // Phi(x) + x phi(x)
      const Phi = mul(C(0.5), add(ONE, un("erf", mul(C(Math.SQRT1_2), x))));
      const phi = mul(C(INV_SQRT_2PI), un("gauss", x));
      return add(Phi, mul(x, phi));
    }
    case "silu": { const s = un("sigmoid", x); return add(s, mul(x, s, sub(ONE, s))); }
    case "softplus": return un("sigmoid", x);
    case "elu": return where(x, ZERO, un("exp", x), ONE, ONE);
    case "erf": return mul(C(TWO_OVER_SQRT_PI), un("gauss", mul(C(Math.SQRT2), x)));
    case "floor": case "ceil": case "round": case "sign": return ZERO;
    case "abs": return un("sign", x);
    case "exp": return un("exp", x);
    case "exp2": return mul(C(LN2), un("exp2", x));
    case "exp10": return mul(C(LN10), un("exp10", x));
    case "log": return un("reciprocal", x);
    case "log2": return div(C(1 / LN2), x);
    case "log10": return div(C(1 / LN10), x);
    case "log1p": return un("reciprocal", add(ONE, x));
    case "expm1": return un("exp", x);
    case "plogp": return add(un("log", x), ONE);
    case "sqrt": return div(C(0.5), un("sqrt", x));
    case "square": return mul(C(2), x);
    case "negate": return C(-1);
    case "reciprocal": return neg(un("reciprocal", un("square", x)));
    case "gauss": return neg(mul(x, un("gauss", x)));
  }
}

/** ∂e/∂x_dim */
export function diffScalar(e: SExpr, dim: number, dimCount: number): SExpr {
  const d = (x: SExpr) => diffScalar(x, dim, dimCount);
  const dv = (v: VExpr) => diffVector(v, dim, dimCount);
  switch (e.k) {
    case "const": return ZERO;
    case "coord": return e.index === dim ? ONE : ZERO;
    case "arg": return { k: "argd", name: e.name, dims: [dim] };
    case "argvi": return { k: "argvid", name: e.name, index: e.index, dims: [dim] };
    case "argd": return { k: "argd", name: e.name, dims: [...e.dims, dim] };
    case "argvid": return { k: "argvid", name: e.name, index: e.index, dims: [...e.dims, dim] };
    case "un": return mul(dUnary(e, e.a), d(e.a));
    case "nary": {
      const args = e.args;
      switch (e.op) {
        case "add": return add(...args.map(d));
        case "mul":
          return add(...args.map((a, i) => mul(d(a), ...args.filter((_, j) => j !== i))));
        case "mean": return mul(C(1 / args.length), add(...args.map(d)));
        case "rms": {
          // d sqrt(mean(a_i^2)) = mean(a_i a_i') / rms
          return div(mul(C(1 / args.length), add(...args.map((a) => mul(a, d(a))))), e);
        }
        case "min": case "max": {
          // fold: min(a, rest)' = a <= rest ? a' : rest'
          let acc: SExpr = args[args.length - 1]!;
          let dacc: SExpr = d(acc);
          for (let i = args.length - 2; i >= 0; i--) {
            const a = args[i]!;
            const da = d(a);
            dacc = e.op === "min" ? where(a, acc, da, da, dacc) : where(a, acc, dacc, da, da);
            acc = { k: "nary", op: e.op, args: [a, acc] };
          }
          return dacc;
        }
      }
      break;
    }
    case "bin": {
      const { a, b } = e;
      const da = d(a), db = d(b);
      switch (e.op) {
        case "sub": return sub(da, db);
        case "div": return div(sub(mul(da, b), mul(a, db)), un("square", b));
        case "pow": {
          // a^b (b a'/a + ln(a) b')
          if (isConst(b)) return mul(b, pow(a, C(b.value - 1)), da);
          const t1 = mul(b, div(da, a));
          const t2 = isConstVal(db, 0) ? ZERO : mul(un("log", a), db);
          return mul(e, add(t1, t2));
        }
        case "logBase": {
          // ln a / ln b
          const lnb = un("log", b);
          const t1 = div(da, mul(a, lnb));
          const t2 = isConstVal(db, 0) ? ZERO : div(mul(un("log", a), db), mul(b, un("square", lnb)));
          return sub(t1, t2);
        }
        case "atan2": return div(sub(mul(b, da), mul(a, db)), add(un("square", a), un("square", b)));
        case "mod": return sub(da, mul(un("floor", div(a, b)), db));
      }
      break;
    }
    case "clamp": {
      const da = d(e.a);
      return where(e.a, e.lo, d(e.lo), da, where(e.a, e.hi, da, da, d(e.hi)));
    }
    case "gaussKernel": {
      const z = div(sub(e.a, e.mu), e.sigma);
      const dz = d(z);
      return mul(neg(z), e, dz);
    }
    case "normalPDF": {
      const asProduct = mul({ k: "gaussKernel", a: e.a, mu: e.mu, sigma: e.sigma }, un("reciprocal", e.sigma), C(INV_SQRT_2PI));
      return d(asProduct);
    }
    case "dot": return add(dotV(dv(e.a), e.b), dotV(e.a, dv(e.b)));
    case "cosineSim": {
      const u = dotV(e.a, e.b);
      const n = mul(normV(e.a), normV(e.b));
      return div(sub(mul(d(u), n), mul(u, d(n))), un("square", n));
    }
    case "norm": return div(dotV(e.v, dv(e.v)), e);
    case "comp": return { k: "comp", v: dv(e.v), index: e.index };
    case "where": return where(e.test, e.ref, d(e.lt), d(e.eq), d(e.gt));
  }
  throw new Error(`unreachable: cannot differentiate ${(e as SExpr).k}`);
}

/** component-wise ∂v/∂x_dim */
export function diffVector(v: VExpr, dim: number, dimCount: number): VExpr {
  const d = (x: SExpr) => diffScalar(x, dim, dimCount);
  const dv = (x: VExpr) => diffVector(x, dim, dimCount);
  switch (v.k) {
    case "constv": case "basisv": return zerosV(dimCount);
    case "coordv": return { k: "basisv", index: dim };
    case "argv": return { k: "compv", comps: Array.from({ length: dimCount }, (_, i) => ({ k: "argvid", name: v.name, index: i, dims: [dim] }) as SExpr) };
    case "scalev": return addV(scaleV(dv(v.v), v.s), scaleV(v.v, d(v.s)));
    case "naryv": return { k: "naryv", op: v.op, args: v.args.map(dv) };
    case "subv": return subV(dv(v.a), dv(v.b));
    case "sumv": return addV({ k: "sumv", vecs: v.vecs.map(dv), coeffs: v.coeffs }, { k: "sumv", vecs: v.vecs, coeffs: v.coeffs.map(d) });
    case "compv": return { k: "compv", comps: v.comps.map(d) };
    case "normalize": {
      // v/|v|  ->  v'/|v| - v (v.v')/|v|^3
      const n = normV(v.v);
      const dvv = dv(v.v);
      return subV(scaleV(dvv, un("reciprocal", n)), scaleV(v.v, div(dotV(v.v, dvv), pow(n, C(3)))));
    }
    case "grad": return { k: "compv", comps: Array.from({ length: dimCount }, (_, i) => d(diffScalar(v.s, i, dimCount))) };
  }
}

/** the gradient of a scalar expression as a vector expression */
export function gradient(s: SExpr, dimCount: number): VExpr {
  return { k: "compv", comps: Array.from({ length: dimCount }, (_, i) => diffScalar(s, i, dimCount)) };
}

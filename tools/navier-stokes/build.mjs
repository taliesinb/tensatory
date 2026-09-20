// Builds apps/viewer/public/bundles/navier-stokes.json: the closed-form pieces of the OpenAI (Sept 2026)
// finite-time-blowup construction for 3D Navier–Stokes, as symbolic Tensatory fields at one fixed time.
//
//   node tools/navier-stokes/build.mjs
//
// Two spaces (see README.md next to this file for the derivation and the caveats):
//   exterior  the exact "heat exterior": a purely azimuthal Navier–Stokes solution whose angular momentum
//             Γ(r,τ) = c∞ (r²/2)^(-h) H(4τ/r²), H(Z) = Γ(1+h)⁻¹ ∫₀^∞ e^(-v) v^h (1+Zv)^(-h) dv, is computed with a
//             10-node generalized Gauss–Laguerre rule (weights already divided by Γ(1+h); |rel. error| < 7·10⁻⁴).
//   core      the leading-order axis profile (paper's Prop. B.2 comparison profile, Λ = 1, C = 1) of the
//             concentrating vortex: q(z,τ) from two Newton steps on q − z² q^(2h) = τ, similarity variables
//             η = z q^(-D), X = r²/(2q), Bessel swirl ϕ = ϕ*(η) f₀(Xχ(η)), axial U = U* − X Z*/(2L), radial
//             velocity from incompressibility (paper's (4.7)), pressure from the centrifugal balance.
// Everything is evaluated at the snapshot time τ = 1 − t given by the const `tau`.

import { writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

/* ---------- fixed construction parameters (baked into the polynomial coefficients below) ---------- */
const h = 0.01;           // anisotropy exponent, 0 < h ≤ 1/100 in the paper
const A = 0.5 + h, D = 0.5 - h;
const j0 = 0.05;          // axial asymmetry, U* = 4η + j0
const sigma = 0.15;       // regularization σ* in χ = H*² / (H*² + σ*²)
const TAU = 0.01;         // snapshot: τ = 1 − t
const CINF = 1;           // exterior amplitude c∞

/* ---------- expression builders (schema/symbolic.ts) ---------- */
const X0 = { op: "coord", index: 0 }, Y0 = { op: "coord", index: 1 }, Z0 = { op: "coord", index: 2 };
const add = (...vals) => ({ op: "add", vals });
const mul = (...vals) => ({ op: "mul", vals });
const sub = (a, b) => ({ op: "sub", vals: [a, b] });
const div = (a, b) => ({ op: "div", vals: [a, b] });
const pow = (a, b) => ({ op: "pow", vals: [a, b] });
const un = (op) => (val) => ({ op, val });
const sq = un("square"), sqrt = un("sqrt"), exp = un("exp"), log = un("log"), atan = un("atan"), neg = un("negate");
const compv = (...coeffs) => ({ op: "compv", coeffs });
const comp = (vec, index) => ({ op: "comp", vec, index });
const grad = (val) => ({ op: "grad", val });
const dot = (a, b) => ({ op: "dot", vecs: [a, b] });
const norm = (vec) => ({ op: "norm", vec });
/** Horner form of Σ c_k x^k */
const poly = (coeffs, x) => {
  let acc = coeffs[coeffs.length - 1];
  for (let k = coeffs.length - 2; k >= 0; k--) acc = add(coeffs[k], mul(x, acc));
  return acc;
};

/* ---------- shared geometry ---------- */
const r2 = add(sq(X0), sq(Y0));                 // r² (exact)
const r2eps = add(sq(X0), sq(Y0), 1e-12);        // r² kept away from 0 where a formula has an r^(-2h) factor

/* ================================================================================================ */
/* exterior: the heat exterior (paper eq. 4.29)                                                      */

// 10-node generalized Gauss–Laguerre rule for weight e^(-v) v^h, h = 1/100; weights divided by Γ(1+h)
const GL_NODES = [0.139500686701, 0.733264787569, 1.81410949723, 3.40903780430, 5.56184548829, 8.34118009540, 11.8564502919, 16.2935513774, 22.0125529738, 29.9385069975];
const GL_WEIGHTS = [0.305502827041, 0.401605266843, 0.219666252985, 0.0627936070203, 0.00963737080577, 0.000765462730046, 0.0000287773383148, 4.33356442054e-7, 1.87841952976e-9, 1.01331055170e-12];
/** H(Z) ≈ Σ w_i (1 + Z v_i)^(-h) */
const Hquad = (Z) => add(...GL_NODES.map((v, i) => mul(GL_WEIGHTS[i], pow(add(1, mul(Z, v)), -h))));

// Γ = r u_θ = c∞ (r²/2)^(-h) H(4τ/r²): finite at the axis (→ c∞ 2^h (4τ)^(-h) / Γ(1+h) as r → 0)
const extAngMom = mul("cinf", pow(mul(0.5, r2eps), -h), Hquad(div(mul(4, "tau"), r2eps)));

/* ================================================================================================ */
/* core: the leading-order concentrating profile                                                     */

// q(z, τ): q − z² q^(2h) = τ, Newton from q₀ = τ + z² (quadratic convergence; two steps are exact in f32 here)
const newton = (q) => {
  const z2q2h = mul(sq(Z0), pow(q, 2 * h));
  return sub(q, div(sub(sub(q, z2q2h), "tau"), sub(1, div(mul(2 * h, z2q2h), q))));
};
const qExpr = newton(newton(add("tau", sq(Z0))));

// Profiles in η (all with j0 = 1/20, h = 1/100, P = 1, i.e. Π₀ = −1/(1+η²)²):
//   U* = 4η + j0                                        (axis axial velocity, paper (B.1))
//   H* = Dη + (1−η²) U*                                 (cubic)
//   Z* = −A(1−2ηU*)U* − 4H* − (1−η²)Π₀' + 4AηΠ₀         = zNum(η) / (20000 (1+η²)³)
//   W  = Z*/L, L = 1 − 2hη²;  W' = wNum(η) / (10⁶ L² (1+η²)⁴)   (η-derivative, needed by the radial velocity)
// Coefficient lists (exact integers from a CAS; see README.md) scaled so magnitudes stay O(1–500).
const Z_NUM = [-4510, -520749, -1370, -514247, 22950, 739353, 31970, 1539251, 12160, 646400].map((c) => c / 20000);
const W_NUM = [-26037450, 1206980, 52529451, 4827920, 258844304, 7241880, 496980906, 4827920, 362364354, 1206980, 93974451, 0, -646400].map((c) => c / 1e6);
//   W'' = wppNum(η) / (200 (η²−50)³ (1+η²)⁵) = −wppNum(η) / (2.5·10⁷ L³ (1+η²)⁵)   (for the vorticity)
const WPP_NUM = [-30174500, -7781887650, -152682970, -17901317501, -310797350, -48925352755, -319849700, -48162221810, -168977200, -24259826760, -39226850, -4947607905, -1810470, -31920051].map((c) => -c / 2.5e7);
// ϕ*(η) = exp(∫₀^η ζ*), ζ* = −L H*/(H*² + σ*²): exact partial fractions over the three complex-pair roots of H*² + σ*²,
//   ∫ ζ* = Σ_k a_k log((η−p_k)² + q_k²) + b_k atan((η−p_k)/q_k)      rows {p, q, a, b}
const PHI_TERMS = [
  [-1.060575234, 0.01685271246, 0.0547980202, -0.005342907781],
  [-0.01111501712, 0.03337720485, -0.1110391817, -0.0002712077591],
  [1.059190252, 0.0165244924, 0.05374116153, 0.005071700022],
];
const G = (eta) => add(...PHI_TERMS.map(([p, q, a, b]) => add(mul(a, log(add(sq(sub(eta, p)), q * q))), mul(b, atan(div(sub(eta, p), q))))));
const G0 = PHI_TERMS.reduce((s, [p, q, a, b]) => s + a * Math.log(p * p + q * q) + b * Math.atan(-p / q), 0);
// f₀(z) = Σ (−z/2)^k / (k!(k+1)!) = 2 J₁(√(2z)) / √(2z), truncated at k = 8 (|error| < 6·10⁻⁸ on [0, 7])
const F0 = [1, -0.25, 0.020833333333333333, -0.00086805555555555556, 0.000021701388888888889, -3.6168981481481481e-7, 4.3058311287477954e-9, -3.844492079239103e-11, 2.669786166138266e-13];
// f₀² truncated at z^8, for the pressure integral ∫₀^X f₀(xχ)² dx = Σ c_k χ^k X^(k+1)/(k+1)
const F0SQ = [1, -0.5, 0.10416666666666667, -0.012152777777777778, 0.00091145833333333333, -0.000047743055555555556, 1.8472015542328042e-6, -5.4976236733119174e-8, 1.2980500339764249e-9];

/** everything the core fields need, given the scalar field `q` as an argument named "q" */
function coreParts(q, eta, X) {
  const d = sub(1, sq(eta));
  const L = sub(1, mul(2 * h, sq(eta)));
  const Ustar = add(mul(4, eta), j0);
  const Hstar = add(mul(D, eta), mul(d, Ustar));
  const s = add(1, sq(eta));                                        // 1 + η²
  const Pi0 = neg(div(1, sq(s)));
  const Zstar = div(poly(Z_NUM, eta), mul(s, s, s));
  const W = div(Zstar, L);
  const Wp = div(poly(W_NUM, eta), mul(sq(L), sq(sq(s))));
  const U = sub(Ustar, mul(0.5, X, W));                             // axial profile
  const AXU = sub(Ustar, mul(0.25, X, W));                          // radial average of U
  const dAXU = sub(4, mul(0.25, X, Wp));                            // its η-derivative
  const v0 = div(add(mul(2, eta, U), mul(-2 * D, eta, AXU), neg(mul(d, dAXU))), L); // V₀ / X, paper (4.7)
  const chi = div(sq(Hstar), add(sq(Hstar), sigma * sigma));
  const phistar = exp(sub(G(eta), G0));
  const phi = mul(phistar, poly(F0, mul(X, chi)));                  // azimuthal amplitude, E = √(2X) ϕ
  const qA = pow(q, -A);
  // Cartesian velocity (paper (4.5)): u = (v₀/2q)(x, y, 0) + q^(−A−½) ϕ (−y, x, 0) + q^(−A) U e_z
  const radial = div(v0, mul(2, q));
  const swirlRate = mul(pow(q, -A - 0.5), phi);                     // u_θ / r
  const u = compv(sub(mul(radial, X0), mul(swirlRate, Y0)), add(mul(radial, Y0), mul(swirlRate, X0)), mul(qA, U));
  // pressure p = q^(−2A) Π, Π = Π₀ + ϕ*² Σ c_k χ^k X^(k+1)/(k+1)
  const Cp = mul(sq(phistar), X, poly(F0SQ.map((c, k) => c / (k + 1)), mul(X, chi)));
  const p = mul(pow(q, -2 * A), add(Pi0, Cp));
  // Vorticity in closed form. z-derivatives go through the similarity map by the paper's Lemma 4.1:
  //   ∂_z (q^b f(X, η)) = q^(b−D) Z_b f,  Z_b f = L⁻¹ (2bη f + d f_η − 2ηX f_X),   and ∂_r X = r/q.
  //   ω_r = −∂_z u_θ,  ω_θ = ∂_z u_r − ∂_r u_z,  ω_z = (1/r) ∂_r (r u_θ)
  const Zb = (f, fEta, fX, b) => div(add(mul(2 * b, eta, f), mul(d, fEta), mul(-2, eta, X, fX)), L);
  const Hp = add(D + 4, mul(-2 * j0, eta), mul(-12, sq(eta)));                        // H*'
  const chiEta = div(mul(2 * sigma * sigma, Hstar, Hp), sq(add(sq(Hstar), sigma * sigma)));
  const zetaStar = neg(div(mul(L, Hstar), add(sq(Hstar), sigma * sigma)));              // ϕ*' = ζ* ϕ*
  const Xchi = mul(X, chi);
  const f0p = poly(F0.slice(1).map((c, k) => c * (k + 1)), Xchi);                         // f₀'
  const phiX = mul(phistar, chi, f0p);
  const phiEta = mul(phistar, add(mul(zetaStar, poly(F0, Xchi)), mul(f0p, X, chiEta)));
  const Wpp = div(poly(WPP_NUM, eta), mul(L, sq(L), mul(s, sq(sq(s)))));
  const N = add(mul(2, eta, U), mul(-2 * D, eta, AXU), neg(mul(d, dAXU)));               // v₀ = N / L
  const NEta = add(mul(2, U), mul(2, eta, sub(4, mul(0.5, X, Wp))), mul(-2 * D, AXU), mul(-2 * D, eta, sub(4, mul(0.25, X, Wp))), mul(2, eta, dAXU), mul(0.25, d, X, Wpp));
  const v0Eta = add(div(NEta, L), div(mul(4 * h, eta, N), sq(L)));
  const v0X = div(add(mul(-1, eta, W), mul(0.5 * D, eta, W), mul(0.25, d, Wp)), L);
  const omRoverR = neg(mul(pow(q, -A - 0.5 - D), Zb(phi, phiEta, phiX, -A - 0.5)));
  const omToverR = mul(0.5, add(mul(pow(q, -1 - D), Zb(v0, v0Eta, v0X, -1)), mul(pow(q, -A - 1), W)));
  const omZ = mul(pow(q, -A - 0.5), add(mul(2, phi), mul(2, X, phiX)));
  const omega = compv(sub(mul(omRoverR, X0), mul(omToverR, Y0)), add(mul(omRoverR, Y0), mul(omToverR, X0)), omZ);
  return { U, phi, qA, swirlRate, u, p, radial, omega, omZ };
}
const etaExpr = mul(Z0, pow("q", -D));
const XExpr = div(r2, mul(2, "q"));
const P = coreParts("q", "eta", "X");
const coreBox = [[-0.15, 0.15], [-0.15, 0.15], [-0.15, 0.15]];   // X ≤ 2.25 at the mid-plane corners: inside the axis profile's range
const extBox = [[-2, 2], [-2, 2], [-1, 1]];
const coreConsts = { tau: TAU };
// the core fields take q, η, X as scalar-field arguments (shared subtrees become shared field evaluations)
const coreArgs = { q: "core/q", eta: "core/eta", X: "core/X" };
const overQ = (expr) => ({ type: "pointwise", expr, scalars: coreArgs });
const overQv = (expr) => ({ type: "pointwisev", expr, scalars: coreArgs });

/* ================================================================================================ */

const bundle = {
  tensatory: "0.1",
  name: "Navier–Stokes blowup (closed-form pieces)",
  summary: "The two closed-form pieces of the 2026 finite-time-blowup construction for 3D Navier–Stokes at one snapshot τ: the exact heat-exterior vortex and the leading-order concentrating core.",
  details:
    "The explicitly computable parts of OpenAI's Sept 2026 finite-time-blowup construction for 3D Navier–Stokes (viscosity 1, singular time t = 1), " +
    `frozen at τ = 1 − t = ${TAU}. "exterior" is the exact heat-exterior vortex (an exact Navier–Stokes solution; u_θ ∝ r·U(1+h, 2, r²/4τ)/τ^(1+h), ` +
    "here via a 10-node Gauss–Laguerre rule); \"core\" is the leading-order axis profile of the concentrating core (Bessel swirl, linear axial profile, " +
    "incompressible radial inflow) with the proof's large parameters set to 1 — a cartoon of the geometry, not a solution. h = 1/100, so the " +
    "anisotropy ℓ_z/ℓ_r = τ^(-h) is invisible; the pulses that make the real construction work live at scales ~10^(-100) and are not represented. " +
    "Generated by tools/navier-stokes/build.mjs.",
  manifolds: {
    exterior: { name: "heat exterior (exact)", numDims: 3, dimNames: ["x", "y", "z"], summary: "the exact azimuthal heat-exterior vortex: angular momentum Γ(r, τ), swirl Γ/r, vorticity, velocity (circular streamlines)" },
    core: { name: "concentrating core (leading order)", numDims: 3, dimNames: ["x", "y", "z"], summary: "the leading-order self-similar core (Λ = C = 1): similarity variables q, η, X, velocity, swirl, pressure, vorticity — a cartoon of the geometry" },
  },
  fields: {
    /* ---------- exterior ---------- */
    "ext/angmom": {
      kind: "scalar", domain: "exterior", name: "r·u_θ (angular momentum)", codomain: "norm",
      details: "Γ(r,τ) = c∞ (r²/2)^(−h) H(4τ/r²), H(Z) = Γ(1+h)⁻¹∫₀^∞ e^(−v) v^h (1+Zv)^(−h) dv. Decreases outward (Rayleigh-unstable), which is what amplifies the paper's pulses. Nearly constant for h = 1/100: a point vortex with a τ^(−h) core.",
      data: { type: "symbolic", box: extBox, consts: { tau: TAU, cinf: CINF }, expr: extAngMom },
    },
    "ext/swirl": {
      kind: "scalar", domain: "exterior", name: "u_θ (swirl)", codomain: "log",
      details: "The azimuthal velocity Γ/r of the heat exterior; ∝ 1/r, an exact solution of the azimuthal heat equation ∂ₜu = u_rr + u_r/r − u/r². Finite at every r > 0 as τ → 0.",
      data: { type: "pointwise", expr: div("g", sqrt(r2eps)), scalars: { g: "ext/angmom" } },
    },
    "ext/vorticity": {
      kind: "scalar", domain: "exterior", name: "ω_z (vorticity)",
      details: "ω_z = (1/r) ∂_r(r u_θ) = ∇Γ·(x,y,0)/r², exact symbolic derivative of the angular momentum. Negative: angular momentum decreases outward.",
      data: { type: "pointwise", expr: div(dot(grad("g"), compv(X0, Y0, 0)), r2eps), scalars: { g: "ext/angmom" } },
    },
    "ext/u": {
      kind: "vector", domain: "exterior", name: "u (velocity)",
      details: "u = Γ (−y, x, 0)/r²: purely azimuthal, independent of z. Streamlines are circles.",
      data: { type: "pointwisev", expr: compv(div(mul(neg("g"), Y0), r2eps), div(mul("g", X0), r2eps), 0), scalars: { g: "ext/angmom" } },
    },
    /* ---------- core ---------- */
    "core/q": {
      kind: "scalar", domain: "core", name: "q (concentration scale)", codomain: "log",
      details: "q(z,τ) solving q − z² q^(2h) = τ (two Newton steps from τ + z²); q ≍ τ + |z|^(1/D). The core's radial scale is √q, its axial scale q^D, its velocity scale q^(−A), A = ½ + h.",
      data: { type: "symbolic", box: coreBox, consts: coreConsts, expr: qExpr },
    },
    "core/eta": {
      kind: "scalar", domain: "core", name: "η (similarity height)", codomain: { min: -1, max: 1 },
      details: "η = z q^(−D) ∈ (−1, 1); η = ±1 is the singular time away from the origin.",
      data: { type: "pointwise", expr: etaExpr, scalars: { q: "core/q" } },
    },
    "core/X": {
      kind: "scalar", domain: "core", name: "X (similarity radius²)", codomain: "norm",
      details: "X = r²/(2q). The leading axis profile is valid for X ≲ 4 (Λ = 1); the paper's pulse annulus would sit at larger X.",
      data: { type: "pointwise", expr: XExpr, scalars: { q: "core/q" } },
    },
    "core/u": {
      kind: "vector", domain: "core", name: "u⁽⁰⁾ (leading velocity)",
      details: "u = (v₀/2q)(x,y,0) + q^(−A−½) ϕ (−y,x,0) + q^(−A) U e_z: inward spiral near z ≈ 0 with axial outflow above and below; exactly divergence-free. Velocity scale τ^(−½−h).",
      data: overQv(P.u),
    },
    "core/speed": {
      kind: "scalar", domain: "core", name: "|u⁽⁰⁾|", codomain: "norm",
      details: "Speed of the leading core flow; isosurfaces show the shrinking column (radius ∝ √τ).",
      data: { type: "pointwise", expr: norm("u"), vectors: { u: "core/u" } },
    },
    "core/swirl": {
      kind: "scalar", domain: "core", name: "u_θ (swirl)",
      details: "u_θ = q^(−A) E, E = √(2X) ϕ*(η) f₀(X χ(η)), f₀(z) = 2J₁(√(2z))/√(2z): a Bessel (Kelvin-like) vortex core whose amplitude ϕ* peaks at the dividing layer η₀ ≈ −0.011.",
      data: overQ(mul(P.swirlRate, sqrt(r2))),
    },
    "core/uz": {
      kind: "scalar", domain: "core", name: "u_z (axial velocity)",
      details: "u_z = q^(−A) U, U = U*(η) − X Z*(η)/(2L), U* = 4η + j₀ (j₀ = 0.05 breaks the z ↦ −z symmetry on purpose). Reverses sign at larger X: the paper's radial-outflow region.",
      data: overQ(mul(P.qA, P.U)),
    },
    "core/ur": {
      kind: "scalar", domain: "core", name: "u_r (radial velocity)",
      details: "u_r = r v₀/(2q) from incompressibility, paper (4.7): negative (inflow) near the mid-plane.",
      data: overQ(mul(P.radial, sqrt(r2))),
    },
    "core/angmom": {
      kind: "scalar", domain: "core", name: "r·u_θ (angular momentum)", codomain: "norm",
      details: "r u_θ = q^(−h) H(X, η), H = 2X ϕ: transported inward by the radial inflow, the spin-up mechanism.",
      data: overQ(mul(P.swirlRate, r2)),
    },
    "core/pressure": {
      kind: "scalar", domain: "core", name: "p⁽⁰⁾ (pressure)",
      details: "p = q^(−2A) Π, Π_X = E²/(2X) integrated from the axis datum Π₀(η) = −1/(1+η²)². Low pressure in the core supplies the centripetal force.",
      data: overQ(P.p),
    },
    "core/vorticity": {
      kind: "vector", domain: "core", name: "ω = curl u⁽⁰⁾",
      details: "Vorticity in closed form (z-derivatives through the similarity map by the paper's Lemma 4.1; checked against the exact symbolic curl). Put it in a scalar slot for |ω| isosurfaces, the vortex-tube picture. The generic {op:\"grad\"} curl of the velocity tree is exact too but its own gradient (isosurface normals) is a second derivative of a 40 kB expression: minutes of shader generation.",
      data: overQv(P.omega),
    },
    "core/vortz": {
      kind: "scalar", domain: "core", name: "ω_z (axial vorticity)",
      details: "ω_z = (1/r) ∂_r(r u_θ) = q^(−A−½) ϕ* [2f₀(Xχ) + 2Xχ f₀'(Xχ)]: the Bessel core's vorticity, positive on the axis, changing sign where the swirl profile turns over.",
      data: overQ(P.omZ),
    },
  },
  pointSets: {
    singularityExt: { domain: "exterior", points: [[0, 0, 0]], labels: ["singular point (t = 1)"] },
    singularityCore: { domain: "core", points: [[0, 0, 0]], labels: ["singular point (t = 1)"] },
  },
};

const out = join(dirname(fileURLToPath(import.meta.url)), "../../apps/viewer/public/bundles/navier-stokes.json");
writeFileSync(out, JSON.stringify(bundle) + "\n");
console.log(`wrote ${out}`);

// Builds apps/viewer/public/bundles/dynamical-systems.json: a "textbook" of continuous dynamical systems
// ẋ = F(x), one space per system, each with the time evolution F as a symbolic vector field.
//
//   node tools/dynamical-systems/build.mjs
//
// Every space has
//   F        the vector field of the system (symbolicv; the manifold's `flow`, so the viewer's streamlines and
//            glyphs follow it by default — turn on the S / V columns of the mappings matrix),
//   |F|      the speed, whose zeros are the equilibria (log codomain: it vanishes at critical points),
// and, where the system has them, a first integral H (its isolines / isosurfaces are the orbits of the
// conservative version), a potential V (F = −∇V; streamlines cut the isolines of V at right angles), the
// energy dissipation rate Ḣ = ∇H·F (a Lyapunov derivative: ≤ 0 everywhere), the divergence ∇·F (Bendixson /
// volume contraction), all as pointwise expressions of F and H so they are exact symbolic derivatives.
// Equilibria are labelled point sets (stable node / saddle / …); limit cycles, strange attractors and
// separatrices are ORDERED point sets integrated here by RK4 from the very same expression trees (a tiny
// evaluator below interprets the subset of the symbolic language that the systems use).
// Parameters (damping, σ ρ β, μ …) are named `consts` of the F field, so they can be edited in the JSON.

import { writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

/* ---------- expression builders (schema/symbolic.ts) ---------- */
const X = { op: "coord", index: 0 }, Y = { op: "coord", index: 1 }, Z = { op: "coord", index: 2 };
const add = (...vals) => ({ op: "add", vals });
const mul = (...vals) => ({ op: "mul", vals });
const sub = (a, b) => ({ op: "sub", vals: [a, b] });
const div = (a, b) => ({ op: "div", vals: [a, b] });
const un = (op) => (val) => ({ op, val });
const sq = un("square"), sin = un("sin"), cos = un("cos"), log = un("log"), neg = un("negate");
const compv = (...coeffs) => ({ op: "compv", coeffs });
const comp = (vec, index) => ({ op: "comp", vec, index });
const grad = (val) => ({ op: "grad", val });
const dot = (a, b) => ({ op: "dot", vecs: [a, b] });
const norm = (vec) => ({ op: "norm", vec });
const argvi = (name, index) => ({ op: "argvi", name, index });
/** ∂f/∂x_i as an exact symbolic derivative */
const d = (f, i) => comp(grad(f), i);
/** ∇·F of a vector-field argument named `F` in D dimensions */
const divergence = (D) => add(...Array.from({ length: D }, (_, i) => d(argvi("F", i), i)));
/** the linear field A·x */
const linear = (A) => compv(...A.map((row) => add(...row.map((a, j) => mul(a, { op: "coord", index: j })))));
const r2 = add(sq(X), sq(Y));
const PI = Math.PI;

/* ---------- a tiny compiler of the subset used here (to a JS function), for integrating trajectories ---------- */
function jsS(e, c) {
  if (typeof e === "number") return `(${e})`;
  if (typeof e === "string") { if (!(e in c)) throw new Error(`unknown const ${e}`); return `(${c[e]})`; }
  const u = (f) => `${f}(${jsS(e.val, c)})`;
  switch (e.op) {
    case "coord": return `p[${e.index}]`;
    case "const": return `(${c[e.name]})`;
    case "add": return `(${e.vals.map((v) => jsS(v, c)).join(" + ")})`;
    case "mul": return `(${e.vals.map((v) => jsS(v, c)).join(" * ")})`;
    case "sub": return `(${jsS(e.vals[0], c)} - ${jsS(e.vals[1], c)})`;
    case "div": return `(${jsS(e.vals[0], c)} / ${jsS(e.vals[1], c)})`;
    case "square": return `((t) => t * t)${`(${jsS(e.val, c)})`}`;
    case "sin": return u("Math.sin");
    case "cos": return u("Math.cos");
    case "log": return u("Math.log");
    case "negate": return `(-${jsS(e.val, c)})`;
    default: throw new Error(`compiler: unsupported scalar op ${e.op}`);
  }
}
function fieldFn(data) {
  const e = data.expr, c = data.consts ?? {};
  if (e.op !== "compv") throw new Error(`compiler: unsupported vector op ${e.op}`);
  return new Function("p", `return [${e.coeffs.map((s) => jsS(s, c)).join(", ")}];`);
}

/* ---------- integration ---------- */
function rk4(F, p, h) {
  const k1 = F(p);
  const k2 = F(p.map((x, i) => x + 0.5 * h * k1[i]));
  const k3 = F(p.map((x, i) => x + 0.5 * h * k2[i]));
  const k4 = F(p.map((x, i) => x + h * k3[i]));
  return p.map((x, i) => x + (h / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]));
}
const round = (p) => p.map((x) => +x.toPrecision(5));
const inBox = (p, box) => p.every((x, i) => x >= box[i][0] && x <= box[i][1]);
/** integrate for `transient`, then record every `every`-th step for time T */
function orbit(F, p0, { dt, transient, T, every }) {
  let p = p0;
  for (let t = 0; t < transient; t += dt) p = rk4(F, p, dt);
  const pts = [];
  const n = Math.round(T / dt);
  for (let i = 0; i < n; i++) { if (i % every === 0) pts.push(round(p)); p = rk4(F, p, dt); }
  return pts;
}
/** one period of a planar limit cycle: after the transient, from one upward crossing of y = 0 to the next */
function limitCycle(F, p0, { dt, transient, every }) {
  let p = p0;
  for (let t = 0; t < transient; t += dt) p = rk4(F, p, dt);
  const crossing = (a, b) => a[1] < 0 && b[1] >= 0;
  let q = rk4(F, p, dt);
  while (!crossing(p, q)) { p = q; q = rk4(F, q, dt); }
  const pts = [round(q)];
  p = q; q = rk4(F, q, dt);
  let i = 0;
  while (!crossing(p, q)) { if (++i % every === 0) pts.push(round(q)); p = q; q = rk4(F, q, dt); }
  pts.push(pts[0]);
  return pts;
}
/** the stable manifold of a planar saddle: both branches integrated BACKWARDS from the saddle along its stable eigenvector,
 *  each until it leaves the box or comes to rest at another equilibrium (in backward time: a source) */
function stableManifold(F, saddle, vStable, box, { dt, every, eps = 1e-4, maxSteps = 100000 }) {
  const branch = (sign) => {
    const pts = [];
    let p = saddle.map((x, i) => x + sign * eps * vStable[i]);
    for (let i = 0; i < maxSteps && inBox(p, box) && Math.hypot(...F(p)) > 1e-6; i++) { if (i % every === 0) pts.push(round(p)); p = rk4(F, p, -dt); }
    return pts;
  };
  return [...branch(-1).reverse(), round(saddle), ...branch(+1)];
}

/* ---------- field builders ---------- */
const bundle = { tensatory: "0.1", manifolds: {}, fields: {}, pointSets: {} };
const speedName = "|F|";
/** a one-line summary from the details when none is given: the details of every field here open with the formula,
 *  set off by ": " or the first sentence end */
function withSummary(spec) {
  if (spec.summary !== undefined || !spec.details) return spec;
  const m = /^(.*?)(?:: |\. (?=[A-Z∀-⋿Α-ω]))/s.exec(spec.details);
  const clause = (m ? m[1] : spec.details).replace(/\.$/, "");
  return clause.length <= 110 ? { ...spec, summary: clause } : spec;
}
/**
 * Add one dynamical system as its own space. The vector field F is registered first (as the manifold's flow), then
 * the speed |F|, then `fields` ([id, spec] pairs) in display order. The FIRST scalar field of a space is the viewer's
 * default colour / isoline field: `lead` names a field of `fields` (an energy, a potential) to put before the speed.
 */
function system(id, { name, dims, box, F, fields = [], points = {}, lead }) {
  bundle.manifolds[id] = { name, numDims: dims.length, dimNames: dims, flow: `${id}/F` };
  const fid = (k) => `${id}/${k}`;
  bundle.fields[fid("F")] = withSummary({ kind: "vector", domain: id, name: "F", summary: F.summary, details: F.details, data: { type: "symbolicv", box, ...(F.consts ? { consts: F.consts } : {}), expr: F.expr } });
  const speed = ["speed", { kind: "scalar", name: speedName, codomain: "log", summary: "the speed of the flow; zero exactly at the equilibria", details: "|F|: the speed of the flow (log codomain). Zero exactly at the equilibria, so its low isolines / isosurfaces ring them.", data: { type: "pointwise", expr: norm("F"), vectors: { F: "F" } } }];
  const leadIdx = fields.findIndex(([k]) => k === lead);
  if (lead !== undefined && leadIdx < 0) throw new Error(`${id}: lead field ${lead} is not among its fields`);
  const ordered = leadIdx < 0 ? [speed, ...fields] : [fields[leadIdx], speed, ...fields.filter((_, i) => i !== leadIdx)];
  for (const [k, spec] of ordered) {
    const data = { ...spec.data };
    if (data.type === "pointwise" || data.type === "pointwisev") {
      if (data.vectors) data.vectors = Object.fromEntries(Object.entries(data.vectors).map(([n, ref]) => [n, fid(ref)]));
      if (data.scalars) data.scalars = Object.fromEntries(Object.entries(data.scalars).map(([n, ref]) => [n, fid(ref)]));
    }
    if (data.type === "symbolic" || data.type === "symbolicv") data.box ??= box;
    const out = withSummary({ ...spec, domain: id, data });
    if (out.exactGradient) out.exactGradient = fid(out.exactGradient);
    bundle.fields[fid(k)] = out;
  }
  for (const [k, ps] of Object.entries(points)) bundle.pointSets[fid(k)] = { domain: id, ...ps };
  return fieldFn(bundle.fields[fid("F")].data);
}
/** the one-line summary of a space (shown by the viewer's space picker and its ⓘ) */
const note = (id, text) => { bundle.manifolds[id].summary = text; };

/* common derived fields (speed is always added by `system`) */
const divField = (D, details) => ["div", { kind: "scalar", name: "∇·F", details: details ?? "Divergence of the flow: negative where phase volume contracts, positive where it expands.", data: { type: "pointwise", expr: divergence(D), vectors: { F: "F" } } }];
const HdotField = (H, details) => ["Hdot", { kind: "scalar", name: "∇H·F", codomain: { max: 0 }, details: details ?? "Ḣ = ∇H·F, the rate of change of the energy along the flow (exact symbolic derivative): ≤ 0, so H is a Lyapunov function.", data: { type: "pointwise", expr: dot(grad("H"), "F"), scalars: { H }, vectors: { F: "F" } } }];
const gradVField = (V) => ["gradV", { kind: "vector", name: "∇V", summary: "the exact gradient of the potential; F = −∇V", data: { type: "pointwisev", expr: grad("V"), scalars: { V } } }];

/* ================================================================================================ */
/* 2D                                                                                                */

/* ---- linear systems: the trace–determinant zoo on one plane ---- */
{
  const id = "linear", box = [[-2, 2], [-2, 2]];
  bundle.manifolds[id] = { name: "linear systems ẋ = A x", numDims: 2, dimNames: ["x", "y"], flow: `${id}/saddle/F` };
  const systems = [
    ["saddle", "saddle", [[0, 1], [1, 0]], "ẋ = y, ẏ = x: eigenvalues ±1 along the diagonals (1, 1) (unstable) and (1, −1) (stable). x² − y² is conserved: the orbits are hyperbolas, the diagonals the separatrices.", ["inv", "x² − y²", sub(sq(X), sq(Y)), "The saddle's first integral: hyperbolic orbits, the asymptotes are the stable and unstable manifolds."]],
    ["node-", "stable node", [[-2, 1], [1, -2]], "A = [[−2, 1], [1, −2]]: eigenvalues −1 (slow, along (1, 1)) and −3 (fast, along (1, −1)). Every orbit approaches the origin tangent to the slow eigendirection.", ["lyap", "|x|²/2", mul(0.5, r2), "A quadratic Lyapunov function: A is symmetric negative definite, so |x|² decreases along every orbit."]],
    ["node+", "unstable node", [[2, -1], [-1, 2]], "The stable node with time reversed: eigenvalues +1 and +3; every orbit leaves the origin (a source)."],
    ["spiral-", "stable spiral", [[-0.3, -1], [1, -0.3]], "ẋ = −0.3x − y, ẏ = x − 0.3y: eigenvalues −0.3 ± i; orbits spiral counter-clockwise into the origin (a stable focus).", ["lyap", "|x|²/2", mul(0.5, r2), "d|x|²/dt = −0.6 |x|² < 0: a Lyapunov function."]],
    ["spiral+", "unstable spiral", [[0.3, -1], [1, 0.3]], "Eigenvalues 0.3 ± i: orbits spiral outward (an unstable focus, a repellor)."],
    ["center", "center", [[0, -1], [1, 0]], "ẋ = −y, ẏ = x: eigenvalues ±i; the harmonic oscillator. Every orbit is a circle, H = |x|²/2 is conserved. Not structurally stable: any damping turns it into a spiral.", ["H", "H = |x|²/2", mul(0.5, r2), "The conserved energy; its isolines ARE the orbits."]],
    ["star", "star node", [[-1, 0], [0, -1]], "A = −I: a double eigenvalue −1 with two independent eigenvectors; orbits are straight rays into the origin."],
    ["improper", "improper node", [[-1, 1], [0, -1]], "A Jordan block, double eigenvalue −1 with a single eigenvector (1, 0): every orbit arrives tangent to the x axis, sheared around."],
    ["line", "equilibrium line", [[-1, 0], [0, 0]], "ẋ = −x, ẏ = 0: det A = 0; every point of the y axis is an equilibrium (non-isolated), reached along horizontal lines."],
  ];
  for (const [k, title, A, details, extra] of systems) {
    const F = linear(A);
    bundle.fields[`${id}/${k}/F`] = withSummary({ kind: "vector", domain: id, name: `${title}/F`, details, data: { type: "symbolicv", box, expr: F } });
    bundle.fields[`${id}/${k}/speed`] = { kind: "scalar", domain: id, name: `${title}/${speedName}`, codomain: "log", summary: `speed of the ${title}`, data: { type: "pointwise", expr: norm("F"), vectors: { F: `${id}/${k}/F` } } };
    if (extra) { const [ek, ename, expr, edesc] = extra; bundle.fields[`${id}/${k}/${ek}`] = withSummary({ kind: "scalar", domain: id, name: `${title}/${ename}`, details: edesc, data: { type: "symbolic", box, expr } }); }
  }
  bundle.pointSets[`${id}/origin`] = { domain: id, points: [[0, 0]], labels: ["equilibrium"] };
  note(id, "the canonical 2×2 linear systems classified by trace and determinant, sharing one plane (pick a system's F for streamlines / glyphs, its |F| or invariant for colour): saddle, stable / unstable node, stable / unstable spiral, center, star, improper node, a line of equilibria");
}

/* ---- gradient system: F = −∇V, V a double-double well ---- */
{
  const V = add(mul(0.25, sq(sub(sq(X), 1))), mul(0.25, sq(sub(sq(Y), 1))));
  const F = compv(sub(X, mul(X, sq(X))), sub(Y, mul(Y, sq(Y))));
  const pts = [], labels = [];
  for (const x of [-1, 0, 1]) for (const y of [-1, 0, 1]) {
    pts.push([x, y]);
    const idx = (x === 0) + (y === 0);
    labels.push(idx === 0 ? "min · stable node" : idx === 1 ? "saddle" : "max · unstable node");
  }
  system("gradient", {
    name: "gradient flow ẋ = −∇V", dims: ["x", "y"], box: [[-1.8, 1.8], [-1.8, 1.8]], lead: "V",
    F: { expr: F, details: "F = −∇V = (x − x³, y − y³): the steepest-descent flow of the double-double well V. Streamlines cross the isolines of V at right angles." },
    fields: [
      ["V", { kind: "scalar", name: "V", codomain: "norm", exactGradient: "gradV", details: "V = ¼(x² − 1)² + ¼(y² − 1)²: four minima (±1, ±1), four saddles (±1, 0), (0, ±1), one maximum (0, 0). A Morse function; the flow's equilibria are its critical points, their index the number of unstable directions.", data: { type: "symbolic", expr: V } }],
      gradVField("V"),
      divField(2, "∇·F = −ΔV = 2 − 3x² − 3y²: the flow expands area near the maximum and contracts it near the minima."),
    ],
    points: { equilibria: { points: pts, labels } },
  });
  note("gradient", "steepest descent on a double-double well: 4 stable nodes, 4 saddles, 1 source, the potential's isolines orthogonal to the flow");
}

/* ---- damped pendulum ---- */
{
  const box = [[-7, 7], [-3.5, 3.5]];
  const H = add(mul(0.5, sq(Y)), sub(1, cos(X)));
  const F = { consts: { gamma: 0.25 }, expr: compv(Y, sub(neg(sin(X)), mul("gamma", Y))), details: "ẋ = y, ẏ = −sin x − γ y (γ = 0.25): angle x, angular velocity y. Stable spirals at x = 2kπ, saddles at x = (2k+1)π; the damping makes the separatrices of the frictionless pendulum spiral into the wells." };
  const Ffn = system("pendulum", {
    name: "damped pendulum", dims: ["θ", "θ̇"], box, F, lead: "H",
    fields: [
      ["H", { kind: "scalar", name: "H", codomain: "norm", details: "H = y²/2 + 1 − cos x, the energy of the frictionless pendulum. Its isolines are the undamped orbits: libration below H = 2, the separatrix at H = 2 (the homoclinic orbits of the saddles), rotation above.", data: { type: "symbolic", expr: H } }],
      ["F0", { kind: "vector", name: "F₀ (undamped)", details: "The frictionless pendulum ẋ = y, ẏ = −sin x: a Hamiltonian flow, H is conserved, the equilibria at x = 2kπ are centers.", data: { type: "symbolicv", expr: compv(Y, neg(sin(X))) } }],
      HdotField("H", "Ḣ = ∇H·F = −γ y²: the energy dissipated by the damping, zero only on the x axis (LaSalle: orbits still converge to the wells)."),
    ],
    points: {
      equilibria: { points: [[-2 * PI, 0], [-PI, 0], [0, 0], [PI, 0], [2 * PI, 0]].map(round), labels: ["stable spiral", "saddle", "stable spiral", "saddle", "stable spiral"] },
    },
  });
  // a pendulum rotating over the top (from θ ≈ −2π) losing energy until it is captured by the well at θ = 0
  bundle.pointSets["pendulum/orbit"] = { domain: "pendulum", ordered: true, name: "a damped orbit", points: orbit(Ffn, [-6.5, 2.9], { dt: 0.01, transient: 0, T: 45, every: 5 }) };
  note("pendulum", "θ̈ = −sin θ − γ θ̇: the energy H (isolines = frictionless orbits, separatrix at H = 2), its dissipation Ḣ ≤ 0, a captured rotating orbit");
}

/* ---- Duffing double well ---- */
{
  const box = [[-2, 2], [-1.6, 1.6]];
  const V = sub(mul(0.25, sq(sq(X))), mul(0.5, sq(X)));
  const H = add(mul(0.5, sq(Y)), V);
  const F = { consts: { delta: 0.25 }, expr: compv(Y, sub(sub(X, mul(X, sq(X))), mul("delta", Y))), details: "ẋ = y, ẏ = x − x³ − δ y (δ = 0.25): a particle in the double well V = x⁴/4 − x²/2 with friction. Stable spirals at (±1, 0), a saddle at the origin whose stable manifold separates the two basins." };
  system("duffing", {
    name: "Duffing double well", dims: ["x", "ẋ"], box, F, lead: "H",
    fields: [
      ["H", { kind: "scalar", name: "H", details: "H = y²/2 + x⁴/4 − x²/2: the energy. The figure-eight isoline H = 0 is the separatrix of the undamped system (two homoclinic loops of the saddle).", data: { type: "symbolic", expr: H } }],
      ["V", { kind: "scalar", name: "V(x)", details: "The double-well potential x⁴/4 − x²/2 (independent of y).", data: { type: "symbolic", expr: V } }],
      ["F0", { kind: "vector", name: "F₀ (undamped)", details: "ẋ = y, ẏ = x − x³: Hamiltonian, H conserved; centers at (±1, 0), the saddle's homoclinic figure-eight at H = 0.", data: { type: "symbolicv", expr: compv(Y, sub(X, mul(X, sq(X)))) } }],
      HdotField("H", "Ḣ = −δ y²: ≤ 0, the friction's dissipation."),
    ],
    points: { equilibria: { points: [[-1, 0], [0, 0], [1, 0]], labels: ["stable spiral", "saddle", "stable spiral"] } },
  });
  note("duffing", "ẍ = x − x³ − δ ẋ: two attracting wells and the saddle between them; the undamped separatrix is the figure-eight H = 0");
}

/* ---- Van der Pol ---- */
{
  const box = [[-4, 4], [-4, 4]];
  const F = { consts: { mu: 1 }, expr: compv(Y, sub(mul("mu", sub(1, sq(X)), Y), X)), details: "ẋ = y, ẏ = μ(1 − x²) y − x (μ = 1): negative damping for |x| < 1, positive outside. The origin is an unstable spiral; every other orbit winds onto the unique limit cycle (Liénard's theorem)." };
  const Ffn = system("vanderpol", {
    name: "Van der Pol oscillator", dims: ["x", "ẋ"], box, F,
    fields: [divField(2, "∇·F = μ(1 − x²): changes sign at |x| = 1 — Bendixson's criterion allows the limit cycle only because the divergence is not of one sign.")],
    points: { equilibria: { points: [[0, 0]], labels: ["unstable spiral"] } },
  });
  bundle.pointSets["vanderpol/cycle"] = { domain: "vanderpol", ordered: true, name: "limit cycle", points: limitCycle(Ffn, [2, 0], { dt: 0.005, transient: 60, every: 4 }) };
  note("vanderpol", "a relaxation oscillator: an unstable spiral inside a globally attracting limit cycle (drawn, integrated to convergence)");
}

/* ---- Hopf normal form ---- */
{
  const box = [[-1.8, 1.8], [-1.8, 1.8]];
  const F = { consts: { mu: 1, omega: 3 }, expr: compv(sub(sub(mul("mu", X), mul("omega", Y)), mul(X, r2)), sub(add(mul("omega", X), mul("mu", Y)), mul(Y, r2))), details: "ṙ = r(μ − r²), θ̇ = ω in Cartesian form (μ = 1, ω = 3): the supercritical Hopf normal form past the bifurcation. The origin is an unstable spiral; the circle r = √μ is an attracting limit cycle. At μ ≤ 0 the origin would be a stable spiral and the cycle gone." };
  system("hopf", {
    name: "Hopf normal form", dims: ["x", "y"], box, F,
    fields: [
      ["r2dot", { kind: "scalar", name: "∇r²·F", details: "∇(r²)·F = 2r²(μ − r²): positive inside the cycle, negative outside, zero on it — the radial dynamics that trap every orbit onto r = √μ.", data: { type: "pointwise", expr: dot(grad(r2), "F"), vectors: { F: "F" } } }],
      divField(2, "∇·F = 2μ − 4r²."),
    ],
    points: { equilibria: { points: [[0, 0]], labels: ["unstable spiral"] } },
  });
  bundle.pointSets["hopf/cycle"] = { domain: "hopf", ordered: true, name: "limit cycle r = √μ", points: Array.from({ length: 97 }, (_, i) => (i % 96 === 0 ? [1, 0] : round([Math.cos((2 * PI * i) / 96), Math.sin((2 * PI * i) / 96)]))) };
  note("hopf", "the supercritical Hopf normal form after the bifurcation: repelling origin, circular limit cycle of radius √μ, the radial rate d(r²)/dt as the trapping mechanism");
}

/* ---- Lotka–Volterra predator–prey ---- */
{
  const box = [[0.02, 3.5], [0.02, 3.5]];
  const F = { consts: { alpha: 1, beta: 1, gamma: 1, delta: 1 }, expr: compv(mul(X, sub("alpha", mul("beta", Y))), mul(Y, sub(mul("delta", X), "gamma"))), details: "ẋ = x(α − βy), ẏ = y(δx − γ) (all 1): prey x, predators y. The coexistence point (γ/δ, α/β) = (1, 1) is a center — every orbit is a closed cycle, the amplitude set by the initial condition; the origin is a saddle (axes invariant)." };
  system("lotka", {
    name: "Lotka–Volterra predator–prey", dims: ["prey", "predators"], box, F, lead: "C",
    fields: [
      ["C", { kind: "scalar", name: "C (first integral)", codomain: { min: 2 }, details: "C = δx − γ ln x + βy − α ln y is conserved (minimum 2 at (1, 1)): its isolines are the population cycles.", data: { type: "symbolic", expr: add(sub(X, log(X)), sub(Y, log(Y))) } }],
    ],
    points: { equilibria: { points: [[1, 1], [0, 0]], labels: ["center", "saddle"] } },
  });
  note("lotka", "predator–prey: a nonlinear center at coexistence, a saddle at extinction, the conserved C whose isolines are the cycles");
}

/* ---- competition (Strogatz's rabbits vs sheep) ---- */
{
  const box = [[0, 3.5], [0, 2.5]];
  const F = { expr: compv(mul(X, sub(sub(3, X), mul(2, Y))), mul(Y, sub(sub(2, X), Y))), details: "ẋ = x(3 − x − 2y), ẏ = y(2 − x − y): two species competing for the same resource (Strogatz §6.4). Bistable: (3, 0) and (0, 2) are stable nodes, (1, 1) a saddle whose stable manifold (drawn) separates the basins — one species always excludes the other; the origin is an unstable node." };
  const Ffn = system("competition", {
    name: "competing species (rabbits vs sheep)", dims: ["rabbits", "sheep"], box, F,
    fields: [divField(2)],
    points: { equilibria: { points: [[0, 0], [3, 0], [0, 2], [1, 1]], labels: ["unstable node", "stable node", "stable node", "saddle"] } },
  });
  // Jacobian at (1, 1): [[−1, −2], [−1, −1]], eigenvalues −1 ± √2; stable eigenvector ∝ (√2, 1)
  const s = Math.hypot(Math.SQRT2, 1);
  bundle.pointSets["competition/separatrix"] = { domain: "competition", ordered: true, name: "separatrix (stable manifold of the saddle)", points: stableManifold(Ffn, [1, 1], [Math.SQRT2 / s, 1 / s], box, { dt: 0.005, every: 10 }) };
  note("competition", "bistable competition: two stable nodes, a saddle, the separatrix between the basins (the saddle's stable manifold, integrated backwards)");
}

/* ================================================================================================ */
/* 3D                                                                                                */

/* ---- saddle-focus ---- */
{
  const box = [[-2, 2], [-2, 2], [-2, 2]];
  const F = { expr: linear([[-0.2, -1, 0], [1, -0.2, 0], [0, 0, 0.4]]), details: "ẋ = −0.2x − y, ẏ = x − 0.2y, ż = 0.4z: eigenvalues −0.2 ± i and +0.4. Orbits spiral in towards the z axis while being pushed away from the xy plane: a saddle-focus, the local picture at the Lorenz and Rössler equilibria (with the roles of the spiral and the axis swapped for Rössler)." };
  system("saddlefocus", {
    name: "saddle-focus (3D linear)", dims: ["x", "y", "z"], box, F, lead: "inv",
    fields: [["inv", { kind: "scalar", name: "z²·r⁴ (invariant)", codomain: "norm", details: "d ln(r²)/dt = −0.4 and d ln(z²)/dt = 0.8, so z²(x² + y²)² is conserved: every orbit stays on one of its level surfaces (hourglasses pinched at the plane and the axis).", data: { type: "symbolic", expr: mul(sq(Z), sq(r2)) } }]],
    points: { equilibria: { points: [[0, 0, 0]], labels: ["saddle-focus"] } },
  });
  note("saddlefocus", "the 3D linear saddle-focus: a stable spiral plane and an unstable axis");
}

/* ---- 3D gradient system ---- */
{
  const box = [[-1.7, 1.7], [-1.7, 1.7], [-1.7, 1.7]];
  const V = add(mul(0.25, sq(sub(sq(X), 1))), mul(0.25, sq(sub(sq(Y), 1))), mul(0.25, sq(sub(sq(Z), 1))));
  const F = compv(sub(X, mul(X, sq(X))), sub(Y, mul(Y, sq(Y))), sub(Z, mul(Z, sq(Z))));
  const pts = [], labels = [];
  for (const x of [-1, 0, 1]) for (const y of [-1, 0, 1]) for (const z of [-1, 0, 1]) {
    pts.push([x, y, z]);
    const idx = (x === 0) + (y === 0) + (z === 0);
    labels.push(["min", "saddle (index 1)", "saddle (index 2)", "max"][idx]);
  }
  system("gradient3", {
    name: "gradient flow ẋ = −∇V (3D)", dims: ["x", "y", "z"], box, lead: "V",
    F: { expr: F, details: "F = −∇V, V = ¼Σ(xᵢ² − 1)²: steepest descent in a volume. 27 equilibria: 8 minima (sinks), 12 index-1 saddles, 6 index-2 saddles, 1 maximum (source); the flow is normal to the isosurfaces of V." },
    fields: [
      ["V", { kind: "scalar", name: "V", codomain: "norm", exactGradient: "gradV", summary: "V = ¼Σ(xᵢ² − 1)², the triple double well", details: "The triple double well; isosurfaces at low levels are 8 blobs around the minima, merging through the saddles as the level rises.", data: { type: "symbolic", expr: V } }],
      gradVField("V"),
      divField(3, "∇·F = −ΔV = 3 − 3(x² + y² + z²)."),
    ],
    points: { equilibria: { points: pts, labels } },
  });
  note("gradient3", "steepest descent on a triple double well: 8 sinks, 18 saddles of index 1 and 2, 1 source, isosurfaces of V normal to the flow");
}

/* ---- Lorenz ---- */
{
  const box = [[-25, 25], [-30, 30], [0, 50]];
  const sigma = 10, rho = 28, beta = 8 / 3;
  const F = { consts: { sigma, rho, beta }, expr: compv(mul("sigma", sub(Y, X)), sub(mul(X, sub("rho", Z)), Y), sub(mul(X, Y), mul("beta", Z))), details: "ẋ = σ(y − x), ẏ = x(ρ − z) − y, ż = xy − βz with σ = 10, ρ = 28, β = 8/3. The origin is a saddle (1 unstable, 2 stable directions); C± = (±√(β(ρ−1)), ±√(β(ρ−1)), ρ − 1) are saddle-foci (a 2D unstable spiral, 1 stable direction). ∇·F = −(σ + 1 + β) = −13.67: phase volume shrinks, so the attractor (drawn) has zero volume." };
  const c = Math.sqrt(beta * (rho - 1));
  const Vtrap = add(mul(rho, sq(X)), mul(sigma, sq(Y)), mul(sigma, sq(sub(Z, 2 * rho))));
  const Ffn = system("lorenz", {
    name: "Lorenz system", dims: ["x", "y", "z"], box, F,
    fields: [
      ["trap", { kind: "scalar", name: "V (trapping)", codomain: "norm", details: "V = ρx² + σy² + σ(z − 2ρ)²: a Lyapunov-like function whose derivative is negative outside a bounded ellipsoid (see V̇), so every orbit enters and stays in a trapping region. Isosurfaces are ellipsoids centred on (0, 0, 2ρ).", data: { type: "symbolic", expr: Vtrap } }],
      ["trapdot", { kind: "scalar", name: "∇V·F", details: "V̇ = ∇V·F = −2σ[ρx² + y² + β(z − ρ)² − βρ²] (exact symbolic derivative): positive only inside the small ellipsoid ρx² + y² + β(z − ρ)² = βρ² — the isosurface V̇ = 0 — where the attractor lives.", data: { type: "pointwise", expr: dot(grad("V"), "F"), scalars: { V: "trap" }, vectors: { F: "F" } } }],
    ],
    points: { equilibria: { points: [[0, 0, 0], round([c, c, rho - 1]), round([-c, -c, rho - 1])], labels: ["saddle", "C₊ saddle-focus", "C₋ saddle-focus"] } },
  });
  bundle.pointSets["lorenz/attractor"] = { domain: "lorenz", ordered: true, name: "strange attractor", points: orbit(Ffn, [1, 1, 1], { dt: 0.005, transient: 20, T: 30, every: 2 }) };
  note("lorenz", "σ = 10, ρ = 28, β = 8/3: three saddle-type equilibria, a strange attractor (30 time units drawn), a trapping ellipsoid and its Lyapunov derivative");
}

/* ---- Rössler ---- */
{
  const box = [[-12, 12], [-12, 12], [0, 25]];
  const a = 0.2, b = 0.2, c = 5.7;
  const F = { consts: { a, b, c }, expr: compv(sub(neg(Y), Z), add(X, mul("a", Y)), add("b", mul(Z, sub(X, "c")))), details: "ẋ = −y − z, ẏ = x + ay, ż = b + z(x − c), a = b = 0.2, c = 5.7. Near the plane z ≈ 0 the flow is the unstable spiral ẋ = −y, ẏ = x + ay; the ż equation folds the spiral's outer part up and back over (a Möbius-band-like attractor, drawn). Two equilibria: P₋ near the origin is a saddle-focus (spiral out in the plane, attracting along one direction), P₊ far outside the box." };
  const disc = Math.sqrt(c * c - 4 * a * b);
  const P = (x) => round([x, -x / a, x / a]);
  const Ffn = system("rossler", {
    name: "Rössler system", dims: ["x", "y", "z"], box, F,
    fields: [divField(3, "∇·F = a + x − c: negative over the attractor except where x > c − a = 5.5, the fold region where the flow expands.")],
    points: { equilibria: { points: [P((c - disc) / 2), P((c + disc) / 2)], labels: ["P₋ saddle-focus", "P₊ saddle-focus"] } },
  });
  bundle.pointSets["rossler/attractor"] = { domain: "rossler", ordered: true, name: "strange attractor", points: orbit(Ffn, [1, 1, 0], { dt: 0.02, transient: 200, T: 150, every: 3 }) };
  note("rossler", "a = b = 0.2, c = 5.7: the spiral-and-fold strange attractor (150 time units drawn), the divergence marking the fold");
}

/* ================================================================================================ */

bundle.name = "dynamical systems (textbook)";
bundle.summary = "Textbook continuous dynamical systems ẋ = F(x), one space each: linear zoo, gradient flows, pendulum, Duffing, Van der Pol, Hopf, Lotka–Volterra, competition, saddle-focus, Lorenz, Rössler.";
bundle.details =
  "Continuous dynamical systems ẋ = F(x), one space per system, the time evolution F as a symbolic vector field (the space's `flow`, so streamlines and glyphs follow it forward in time; turn on the S and V columns of the fields panel).\n\n" +
  "In every space: |F| is the speed (zero at the equilibria); H / V / C are energies, potentials and first integrals whose level sets are orbits or are cut orthogonally by them; ∇H·F, ∇V·F, ∇r²·F are exact symbolic Lyapunov derivatives; ∇·F the divergence. " +
  "Equilibria are labelled by type; limit cycles, strange attractors and separatrices are ordered point sets integrated by RK4 from the same expression trees. Parameters (γ, δ, μ, σ ρ β, …) are `consts` of each F and can be edited in the JSON.\n\n" +
  "Each space's ⓘ says what it shows. Generated by tools/dynamical-systems/build.mjs.";

const out = join(dirname(fileURLToPath(import.meta.url)), "../../apps/viewer/public/bundles/dynamical-systems.json");
writeFileSync(out, JSON.stringify(bundle) + "\n");
const nPts = Object.values(bundle.pointSets).reduce((s, p) => s + p.points.length, 0);
console.log(`wrote ${out}: ${Object.keys(bundle.manifolds).length} spaces, ${Object.keys(bundle.fields).length} fields, ${Object.keys(bundle.pointSets).length} point sets (${nPts} points)`);

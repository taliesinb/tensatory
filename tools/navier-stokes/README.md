# Navier–Stokes blowup bundle

`node tools/navier-stokes/build.mjs` writes `apps/viewer/public/bundles/navier-stokes.json`
(compact JSON: the trees are deep, indentation would multiply the size by 7).

Source: OpenAI, *Finite time blowup for Navier–Stokes* (8 Sept 2026,
[paper](https://cdn.openai.com/pdf/32d9f210-8b73-45e0-91bc-82a30aef8a9a/navier-stokes.pdf),
[Lean](https://github.com/openai/NavierStokesAndEuler)): for every ν > 0 a smooth compactly supported
force drives a fluid at rest to unbounded velocity at t = 1 with bounded energy (Clay alternatives C, D).
Equation numbers below refer to that paper (viscosity 1, τ = 1 − t, h ≤ 1/100, A = ½ + h, D = ½ − h).

## What is in the bundle, and what is not

The proof's solution is a sum of layers — a self-similar axisymmetric core, corrections to every order in
q^{2h}, dyadic families of WKB pulses with carrier frequency ⌈Q^{−h/2}⌉ whose averaged Reynolds stress
supplies the force the core cannot balance, mean corrections, cutoffs. Only two pieces are closed-form,
and the bundle holds exactly those, at one snapshot τ (const `tau`, default 0.01):

* **exterior** — the *heat exterior* (4.29), X ≥ X_b: purely azimuthal, z-independent, an exact
  Navier–Stokes solution. Angular momentum

  Γ(r, τ) = r u_θ = c_∞ (r²/2)^{−h} H(4τ/r²),  H(Z) = Γ(1+h)^{−1} ∫₀^∞ e^{−v} v^h (1+Zv)^{−h} dv
  = Z^{−1−h} U(1+h, 2, 1/Z),

  so u_θ = c_∞ 2^{½+h} (4τ)^{−1−h} r U(1+h, 2, r²/4τ) (Tricomi U; verified to satisfy
  ∂ₜu = u_rr + u_r/r − u/r² to round-off). Tensatory has no U, so H is a 10-node generalized
  Gauss–Laguerre sum Σ wᵢ (1 + Z vᵢ)^{−h} (weight e^{−v} v^h; relative error < 7·10⁻⁴ for
  Z ∈ [10⁻³, 10⁶]). Fields: Γ, u_θ = Γ/r, ω_z = ∇Γ·(x,y,0)/r² (exact symbolic derivative), u.
  Γ decreases outward (Rayleigh-unstable) — the pulses' energy source.

* **core** — the leading-order axis profile of Prop. B.2 with the proof's "sufficiently large" parameters
  set to Λ = 1, C = 1, P* = 1 (Π₀ = −1/(1+η²)²), j₀ = 0.05, σ* = 0.15. Similarity variables (3.2):
  q − z² q^{2h} = τ (two Newton steps from τ + z²), η = z q^{−D}, X = r²/(2q). Profiles:

  * U* = 4η + j₀, H* = Dη + (1−η²)U*, χ = H*²/(H*² + σ*²) (B.1–B.2);
  * Z* = −A(1−2ηU*)U* − 4H* − (1−η²)Π₀′ + 4AηΠ₀ (B.1), as a rational function of η;
  * ϕ* = exp ∫₀^η ζ*, ζ* = −L H*/(H*² + σ*²) (B.3): the integral is done exactly by partial fractions
    over the three complex-pair roots of H*² + σ*² (three log + three atan terms);
  * ϕ = ϕ*(η) f₀(Xχ), f₀(z) = Σ (−z/2)^k/(k!(k+1)!) = 2J₁(√(2z))/√(2z) (B.11), series to k = 8;
  * U = U* − X Z*/(2L) (B.13), V₀ = X v₀ from incompressibility (4.7) — needs ∂_η(Z*/L), also a
    rational function; Π = Π₀ + ∫₀^X E²/(2x) dx with E = √(2X) ϕ, the f₀² series integrated termwise.

  Cartesian velocity (4.5): u = (v₀/2q)(x, y, 0) + q^{−A−½} ϕ (−y, x, 0) + q^{−A} U e_z, smooth
  across the axis; checked divergence-free to finite-difference noise and against the CAS reference at
  four points. Vorticity in closed form: z-derivatives go through the similarity map by Lemma 4.1,
  ∂_z(q^b f) = q^{b−D} Z_b f with Z_b f = L⁻¹(2bη f + d f_η − 2ηX f_X), so
  ω_r = −∂_z u_θ, ω_θ = ∂_z u_r − ∂_r u_z, ω_z = (1/r)∂_r(r u_θ) need ϕ_η (ϕ*′ = ζ*ϕ*, χ′, f₀′),
  v₀_η (hence W″, one more rational function) and v₀_X; checked against core's exact symbolic curl of
  `core/u` to 10⁻¹⁰. Fields: q, η, X, u, |u|, u_θ, u_z, u_r, r u_θ, p, ω, ω_z (|ω| = ω in a scalar slot).
  Box ±0.15 keeps X ≤ 2.25 at the mid-plane corners, inside the profile's range Y = ΛX ≤ 4.1.

  Why not `{op: "grad"}` for the curl: it is exact and cheap to *evaluate* (~0.1 ms per point on the
  CPU), but the 3D isosurface kernel also needs the iso field's gradient (normals, ∇-projection), and
  the gradient of |curl u| is a second derivative of the 40 kB velocity tree — with `diff` copying
  subtrees that is 25 s of WGSL generation and a 1.9 MB shader, i.e. a frozen page. The closed form
  keeps |ω| at 1.2 s / 200 kB (still the heaviest field of the bundle). Sharing subtrees in `diff`
  (roadmap item 12) is the real fix.

Not represented, on purpose: the pulses and all corrections (they are asymptotic in ε = Q^h, i.e. valid
at Q ~ 10⁻¹⁰⁰ for h = 1/100), the joining annulus between core and exterior, the force. With h = 1/100
the anisotropy ℓ_z/ℓ_r = τ^{−h} is invisible at any plottable τ; the exterior is visually a point vortex
whose core circulation creeps up like τ^{−h}. The core is a cartoon of the geometry (inward spiral at
the dividing layer, axial outflow, Bessel swirl, low-pressure core), not a solution.

## Regenerating with other constants

`h`, `j0`, `sigma` and P* are baked into the coefficient lists `Z_NUM`, `W_NUM`, `PHI_TERMS`,
`GL_NODES` / `GL_WEIGHTS`; change them with a CAS (the Mathematica session that produced them is
recorded in the header comment of build.mjs) and re-run. `tau` and `cinf` are ordinary bundle consts and can be edited
in the JSON. A time slider would need a non-folded parameter slot in the symbolic language
(`consts` are folded to literals at parse time, so today a changed `tau` means a shader recompile).

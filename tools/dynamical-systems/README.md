# Dynamical systems bundle

`node tools/dynamical-systems/build.mjs` writes `apps/viewer/public/bundles/dynamical-systems.json`
(compact JSON; ~200 kB, most of it the integrated attractors).

A "textbook" of continuous dynamical systems ẋ = F(x), one space per system. Each space declares its
vector field as the manifold's `flow`, so the viewer's streamlines and glyphs follow F by default (turn
on the S and V columns of the mappings matrix) and run forward in time (`dir` starts *ascending*).

## What every space has

| field | meaning |
|---|---|
| `F` | the flow, `symbolicv`; parameters (γ, δ, μ, σ ρ β, …) are named `consts`, editable in the JSON |
| `\|F\|` | the speed (log codomain): zero exactly at the equilibria |
| `H` / `V` / `C` | energy, potential or first integral where the system has one — isolines / isosurfaces are the conservative orbits (H, C), or are cut orthogonally by the flow (V, with `exactGradient` = `∇V`) |
| `∇H·F`, `∇V·F`, `∇r²·F` | Lyapunov derivatives, exact symbolic derivatives through the field arguments (`pointwise` over `H` and `F`) |
| `∇·F` | divergence as `Σ ∂Fᵢ/∂xᵢ` of the vector argument (Bendixson, volume contraction) |
| point sets | equilibria labelled by type; ordered sets for limit cycles, strange attractors, separatrices |

The ordered point sets are integrated here by RK4 **from the same expression trees** the bundle carries
(`jsS` compiles the subset of the symbolic language the systems use to a JS function): a limit cycle is
one period after a transient (detected by y = 0 crossings), an attractor is a stretch of orbit after a
transient, a separatrix is the saddle's stable manifold integrated backwards along its stable
eigenvector until it leaves the box or comes to rest at a source.

## Spaces

2D — `linear` (nine canonical ẋ = Ax systems on one plane: saddle, stable / unstable node, stable /
unstable spiral, center, star, improper node, a line of equilibria; pick one system's `F`),
`gradient` (F = −∇V for the double-double well: 4 sinks, 4 saddles, 1 source), `pendulum` (damped;
H, F₀ undamped, ∇H·F = −γy², a captured rotating orbit), `duffing` (double well; figure-eight
separatrix H = 0), `vanderpol` (limit cycle, ∇·F = μ(1 − x²)), `hopf` (normal form past the
bifurcation; cycle r = √μ, ∇r²·F), `lotka` (predator–prey center with first integral C),
`competition` (Strogatz's rabbits vs sheep: bistable, separatrix drawn).

3D — `saddlefocus` (linear; first integral z²r⁴), `gradient3` (triple double well: 27 equilibria by
Morse index), `lorenz` (σ = 10, ρ = 28, β = 8/3; trapping function and its derivative), `rossler`
(a = b = 0.2, c = 5.7; ∇·F marks the fold).

## Adding a system

Call `system(id, { name, dims, box, F: { expr, consts?, details }, fields, points, lead? })`:
`fields` are `[key, spec]` pairs with references to sibling fields by key (`system` prefixes them); a field without a `summary` gets one from the opening clause of its `details` (`withSummary`);
`lead` names the scalar to put first (the viewer's default colour / isoline field), else `|F|` leads.
`divField(D)`, `HdotField(H)`, `gradVField(V)` build the common derived fields. Field NAMES are tree
paths in the viewer (`/` separates levels), so keep slashes out of names unless a subtree is intended.

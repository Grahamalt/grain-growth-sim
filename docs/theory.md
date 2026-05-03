# Theory

This document collects every equation used in the project, in the
notation the code uses. Implementation pointers are given inline.

---

## 1. Potts model

The microstructure is a 2D L×L lattice. Each site (i, j) carries an
integer orientation `s(i, j) in {1, ..., Q}`. We use Q = 48, L = 64
(L = 96 for the showcase), 4-nearest-neighbor connectivity with
periodic boundary conditions everywhere.

### Site energy

$$E_{site}(i, j) = J \sum_{(i', j') \in \mathcal{N}(i, j)} \bigl[1 - \delta_{s(i,j),\, s(i', j')}\bigr]$$

Zero when all neighbors share the site's orientation; J per unlike
neighbor. We use J = 1 throughout.

### Total Hamiltonian (each bond counted once)

$$H = \tfrac{1}{2}\sum_{(i, j)} E_{site}(i, j) = J \sum_{\langle (i,j), (i', j')\rangle} \bigl[1 - \delta_{s(i,j),\, s(i', j')}\bigr]$$

### Boundary site

A site (i, j) is a *grain-boundary site* iff at least one neighbor has
a different orientation. The binary boundary mask `b(i, j) = 1` for
boundary sites, 0 otherwise. Implementation: `code/potts.py`
(`site_energy`, `total_energy`, `boundary_mask`).

---

## 2. Monte Carlo dynamics (Metropolis)

### Proposal rule

At a randomly chosen site (i, j), draw a proposed new orientation
`s'` uniformly from the set of *distinct neighbor orientations*. If
no neighbor differs, the proposal is a no-op.

### Acceptance rule

Let `dE = E_site(s') - E_site(s)` be the local energy change.

$$P_\text{accept} = \begin{cases} 1 & \Delta E \le 0 \\ \exp(-\Delta E / k_B T) & \Delta E > 0 \end{cases}$$

We use `kT = 0.5` throughout. One Monte Carlo sweep (MCS) is N = L^2
random site visits. Implementation: `code/mc_step.py`.

---

## 3. Solute field

The solute concentration `C(i, j) in [0, 1]` is a continuous scalar
on the same lattice as the orientations (the planning notes' "Option
A").

### Diffusion (explicit FD Laplacian, periodic BCs)

$$C^{n+1}_{i,j} = C^n_{i,j} + D_{sol}\,\Delta t \cdot \nabla^2 C^n_{i,j}$$

with the 5-point stencil
`Lap C[i,j] = C[i+1,j] + C[i-1,j] + C[i,j+1] + C[i,j-1] - 4 C[i,j]`.

CFL stability in 2D requires `D_sol * dt <= 1/4`; the code rejects
larger steps. Defaults: `D_sol = 0.1`, `dt = 1`. The total mass
`M = sum C(i, j)` is preserved exactly because the periodic Laplacian
sums to zero.

### Equilibrium segregation isotherm

A solute atom at a boundary site has energy `E_seg < 0` relative to a
bulk site. At thermal equilibrium between a boundary and a bulk site
the populations satisfy a Boltzmann ratio:

$$\frac{C_{GB}}{C_{bulk}} = \exp\!\left(-\frac{E_{seg}}{k_B T}\right).$$

### Mass-conserving relaxation step

Define a per-site Boltzmann weight

$$w(i, j) = \begin{cases} \exp(-E_{seg}/k_B T) & b(i, j) = 1 \\ 1 & b(i, j) = 0 \end{cases}$$

The equilibrium concentration field, normalized so total mass equals
the current total `M = sum C`, is

$$C_{eq}(i, j) = \frac{M}{\sum_{i', j'} w(i', j')}\, w(i, j).$$

A relaxation step with rate `r in [0, 1]` is

$$C^{n+1} = C^n + r \cdot \bigl(C_{eq} - C^n\bigr),$$

which preserves total mass exactly and reduces to a direct jump to
equilibrium for `r = 1`. Implementation: `code/solute.py`.

---

## 4. Coupling: solute drag in the Metropolis step

The "simple coupling" used here multiplies the standard Metropolis
acceptance probability by a local drag factor:

$$P_\text{accept}^\text{coupled}(i, j) = P_\text{accept}^\text{Potts}(i, j) \cdot \max\!\bigl(0,\ \min(1,\ 1 - C(i, j))\bigr).$$

When `C(i, j) = 0` the rule reduces to the standard Metropolis
dynamics (Sec. 2); when `C(i, j) = 1` the site cannot flip at all.
This produces a qualitative Cahn-style drag-vs-velocity curve (Sec. 7)
but lacks the explicit velocity dependence of the full Cahn integral.
Implementation: `code/mc_step.py` (`coupled_metropolis_step`,
`coupled_mc_step`).

### One coupled MCS, in order

1. One coupled MC sweep: `N = L^2` site visits using
   `coupled_metropolis_step`.
2. One diffusion step with parameters `(D_sol, dt)`.
3. One segregation relaxation toward `C_eq` with rate `r`.

Implementation: `code/simulation.py` (`run_coupled_simulation`).

---

## 5. Grain identification and statistics

### Connected components with periodic BCs

We flood-fill (`scipy.ndimage.label`, 4-connectivity) the same-
orientation regions per orientation, then merge clusters that touch
across the periodic top/bottom and left/right seams via union-find.
The result is a label array `L(i, j)` with each grain assigned a
unique ID `1, ..., K`. Implementation: `code/analysis.py`
(`identify_grains`).

### Grain size and effective diameter

The size `A_g` of grain `g` is the number of sites with label `g`.
Its effective diameter (assuming a circular footprint) is

$$D_g = 2\sqrt{A_g / \pi}.$$

The mean grain diameter for the lattice is

$$\langle D \rangle = \frac{1}{K}\sum_{g=1}^{K} D_g.$$

Implementation: `code/analysis.py` (`grain_size_distribution`,
`mean_grain_diameter`).

---

## 6. Validation: the parabolic growth law

In the curvature-driven regime, the mean grain diameter grows as

$$\langle D \rangle^2 = \langle D_0 \rangle^2 + k\, t,$$

with `t` measured in MCS. The slope `k` is a model-internal kinetic
constant. For our 64x64, Q = 48, kT = 0.5 runs (linear-regime fit
skipping the first 20% of snapshots): **k ~ 1.10, R^2 ~ 0.96**
(figure: `results/figures/exp1_d2_vs_t.png`).

---

## 7. Solute drag curve (Cahn-style)

We extract a drag estimate at matched mean diameter (matched curvature
driving force):

$$F_i(D) \equiv v_\text{pure}(D) - v_\text{coupled}(D),$$

where each velocity is a finite-difference

$$v(D) \approx \frac{\langle D\rangle_{n+1} - \langle D\rangle_n}{t_{n+1} - t_n}$$

evaluated at the secant midpoint of bracketing snapshots, and
`v_pure(D)` is linearly interpolated onto the mean-diameter samples
from the coupled run. Plotting `F_i` vs `v_coupled` across a sweep of
`C_bulk` values produces the Cahn-style non-monotonic curve.
Implementation: `code/drag.py`.

---

## 8. Optical scattering proxy (illustrative)

Strictly a 2D, geometric-regime proxy - not a Mie or photon-transport
calculation. See `docs/final_report.md` Sec. 9 and Sec. 3.4 for the
full framing.

### Boundary map

`b(i, j) = 1` if any neighbor of `(i, j)` has a different orientation,
else 0.

### Scattering intensity field

Periodic Gaussian convolution of the boundary map, representing the
diffraction-limited halo around each point scatterer:

$$I(x) = (G_\sigma \star b)(x),\qquad G_\sigma(x) \propto \exp(-|x|^2 / 2\sigma^2).$$

We use `sigma = 2.5` lattice units in the showcase. The convolution
is performed with `mode = 'wrap'` for periodic BCs; total intensity
`sum I(x) = sum b(x)` is preserved.

### Effective attenuation coefficient

$$\mu_\text{eff} = (\delta n)^2 \cdot f_\text{boundary},\qquad f_\text{boundary} = \frac{1}{L^2}\sum_{i,j} b(i, j).$$

For a polycrystal in the geometric regime, `f_boundary` scales as
`1/<D>`, so `mu_eff ~ 1/<D>`. Verified in
`docs/testing_and_validation.md` Sec. 4.2: log-log slope of `mu_eff`
vs `<D>` is `-0.78` (expected `-1`); the product `mu_eff * <D>` is
constant to within ~15% across the dopant sweep. The `(delta n)^2`
prefactor is a placeholder; absolute magnitudes are arbitrary.

### Beer-Lambert transmission (illustrative)

$$T = \exp(-\mu_\text{eff} \cdot L_\text{fiber}).$$

Implementation: `code/optical_proxy.py`.

---

## 9. Symbol summary

| symbol | meaning | default value |
|---|---|---|
| L | lattice side | 64 (96 for showcase) |
| Q | number of orientations | 48 |
| J | Potts coupling constant | 1 |
| kT | Monte Carlo temperature | 0.5 |
| C(i, j) | local solute concentration | initialized to `C_bulk` |
| C_bulk | initial bulk solute concentration | swept |
| E_seg | solute segregation energy at boundary | swept |
| D_sol | solute diffusion coefficient | 0.1 |
| dt | diffusion time step | 1 |
| r | segregation relaxation rate | 1.0 |
| sigma | Gaussian scattering halo width | 2.5 lattice units |
| delta_n | refractive-index mismatch (placeholder) | 0.01 |

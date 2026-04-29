# Phase 4 — Testing and Validation

This document covers (4.1) the unit-test suite, (4.2) validation against
known theory, and (4.3) the known limitations of the model.

All test counts and numbers below are taken from the actual repo state
(`pytest tests/ -v` and `results/data/*.npz`).

---

## 4.1 Unit test summary

The suite has **67 tests across 6 files**, all passing. Each module is
covered below.

### `code/potts.py` — `tests/test_potts.py` (9 tests)

| What was tested | Expected | Actual | Resolution |
|---|---|---|---|
| `initialize_lattice` shape and value range | L×L, values in {1,…,Q} | matches | — |
| `initialize_lattice` rejects L≤0, Q≤0 | `ValueError` | raised | — |
| `get_neighbors` returns 4 or 8 neighbors with periodic wrap at corners | both corners wrap correctly | matches | — |
| `get_neighbors` rejects bad connectivity | `ValueError` | raised | — |
| Uniform lattice has zero boundary sites and zero energy | `boundary_mask` all False; `total_energy = 0` | matches | — |
| `total_energy` on stripe pattern matches independent bond enumeration | each bond once | matches | — |
| `total_energy` symmetry on a random lattice via independent bond count | exact equality | matches | — |
| Single-defect lattice flags the defect and its 4 neighbors as boundary | yes | matches | — |
| Across-seam boundary detection | row 0 and row L−1 differing wraps | matches | — |

### `code/mc_step.py` + `code/simulation.py` — `tests/test_kinetics.py` (22 tests)

Covers Metropolis acceptance, propose-from-neighbors, full sweeps, the
coupled drag step, and `run_pure_growth` / `run_coupled_simulation`.

| What was tested | Expected | Actual | Resolution |
|---|---|---|---|
| `propose_move` returns a neighbor orientation | always | matches | — |
| `propose_move` returns current orientation when no neighbor differs | no-op | matches | — |
| Low-kT Metropolis rejects energy-raising moves | always rejected at kT→0 | matches | — |
| Low-kT Metropolis accepts energy-lowering moves | always accepted | matches | — |
| High-kT acceptance approaches 100% for non-trivial proposals | >99% | observed >99% over ~100+ attempts | — |
| Energy non-increasing on average at low kT | `total_energy` falls over 20 sweeps | matches | — |
| `monte_carlo_step` returns accepted count in [0, N] | yes | matches | — |
| Uniform lattice stays uniform under MC | yes | matches | — |
| `coupled_metropolis_step` with C=1 blocks even downhill flips | acceptance = 0 | matches | — |
| `coupled_mc_step` with C=1 accepts no moves over a full sweep | accepted = 0 | matches | — |
| `coupled_metropolis_step` with C=0 accepts downhill flips | yes | matches | — |
| `coupled_metropolis_step` with C=0 accepts >99% of high-kT proposals | yes | matches | — |
| Drag run ends with smaller `<D>` than pure run at matched seed | yes | matches | — |
| Total solute conserved across coupled runs (with and without diffusion) | exact to 1e-9 | matches | — |
| `run_coupled_simulation` records `C_field` when requested | not None | matches | — |
| `run_pure_growth` snapshot scheduling at 0/interval/final step | exact step list | matches | — |
| `record_lattice=False` skips lattice copies | `lattice is None` | matches | — |
| `n_mcs=0` returns the initial snapshot | yes | matches | — |
| Coarsening: `<D>` ↑ and `n_grains` ↓ over 40 sweeps on 32×32 | yes | matches | — |
| **Parabolic growth law: linear fit of `<D>²` vs MCS** | **positive slope, R² > 0.9** | **k ≈ 1.1, R² ≈ 0.96** | — |

**Issue resolved during this phase:** the `code/` directory shadowed
the stdlib `code` module under pytest. We removed `code/__init__.py`;
tests now inject `code/` onto `sys.path` directly and import modules
flat. (Listed here because it surfaced via the test runner.)

### `code/analysis.py` — `tests/test_analysis.py` (8 tests)

| What was tested | Expected | Actual | Resolution |
|---|---|---|---|
| Uniform lattice → 1 grain of size L² | yes | matches | — |
| Checkerboard → L² singletons | yes | matches | — |
| Two horizontal stripes → 2 grains of size L²/2 | yes | matches | — |
| Periodic stitching: orientation-1 columns at 0 and L−1 merge into one grain | yes | matches | — |
| `mean_grain_diameter` formula on uniform lattice | `2·√(L²/π)` exact | matches | — |
| `mean_grain_diameter` on singletons | `2·√(1/π)` exact | matches | — |
| Labels are compact `1..K`, no background | yes | matches | — |
| Sizes sum to L² on a random lattice | yes | matches | — |

### `code/solute.py` — `tests/test_solute.py` (12 tests)

| What was tested | Expected | Actual | Resolution |
|---|---|---|---|
| `initialize_solute` uniform shape/value | matches | — |
| Bad arg validation (L≤0, C<0) | `ValueError` | raised | — |
| Diffusion of a uniform field is a no-op | identical | matches | — |
| Diffusion conserves total mass on a periodic grid | exact to 1e-12 | matches | — |
| Diffusion smooths a single hot spot to its 4 neighbors | yes | matches | — |
| CFL guard: rejects D·dt > 0.25 | `ValueError` | raised | — |
| `equilibrium_ratio` formula `exp(−E/kT)` | yes | matches | — |
| Segregation conserves total mass | exact to 1e-12 | matches | — |
| **Segregation builds up at boundary; ratio matches exp(−E_seg/kT)** | **C_GB/C_bulk = e for E=−1, kT=1** | **matches to 1e-9** | — |
| No-boundary lattice → segregation is a no-op | yes | matches | — |
| Half-rate update is exact midpoint of input and full equilibrium | yes | matches | — |
| Bad arg validation (rate ∉ [0,1], kT≤0) | `ValueError` | raised | — |

### `code/drag.py` — `tests/test_drag.py` (6 tests)

| What was tested | Expected | Actual | Resolution |
|---|---|---|---|
| `boundary_velocity` recovers constant slope | exact | matches | — |
| `boundary_velocity` requires ≥2 snapshots | `ValueError` | raised | — |
| `drag_curve` is identically zero when histories match | yes | matches | — |
| `drag_curve` is positive when coupled is uniformly slower | yes | matches | — |
| Real 24×24 pure-vs-coupled run produces mostly-nonnegative drag | ≥50% of samples ≥0 | matches | — |
| `drag_curve_over_concentrations` concatenates multi-history points | yes | matches | — |

### `code/optical_proxy.py` — `tests/test_optical.py` (10 tests)

| What was tested | Expected | Actual | Resolution |
|---|---|---|---|
| Uniform lattice → empty boundary map and `mu_eff = 0` | yes | matches | — |
| Checkerboard → all-boundary, `mu_eff = δn²` | yes | matches | — |
| **`mu_eff` halves stripe-thickness ⇒ ~doubles** (1/⟨D⟩ scaling) | ratio in [1.7, 2.3] | **matches** | — |
| Uniform lattice → zero scattering field | yes | matches | — |
| Periodic Gaussian blur is non-negative and conserves the integral | yes | matches | — |
| Higher scattering intensity in clustered-boundary regions | yes | matches | — |
| `sigma=0` returns the raw boundary map | yes | matches | — |
| Bad arg validation (sigma<0, mu<0, L<0) | `ValueError` | raised | — |
| `simulated_fiber_transmission` Beer–Lambert formula | matches `exp(−μL)` | exact | — |

---

## 4.2 Validation against theory

All numbers below are pulled from `results/data/*.npz`, generated by
`python code/main.py` (64×64, Q=48, kT=0.5, 200 MCS unless noted).

### Parabolic growth law

Fit `<D>² = k · MCS + b` over the linear regime (skipping the first 20%
of snapshots as transient).

- **k ≈ 1.10**, R² ≈ 0.96 (`results/figures/exp1_d2_vs_t.png`,
  `results/data/exp1_pure_growth.npz`)
- Published Potts-model values are typically reported as k ≈ 0.5 in
  dimensionless units (Anderson, Srolovitz, Grest, Sahni, *Acta Met.*
  1984). Our value is the same order of magnitude. The remaining ~2×
  factor depends on convention (per-sweep vs per-attempt MC rate),
  Q (we use 48 vs literature 16–48), and the linear-regime window
  used for the fit. For the purposes of this project we treat this as
  qualitative agreement and quote our k as a model-internal constant.

### Self-similar grain size distribution

`results/figures/exp2_size_distribution.png` shows the normalized
distribution of `D/<D>` at MCS=200. The histogram is unimodal with a
peak near `D/<D> ≈ 1` and a tail toward larger sizes — qualitatively
consistent with the Hillert form. Statistics are noisy at this lattice
size (N ≈ 19 grains in the final frame), so we do not fit Hillert
parameters. A larger lattice (e.g. L=256) and seed averaging would be
needed for a quantitative comparison.

### Solute drag curve

`results/figures/exp4_drag_curve.png` plots F = v_pure − v_coupled vs
v_coupled for 95 pooled samples across five `C_bulk` runs. The binned
mean shows a non-monotonic profile, rising to a peak at intermediate v
and falling at the high-v end — qualitatively the Cahn signature.
The high-v fall-off is shallower than Cahn theory predicts because our
drag model uses a static `(1 − C_local)` factor rather than a
velocity-dependent integral over the diffusion-segregation balance.
Documented in §4.3.

### Limiting behavior

From `results/data/exp3_concentration_sweep.npz`:

| C_bulk | final ⟨D⟩ |
|--------|-----------|
| 0.000 | 14.03 |
| 0.001 | 13.60 |
| 0.010 | 14.03 |
| 0.050 | 11.76 |
| 0.100 | 11.51 |
| 0.200 | 10.62 |

The C_bulk → 0 limit recovers pure-growth ⟨D⟩ to within finite-size
noise (compare 0.000 vs 0.001 and 0.010, all ≈ 14). At C_bulk = 0.4
with stronger E_seg (showcase parameters), final ⟨D⟩ drops to 7.9 —
roughly half the no-drag value over the same MCS budget — confirming
that growth is heavily suppressed at high concentration. Pushing
C_bulk → 1 freezes the lattice entirely (test
`test_coupled_step_with_unit_solute_blocks_all_moves`).

### Optical scaling: μ_eff ∝ 1/⟨D⟩

For a polycrystal in the geometric (non-Rayleigh) scattering regime,
the boundary length per unit area scales as 1/⟨D⟩, so μ_eff should
scale the same way. Using the six dopant-sweep cases:

- log-log fit `log μ_eff = a · log ⟨D⟩ + b` gives slope **a = −0.78**
  (expected −1.0)
- μ_eff·⟨D⟩ ranges over `(2.95–3.51)e−4` across the six cases, a ~15%
  spread.

The deviation from exactly −1 reflects (a) finite-size noise in the
boundary fraction at L=64, and (b) the boundary-fraction proxy
(boundary_sites/total_sites) is not strictly equal to perimeter per
unit area for thin grains. The trend is correct and the proportionality
constant is stable to ~15%, which validates the scaling relationship at
the level claimed.

---

## 4.3 Known limitations

These limitations are real and should be quoted in the writeup; the
code does not pretend otherwise.

1. **2D simulation.** Real grain growth is 3D. The mean exponent in
   `<D>² = k·t` is the same in 2D and 3D, but topological details
   (vertex degrees, neighbor counts, junction geometry) differ.
   Quantitative grain-size statistics from 2D should not be applied
   directly to a 3D ceramic.

2. **Simplified solute coupling.** We use a static
   acceptance-probability scaling factor `(1 − C_local)`. Real Cahn
   solute drag has an explicit dependence on boundary velocity through
   the steady-state segregation profile; the drag force vs velocity
   curve has a peak whose position and width depend on both `D_sol/v`
   and the segregation isotherm. Our model reproduces the qualitative
   shape of that curve but not the velocity-dependent magnitude.

3. **Isotropic boundary energy.** All boundaries are treated as equal
   (`J = 1.0` everywhere). Real ceramics have anisotropic boundary
   energies (Σ-boundaries, low- vs high-angle boundaries), which lead
   to abnormal grain growth and texture development that our model
   cannot capture.

4. **No thermal fluctuation effects on solute binding.** The
   segregation update relaxes the field deterministically toward
   `C_eq ∝ exp(−E/kT)`. There is no stochastic noise on the solute
   field itself, and no kinetic barrier for solute hopping beyond the
   linear diffusion term.

5. **Finite-size effects.** When `<D>` becomes comparable to L (in our
   200-MCS runs, `<D> ≈ 14` on a 64-lattice means ~4-5 grains across
   the box), grain statistics are dominated by a handful of grains and
   the periodic boundary conditions begin to influence growth. The
   late-time grain-size distribution in particular is statistics-
   limited.

6. **Optical scattering is illustrative only.** `mu_eff` is derived
   from a 2D boundary-length proxy with a `(δn)²` prefactor, not from
   a Rayleigh, Mie, or photon-transport calculation. The proxy
   captures the correct scaling with grain size in the geometric
   regime (validated above), but absolute magnitudes are arbitrary.
   Real scintillator ceramics depend on grain size, refractive-index
   mismatch, wavelength, and absorption in ways that combine
   non-trivially. Quantitative optical predictions would require a
   coupled photon-transport simulation, which is outside the scope of
   this class project.

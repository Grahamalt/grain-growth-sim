"""Solute field: continuous concentration on the lattice (Option A).

The solute is a scalar C(i, j) that diffuses by an explicit
finite-difference Laplacian (5-point stencil, periodic BCs) and
segregates to grain-boundary sites with energy E_seg < 0.

Equilibrium between a boundary site and a bulk site obeys

    C_GB / C_bulk = exp(-E_seg / kT).

Both ``diffuse_solute`` and ``segregation_update`` conserve the total
solute on the lattice exactly.
"""
from __future__ import annotations

import math

import numpy as np

from potts import boundary_mask


def initialize_solute(L, C_bulk, rng=None):
    """Uniform solute field at concentration C_bulk on an L x L lattice.

    ``rng`` is accepted for API symmetry with the lattice initializer but
    unused for the uniform initial condition.
    """
    if L <= 0:
        raise ValueError("L must be positive")
    if C_bulk < 0:
        raise ValueError("C_bulk must be non-negative")
    return np.full((L, L), float(C_bulk), dtype=np.float64)


def diffuse_solute(C_field, D_sol, dt):
    """One explicit FD diffusion step on a periodic L x L grid.

    Uses unit grid spacing. Returns a new array; total solute is conserved
    to floating-point precision because the periodic Laplacian sums to zero.

    Stability requires D_sol * dt <= 0.25 in 2D (CFL).
    """
    if D_sol < 0:
        raise ValueError("D_sol must be non-negative")
    if dt < 0:
        raise ValueError("dt must be non-negative")
    if D_sol * dt > 0.25 + 1e-12:
        raise ValueError(
            f"diffusion step unstable: D_sol*dt = {D_sol * dt} > 0.25 (CFL)"
        )

    laplacian = (
        np.roll(C_field, 1, axis=0)
        + np.roll(C_field, -1, axis=0)
        + np.roll(C_field, 1, axis=1)
        + np.roll(C_field, -1, axis=1)
        - 4.0 * C_field
    )
    return C_field + D_sol * dt * laplacian


def equilibrium_ratio(E_seg, kT):
    """C_GB / C_bulk at equilibrium: exp(-E_seg / kT)."""
    if kT <= 0:
        raise ValueError("kT must be positive")
    return math.exp(-E_seg / kT)


def segregation_update(lattice, C_field, E_seg, kT, rate=1.0, connectivity=4):
    """Relax C_field toward the segregation equilibrium, conserving mass.

    Each site has a Boltzmann weight w_i = exp(-E_i / kT) where E_i = E_seg
    on grain-boundary sites and 0 in the bulk. The equilibrium concentration
    field is C_eq_i = (M / sum_j w_j) * w_i, where M is the total solute
    (which is conserved). We move ``rate`` of the way toward C_eq each call:

        C_new = C + rate * (C_eq - C).

    ``rate=1.0`` jumps directly to equilibrium; smaller values give a
    smooth relaxation. Returns a new array.
    """
    if not (0.0 <= rate <= 1.0):
        raise ValueError("rate must be in [0, 1]")
    if kT <= 0:
        raise ValueError("kT must be positive")

    is_boundary = boundary_mask(lattice, connectivity=connectivity)
    weights = np.where(is_boundary, math.exp(-E_seg / kT), 1.0)

    total = C_field.sum()
    weight_sum = weights.sum()
    if weight_sum == 0:
        return C_field.copy()
    C_eq = (total / weight_sum) * weights
    return C_field + rate * (C_eq - C_field)

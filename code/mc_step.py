"""Monte Carlo step with Metropolis acceptance.

Grain-growth variant: proposed new orientation is drawn from the set of
distinct neighbor orientations. This restricts reorientation to the
local neighborhood (spin-exchange style) and converges far faster than
random sampling over all Q for grain-coarsening dynamics.
"""
from __future__ import annotations

import math

import numpy as np

from potts import get_neighbors, site_energy


def propose_move(lattice, i, j, rng, connectivity=4):
    """Pick a proposed new orientation from neighbor orientations.

    Returns the current orientation if no neighbor differs (no-op move).
    """
    L = lattice.shape[0]
    current = lattice[i, j]
    neighbor_orientations = {
        lattice[ni, nj] for ni, nj in get_neighbors(i, j, L, connectivity)
    }
    candidates = [s for s in neighbor_orientations if s != current]
    if not candidates:
        return current
    return int(candidates[rng.integers(0, len(candidates))])


def metropolis_step(lattice, i, j, kT, rng, J=1.0, connectivity=4):
    """Execute one Metropolis move at site (i, j). Returns True if accepted."""
    current = lattice[i, j]
    proposed = propose_move(lattice, i, j, rng, connectivity)
    if proposed == current:
        return False

    e_before = site_energy(lattice, i, j, J=J, connectivity=connectivity)
    lattice[i, j] = proposed
    e_after = site_energy(lattice, i, j, J=J, connectivity=connectivity)
    dE = e_after - e_before

    if dE <= 0.0:
        return True
    if kT <= 0.0:
        lattice[i, j] = current
        return False
    if rng.random() < math.exp(-dE / kT):
        return True
    lattice[i, j] = current
    return False


def monte_carlo_step(lattice, kT, rng, J=1.0, connectivity=4):
    """One full Monte Carlo sweep: N = L*L random site visits.

    Returns the number of accepted moves in the sweep.
    """
    L = lattice.shape[0]
    N = L * L
    accepted = 0
    rows = rng.integers(0, L, size=N)
    cols = rng.integers(0, L, size=N)
    for k in range(N):
        if metropolis_step(lattice, int(rows[k]), int(cols[k]), kT, rng,
                           J=J, connectivity=connectivity):
            accepted += 1
    return accepted


def coupled_metropolis_step(lattice, C_field, i, j, kT, rng,
                            J=1.0, connectivity=4):
    """Metropolis step with solute drag.

    Otherwise-accepted moves are scaled by (1 - C_local), clamped to
    [0, 1], so high local solute concentration suppresses boundary
    motion. C_local is the solute concentration at the proposed site.
    Returns True if accepted.
    """
    current = lattice[i, j]
    proposed = propose_move(lattice, i, j, rng, connectivity)
    if proposed == current:
        return False

    e_before = site_energy(lattice, i, j, J=J, connectivity=connectivity)
    lattice[i, j] = proposed
    e_after = site_energy(lattice, i, j, J=J, connectivity=connectivity)
    dE = e_after - e_before

    if dE <= 0.0:
        p_potts = 1.0
    elif kT <= 0.0:
        lattice[i, j] = current
        return False
    else:
        p_potts = math.exp(-dE / kT)

    drag = 1.0 - float(C_field[i, j])
    if drag < 0.0:
        drag = 0.0
    elif drag > 1.0:
        drag = 1.0
    p_accept = p_potts * drag

    if rng.random() < p_accept:
        return True
    lattice[i, j] = current
    return False


def coupled_mc_step(lattice, C_field, kT, rng, J=1.0, connectivity=4):
    """One full coupled Monte Carlo sweep: N = L*L visits with solute drag.

    Returns the number of accepted moves.
    """
    L = lattice.shape[0]
    N = L * L
    accepted = 0
    rows = rng.integers(0, L, size=N)
    cols = rng.integers(0, L, size=N)
    for k in range(N):
        if coupled_metropolis_step(lattice, C_field,
                                   int(rows[k]), int(cols[k]),
                                   kT, rng, J=J, connectivity=connectivity):
            accepted += 1
    return accepted

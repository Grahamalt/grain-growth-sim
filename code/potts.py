"""Core Potts model: lattice, neighbors, site energy, boundary detection.

2D square lattice with periodic boundary conditions. Each site holds an
integer orientation in {1, ..., Q}. Supports 4- or 8-neighbor stencils.
"""
from __future__ import annotations

import numpy as np


NEIGHBOR_OFFSETS_4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
NEIGHBOR_OFFSETS_8 = NEIGHBOR_OFFSETS_4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def initialize_lattice(L, Q, rng):
    """L x L lattice with orientations drawn uniformly from {1, ..., Q}."""
    if L <= 0:
        raise ValueError("L must be positive")
    if Q <= 0:
        raise ValueError("Q must be positive")
    return rng.integers(low=1, high=Q + 1, size=(L, L), dtype=np.int32)


def get_neighbors(i, j, L, connectivity=4):
    """Neighbor coordinates of (i, j) on an L x L lattice with periodic BCs."""
    if connectivity == 4:
        offsets = NEIGHBOR_OFFSETS_4
    elif connectivity == 8:
        offsets = NEIGHBOR_OFFSETS_8
    else:
        raise ValueError("connectivity must be 4 or 8")
    return [((i + di) % L, (j + dj) % L) for di, dj in offsets]


def site_energy(lattice, i, j, J=1.0, connectivity=4):
    """Potts site energy: J * sum_neighbors (1 - delta(s_i, s_n)).

    Zero when all neighbors match; grows with the count of unlike neighbors.
    """
    L = lattice.shape[0]
    s = lattice[i, j]
    unlike = 0
    for ni, nj in get_neighbors(i, j, L, connectivity):
        if lattice[ni, nj] != s:
            unlike += 1
    return J * unlike


def total_energy(lattice, J=1.0, connectivity=4):
    """Sum of site energies divided by 2 (each bond counted once)."""
    L = lattice.shape[0]
    acc = 0.0
    for i in range(L):
        for j in range(L):
            acc += site_energy(lattice, i, j, J=J, connectivity=connectivity)
    return 0.5 * acc


def is_boundary_site(lattice, i, j, connectivity=4):
    """True if any neighbor has a different orientation than (i, j)."""
    L = lattice.shape[0]
    s = lattice[i, j]
    for ni, nj in get_neighbors(i, j, L, connectivity):
        if lattice[ni, nj] != s:
            return True
    return False


def boundary_mask(lattice, connectivity=4):
    """Boolean L x L mask, True where the site sits on a grain boundary."""
    L = lattice.shape[0]
    mask = np.zeros_like(lattice, dtype=bool)
    for i in range(L):
        for j in range(L):
            mask[i, j] = is_boundary_site(lattice, i, j, connectivity)
    return mask

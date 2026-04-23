"""Tests for code/potts.py."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from potts import (  # noqa: E402
    boundary_mask,
    get_neighbors,
    initialize_lattice,
    is_boundary_site,
    site_energy,
    total_energy,
)


def test_initialize_lattice_shape_and_range():
    rng = np.random.default_rng(0)
    L, Q = 16, 48
    lat = initialize_lattice(L, Q, rng)
    assert lat.shape == (L, L)
    assert lat.min() >= 1
    assert lat.max() <= Q


def test_initialize_lattice_rejects_bad_args():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        initialize_lattice(0, 5, rng)
    with pytest.raises(ValueError):
        initialize_lattice(5, 0, rng)


def test_get_neighbors_count_and_periodic_bc():
    L = 8
    assert len(get_neighbors(3, 3, L, connectivity=4)) == 4
    assert len(get_neighbors(3, 3, L, connectivity=8)) == 8

    # corner (0, 0) wraps around to the far edges
    nbrs4 = set(get_neighbors(0, 0, L, connectivity=4))
    assert (L - 1, 0) in nbrs4
    assert (0, L - 1) in nbrs4
    assert (1, 0) in nbrs4
    assert (0, 1) in nbrs4

    # opposite corner (L-1, L-1) wraps back to (0, *) and (*, 0)
    nbrs8 = set(get_neighbors(L - 1, L - 1, L, connectivity=8))
    assert (0, 0) in nbrs8
    assert (0, L - 2) in nbrs8


def test_get_neighbors_rejects_bad_connectivity():
    with pytest.raises(ValueError):
        get_neighbors(0, 0, 4, connectivity=6)


def test_uniform_lattice_has_zero_boundary_sites():
    lat = np.full((10, 10), 7, dtype=np.int32)
    mask = boundary_mask(lat)
    assert not mask.any()
    assert total_energy(lat) == 0.0
    assert site_energy(lat, 5, 5) == 0.0
    assert not is_boundary_site(lat, 0, 0)


def test_total_energy_counts_each_bond_once():
    # Stripe pattern: every horizontal neighbor pair is unlike, vertical pairs are like.
    L = 6
    lat = np.tile(np.array([[1, 2]], dtype=np.int32), (L, L // 2))
    # Each site has exactly 2 unlike neighbors (left, right) under 4-connectivity.
    # total_energy = 0.5 * sum(site_energy) = 0.5 * (L*L * 2 * J) = L*L with J=1.
    assert total_energy(lat, J=1.0) == float(L * L)


def test_total_energy_symmetry_matches_bond_enumeration():
    # Independent bond-counting reference: iterate each bond once.
    rng = np.random.default_rng(42)
    L = 8
    lat = initialize_lattice(L, 5, rng)
    bonds = 0
    for i in range(L):
        for j in range(L):
            # Count the down and right bonds only, with periodic wrap.
            if lat[i, j] != lat[(i + 1) % L, j]:
                bonds += 1
            if lat[i, j] != lat[i, (j + 1) % L]:
                bonds += 1
    assert total_energy(lat, J=1.0, connectivity=4) == float(bonds)


def test_boundary_site_detects_single_defect():
    lat = np.full((5, 5), 1, dtype=np.int32)
    lat[2, 2] = 2
    assert is_boundary_site(lat, 2, 2)
    # Its 4 neighbors are also boundary sites; sites further away are not.
    assert is_boundary_site(lat, 1, 2)
    assert is_boundary_site(lat, 2, 1)
    assert not is_boundary_site(lat, 0, 0)


def test_periodic_bc_boundary_detection_across_edges():
    # Two differing orientations split across the periodic seam should still
    # register as boundary sites.
    L = 4
    lat = np.ones((L, L), dtype=np.int32)
    lat[0, :] = 2  # row 0 differs from row L-1 via wrap-around
    assert is_boundary_site(lat, 0, 1)
    assert is_boundary_site(lat, L - 1, 1)

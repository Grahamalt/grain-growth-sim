"""Tests for code/analysis.py."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from analysis import (  # noqa: E402
    grain_size_distribution,
    identify_grains,
    mean_grain_diameter,
    num_grains,
)


def test_uniform_lattice_is_one_grain():
    L = 12
    lat = np.full((L, L), 4, dtype=np.int32)
    labels = identify_grains(lat)
    assert num_grains(labels) == 1
    sizes = grain_size_distribution(labels)
    assert sizes.tolist() == [L * L]


def test_checkerboard_is_all_singleton_grains():
    # Two orientations in a checkerboard -> L*L grains of size 1 under 4-connectivity.
    L = 8
    lat = ((np.indices((L, L)).sum(axis=0)) % 2 + 1).astype(np.int32)
    labels = identify_grains(lat)
    assert num_grains(labels) == L * L
    sizes = grain_size_distribution(labels)
    assert sizes.size == L * L
    assert (sizes == 1).all()


def test_two_horizontal_stripes_is_two_grains():
    L = 6
    lat = np.ones((L, L), dtype=np.int32)
    lat[L // 2:, :] = 2
    labels = identify_grains(lat)
    assert num_grains(labels) == 2
    sizes = sorted(grain_size_distribution(labels).tolist())
    assert sizes == [L * L // 2, L * L // 2]


def test_periodic_stitching_merges_across_seam():
    # Column 0 and column L-1 are both orientation 1, connected via the periodic seam.
    # Interior columns are orientation 2. Expect 2 grains total (not 3).
    L = 6
    lat = np.full((L, L), 2, dtype=np.int32)
    lat[:, 0] = 1
    lat[:, L - 1] = 1
    labels = identify_grains(lat)
    assert num_grains(labels) == 2
    sizes = sorted(grain_size_distribution(labels).tolist())
    # Orientation-1 grain spans both edge columns = 2*L sites; orientation-2 fills the rest.
    assert sizes == [2 * L, L * (L - 2)]


def test_mean_grain_diameter_uniform_lattice():
    L = 10
    lat = np.full((L, L), 1, dtype=np.int32)
    labels = identify_grains(lat)
    d = mean_grain_diameter(labels)
    # Single grain of area L*L -> D = 2*sqrt(L*L/pi).
    assert math.isclose(d, 2.0 * math.sqrt(L * L / math.pi), rel_tol=1e-12)


def test_mean_grain_diameter_singletons():
    L = 4
    lat = ((np.indices((L, L)).sum(axis=0)) % 2 + 1).astype(np.int32)
    labels = identify_grains(lat)
    d = mean_grain_diameter(labels)
    # Every grain has area 1 -> D = 2*sqrt(1/pi) for every grain.
    assert math.isclose(d, 2.0 * math.sqrt(1.0 / math.pi), rel_tol=1e-12)


def test_labels_are_compact_and_start_at_one():
    L = 6
    lat = np.ones((L, L), dtype=np.int32)
    lat[L // 2:, :] = 2
    labels = identify_grains(lat)
    unique = np.unique(labels)
    # Background (0) should not appear when every site belongs to some grain.
    assert 0 not in unique
    assert unique.tolist() == [1, 2]


def test_grain_count_matches_distribution_length():
    rng = np.random.default_rng(0)
    L, Q = 16, 8
    lat = rng.integers(1, Q + 1, size=(L, L), dtype=np.int32)
    labels = identify_grains(lat)
    assert num_grains(labels) == grain_size_distribution(labels).size
    # Sanity: every site is labelled, so sizes sum to L*L.
    assert grain_size_distribution(labels).sum() == L * L

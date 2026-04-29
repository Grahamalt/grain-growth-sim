"""Tests for code/optical_proxy.py."""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from optical_proxy import (  # noqa: E402
    effective_attenuation_coefficient,
    grain_boundary_map,
    scattering_intensity_field,
    simulated_fiber_transmission,
)


def test_uniform_lattice_has_empty_boundary_map():
    L = 10
    lat = np.full((L, L), 5, dtype=np.int32)
    bmap = grain_boundary_map(lat)
    assert bmap.shape == (L, L)
    assert bmap.dtype == np.uint8
    assert bmap.sum() == 0
    assert effective_attenuation_coefficient(lat) == 0.0


def test_checkerboard_is_all_boundary_and_maximal_attenuation():
    L = 8
    lat = ((np.indices((L, L)).sum(axis=0)) % 2 + 1).astype(np.int32)
    bmap = grain_boundary_map(lat)
    assert (bmap == 1).all()
    # boundary_fraction = 1 -> mu_eff = delta_n^2.
    delta_n = 0.05
    mu = effective_attenuation_coefficient(lat, delta_n=delta_n)
    assert math.isclose(mu, delta_n ** 2, rel_tol=1e-12)


def test_attenuation_scales_inversely_with_grain_size():
    # Two stripe lattices: one with thicker stripes than the other.
    # Boundary fraction (= mu_eff / delta_n^2) should be roughly inversely
    # proportional to stripe thickness.
    L = 32
    delta_n = 0.01

    def stripe_lattice(stripe_width):
        lat = np.zeros((L, L), dtype=np.int32)
        for i in range(L):
            lat[i, :] = (i // stripe_width) % 2 + 1
        return lat

    mu_thick = effective_attenuation_coefficient(stripe_lattice(8), delta_n)
    mu_thin = effective_attenuation_coefficient(stripe_lattice(4), delta_n)
    # Halving stripe thickness should roughly double the boundary density.
    ratio = mu_thin / mu_thick
    assert 1.7 < ratio < 2.3


def test_scattering_intensity_field_uniform_lattice_is_zero():
    L = 16
    lat = np.full((L, L), 3, dtype=np.int32)
    field = scattering_intensity_field(lat, sigma=2.0)
    assert field.shape == (L, L)
    assert np.allclose(field, 0.0)


def test_scattering_intensity_field_is_smooth_and_nonnegative():
    L = 24
    rng = np.random.default_rng(0)
    lat = rng.integers(1, 5, size=(L, L), dtype=np.int32)
    field = scattering_intensity_field(lat, sigma=2.0)
    assert (field >= 0.0).all()
    # Smoothing brings the maximum below 1 (the boundary-map cap) for any
    # non-trivial sigma, and the mean of the smoothed field equals the mean
    # of the boundary map (Gaussian convolution conserves the integral on a
    # periodic domain).
    bmap = grain_boundary_map(lat).astype(float)
    assert math.isclose(field.mean(), bmap.mean(), rel_tol=1e-9, abs_tol=1e-12)
    assert field.max() <= bmap.max() + 1e-12


def test_scattering_intensity_higher_where_boundaries_cluster():
    # Construct a lattice where one half is a high-boundary-density checkerboard
    # and the other half is a single uniform grain.
    L = 32
    lat = np.full((L, L), 1, dtype=np.int32)
    # Right half: checkerboard pattern of orientations 1 and 2.
    for i in range(L):
        for j in range(L // 2, L):
            lat[i, j] = ((i + j) % 2) + 1
    field = scattering_intensity_field(lat, sigma=2.0)
    # Sample interior points away from the seam to avoid wrap-around contributions.
    left_sample = field[L // 2, L // 8]      # in the uniform region
    right_sample = field[L // 2, 3 * L // 4]  # in the checkerboard region
    assert right_sample > left_sample


def test_scattering_intensity_sigma_zero_returns_boundary_map():
    L = 8
    lat = ((np.indices((L, L)).sum(axis=0)) % 2 + 1).astype(np.int32)
    field = scattering_intensity_field(lat, sigma=0.0)
    bmap = grain_boundary_map(lat).astype(float)
    assert np.array_equal(field, bmap)


def test_scattering_rejects_negative_sigma():
    lat = np.ones((4, 4), dtype=np.int32)
    with pytest.raises(ValueError):
        scattering_intensity_field(lat, sigma=-1.0)


def test_simulated_fiber_transmission_basic():
    assert simulated_fiber_transmission(0.0, 100.0) == 1.0
    assert math.isclose(
        simulated_fiber_transmission(0.5, 2.0), math.exp(-1.0), rel_tol=1e-12
    )


def test_simulated_fiber_transmission_validation():
    with pytest.raises(ValueError):
        simulated_fiber_transmission(-0.1, 1.0)
    with pytest.raises(ValueError):
        simulated_fiber_transmission(0.1, -1.0)

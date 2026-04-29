"""Tests for code/solute.py."""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from potts import boundary_mask  # noqa: E402
from solute import (  # noqa: E402
    diffuse_solute,
    equilibrium_ratio,
    initialize_solute,
    segregation_update,
)


def test_initialize_solute_uniform():
    C = initialize_solute(L=8, C_bulk=0.05)
    assert C.shape == (8, 8)
    assert (C == 0.05).all()


def test_initialize_solute_rejects_bad_args():
    with pytest.raises(ValueError):
        initialize_solute(0, 0.1)
    with pytest.raises(ValueError):
        initialize_solute(8, -0.1)


def test_pure_diffusion_uniform_field_remains_uniform():
    C = initialize_solute(L=10, C_bulk=0.2)
    out = diffuse_solute(C, D_sol=0.1, dt=1.0)
    assert np.allclose(out, C)


def test_diffusion_conserves_total_mass():
    rng = np.random.default_rng(0)
    C = rng.random((12, 12)) * 0.1
    total_before = C.sum()
    out = diffuse_solute(C, D_sol=0.2, dt=1.0)
    assert math.isclose(out.sum(), total_before, rel_tol=1e-12, abs_tol=1e-12)


def test_diffusion_smooths_a_bump():
    C = np.zeros((9, 9))
    C[4, 4] = 1.0
    out = diffuse_solute(C, D_sol=0.1, dt=1.0)
    assert out[4, 4] < 1.0
    assert out[3, 4] > 0.0
    assert out[5, 4] > 0.0
    assert out[4, 3] > 0.0
    assert out[4, 5] > 0.0


def test_diffusion_rejects_unstable_step():
    C = np.zeros((6, 6))
    with pytest.raises(ValueError):
        diffuse_solute(C, D_sol=1.0, dt=1.0)  # 1.0 > 0.25 CFL bound


def test_equilibrium_ratio_formula():
    assert math.isclose(equilibrium_ratio(-1.0, 1.0), math.e, rel_tol=1e-12)
    assert equilibrium_ratio(0.0, 1.0) == 1.0


def test_segregation_conserves_total_mass():
    L = 10
    lat = np.ones((L, L), dtype=np.int32)
    lat[L // 2:, :] = 2
    C = np.full((L, L), 0.05)
    total_before = C.sum()
    out = segregation_update(lat, C, E_seg=-1.0, kT=1.0, rate=1.0)
    assert math.isclose(out.sum(), total_before, rel_tol=1e-12, abs_tol=1e-12)


def test_segregation_builds_up_at_grain_boundary():
    L = 10
    lat = np.ones((L, L), dtype=np.int32)
    lat[L // 2:, :] = 2
    C = np.full((L, L), 0.05)
    C_new = segregation_update(lat, C, E_seg=-1.0, kT=1.0, rate=1.0)

    is_boundary = boundary_mask(lat)
    C_gb = C_new[is_boundary].mean()
    C_bulk = C_new[~is_boundary].mean()
    assert C_gb > C_bulk
    expected_ratio = math.exp(1.0)  # E_seg = -1, kT = 1 -> ratio = e
    assert math.isclose(C_gb / C_bulk, expected_ratio, rel_tol=1e-9)


def test_segregation_no_boundaries_leaves_field_unchanged():
    L = 8
    lat = np.full((L, L), 3, dtype=np.int32)
    C = np.full((L, L), 0.1)
    out = segregation_update(lat, C, E_seg=-2.0, kT=0.5, rate=1.0)
    assert np.allclose(out, C)


def test_segregation_partial_rate_moves_partially():
    L = 6
    lat = np.ones((L, L), dtype=np.int32)
    lat[L // 2:, :] = 2
    C = np.full((L, L), 0.1)
    full = segregation_update(lat, C, E_seg=-1.0, kT=1.0, rate=1.0)
    half = segregation_update(lat, C, E_seg=-1.0, kT=1.0, rate=0.5)
    assert np.allclose(half, 0.5 * (C + full))


def test_segregation_rate_validation():
    L = 4
    lat = np.ones((L, L), dtype=np.int32)
    C = np.full((L, L), 0.1)
    with pytest.raises(ValueError):
        segregation_update(lat, C, E_seg=-1.0, kT=1.0, rate=1.5)
    with pytest.raises(ValueError):
        segregation_update(lat, C, E_seg=-1.0, kT=0.0, rate=1.0)

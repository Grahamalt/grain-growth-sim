"""Tests for code/drag.py."""
import os
import sys
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from drag import (  # noqa: E402
    boundary_velocity,
    drag_curve,
    drag_curve_over_concentrations,
)
from simulation import run_coupled_simulation, run_pure_growth  # noqa: E402


@dataclass
class _FakeSnap:
    step: int
    mean_diameter: float


def test_boundary_velocity_constant_growth():
    history = [_FakeSnap(step=k * 5, mean_diameter=2.0 + 0.1 * k * 5)
               for k in range(5)]
    steps_mid, D_mid, v = boundary_velocity(history)
    assert np.allclose(v, 0.1)
    assert steps_mid.shape == v.shape == D_mid.shape


def test_boundary_velocity_requires_two_snapshots():
    try:
        boundary_velocity([_FakeSnap(0, 1.0)])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_drag_curve_zero_when_histories_match():
    # Identical synthetic histories -> drag identically zero.
    h = [_FakeSnap(step=k, mean_diameter=1.0 + 0.5 * k) for k in range(8)]
    D_c, v_c, F = drag_curve(h, h)
    assert np.allclose(F, 0.0)
    assert np.allclose(v_c, 0.5)


def test_drag_curve_positive_when_coupled_is_slower():
    pure = [_FakeSnap(step=k, mean_diameter=1.0 + 1.0 * k) for k in range(8)]
    slow = [_FakeSnap(step=k, mean_diameter=1.0 + 0.3 * k) for k in range(8)]
    D_c, v_c, F = drag_curve(pure, slow)
    # In the overlapping D range (1.0 .. ~3.1) the slow run is slower at every D.
    assert (F > 0).all()
    assert np.allclose(v_c, 0.3)


def test_drag_curve_on_real_simulations_is_mostly_nonnegative():
    pure = run_pure_growth(L=24, Q=16, kT=0.5, n_mcs=30, snapshot_interval=5,
                           seed=4, record_lattice=False)
    coupled = run_coupled_simulation(
        L=24, Q=16, kT=0.5, C_bulk=0.4, E_seg=-2.0, D_sol=0.1,
        n_mcs=30, snapshot_interval=5, seed=4, record_lattice=False,
    )
    D_c, v_c, F = drag_curve(pure, coupled)
    assert F.size > 0
    # Solute drag should make most matched-D drag samples non-negative; allow
    # a small fraction of negatives from finite-size velocity noise.
    if F.size >= 4:
        assert (F >= 0).mean() >= 0.5


def test_drag_curve_over_concentrations_concatenates():
    pure = [_FakeSnap(step=k, mean_diameter=1.0 + 1.0 * k) for k in range(6)]
    h_a = [_FakeSnap(step=k, mean_diameter=1.0 + 0.5 * k) for k in range(6)]
    h_b = [_FakeSnap(step=k, mean_diameter=1.0 + 0.7 * k) for k in range(6)]
    v, F = drag_curve_over_concentrations(pure, [h_a, h_b])
    # Two histories of 5 velocity samples each -> up to 10 pooled points.
    assert v.size <= 10
    assert F.size == v.size
    assert (F >= 0).all()

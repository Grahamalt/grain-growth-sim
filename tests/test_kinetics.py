"""Tests for MC kinetics (mc_step, simulation)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from mc_step import metropolis_step, monte_carlo_step, propose_move  # noqa: E402
from potts import initialize_lattice, total_energy  # noqa: E402


def test_propose_move_returns_neighbor_orientation():
    # Checkerboard of 1 and 2: every site sees only the other orientation as a neighbor.
    lat = np.indices((6, 6)).sum(axis=0) % 2 + 1  # values in {1, 2}
    lat = lat.astype(np.int32)
    rng = np.random.default_rng(0)
    for _ in range(20):
        proposed = propose_move(lat, 3, 3, rng)
        assert proposed != lat[3, 3]
        assert proposed in (1, 2)


def test_propose_move_uniform_lattice_is_noop():
    lat = np.full((5, 5), 7, dtype=np.int32)
    rng = np.random.default_rng(0)
    assert propose_move(lat, 2, 2, rng) == 7


def test_low_temperature_rejects_energy_increasing_moves():
    # Uniform lattice except one differing neighbor, site (2,2) is surrounded by 1s.
    # Any proposed flip to 2 raises energy; at kT -> 0 it must be rejected.
    lat = np.full((5, 5), 1, dtype=np.int32)
    lat[2, 3] = 2  # one unlike neighbor of (2,2)
    rng = np.random.default_rng(0)
    before = lat[2, 2]
    # Repeated attempts; with kT = 1e-12, no energy-raising move can pass.
    for _ in range(50):
        metropolis_step(lat, 2, 2, kT=1e-12, rng=rng)
    # Either unchanged, or moved to 2 only if it lowered energy. Here flipping (2,2) to 2
    # gives 3 unlike neighbors instead of 1 -> dE = +2J > 0, must be rejected.
    assert lat[2, 2] == before


def test_low_temperature_accepts_energy_lowering_moves():
    # Site (2,2) is a lone defect: all 4 neighbors are 1, site is 2.
    # Flipping to 1 lowers energy from 4J to 0; at kT -> 0 it must be accepted on
    # the first attempt where the proposal is 1.
    lat = np.full((5, 5), 1, dtype=np.int32)
    lat[2, 2] = 2
    rng = np.random.default_rng(0)
    metropolis_step(lat, 2, 2, kT=1e-12, rng=rng)
    assert lat[2, 2] == 1


def test_high_temperature_acceptance_approaches_one_for_available_moves():
    # At very high kT, even energy-raising moves are accepted with probability ~1.
    # Using a uniform-with-defect setup so proposals are well-defined.
    rng = np.random.default_rng(123)
    lat = initialize_lattice(L=16, Q=8, rng=rng)
    attempts = 0
    accepted = 0
    for _ in range(2000):
        i = int(rng.integers(0, 16))
        j = int(rng.integers(0, 16))
        # Only count attempts where a real proposal exists.
        if len({lat[a, b] for a, b in [
            ((i - 1) % 16, j), ((i + 1) % 16, j),
            (i, (j - 1) % 16), (i, (j + 1) % 16),
        ]} - {lat[i, j]}) == 0:
            continue
        attempts += 1
        if metropolis_step(lat, i, j, kT=1e6, rng=rng):
            accepted += 1
    # At kT = 1e6, every non-trivial proposal is accepted.
    assert attempts > 100
    assert accepted / attempts > 0.99


def test_energy_non_increasing_on_average_at_low_temperature():
    rng = np.random.default_rng(7)
    lat = initialize_lattice(L=24, Q=16, rng=rng)
    e0 = total_energy(lat)
    for _ in range(20):
        monte_carlo_step(lat, kT=0.05, rng=rng)
    e1 = total_energy(lat)
    # At low kT the system coarsens; energy must not increase.
    assert e1 <= e0


def test_monte_carlo_step_returns_accepted_count_and_runs():
    rng = np.random.default_rng(3)
    L = 10
    lat = initialize_lattice(L=L, Q=8, rng=rng)
    accepted = monte_carlo_step(lat, kT=0.5, rng=rng)
    assert 0 <= accepted <= L * L


def test_uniform_lattice_stays_uniform():
    rng = np.random.default_rng(0)
    lat = np.full((8, 8), 3, dtype=np.int32)
    monte_carlo_step(lat, kT=1.0, rng=rng)
    assert (lat == 3).all()

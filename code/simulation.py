"""Main simulation loop for pure grain growth.

Drives the Potts MC kinetics for a fixed number of Monte Carlo sweeps,
records grain-size snapshots on a configurable interval, and returns
the time series for downstream analysis (e.g. parabolic-law validation).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from analysis import identify_grains, mean_grain_diameter, num_grains
from mc_step import monte_carlo_step
from potts import initialize_lattice, total_energy


@dataclass
class Snapshot:
    step: int
    mean_diameter: float
    num_grains: int
    energy: float
    lattice: np.ndarray


def run_pure_growth(L, Q, kT, n_mcs, snapshot_interval, seed=0,
                    J=1.0, connectivity=4, record_lattice=True):
    """Run pure grain growth for n_mcs Monte Carlo sweeps.

    Records a Snapshot at step 0 and every ``snapshot_interval`` sweeps
    thereafter (and always at the final step). Returns a list of Snapshots.
    Set ``record_lattice=False`` to omit lattice copies for long runs.
    """
    if snapshot_interval <= 0:
        raise ValueError("snapshot_interval must be positive")
    if n_mcs < 0:
        raise ValueError("n_mcs must be non-negative")

    rng = np.random.default_rng(seed)
    lattice = initialize_lattice(L, Q, rng)

    history = [_snapshot(0, lattice, J, connectivity, record_lattice)]

    for step in range(1, n_mcs + 1):
        monte_carlo_step(lattice, kT, rng, J=J, connectivity=connectivity)
        if step % snapshot_interval == 0 or step == n_mcs:
            history.append(_snapshot(step, lattice, J, connectivity, record_lattice))

    return history


def _snapshot(step, lattice, J, connectivity, record_lattice):
    labels = identify_grains(lattice)
    return Snapshot(
        step=step,
        mean_diameter=mean_grain_diameter(labels),
        num_grains=num_grains(labels),
        energy=total_energy(lattice, J=J, connectivity=connectivity),
        lattice=lattice.copy() if record_lattice else None,
    )

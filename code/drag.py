"""Solute-drag curve extraction.

Given a snapshot history (list of Snapshot from simulation.py), compute:

  - the boundary velocity v(t) ~ d<D>/dt by finite differences on the
    mean grain diameter,
  - a drag estimate F_i(D) ~ v_pure(D) - v_coupled(D), evaluated at
    matched mean grain diameter so that the curvature-driving-force is
    the same in both runs (Cahn-style comparison).

Sweeping coupled runs over solute concentrations (or letting a single
run coast as grains coarsen, sampling many D values along the way)
yields the characteristic non-monotonic F_i vs v curve.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def boundary_velocity(history, dt_per_mcs=1.0):
    """Finite-difference dD/dt from a snapshot history.

    Returns (steps_mid, D_mid, v) arrays, where each v[k] is the central
    secant slope (D[k+1] - D[k]) / (steps[k+1] - steps[k]) * dt_per_mcs,
    and steps_mid / D_mid are the midpoints of the bracketing snapshots.
    """
    if len(history) < 2:
        raise ValueError("need at least two snapshots to compute velocity")
    steps = np.array([snap.step for snap in history], dtype=float)
    D = np.array([snap.mean_diameter for snap in history], dtype=float)
    dD = np.diff(D)
    dt = np.diff(steps) * dt_per_mcs
    v = dD / dt
    steps_mid = 0.5 * (steps[1:] + steps[:-1])
    D_mid = 0.5 * (D[1:] + D[:-1])
    return steps_mid, D_mid, v


def drag_curve(pure_history, coupled_history, dt_per_mcs=1.0):
    """Cahn-style drag curve: F_i(D) = v_pure(D) - v_coupled(D).

    Velocities are computed from each history, then v_pure is linearly
    interpolated onto the coupled run's mean-diameter samples so the two
    are compared at matched D (and therefore matched curvature driving
    force, modulo finite-size noise).

    Returns (D_coupled, v_coupled, F_drag).
    """
    _, D_p, v_p = boundary_velocity(pure_history, dt_per_mcs)
    _, D_c, v_c = boundary_velocity(coupled_history, dt_per_mcs)

    # Sort the pure-growth (D, v) samples for monotone interpolation.
    order = np.argsort(D_p)
    D_p_sorted = D_p[order]
    v_p_sorted = v_p[order]

    # Restrict to the overlapping D range so interpolation does not extrapolate.
    lo = max(D_p_sorted.min(), D_c.min())
    hi = min(D_p_sorted.max(), D_c.max())
    in_range = (D_c >= lo) & (D_c <= hi)
    D_c_use = D_c[in_range]
    v_c_use = v_c[in_range]

    v_p_at_Dc = np.interp(D_c_use, D_p_sorted, v_p_sorted)
    F_drag = v_p_at_Dc - v_c_use
    return D_c_use, v_c_use, F_drag


def drag_curve_over_concentrations(
    pure_history,
    coupled_histories: Sequence,
    dt_per_mcs: float = 1.0,
):
    """Aggregate (v, F_drag) points across a sweep of coupled runs.

    Each coupled history contributes its (v_coupled, F_drag) samples,
    pooled into a single (v, F) array suitable for plotting the
    characteristic Cahn curve.
    """
    vs = []
    fs = []
    for h in coupled_histories:
        _, v_c, F = drag_curve(pure_history, h, dt_per_mcs)
        vs.append(v_c)
        fs.append(F)
    if not vs:
        return np.zeros(0), np.zeros(0)
    return np.concatenate(vs), np.concatenate(fs)

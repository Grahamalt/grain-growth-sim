"""Experiment runners for Phase 3.

Each experiment is a function ``exp_<name>(out_dir)`` that runs the
required simulations, writes a figure to ``out_dir/figures/`` and a
data file to ``out_dir/data/``, and returns a small dict of summary
results. ``run_all(out_dir)`` runs everything in order.
"""
from __future__ import annotations

import os
import time
from typing import Dict, List

import numpy as np

from analysis import grain_size_distribution, identify_grains
from drag import drag_curve_over_concentrations
from optical_proxy import effective_attenuation_coefficient
from plotting import (
    plot_attenuation_vs_concentration,
    plot_d2_vs_t,
    plot_design_curve,
    plot_diameter_vs_time_multi,
    plot_drag_curve,
    plot_evolution_panels,
    plot_grain_size_distribution,
    plot_lattice_snapshot,
    plot_microstructure_scattering_grid,
    plot_showcase_figure,
)
from simulation import run_coupled_simulation, run_pure_growth


# Defaults tuned for ~minutes-scale total runtime on a laptop.
DEFAULTS = dict(L=64, Q=48, kT=0.5, n_mcs=200, snapshot_interval=10, seed=0)


def _paths(out_dir, name, kind):
    sub = "figures" if kind == "fig" else "data"
    d = os.path.join(out_dir, sub)
    os.makedirs(d, exist_ok=True)
    ext = "png" if kind == "fig" else "npz"
    return os.path.join(d, f"{name}.{ext}")


# ---------------------------------------------------------------------------
# Experiment 1 - pure grain growth validation
# ---------------------------------------------------------------------------

def exp_pure_growth(out_dir):
    p = DEFAULTS
    history = run_pure_growth(L=p["L"], Q=p["Q"], kT=p["kT"], n_mcs=p["n_mcs"],
                              snapshot_interval=p["snapshot_interval"],
                              seed=p["seed"], record_lattice=False)
    steps = np.array([s.step for s in history], dtype=float)
    D = np.array([s.mean_diameter for s in history])
    # Fit on the linear regime: skip the first few snapshots (transient).
    fit_mask = steps >= 0.2 * steps.max()
    slope, intercept = np.polyfit(steps[fit_mask], D[fit_mask] ** 2, 1)
    fit = slope * steps + intercept
    ss_res = ((D[fit_mask] ** 2 - (slope * steps[fit_mask] + intercept)) ** 2).sum()
    ss_tot = ((D[fit_mask] ** 2 - (D[fit_mask] ** 2).mean()) ** 2).sum()
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    plot_d2_vs_t(steps, D, _paths(out_dir, "exp1_d2_vs_t", "fig"),
                 slope=slope, intercept=intercept,
                 title=f"Pure growth: <D>^2 vs MCS (k = {slope:.3g}, R^2 = {r_squared:.3f})")
    np.savez(_paths(out_dir, "exp1_pure_growth", "data"),
             steps=steps, mean_diameter=D, slope=slope, intercept=intercept,
             r_squared=r_squared)
    return dict(slope=slope, intercept=intercept, r_squared=r_squared,
                final_D=float(D[-1]))


# ---------------------------------------------------------------------------
# Experiment 2 - normalized grain size distribution at late times
# ---------------------------------------------------------------------------

def exp_size_distribution(out_dir):
    p = DEFAULTS
    history = run_pure_growth(L=p["L"], Q=p["Q"], kT=p["kT"],
                              n_mcs=p["n_mcs"],
                              snapshot_interval=p["n_mcs"],  # one final snapshot
                              seed=p["seed"], record_lattice=True)
    final = history[-1]
    labels = identify_grains(final.lattice)
    sizes = grain_size_distribution(labels)
    plot_grain_size_distribution(sizes, _paths(out_dir, "exp2_size_distribution", "fig"),
                                 title=(f"Normalized grain size distribution "
                                        f"at MCS={final.step} (N={sizes.size})"))
    np.savez(_paths(out_dir, "exp2_size_distribution", "data"),
             sizes=sizes, mean_diameter=final.mean_diameter)
    return dict(num_grains=int(sizes.size), final_D=float(final.mean_diameter))


# ---------------------------------------------------------------------------
# Experiment 3 - effect of solute concentration
# ---------------------------------------------------------------------------

C_BULK_VALUES = (0.0, 0.001, 0.01, 0.05, 0.1, 0.2)


def exp_concentration_sweep(out_dir):
    p = DEFAULTS
    histories = {}
    for C in C_BULK_VALUES:
        if C == 0.0:
            h = run_pure_growth(L=p["L"], Q=p["Q"], kT=p["kT"],
                                n_mcs=p["n_mcs"],
                                snapshot_interval=p["snapshot_interval"],
                                seed=p["seed"], record_lattice=False)
        else:
            h = run_coupled_simulation(
                L=p["L"], Q=p["Q"], kT=p["kT"], C_bulk=C,
                E_seg=-1.0, D_sol=0.1,
                n_mcs=p["n_mcs"], snapshot_interval=p["snapshot_interval"],
                seed=p["seed"], record_lattice=False,
            )
        histories[f"C={C}"] = h
    plot_diameter_vs_time_multi(
        histories, _paths(out_dir, "exp3_concentration_sweep", "fig"),
        title="Mean grain diameter vs MCS for varying C_bulk",
    )
    final_D = {C: histories[f"C={C}"][-1].mean_diameter for C in C_BULK_VALUES}
    np.savez(_paths(out_dir, "exp3_concentration_sweep", "data"),
             C_values=np.array(C_BULK_VALUES),
             final_D=np.array([final_D[C] for C in C_BULK_VALUES]))
    return dict(final_D=final_D, histories=histories)


# ---------------------------------------------------------------------------
# Experiment 4 - solute drag curve
# ---------------------------------------------------------------------------

def exp_drag_curve(out_dir, sweep_results=None):
    p = DEFAULTS
    pure = run_pure_growth(L=p["L"], Q=p["Q"], kT=p["kT"], n_mcs=p["n_mcs"],
                           snapshot_interval=p["snapshot_interval"],
                           seed=p["seed"], record_lattice=False)
    coupled_histories = []
    # Reuse the concentration sweep histories if provided; else regenerate.
    sweep_C = (0.001, 0.01, 0.05, 0.1, 0.2)
    if sweep_results is not None:
        coupled_histories = [sweep_results["histories"][f"C={C}"] for C in sweep_C]
    else:
        for C in sweep_C:
            coupled_histories.append(run_coupled_simulation(
                L=p["L"], Q=p["Q"], kT=p["kT"], C_bulk=C,
                E_seg=-1.0, D_sol=0.1,
                n_mcs=p["n_mcs"], snapshot_interval=p["snapshot_interval"],
                seed=p["seed"], record_lattice=False,
            ))
    v, F = drag_curve_over_concentrations(pure, coupled_histories)
    plot_drag_curve(v, F, _paths(out_dir, "exp4_drag_curve", "fig"))
    np.savez(_paths(out_dir, "exp4_drag_curve", "data"), v=v, F=F)
    return dict(num_points=int(v.size))


# ---------------------------------------------------------------------------
# Experiment 5 - effect of segregation energy
# ---------------------------------------------------------------------------

E_SEG_VALUES = (0.0, -0.5, -1.0, -2.0, -3.0)


def exp_segregation_energy(out_dir):
    p = DEFAULTS
    histories = {}
    for E in E_SEG_VALUES:
        h = run_coupled_simulation(
            L=p["L"], Q=p["Q"], kT=p["kT"], C_bulk=0.05,
            E_seg=E, D_sol=0.1,
            n_mcs=p["n_mcs"], snapshot_interval=p["snapshot_interval"],
            seed=p["seed"], record_lattice=False,
        )
        histories[f"E_seg={E}"] = h
    plot_diameter_vs_time_multi(
        histories, _paths(out_dir, "exp5_segregation_energy", "fig"),
        title="Effect of segregation energy on growth (C_bulk = 0.05)",
    )
    final_D = {E: histories[f"E_seg={E}"][-1].mean_diameter for E in E_SEG_VALUES}
    np.savez(_paths(out_dir, "exp5_segregation_energy", "data"),
             E_seg=np.array(E_SEG_VALUES),
             final_D=np.array([final_D[E] for E in E_SEG_VALUES]))
    return dict(final_D=final_D)


# ---------------------------------------------------------------------------
# Experiments 6 & 7 - design curve and optical attenuation vs C_bulk
# (requires final lattices, so we run a dedicated record_lattice=True sweep)
# ---------------------------------------------------------------------------

def exp_design_and_attenuation(out_dir):
    p = DEFAULTS
    final_D = []
    mu_eff_values = []
    final_lattices = {}
    for C in C_BULK_VALUES:
        if C == 0.0:
            h = run_pure_growth(L=p["L"], Q=p["Q"], kT=p["kT"], n_mcs=p["n_mcs"],
                                snapshot_interval=p["n_mcs"],
                                seed=p["seed"], record_lattice=True)
        else:
            h = run_coupled_simulation(
                L=p["L"], Q=p["Q"], kT=p["kT"], C_bulk=C,
                E_seg=-1.0, D_sol=0.1,
                n_mcs=p["n_mcs"], snapshot_interval=p["n_mcs"],
                seed=p["seed"], record_lattice=True,
            )
        last = h[-1]
        final_D.append(last.mean_diameter)
        mu_eff_values.append(effective_attenuation_coefficient(last.lattice))
        final_lattices[C] = last.lattice

    plot_design_curve(C_BULK_VALUES, final_D,
                      _paths(out_dir, "exp6_design_curve", "fig"))
    plot_attenuation_vs_concentration(
        C_BULK_VALUES, mu_eff_values,
        _paths(out_dir, "exp7_attenuation_vs_concentration", "fig"),
    )
    np.savez(_paths(out_dir, "exp6_7_design_and_attenuation", "data"),
             C_values=np.array(C_BULK_VALUES),
             final_D=np.array(final_D),
             mu_eff=np.array(mu_eff_values))
    return dict(final_D=dict(zip(C_BULK_VALUES, final_D)),
                mu_eff=dict(zip(C_BULK_VALUES, mu_eff_values)),
                final_lattices=final_lattices)


# ---------------------------------------------------------------------------
# Experiment 8 - showcase 3x2 grid
# ---------------------------------------------------------------------------

def exp_showcase_figure(out_dir):
    """Phase 3.3 showcase: 3 dopant cases with consistent colormaps and a caption.

    Parameters chosen after a probe sweep so the three cases are visually
    distinct (D approximately 15, 12, 8 lattice units) at fixed simulation
    time, dropout temperature, and segregation energy.
    """
    L, Q, kT = 96, 48, 0.5
    n_mcs = 300
    E_seg, D_sol = -2.5, 0.1
    cases_spec = [
        ("low dopant: C = 0.05", 0.05),
        ("medium dopant: C = 0.20", 0.20),
        ("high dopant: C = 0.40", 0.40),
    ]
    cases = []
    for label, C in cases_spec:
        h = run_coupled_simulation(
            L=L, Q=Q, kT=kT, C_bulk=C, E_seg=E_seg, D_sol=D_sol,
            n_mcs=n_mcs, snapshot_interval=n_mcs,
            seed=0, record_lattice=True,
        )
        last = h[-1]
        cases.append(dict(
            label=label,
            mean_diameter=float(last.mean_diameter),
            mu_eff=float(effective_attenuation_coefficient(last.lattice)),
            lattice=last.lattice,
        ))

    caption = (
        "Simulated microstructure (top row) and optical scattering intensity field "
        "(bottom row) for three dopant concentrations after 300 Monte Carlo sweeps "
        "(L = 96, Q = 48, kT = 0.5, E_seg = -2.5, D_sol = 0.1). As dopant "
        "concentration increases, solute drag reduces the mean grain size, "
        "producing a denser network of grain boundaries and correspondingly "
        "stronger optical scattering. The scattering field is computed as a "
        "Gaussian convolution (sigma = 2.5 lattice units) of the binary grain-"
        "boundary map; absolute values are illustrative only."
    )

    plot_showcase_figure(
        cases,
        _paths(out_dir, "showcase_figure", "fig"),
        sigma=2.5,
        caption=caption,
        title="Microstructure and optical scattering vs dopant concentration",
    )
    np.savez(_paths(out_dir, "showcase_figure", "data"),
             C=np.array([0.05, 0.20, 0.40]),
             mean_diameter=np.array([c["mean_diameter"] for c in cases]),
             mu_eff=np.array([c["mu_eff"] for c in cases]))
    return dict(cases=[(c["label"], c["mean_diameter"], c["mu_eff"]) for c in cases])


def exp_snapshot_frames(out_dir):
    """Phase 8 pre-flight: individual microstructure frames for an animation.

    Writes 4 separate PNGs at MCS = 0, n_mcs/8, n_mcs/2, n_mcs from one
    representative coupled run, in addition to the combined 4-panel
    figure produced by exp_evolution_panels.
    """
    p = DEFAULTS
    n_mcs = p["n_mcs"]
    history = run_coupled_simulation(
        L=p["L"], Q=p["Q"], kT=p["kT"], C_bulk=0.05,
        E_seg=-1.5, D_sol=0.1,
        n_mcs=n_mcs, snapshot_interval=10,
        seed=p["seed"], record_lattice=True,
    )
    target_steps = [0, n_mcs // 8, n_mcs // 2, n_mcs]
    out_dir_frames = os.path.join(out_dir, "figures", "snapshots")
    os.makedirs(out_dir_frames, exist_ok=True)
    written = []
    for k, t in enumerate(target_steps):
        snap = min(history, key=lambda s: abs(s.step - t))
        path = os.path.join(out_dir_frames, f"snapshot_{k:02d}_mcs{snap.step:04d}.png")
        plot_lattice_snapshot(snap, path)
        written.append(path)
    return dict(frames=written)


def exp_evolution_panels(out_dir):
    """Phase 3.2: representative coupled run with snapshots at early/mid/late MCS."""
    p = DEFAULTS
    n_mcs = p["n_mcs"]
    # Capture a snapshot every 10 MCS (cheap), then pick four representative ones.
    history = run_coupled_simulation(
        L=p["L"], Q=p["Q"], kT=p["kT"], C_bulk=0.05,
        E_seg=-1.5, D_sol=0.1,
        n_mcs=n_mcs, snapshot_interval=10,
        seed=p["seed"], record_lattice=True,
    )
    target_steps = [0, n_mcs // 8, n_mcs // 2, n_mcs]
    chosen = []
    for t in target_steps:
        snap = min(history, key=lambda s: abs(s.step - t))
        chosen.append(snap)
    plot_evolution_panels(
        chosen, _paths(out_dir, "exp_evolution_panels", "fig"),
        title=("Microstructure and solute evolution "
               "(C_bulk=0.05, E_seg=-1.5, D_sol=0.1)"),
    )
    np.savez(_paths(out_dir, "exp_evolution_panels", "data"),
             steps=np.array([s.step for s in chosen]),
             mean_diameter=np.array([s.mean_diameter for s in chosen]),
             total_solute=np.array([s.total_solute for s in chosen]))
    return dict(steps=[s.step for s in chosen],
                mean_diameter=[s.mean_diameter for s in chosen])


def exp_showcase_grid(out_dir, design_results):
    final_lattices = design_results["final_lattices"]
    cases = [
        ("low dopant: C=0.0", final_lattices[0.0]),
        ("medium dopant: C=0.05", final_lattices[0.05]),
        ("high dopant: C=0.2", final_lattices[0.2]),
    ]
    plot_microstructure_scattering_grid(
        cases, _paths(out_dir, "exp8_showcase_grid", "fig"), sigma=2.5,
        title="Microstructure (top) and scattering intensity (bottom)",
    )
    return dict(cases=[c[0] for c in cases])


# ---------------------------------------------------------------------------

def run_all(out_dir):
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    summary = {}
    t0 = time.time()

    print("[1/8] pure growth validation...")
    summary["exp1"] = exp_pure_growth(out_dir)
    print(f"    k = {summary['exp1']['slope']:.4f}, R^2 = {summary['exp1']['r_squared']:.3f}")

    print("[2/8] grain size distribution...")
    summary["exp2"] = exp_size_distribution(out_dir)
    print(f"    final N grains = {summary['exp2']['num_grains']}")

    print("[3/8] concentration sweep...")
    summary["exp3"] = exp_concentration_sweep(out_dir)
    for C, D in summary["exp3"]["final_D"].items():
        print(f"    C={C}: final <D>={D:.2f}")

    print("[4/8] solute drag curve...")
    summary["exp4"] = exp_drag_curve(out_dir, sweep_results=summary["exp3"])
    print(f"    drag samples = {summary['exp4']['num_points']}")

    print("[5/8] segregation energy sweep...")
    summary["exp5"] = exp_segregation_energy(out_dir)
    for E, D in summary["exp5"]["final_D"].items():
        print(f"    E_seg={E}: final <D>={D:.2f}")

    print("[6/8 + 7/8] design curve and attenuation vs C_bulk...")
    summary["exp6_7"] = exp_design_and_attenuation(out_dir)

    print("[8/8] showcase microstructure/scattering grid...")
    summary["exp8"] = exp_showcase_grid(out_dir, summary["exp6_7"])

    print("[3.2] microstructure + solute evolution panels...")
    summary["exp_evolution"] = exp_evolution_panels(out_dir)
    print(f"    snapshots at steps {summary['exp_evolution']['steps']}")

    print("[3.3] showcase figure...")
    summary["showcase"] = exp_showcase_figure(out_dir)
    for label, D, mu in summary["showcase"]["cases"]:
        print(f"    {label}: <D>={D:.2f}, mu_eff={mu:.3e}")

    print("[8.0] individual snapshot frames for animation...")
    summary["snapshots"] = exp_snapshot_frames(out_dir)
    for f in summary["snapshots"]["frames"]:
        print(f"    wrote {os.path.relpath(f, out_dir)}")

    # exp3.histories carries lattice=None copies; drop before returning to keep summary small.
    summary["exp3"].pop("histories", None)
    summary["exp6_7"].pop("final_lattices", None)

    print(f"\nDone in {time.time() - t0:.1f}s. Figures in {os.path.join(out_dir, 'figures')}.")
    return summary

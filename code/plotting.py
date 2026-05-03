"""All visualization for the grain-growth project.

Each function writes a PNG to ``out_path`` (creating the parent dir as
needed) and returns the figure for inline use. Matplotlib only.
"""
from __future__ import annotations

import os
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis import grain_size_distribution, identify_grains, mean_grain_diameter
from optical_proxy import scattering_intensity_field


def _ensure_parent(path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def plot_d2_vs_t(steps, mean_diameter, out_path, slope=None, intercept=None,
                 title="Pure grain growth: <D>^2 vs MCS"):
    _ensure_parent(out_path)
    steps = np.asarray(steps, dtype=float)
    d2 = np.asarray(mean_diameter, dtype=float) ** 2
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(steps, d2, "o-", label="<D>^2")
    if slope is not None and intercept is not None:
        ax.plot(steps, slope * steps + intercept, "k--",
                label=f"linear fit: k = {slope:.3g}")
    ax.set_xlabel("Monte Carlo sweeps")
    ax.set_ylabel("<D>^2 (lattice units^2)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_grain_size_distribution(sizes, out_path,
                                 title="Normalized grain size distribution"):
    _ensure_parent(out_path)
    sizes = np.asarray(sizes, dtype=float)
    diameters = 2.0 * np.sqrt(sizes / np.pi)
    mean_d = diameters.mean()
    x = diameters / mean_d
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(x, bins=20, density=True, edgecolor="black", alpha=0.75)
    ax.set_xlabel("D / <D>")
    ax.set_ylabel("Probability density")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_diameter_vs_time_multi(histories_by_label, out_path,
                                title="Grain diameter vs MCS"):
    _ensure_parent(out_path)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for label, history in histories_by_label.items():
        steps = [snap.step for snap in history]
        D = [snap.mean_diameter for snap in history]
        ax.plot(steps, D, "o-", label=label, markersize=3)
    ax.set_xlabel("Monte Carlo sweeps")
    ax.set_ylabel("Mean grain diameter <D> (lattice units)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_drag_curve(v, F, out_path, title="Solute drag curve: F_i vs v"):
    _ensure_parent(out_path)
    v = np.asarray(v, dtype=float)
    F = np.asarray(F, dtype=float)
    order = np.argsort(v)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(v, F, s=18, alpha=0.55, label="samples")
    # Bin-average to expose the peak shape.
    if v.size >= 8:
        bins = np.linspace(v.min(), v.max(), 12)
        idx = np.digitize(v, bins)
        v_mean, F_mean = [], []
        for b in range(1, len(bins)):
            sel = idx == b
            if sel.sum() >= 2:
                v_mean.append(v[sel].mean())
                F_mean.append(F[sel].mean())
        if v_mean:
            ax.plot(v_mean, F_mean, "k-o", label="binned mean")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("Boundary velocity v (d<D>/dMCS)")
    ax.set_ylabel("Drag F_i = v_pure - v_coupled")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_design_curve(C_values, final_D, out_path,
                      title="Design curve: final <D> vs dopant concentration"):
    _ensure_parent(out_path)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(C_values, final_D, "o-")
    ax.set_xlabel("Dopant concentration C_bulk")
    ax.set_ylabel("Final mean grain diameter <D>")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_attenuation_vs_concentration(C_values, mu_eff, out_path,
                                      title="Optical attenuation vs dopant concentration"):
    _ensure_parent(out_path)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(C_values, mu_eff, "s-")
    ax.set_xlabel("Dopant concentration C_bulk")
    ax.set_ylabel("Effective attenuation coefficient mu_eff (1/lattice unit)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_lattice_snapshot(snapshot, out_path, title=None):
    """Single-frame microstructure snapshot suitable for an animation series."""
    _ensure_parent(out_path)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.imshow(snapshot.lattice, cmap="tab20", interpolation="nearest")
    if title is None:
        title = (f"MCS = {snapshot.step}, <D> = {snapshot.mean_diameter:.1f}, "
                 f"N = {snapshot.num_grains}")
    ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_evolution_panels(snapshots, out_path,
                          title="Microstructure and solute evolution"):
    """2 x N panels: top row = orientation maps, bottom row = solute field.

    ``snapshots`` is an iterable of Snapshot objects (each must carry both
    ``lattice`` and ``C_field``). One column per snapshot.
    """
    _ensure_parent(out_path)
    snaps = list(snapshots)
    n = len(snaps)
    fig, axes = plt.subplots(2, n, figsize=(3.6 * n, 7.0))
    if n == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    # Use a single shared color scale for the solute panels so the
    # boundary build-up is comparable across times.
    c_max = max(snap.C_field.max() for snap in snaps if snap.C_field is not None)
    for k, snap in enumerate(snaps):
        ax_top = axes[0, k]
        ax_bot = axes[1, k]
        ax_top.imshow(snap.lattice, cmap="tab20", interpolation="nearest")
        ax_top.set_title(f"MCS = {snap.step}\n<D> = {snap.mean_diameter:.1f}, "
                         f"N = {snap.num_grains}", fontsize=9)
        ax_top.set_xticks([]); ax_top.set_yticks([])
        im = ax_bot.imshow(snap.C_field, cmap="viridis",
                           interpolation="nearest", vmin=0.0, vmax=c_max)
        ax_bot.set_xticks([]); ax_bot.set_yticks([])
        plt.colorbar(im, ax=ax_bot, fraction=0.046, pad=0.04)
    axes[0, 0].set_ylabel("Microstructure", fontsize=11)
    axes[1, 0].set_ylabel("Solute field C(x, y)", fontsize=11)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_showcase_figure(cases, out_path, sigma=2.5,
                         caption=None,
                         title="Microstructure and optical scattering vs dopant concentration"):
    """Polished 3x2 showcase: microstructure (top) and scattering field (bottom).

    ``cases`` is a list of dicts with keys: ``label`` (e.g. "C = 0.2"),
    ``mean_diameter``, ``mu_eff``, ``lattice``. The scattering color
    scale is shared across all columns so the dopant-trade-off is
    visually comparable. ``caption`` is rendered beneath the figure.
    """
    _ensure_parent(out_path)
    n = len(cases)
    fields = [scattering_intensity_field(c["lattice"], sigma=sigma) for c in cases]
    field_max = max(f.max() for f in fields)

    fig, axes = plt.subplots(2, n, figsize=(4.2 * n, 8.5))
    if n == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for k, (case, field) in enumerate(zip(cases, fields)):
        ax_top = axes[0, k]
        ax_bot = axes[1, k]
        ax_top.imshow(case["lattice"], cmap="tab20", interpolation="nearest")
        ax_top.set_title(case["label"], fontsize=13, fontweight="bold")
        ax_top.set_xticks([]); ax_top.set_yticks([])

        im = ax_bot.imshow(field, cmap="inferno", interpolation="nearest",
                           vmin=0.0, vmax=field_max)
        ax_bot.set_xticks([]); ax_bot.set_yticks([])

        sub = (f"<D> = {case['mean_diameter']:.1f} lattice units\n"
               f"mu_eff = {case['mu_eff']:.2e} (1/lattice unit)")
        ax_bot.set_xlabel(sub, fontsize=10)

    axes[0, 0].set_ylabel("Microstructure", fontsize=12)
    axes[1, 0].set_ylabel("Scattering intensity (a.u.)", fontsize=12)

    # Shared colorbar for the scattering row.
    cbar_ax = fig.add_axes([0.92, 0.10, 0.015, 0.36])
    sm = plt.cm.ScalarMappable(cmap="inferno",
                               norm=plt.Normalize(vmin=0.0, vmax=field_max))
    fig.colorbar(sm, cax=cbar_ax, label="Scattering intensity (a.u.)")

    fig.suptitle(title, fontsize=14)

    if caption is not None:
        # Wrapped caption rendered below the panels.
        fig.text(0.04, 0.02, caption, fontsize=9, wrap=True,
                 ha="left", va="bottom", style="italic")
        bottom = 0.18
    else:
        bottom = 0.08

    fig.subplots_adjust(left=0.05, right=0.90, top=0.92, bottom=bottom,
                        wspace=0.05, hspace=0.05)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_microstructure_scattering_grid(cases, out_path, sigma=2.0,
                                        title="Microstructure and scattering"):
    """``cases`` is a list of (label, lattice) tuples (length 3 expected)."""
    _ensure_parent(out_path)
    n = len(cases)
    fig, axes = plt.subplots(2, n, figsize=(4.0 * n, 7.5))
    if n == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    for k, (label, lat) in enumerate(cases):
        ax_top = axes[0, k]
        ax_bot = axes[1, k]
        ax_top.imshow(lat, cmap="tab20", interpolation="nearest")
        ax_top.set_title(label)
        ax_top.set_xticks([]); ax_top.set_yticks([])
        field = scattering_intensity_field(lat, sigma=sigma)
        im = ax_bot.imshow(field, cmap="inferno", interpolation="nearest")
        ax_bot.set_xticks([]); ax_bot.set_yticks([])
        plt.colorbar(im, ax=ax_bot, fraction=0.046, pad=0.04)
    axes[0, 0].set_ylabel("Microstructure", fontsize=11)
    axes[1, 0].set_ylabel("Scattering field", fontsize=11)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path

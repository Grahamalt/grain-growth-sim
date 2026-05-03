"""Render the individual showcase panels needed by web/index.html.

Produces six PNGs into web/assets/figures/ at the same parameters used
by exp_showcase_figure (L=96, Q=48, kT=0.5, E_seg=-2.5, D_sol=0.1,
300 MCS, seed=0). Each is an unannotated square image so the JS slider
+ toggle can swap them cleanly under shared HTML controls.

  showcase_low_microstructure.png      (C = 0.05, orientation map)
  showcase_low_scattering.png          (C = 0.05, scattering field)
  showcase_medium_microstructure.png   (C = 0.20, orientation map)
  showcase_medium_scattering.png       (C = 0.20, scattering field)
  showcase_high_microstructure.png     (C = 0.40, orientation map)
  showcase_high_scattering.png         (C = 0.40, scattering field)

Run from the repo root:  python3 code/generate_web_panels.py
"""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from optical_proxy import (  # noqa: E402
    effective_attenuation_coefficient,
    scattering_intensity_field,
)
from simulation import run_coupled_simulation  # noqa: E402


CASES = [
    ("low",    0.05),
    ("medium", 0.20),
    ("high",   0.40),
]
PARAMS = dict(L=96, Q=48, kT=0.5, E_seg=-2.5, D_sol=0.1,
              n_mcs=300, snapshot_interval=300, seed=0)
SIGMA = 2.5
OUT_DIR = os.path.normpath(os.path.join(HERE, "..", "web", "assets", "figures"))


def _save_panel(arr, cmap, out_path, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.imshow(arr, cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    metadata = []

    # First pass: run all sims and collect lattices + scattering fields.
    panels = []
    for label, C in CASES:
        history = run_coupled_simulation(C_bulk=C, record_lattice=True, **PARAMS)
        last = history[-1]
        field = scattering_intensity_field(last.lattice, sigma=SIGMA)
        mu = effective_attenuation_coefficient(last.lattice)
        panels.append((label, C, last.mean_diameter, mu, last.lattice, field))

    # Shared color scale across the three scattering panels (matches the
    # composite showcase figure's behavior).
    field_max = max(p[5].max() for p in panels)

    for label, C, D, mu, lat, field in panels:
        micro_path = os.path.join(OUT_DIR, f"showcase_{label}_microstructure.png")
        scat_path  = os.path.join(OUT_DIR, f"showcase_{label}_scattering.png")
        _save_panel(lat, "tab20", micro_path)
        _save_panel(field, "inferno", scat_path, vmin=0.0, vmax=field_max)
        metadata.append({
            "label": label, "C": C, "D": float(D), "mu_eff": float(mu),
            "micro": os.path.basename(micro_path),
            "scat":  os.path.basename(scat_path),
        })
        print(f"  {label}: C={C}, D={D:.2f}, mu_eff={mu:.3e}")

    print(f"\nWrote 6 panels to {OUT_DIR}")
    print("Shared scattering color scale max:", float(field_max))


if __name__ == "__main__":
    main()

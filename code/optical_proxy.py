"""Optical scattering proxy from microstructure.

Illustrative — NOT a full optical-transport simulation. The goal is to
visualize how microstructure evolution translates into qualitative
optical scattering for the scintillator-fiber motivation.

  - grain_boundary_map(lattice): binary map of boundary sites
  - scattering_intensity_field(lattice, sigma): Gaussian-blurred boundary
    map representing diffraction-limited halos around point scatterers
  - effective_attenuation_coefficient(lattice, delta_n): scalar mu_eff
    proportional to boundary density and (Delta n)^2
  - simulated_fiber_transmission(mu_eff, L_fiber): T = exp(-mu_eff * L);
    explicitly illustrative only
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

from potts import boundary_mask


def grain_boundary_map(lattice, connectivity=4):
    """Binary L x L array: 1 where any neighbor has a different orientation."""
    return boundary_mask(lattice, connectivity=connectivity).astype(np.uint8)


def scattering_intensity_field(lattice, sigma=2.0, connectivity=4):
    """Gaussian-blurred boundary map (periodic BCs).

    Physically motivated: each boundary segment scatters light into a
    diffraction-limited halo of width ~sigma lattice units. Returns a
    non-negative L x L float array.
    """
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    bmap = grain_boundary_map(lattice, connectivity=connectivity).astype(float)
    if sigma == 0:
        return bmap
    return ndimage.gaussian_filter(bmap, sigma=sigma, mode="wrap")


def effective_attenuation_coefficient(lattice, delta_n=0.01, connectivity=4):
    """Scalar mu_eff for the whole microstructure, in inverse lattice units.

    Proportional to boundary fraction (boundary sites / total sites) times
    (delta_n)^2. The boundary fraction scales as 1/<D> for a grain
    structure with mean diameter <D>, so mu_eff ~ (delta_n)^2 / <D>.
    """
    bmap = grain_boundary_map(lattice, connectivity=connectivity)
    boundary_fraction = bmap.mean()  # in [0, 1]
    return float((delta_n ** 2) * boundary_fraction)


def simulated_fiber_transmission(mu_eff, L_fiber_cm):
    """Illustrative Beer-Lambert transmission: T = exp(-mu_eff * L_fiber).

    Returns a value in [0, 1]. Labeled illustrative because mu_eff is
    derived from a 2D microstructure proxy, not from a calibrated 3D
    optical-transport calculation.
    """
    if mu_eff < 0:
        raise ValueError("mu_eff must be non-negative")
    if L_fiber_cm < 0:
        raise ValueError("L_fiber_cm must be non-negative")
    return math.exp(-mu_eff * L_fiber_cm)

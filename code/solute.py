"""Solute field: segregation and diffusion."""


def init_solute(size, c0):
    raise NotImplementedError


def diffuse(solute, D, dt):
    raise NotImplementedError


def segregate(solute, lattice):
    raise NotImplementedError

"""Core Potts model: lattice, neighbors, energy."""


def make_lattice(size, q):
    raise NotImplementedError


def neighbors(lattice, i, j):
    raise NotImplementedError


def local_energy(lattice, i, j):
    raise NotImplementedError

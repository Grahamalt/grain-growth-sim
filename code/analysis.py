"""Grain size extraction and statistics.

A "grain" is a maximally connected cluster of sites that share the same
orientation, under 4-connectivity with periodic boundary conditions.
We label grains via scipy.ndimage.label per-orientation and then stitch
clusters that touch across the periodic seams using a union-find pass.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage


class _UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def identify_grains(lattice):
    """Label connected same-orientation regions with periodic boundary conditions.

    Returns an int array of the same shape as ``lattice`` where each grain
    has a unique ID starting at 1. Labels are compacted and contiguous.
    """
    L = lattice.shape[0]
    labels = np.zeros_like(lattice, dtype=np.int32)
    structure = np.array([[0, 1, 0],
                          [1, 1, 1],
                          [0, 1, 0]], dtype=np.int32)

    next_label = 1
    unique_orientations = np.unique(lattice)
    for s in unique_orientations:
        mask = lattice == s
        comp, n = ndimage.label(mask, structure=structure)
        if n == 0:
            continue
        # Offset so IDs are globally unique before periodic stitching.
        comp[comp > 0] += next_label - 1
        labels[mask] = comp[mask]
        next_label += n

    total = next_label - 1
    if total == 0:
        return labels

    uf = _UnionFind(total + 1)

    # Stitch across the top/bottom seam.
    top = labels[0, :]
    bot = labels[L - 1, :]
    top_orient = lattice[0, :]
    bot_orient = lattice[L - 1, :]
    for j in range(L):
        if top_orient[j] == bot_orient[j] and top[j] > 0 and bot[j] > 0:
            uf.union(top[j], bot[j])

    # Stitch across the left/right seam.
    left = labels[:, 0]
    right = labels[:, L - 1]
    left_orient = lattice[:, 0]
    right_orient = lattice[:, L - 1]
    for i in range(L):
        if left_orient[i] == right_orient[i] and left[i] > 0 and right[i] > 0:
            uf.union(left[i], right[i])

    # Relabel using union-find roots, then compact to 1..K.
    flat = labels.reshape(-1)
    roots = np.array([uf.find(v) if v > 0 else 0 for v in range(total + 1)])
    remapped = roots[flat]
    unique_roots = np.unique(remapped[remapped > 0])
    compact = {root: idx + 1 for idx, root in enumerate(unique_roots)}
    out = np.zeros_like(remapped)
    for root, new_id in compact.items():
        out[remapped == root] = new_id
    return out.reshape(L, L).astype(np.int32)


def grain_size_distribution(labels):
    """Array of grain sizes (sites per grain), one entry per distinct grain."""
    if labels.max() == 0:
        return np.zeros(0, dtype=np.int64)
    counts = np.bincount(labels.ravel())
    # Drop the background count at index 0.
    return counts[1:].astype(np.int64)


def mean_grain_diameter(labels):
    """Mean effective grain diameter D = 2*sqrt(A/pi), averaged over grains."""
    sizes = grain_size_distribution(labels)
    if sizes.size == 0:
        return 0.0
    diameters = 2.0 * np.sqrt(sizes / math.pi)
    return float(diameters.mean())


def num_grains(labels):
    """Number of distinct grains in a label array."""
    return int(labels.max())

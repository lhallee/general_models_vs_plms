"""Groups of rows that share a trivial predictor of the label, for within-group scores.

A pooled rank correlation can come from group membership alone: which protein a sequence belongs to, which library
it comes from, or how many substitutions it carries. A stratum holds rows that are alike in that respect.
"""

from __future__ import annotations

import numpy as np

from .encoding import Layout, mutation_counts
from .flip2 import SplitInfo


IDENTITY_LINK = 0.95


def linked_components(codes: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Cluster rows of equal length that match at 95% of positions or more, by single linkage."""
    # codes: (n, l); lengths: (n,)
    parent = np.arange(len(codes))  # (n,)

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = int(parent[node])
        return node

    for length in np.unique(lengths):
        members = np.flatnonzero(lengths == length)  # (m,)
        block = codes[members, :length]  # (m, length)
        matches = (block[:, None, :] == block[None, :, :]).mean(axis=2)  # (m, m)
        for left, right in zip(*np.nonzero(np.triu(matches >= IDENTITY_LINK, k=1)), strict=True):
            parent[find(int(members[left]))] = find(int(members[right]))
    return np.array([find(node) for node in range(len(codes))])  # (n,)


def strata_of(split_info: SplitInfo, layout: Layout) -> np.ndarray:
    """One stratum name per row.

    TrpB: the set of substituted positions, which names the combinatorial library at full order. Other parents: the
    number of substitutions. Hydro: the protein, read from the sequence length. Rhomax: a cluster of near-identical
    sequences, which stands for a wild type and its variants.
    """
    gap = ord("-")
    lengths = (layout.codes != gap).sum(axis=1)  # (n,)
    if layout.parent is not None and split_info.dataset == "trpb":
        differs = layout.codes != layout.parent  # (n, l)
        return np.array(["-".join(layout.labels[column] for column in np.flatnonzero(row)) or "parent" for row in differs])  # (n,)
    if layout.parent is not None:
        return mutation_counts(layout).astype(str)  # (n,)
    if split_info.dataset == "hydro":
        return lengths.astype(str)  # (n,)
    return linked_components(layout.codes, lengths).astype(str)  # (n,)

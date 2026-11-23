"""A split's sequences in one padded frame: parents, substitution text, the cipher control and one-hot features."""

from __future__ import annotations

import hashlib
import numpy as np

from collections.abc import Sequence
from dataclasses import dataclass


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
GAP = "-"
CHAIN_BREAK = ":"
SYMBOLS = AMINO_ACIDS + GAP
GAP_CODE = ord(GAP)
BREAK_CODE = ord(CHAIN_BREAK)
RESIDUE_CODES = np.frombuffer(AMINO_ACIDS.encode("ascii"), dtype=np.uint8)  # (20,)


@dataclass(frozen=True)
class Layout:
    """Rows of a split as byte codes, right-padded with the gap to one width.

    `parent` is the reference sequence of a `single_parent` or `pair` layout and None for `sequences`. `labels` names
    each column in text: 1, 2, ... along a chain, and p1, p2, ... along the peptide of a pair.
    """

    kind: str
    codes: np.ndarray  # (n, l) uint8
    parent: np.ndarray | None  # (l,) uint8
    labels: tuple[str, ...]


def column_majority(codes: np.ndarray) -> np.ndarray:
    """The most frequent byte of each column."""
    # codes: (n, l)
    majority = np.empty(codes.shape[1], dtype=np.uint8)  # (l,)
    for column in range(codes.shape[1]):
        majority[column] = np.bincount(codes[:, column], minlength=128).argmax()
    return majority  # (l,)


def build_layout(sequences: Sequence[str], kind: str) -> Layout:
    """Pad a split's sequences to one width and find the parent its variants share, if the layout has one."""
    if kind not in ("single_parent", "pair", "sequences"):
        raise ValueError(f"unknown layout {kind!r}")

    if kind == "pair":
        chains = [sequence.split(CHAIN_BREAK) for sequence in sequences]
        if any(len(pair) != 2 for pair in chains) or len({len(domain) for domain, _ in chains}) != 1:
            raise ValueError("a pair layout needs domain:peptide strings with one domain length")
        peptide_width = max(len(peptide) for _, peptide in chains)
        padded = [domain + CHAIN_BREAK + peptide.ljust(peptide_width, GAP) for domain, peptide in chains]
        domain_width = len(chains[0][0])
        labels = (*(str(index + 1) for index in range(domain_width)), CHAIN_BREAK,
                  *(f"p{index + 1}" for index in range(peptide_width)))
    else:
        if kind == "single_parent" and len({len(sequence) for sequence in sequences}) != 1:
            raise ValueError("a single_parent layout needs sequences of one length")
        width = max(len(sequence) for sequence in sequences)
        padded = [sequence.ljust(width, GAP) for sequence in sequences]
        labels = tuple(str(index + 1) for index in range(width))

    codes = np.frombuffer("".join(padded).encode("ascii"), dtype=np.uint8).reshape(len(padded), -1)  # (n, l)
    known = np.isin(codes, RESIDUE_CODES) | (codes == GAP_CODE) | (codes == BREAK_CODE)  # (n, l)
    if not known.all():
        raise ValueError("a sequence holds a character outside the 20 amino acids, the gap and the chain break")

    parent = None if kind == "sequences" else column_majority(codes)  # (l,)
    return Layout(kind, codes, parent, tuple(labels))


def mutation_counts(layout: Layout) -> np.ndarray:
    """How many columns of each row differ from the parent."""
    assert layout.parent is not None, "a sequences layout has no parent to count from"
    return (layout.codes != layout.parent).sum(axis=1)  # (n,)


def parent_text(layout: Layout) -> str:
    """The parent as text, without padding."""
    assert layout.parent is not None, "a sequences layout has no parent"
    return layout.parent.tobytes().decode("ascii").replace(GAP, "")


def sequence_text(layout: Layout, row: int) -> str:
    """A row as text, without padding."""
    return layout.codes[row].tobytes().decode("ascii").replace(GAP, "")


def substitution_text(layout: Layout, row: int) -> str:
    """A row as its differences from the parent, such as `A12V G40D`, or `parent` when there are none."""
    assert layout.parent is not None, "a sequences layout has no parent to differ from"
    columns = np.flatnonzero(layout.codes[row] != layout.parent)  # (m,)
    if columns.size == 0:
        return "parent"
    return " ".join(f"{chr(layout.parent[column])}{layout.labels[column]}{chr(layout.codes[row, column])}"
                    for column in columns)


def variant_text(layout: Layout, row: int, representation: str) -> str:
    """A row in the requested text form."""
    if representation == "substitutions":
        return substitution_text(layout, row)
    if representation == "sequence":
        return sequence_text(layout, row)
    raise ValueError(f"unknown representation {representation!r}")


def draw_cipher(layout: Layout, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """A random relabeling of the 20 amino acids and a random column order for a layout.

    The relabeling moves every amino acid. Columns move only where rows are aligned to a parent: all of them for a
    single parent, the domain and the peptide each among themselves for a pair, and none for free sequences.
    """
    generator = np.random.default_rng([seed, 811])
    shuffled = RESIDUE_CODES.copy()  # (20,)
    while (shuffled == RESIDUE_CODES).any():
        shuffled = generator.permutation(RESIDUE_CODES)  # (20,)
    alphabet = np.arange(256, dtype=np.uint8)  # (256,)
    alphabet[RESIDUE_CODES] = shuffled

    width = layout.codes.shape[1]  # l
    order = np.arange(width)  # (l,)
    if layout.kind == "single_parent":
        order = generator.permutation(width)  # (l,)
    elif layout.kind == "pair":
        chain_break = layout.labels.index(CHAIN_BREAK)
        order[:chain_break] = generator.permutation(chain_break)
        order[chain_break + 1:] = chain_break + 1 + generator.permutation(width - chain_break - 1)
    return alphabet, order  # (256,), (l,)


def apply_cipher(layout: Layout, alphabet: np.ndarray, order: np.ndarray) -> Layout:
    """The layout with its residues relabeled and its columns reordered; labels keep counting from 1."""
    # alphabet: (256,); order: (l,)
    codes = alphabet[layout.codes][:, order]  # (n, l)
    parent = None if layout.parent is None else alphabet[layout.parent][order]  # (l,)
    return Layout(layout.kind, codes, parent, layout.labels)


def one_hot(layout: Layout, fit_rows: np.ndarray, query_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Indicator features of residue by column for the fit and query rows, over the columns that vary among them."""
    # fit_rows: (k,); query_rows: (q,)
    rows = np.concatenate([fit_rows, query_rows])  # (k + q,)
    codes = layout.codes[rows]  # (k + q, l)
    variable = (codes != codes[0]).any(axis=0)  # (l,)
    codes = codes[:, variable]  # (k + q, v)
    symbol_index = np.full(256, -1, dtype=np.int64)  # (256,)
    symbol_index[np.frombuffer(SYMBOLS.encode("ascii"), dtype=np.uint8)] = np.arange(len(SYMBOLS))
    indices = symbol_index[codes]  # (k + q, v)
    assert (indices >= 0).all(), "a varying column holds the chain break"

    features = np.zeros((codes.shape[0], codes.shape[1], len(SYMBOLS)), dtype=np.float32)  # (k + q, v, 21)
    np.put_along_axis(features, indices[:, :, None], 1.0, axis=2)
    features = features.reshape(codes.shape[0], -1)  # (k + q, 21 v)
    return features[:len(fit_rows)], features[len(fit_rows):]  # (k, 21 v), (q, 21 v)


def composition(layout: Layout, rows: np.ndarray) -> np.ndarray:
    """The fraction of each amino acid in each chain of the rows: one-hot residues averaged over a chain's positions."""
    # rows: (n,)
    codes = layout.codes[rows]  # (n, l)
    breaks = [column for column, label in enumerate(layout.labels) if label == CHAIN_BREAK]
    edges = [-1, *breaks, codes.shape[1]]
    blocks = []
    for start, stop in zip(edges[:-1], edges[1:], strict=True):
        chain = codes[:, start + 1:stop]  # (n, l_chain)
        counts = np.stack([(chain == code).sum(axis=1) for code in RESIDUE_CODES], axis=1)  # (n, 20)
        lengths = counts.sum(axis=1, keepdims=True)  # (n, 1); 0 for an empty peptide
        blocks.append(counts / np.maximum(lengths, 1))
    return np.hstack(blocks)  # (n, 20 c) for c chains


def random_vectors(layout: Layout, rows: np.ndarray, width: int) -> np.ndarray:
    """One N(0, 1) vector per row, drawn from a seed that is the SHA-256 of the row's sequence."""
    # rows: (n,)
    vectors = np.empty((len(rows), width))  # (n, w)
    for index, row in enumerate(rows):
        digest = hashlib.sha256(sequence_text(layout, int(row)).encode("ascii")).digest()
        vectors[index] = np.random.default_rng(int.from_bytes(digest[:8], "big")).standard_normal(width)
    return vectors  # (n, w)

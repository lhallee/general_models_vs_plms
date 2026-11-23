"""Layouts, substitution text, the cipher and one-hot features."""

from __future__ import annotations

import numpy as np
import pytest

from general_models_vs_plms.baselines import marginal_scores, nearest_neighbor, ridge, stratum_mean
from general_models_vs_plms.encoding import (
    AMINO_ACIDS, apply_cipher, build_layout, draw_cipher, mutation_counts, one_hot, parent_text, sequence_text,
    substitution_text,
)


PARENT = "MKTAYIAKQR"
VARIANTS = [PARENT, "MKTAYIAKQR", "AKTAYIAKQR", "MKTAYIAKQW", "AKTAYCAKQR", "MKTAYIAKQR"]


def test_single_parent_layout_finds_the_majority_parent_and_names_substitutions() -> None:
    layout = build_layout(VARIANTS, "single_parent")

    assert parent_text(layout) == PARENT
    assert mutation_counts(layout).tolist() == [0, 0, 1, 1, 2, 0]
    assert substitution_text(layout, 0) == "parent"
    assert substitution_text(layout, 2) == "M1A"
    assert substitution_text(layout, 4) == "M1A I6C"
    assert sequence_text(layout, 3) == "MKTAYIAKQW"


def test_single_parent_layout_refuses_unequal_lengths_and_unknown_characters() -> None:
    with pytest.raises(ValueError, match="one length"):
        build_layout(["MKT", "MK"], "single_parent")
    with pytest.raises(ValueError, match="outside the 20 amino acids"):
        build_layout(["MKX", "MKT"], "sequences")


def test_pair_layout_numbers_the_peptide_apart_and_marks_absent_residues() -> None:
    sequences = ["GEED:KNYK", "GEED:KNYK", "GEAD:KNYK", "GEED:KNWK", "GEED:K", "GEED:"]
    layout = build_layout(sequences, "pair")

    assert parent_text(layout) == "GEED:KNYK"
    assert substitution_text(layout, 2) == "E3A"
    assert substitution_text(layout, 3) == "Yp3W"
    assert substitution_text(layout, 4) == "Np2- Yp3- Kp4-"
    assert sequence_text(layout, 4) == "GEED:K"
    assert sequence_text(layout, 5) == "GEED:"


def test_sequences_layout_pads_and_has_no_parent() -> None:
    layout = build_layout(["MKT", "MKTAY"], "sequences")

    assert layout.parent is None
    assert layout.codes.shape == (2, 5)
    assert sequence_text(layout, 0) == "MKT"


def test_cipher_moves_every_residue_and_keeps_each_row_its_distance_from_the_parent() -> None:
    layout = build_layout(VARIANTS, "single_parent")
    alphabet, order = draw_cipher(layout, seed=3)
    hidden = apply_cipher(layout, alphabet, order)

    residues = np.frombuffer(AMINO_ACIDS.encode("ascii"), dtype=np.uint8)
    assert (alphabet[residues] != residues).all()
    assert sorted(alphabet[residues].tolist()) == sorted(residues.tolist())
    assert sorted(order.tolist()) == list(range(len(PARENT)))
    assert mutation_counts(hidden).tolist() == mutation_counts(layout).tolist()
    assert parent_text(hidden) != PARENT


def test_cipher_keeps_a_pair_in_its_chains_and_free_sequences_in_their_columns() -> None:
    pair = build_layout(["GEED:KNYK", "GEAD:KNYK", "GEED:KNWK"], "pair")
    _, order = draw_cipher(pair, seed=1)
    assert sorted(order[:4].tolist()) == [0, 1, 2, 3]
    assert order[4] == 4
    assert sorted(order[5:].tolist()) == [5, 6, 7, 8]

    free = build_layout(["MKT", "MKTAY"], "sequences")
    _, free_order = draw_cipher(free, seed=1)
    assert free_order.tolist() == [0, 1, 2, 3, 4]


def test_one_hot_ridge_is_unchanged_by_the_cipher() -> None:
    generator = np.random.default_rng(0)
    letters = np.array(list(AMINO_ACIDS))
    sequences = ["".join(generator.choice(letters[:6], size=12)) for _ in range(80)]
    layout = build_layout(sequences, "single_parent")
    hidden = apply_cipher(layout, *draw_cipher(layout, seed=5))
    fit_rows, query_rows = np.arange(60), np.arange(60, 80)
    values = generator.normal(size=60)

    plain = ridge(*one_hot(layout, fit_rows, query_rows)[:1], values, one_hot(layout, fit_rows, query_rows)[1])
    enciphered = ridge(*one_hot(hidden, fit_rows, query_rows)[:1], values, one_hot(hidden, fit_rows, query_rows)[1])

    np.testing.assert_allclose(plain, enciphered, rtol=1e-4, atol=1e-5)


def test_one_hot_covers_only_columns_that_vary_among_the_rows_used() -> None:
    layout = build_layout(VARIANTS, "single_parent")
    fit, query = one_hot(layout, np.array([0, 2, 3]), np.array([4]))

    assert fit.shape == (3, 3 * 21)
    assert query.shape == (1, 3 * 21)
    assert fit.sum(axis=1).tolist() == [3.0, 3.0, 3.0]


def test_trivial_baselines_read_only_the_context() -> None:
    layout = build_layout(VARIANTS, "single_parent")
    context_rows, values = np.array([0, 2, 3]), np.array([1.0, 5.0, 9.0])

    assert nearest_neighbor(layout, context_rows, values, np.array([4])).tolist() == [5.0]
    assert nearest_neighbor(layout, context_rows, values, np.array([5])).tolist() == [1.0]
    strata = np.array(["a", "a", "b", "b", "b", "c"])
    assert stratum_mean(strata, context_rows, values, np.array([4, 5])).tolist() == [7.0, 5.0]


def test_marginal_scores_sum_log_ratios_over_substitutions_and_skip_absent_residues() -> None:
    layout = build_layout(["GEED:KNYK", "GEED:KNYK", "GEAD:KNYK", "GEED:KNWK", "GEED:K"], "pair")
    marginals = np.zeros((8, 20), dtype=np.float32)
    marginals[2, AMINO_ACIDS.index("A")] = 0.5
    marginals[2, AMINO_ACIDS.index("E")] = 2.0
    marginals[6, AMINO_ACIDS.index("W")] = -1.0
    marginals[6, AMINO_ACIDS.index("Y")] = 3.0

    scores = marginal_scores(layout, marginals, np.arange(5))

    np.testing.assert_allclose(scores, [0.0, 0.0, -1.5, -4.0, 0.0])

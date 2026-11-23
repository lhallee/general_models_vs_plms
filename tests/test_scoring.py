"""Metrics, strata and the paired bootstrap."""

from __future__ import annotations

import numpy as np
import pytest

from scipy.stats import spearmanr

from general_models_vs_plms.encoding import build_layout
from general_models_vs_plms.flip2 import SplitInfo
from general_models_vs_plms.scoring import bootstrap_spearman, ndcg, score, spearman, top_decile_metrics, within_strata_spearman
from general_models_vs_plms.strata import linked_components, strata_of


def test_spearman_is_zero_for_a_constant_side_and_matches_scipy_otherwise() -> None:
    truth = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    assert spearman(truth, np.ones(5)) == 0.0
    assert spearman(np.ones(5), truth) == 0.0
    assert spearman(truth, truth[::-1].copy()) == pytest.approx(-1.0)


def test_ndcg_is_one_for_a_perfect_ranking_and_lower_for_its_reverse() -> None:
    truth = np.array([0.1, 0.4, 0.2, 0.9, 0.5])

    assert ndcg(truth, truth) == pytest.approx(1.0)
    assert ndcg(truth, -truth) < 0.8
    assert ndcg(np.ones(4), np.arange(4.0)) == 1.0


def test_within_strata_spearman_removes_a_between_group_trend() -> None:
    truth = np.concatenate([np.arange(10.0), 100 + np.arange(10.0)[::-1]])
    predictions = np.concatenate([np.arange(10.0), 100 + np.arange(10.0)])
    strata = np.array(["a"] * 10 + ["b"] * 10)

    assert spearman(truth, predictions) > 0.7
    within, covered = within_strata_spearman(truth, predictions, strata)
    assert covered == 20
    assert abs(within) < 1e-9

    _, covered_small = within_strata_spearman(truth, predictions, np.array(["a"] * 15 + ["b"] * 5))
    assert covered_small == 15


def test_top_decile_metrics_are_perfect_for_the_truth_and_undefined_without_two_classes() -> None:
    truth = np.arange(100.0)
    perfect = top_decile_metrics(truth, truth)

    assert perfect["top_prevalence"] == 0.1
    assert perfect["roc_auc"] == pytest.approx(1.0) and perfect["pr_auc"] == pytest.approx(1.0)
    assert perfect["balanced_mcc"] == pytest.approx(1.0)
    assert np.isnan(top_decile_metrics(np.ones(20), np.arange(20.0))["roc_auc"])


def test_score_reports_every_metric() -> None:
    generator = np.random.default_rng(0)
    truth = generator.normal(size=120)
    metrics = score(truth, truth + generator.normal(size=120), np.array(["x"] * 60 + ["y"] * 60))

    assert set(metrics) == {"spearman", "ndcg", "within_strata_spearman", "within_strata_queries",
                            "distinct_predictions", "top_prevalence", "roc_auc", "pr_auc", "balanced_mcc"}
    assert 0.4 < metrics["spearman"] < 0.95 and metrics["within_strata_queries"] == 120.0


def test_bootstrap_spearman_pairs_methods_on_the_same_resamples() -> None:
    generator = np.random.default_rng(1)
    truth = generator.normal(size=80)
    predictions = np.stack([truth + generator.normal(size=80), generator.normal(size=80), np.ones(80)])

    values = bootstrap_spearman(truth, predictions, draws=300, seed=4)

    assert values.shape == (300, 3)
    assert abs(values[:, 0].mean() - spearmanr(truth, predictions[0]).statistic) < 0.03
    assert abs(values[:, 1].mean()) < 0.15
    assert (values[:, 2] == 0).all()
    np.testing.assert_allclose(values, bootstrap_spearman(truth, predictions, draws=300, seed=4))


def test_linked_components_join_near_identical_sequences_of_one_length() -> None:
    base = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"
    near = base[:5] + "W" + base[6:]
    far = base[::-1]
    layout = build_layout([base, near, far, base[:30]], "sequences")
    lengths = np.array([40, 40, 40, 30])

    clusters = linked_components(layout.codes, lengths)

    assert clusters[0] == clusters[1]
    assert len({clusters[0], clusters[2], clusters[3]}) == 3


def test_strata_follow_the_dataset() -> None:
    single = build_layout(["MKTAY", "MKTAY", "AKTAY", "AKTAW", "MKTAY"], "single_parent")
    assert strata_of(SplitInfo("amylase", "s", "number", "single_parent", ""), single).tolist() == ["0", "0", "1", "2", "0"]
    assert strata_of(SplitInfo("trpb", "s", "number", "single_parent", ""), single).tolist() == ["parent", "parent", "1", "1-5", "parent"]
    hydro = build_layout(["MKT", "MKTAY", "AKT"], "sequences")
    assert strata_of(SplitInfo("hydro", "s", "number", "sequences", ""), hydro).tolist() == ["3", "5", "3"]

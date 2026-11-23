"""Scores for a set of predictions against measured values: rank correlation, NDCG and a top-decile classification.

A method is scored on every query of a task. A query the method did not answer takes the fill value the caller
gives, so leaving hard queries unanswered cannot raise a score.
"""

from __future__ import annotations

import numpy as np

from scipy.stats import rankdata, spearmanr
from sklearn.metrics import ndcg_score

from foundry.metrics.binary import balanced_mcc
from foundry.metrics.ranking import binary_ranking_metrics


TOP_FRACTION = 0.1
MIN_STRATUM = 10


def spearman(truth: np.ndarray, predictions: np.ndarray) -> float:
    """Spearman rank correlation with ties at their mean rank, and 0.0 when either side is constant."""
    # truth, predictions: (q,)
    if np.ptp(truth) == 0 or np.ptp(predictions) == 0:
        return 0.0
    return float(spearmanr(truth, predictions).statistic)


def ndcg(truth: np.ndarray, predictions: np.ndarray) -> float:
    """NDCG over the whole ranking, with the measured value scaled to 0 through 1 as the gain."""
    # truth, predictions: (q,)
    span = np.ptp(truth)
    if span == 0:
        return 1.0
    gains = (truth - truth.min()) / span  # (q,)
    return float(ndcg_score(gains[None, :], predictions[None, :]))


def within_strata_spearman(truth: np.ndarray, predictions: np.ndarray, strata: np.ndarray) -> tuple[float, int]:
    """Mean Spearman inside strata of ten queries or more, weighted by stratum size, and how many queries it covers."""
    # truth, predictions, strata: (q,)
    total, covered = 0.0, 0
    for stratum in np.unique(strata):
        members = strata == stratum  # (q,)
        if members.sum() >= MIN_STRATUM:
            total += spearman(truth[members], predictions[members]) * members.sum()
            covered += int(members.sum())
    return (total / covered if covered else float("nan")), covered


def top_decile_metrics(truth: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    """Ranking the queries at or above the 90th percentile of the measured value against the rest."""
    # truth, predictions: (q,)
    labels = (truth >= np.quantile(truth, 1.0 - TOP_FRACTION)).astype(int)  # (q,)
    if labels.min() == labels.max():
        return {"top_prevalence": float(labels.mean()), "roc_auc": float("nan"), "pr_auc": float("nan"),
                "balanced_mcc": float("nan")}
    ranking = binary_ranking_metrics(labels, predictions)
    return {"top_prevalence": float(labels.mean()), "roc_auc": float(ranking["auroc"]),
            "pr_auc": float(ranking["average_precision"]),
            "balanced_mcc": float(balanced_mcc(labels, predictions)["balanced_mcc"])}


def score(truth: np.ndarray, predictions: np.ndarray, strata: np.ndarray) -> dict[str, float]:
    """Every metric for one method on one task."""
    # truth, predictions, strata: (q,)
    within, covered = within_strata_spearman(truth, predictions, strata)
    return {"spearman": spearman(truth, predictions), "ndcg": ndcg(truth, predictions),
            "within_strata_spearman": within, "within_strata_queries": float(covered),
            "distinct_predictions": float(len(np.unique(predictions))), **top_decile_metrics(truth, predictions)}


def bootstrap_spearman(truth: np.ndarray, predictions: np.ndarray, draws: int, seed: int) -> np.ndarray:
    """Spearman of each method on resampled queries; methods share the resamples, so their differences are paired."""
    # truth: (q,); predictions: (m, q)
    generator = np.random.default_rng(seed)
    samples = generator.integers(0, len(truth), size=(draws, len(truth)))  # (r, q)
    truth_ranks = rankdata(truth[samples], axis=1)  # (r, q)
    prediction_ranks = rankdata(predictions[:, samples], axis=2)  # (m, r, q)
    centered_truth = truth_ranks - truth_ranks.mean(axis=1, keepdims=True)  # (r, q)
    centered = prediction_ranks - prediction_ranks.mean(axis=2, keepdims=True)  # (m, r, q)
    products = (centered * centered_truth[None, :, :]).sum(axis=2)  # (m, r)
    norms = np.sqrt((centered ** 2).sum(axis=2) * (centered_truth ** 2).sum(axis=1)[None, :])  # (m, r)
    # A resample with a constant side has no rank correlation and scores 0, as `spearman` does.
    return np.divide(products, norms, out=np.zeros_like(products), where=norms > 0).T  # (r, m)

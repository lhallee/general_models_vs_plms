"""The join of knowledge and error over sequences, on a toy table with a planted relation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from general_models_vs_plms import knowledge_report
from general_models_vs_plms.sequence_knowledge import RHOMAX_TASK, Entry


def test_a_planted_relation_is_found_and_an_unrelated_measure_is_not() -> None:
    generator = np.random.default_rng(5)
    knowledge = generator.normal(size=80)  # (80,)
    table = pd.DataFrame({
        "known": knowledge, "error_follows": -knowledge + 0.3 * generator.normal(size=80),
        "error_unrelated": generator.normal(size=80), "unusual": generator.normal(size=80)})

    planted = knowledge_report.relation(table, "known", "error_follows", "planted")
    absent = knowledge_report.relation(table, "known", "error_unrelated", "absent")

    assert planted["spearman"] < -0.8 and planted["high"] < 0 and planted["partial_given_unusual"] < -0.8
    assert absent["low"] < 0 < absent["high"] and planted["sequences"] == 80


def test_the_rhodopsin_table_joins_knowledge_with_absolute_error_by_row() -> None:
    entries = [Entry(f"rhodopsin/{row}", "rhodopsin", "ACDEFGHIKL", asked=row < 12, row=row) for row in (10, 11, 12)]
    answers = pd.DataFrame({"name": ["rhodopsin/10", "rhodopsin/10", "rhodopsin/11"], "correct": [2, 4, 5], "masked": [4, 8, 10]})
    masked = {"esmc_600m": pd.DataFrame({
        "name": [entry.name for entry in entries] * 2, "level": [0.0] * 3 + [0.5] * 3, "masked": [10] * 3 + [5] * 3,
        "correct": [9, 8, 7, 5, 4, 3], "mean_log_probability": [-0.1, -0.2, -0.3, -1.0, -1.1, -1.2]})}
    methods = [(knowledge_report.LANGUAGE_MODEL, "k0_described", 12.0), (knowledge_report.LANGUAGE_MODEL, "k256", 5.0),
               ("ridge_esmc_600m", "k256", 20.0), ("ridge_onehot", "k256", 30.0)]
    predictions = pd.DataFrame([
        {"task_id": RHOMAX_TASK.replace("__k256__", f"__{arm}__"), "method": method, "row": row, "truth": 500.0 + row,
         "prediction": 500.0 + row + offset}
        for method, arm, offset in methods for row in (10, 11, 12)])

    table = knowledge_report.rhodopsin_table(entries, answers, masked, predictions).set_index("row")

    assert table.loc[10, "recovery_language_model"] == 0.5 and np.isnan(table.loc[12, "recovery_language_model"])
    assert table.loc[11, "ppll_esmc_600m"] == -0.2 and table.loc[12, "recovery_esmc_600m"] == 0.6
    assert (table["error_language_model_k0_described"] == 12.0).all() and (table["error_ridge_esmc_600m"] == 20.0).all()
    assert table.loc[11, "unusual"] == 0.0 and table.loc[10, "unusual"] == 1.0
    assert table.loc[10, "levels_recited"] == 0 and np.isnan(table.loc[12, "levels_recited"])


def test_the_error_is_grouped_by_how_many_levels_came_back_whole() -> None:
    rhodopsins = pd.DataFrame({
        "levels_recited": [0, 0, 3, 3, 3, np.nan], "error_language_model_k0_described": [10.0, 20.0, 1.0, 2.0, 9.0, 99.0],
        "error_language_model_k256": [4.0, 6.0, 3.0, 3.0, 3.0, 99.0]})

    table = knowledge_report.recited_table(rhodopsins).set_index("levels_recited")

    assert table.loc[0, "sequences"] == 2 and table.loc[3, "sequences"] == 3
    assert table.loc[0, "median_error_k0_described"] == 15.0 and table.loc[3, "median_error_k256"] == 3.0

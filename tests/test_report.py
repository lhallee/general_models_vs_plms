"""The paired comparison of series, and the prompt files of a model reached through subagents."""

from __future__ import annotations

import hashlib
import json
import numpy as np
import pandas as pd
import pytest

from pathlib import Path

from general_models_vs_plms import flip2, plan, report, subagent_arm
from general_models_vs_plms.report import Series
from general_models_vs_plms.tasks import Arm, build_task, load_layout, write_task
from test_collect import write_toy_split


def toy_predictions() -> pd.DataFrame:
    """Two splits of 80 queries. `good` tracks the truth, `noisy` tracks it loosely in two seeds, `absent` has one split."""
    generator = np.random.default_rng(5)
    rows = []
    for split in ("a/one", "b/two"):
        truth = generator.normal(size=80)
        series = {("good", "k256", 0): truth + 0.1 * generator.normal(size=80),
                  ("noisy", "k256", 0): truth + 2.0 * generator.normal(size=80),
                  ("noisy", "k256", 1): truth + 2.0 * generator.normal(size=80),
                  ("good", "k256_flipped", 0): -truth}
        if split == "a/one":
            series[("absent", "k256", 0)] = truth
        for (method, arm, seed), values in series.items():
            rows.append(pd.DataFrame({"task_id": f"{split}__{arm}__s{seed}", "split": split, "method": method, "arm": arm,
                                      "seed": seed, "truth": truth,
                                      "query_id": [f"Q{index + 1:03d}" for index in range(80)], "prediction": values}))
    return pd.concat(rows, ignore_index=True)


def test_compare_pairs_series_on_shared_queries_and_averages_seeds() -> None:
    predictions = toy_predictions()
    series = [Series("good", "good", "k256"), Series("noisy", "noisy", "k256")]

    comparison = report.compare(report.Scores(predictions), ["a/one", "b/two"], series)

    assert comparison["splits"] == ["a/one", "b/two"] and comparison["draws"].shape == (report.DRAWS, 2)
    assert comparison["mean"][0] > 0.95 > comparison["mean"][1]
    assert np.all(comparison["low"] <= comparison["mean"]) and np.all(comparison["mean"] <= comparison["high"])
    one_seed = report.compare(report.Scores(predictions[predictions["seed"] == 0]), ["a/one"], series)
    both_seeds = report.compare(report.Scores(predictions), ["a/one"], series)
    assert one_seed["mean"][0] == pytest.approx(both_seeds["mean"][0])
    assert one_seed["mean"][1] != pytest.approx(both_seeds["mean"][1])

    gap = report.difference(comparison, "good", "noisy")
    assert gap["difference"] == pytest.approx(comparison["mean"][0] - comparison["mean"][1])
    assert gap["low"] > 0 and gap["share_above_zero"] == 1.0
    pair = report.paired(report.Scores(predictions), ["a/one", "b/two"], series[0], series[1])
    assert pair["difference"] == pytest.approx(gap["difference"]) and pair["low"] == pytest.approx(gap["low"])


def test_compare_drops_a_split_that_a_series_lacks_and_applies_the_sign() -> None:
    scores = report.Scores(toy_predictions())

    with_absent = report.compare(scores, ["a/one", "b/two"], [Series("good", "good", "k256"), Series("absent", "absent", "k256")])
    flipped = report.compare(scores, ["a/one", "b/two"], [Series("raw", "good", "k256_flipped"),
                                                           Series("signed", "good", "k256_flipped", sign=-1.0)])

    assert with_absent["splits"] == ["a/one"]
    assert flipped["mean"].tolist() == pytest.approx([-1.0, 1.0])
    assert report.paired(scores, ["b/two"], Series("good", "good", "k256"), Series("absent", "absent", "k256")) is None


def test_compare_on_the_first_queries_keeps_only_those() -> None:
    predictions = toy_predictions()
    late = predictions["query_id"] > "Q040"
    predictions.loc[late & (predictions["method"] == "noisy"), "prediction"] = 0.0

    comparison = report.compare(report.Scores(predictions), ["a/one"],
                                [Series("good", "good", "k256"), Series("noisy", "noisy", "k256")], first_queries=40)
    first = predictions[~late & (predictions["split"] == "a/one") & (predictions["method"] == "good")
                        & (predictions["arm"] == "k256")]

    assert comparison["mean"][0] == pytest.approx(report.spearman(first["truth"].to_numpy(), first["prediction"].to_numpy()))


def test_partial_spearman_is_zero_for_a_copy_of_the_control_and_high_for_new_signal() -> None:
    generator = np.random.default_rng(11)
    first, second = generator.normal(size=(2, 400))
    truth = first + second

    copy = report.partial_spearman(truth[None, :], first[None, :] * 2.0 + 1.0, first[None, :])
    fresh = report.partial_spearman(truth[None, :], second[None, :], first[None, :])

    assert copy[0] == pytest.approx(0.0, abs=1e-9)
    assert fresh[0] > 0.9


def test_rank_average_of_two_noisy_methods_beats_each() -> None:
    predictions = toy_predictions()
    blend = report.rank_average(predictions, "noisy", "good", "noisy+good")
    scores = report.Scores(pd.concat([predictions, blend], ignore_index=True))

    comparison = report.compare(scores, ["a/one", "b/two"], [Series("blend", "noisy+good", "k256"), Series("noisy", "noisy", "k256")])

    # `good` ran one seed, so the blend exists for that seed of each split alone.
    assert set(blend["arm"]) == {"k256"} and set(blend["seed"]) == {0} and len(blend) == 2 * 80
    assert comparison["mean"][0] > comparison["mean"][1]


def test_answered_only_scores_both_sides_on_the_queries_a_model_answered() -> None:
    predictions = toy_predictions().assign(answered=True)
    # `noisy` refused the second half of one split, and its fill value there is a constant.
    refused = (predictions["method"] == "noisy") & (predictions["split"] == "a/one") & (predictions["query_id"] > "Q040")
    predictions.loc[refused, ["prediction", "answered"]] = [0.0, False]
    # On the other split it answered too few queries to score.
    few = (predictions["method"] == "noisy") & (predictions["split"] == "b/two") & (predictions["query_id"] > "Q010")
    predictions.loc[few, ["prediction", "answered"]] = [0.0, False]

    rows = report.answered_only(predictions, ["a/one", "b/two"], Series("noisy", "noisy", "k256"), [Series("good", "good", "k256")])
    nothing = report.answered_only(predictions, ["a/one", "b/two"], Series("good", "good", "k256"), [Series("noisy", "noisy", "k256")])

    answered = predictions[(predictions["split"] == "a/one") & (predictions["seed"] == 0) & (predictions["arm"] == "k256")
                           & (predictions["query_id"] <= "Q040")]
    own = answered[answered["method"] == "noisy"]
    assert len(rows) == 1 and rows[0]["splits"] == 1 and rows[0]["splits_of"] == 2
    assert rows[0]["queries"] == 40 and rows[0]["queries_asked"] == 160
    assert rows[0]["left_mean"] == pytest.approx(report.spearman(own["truth"].to_numpy(), own["prediction"].to_numpy()))
    assert rows[0]["right_mean"] > 0.95
    assert nothing == []


def test_means_and_split_differences_cover_the_splits_a_series_ran() -> None:
    scores = report.Scores(toy_predictions())
    groups = {"all": ["a/one", "b/two"]}

    means = report.means_table(scores, groups).set_index(["method", "arm"])
    gaps = report.split_differences(scores, groups["all"], Series("good", "good", "k256"), Series("absent", "absent", "k256"))

    assert means.loc[("absent", "k256"), "splits"] == 1 and means.loc[("good", "k256"), "splits"] == 2
    assert means.loc[("good", "k256"), "low"] <= means.loc[("good", "k256"), "mean"] <= means.loc[("good", "k256"), "high"]
    assert [row["split"] for row in gaps] == ["a/one"]
    assert gaps[0]["difference"] == pytest.approx(gaps[0]["left_spearman"] - gaps[0]["right_spearman"])
    assert gaps[0]["low"] <= gaps[0]["difference"] <= gaps[0]["high"]


def test_refusal_table_counts_refused_prompts_by_split() -> None:
    calls = pd.DataFrame({
        "record_id": ["r1", "r2", "r3", "r4"], "model_requested": "m", "model_served": "m", "effort": "low",
        "arm": "k256", "split": ["a/one", "a/one", "b/two", "b/two"],
        "stop_reason": ["refusal", "end_turn", "end_turn", "end_turn"], "stop_category": ["bio", None, None, None],
        "answered": [0, 100, 100, 0]})

    table = report.refusal_table(calls).set_index("split")

    assert table.loc["a/one", ["prompts", "refused", "nothing_read"]].tolist() == [2, 1, 0]
    assert table.loc["a/one", "categories"] == "bio"
    assert table.loc["b/two", ["prompts", "refused", "nothing_read"]].tolist() == [2, 0, 1]


def test_baseline_questions_pair_a_control_with_the_model_it_was_made_from() -> None:
    base = toy_predictions()
    base = base[(base["method"] == "good") & (base["arm"] == "k256")]
    methods = ("ridge_onehot", "ridge_esmc_300m", "ridge_esmc_600m", "zero_shot_esmc_300m", "ridge_esmc_300m_permuted",
               "ridge_composition")
    scores = report.Scores(pd.concat([base.assign(method=method) for method in methods], ignore_index=True))

    pairs = {(left.method, right.method) for _, left, right in report.baseline_questions(scores)}

    assert pairs == {("ridge_esmc_300m", "ridge_esmc_600m"), ("ridge_esmc_300m", "zero_shot_esmc_300m"),
                     ("ridge_esmc_300m", "ridge_esmc_300m_permuted"), ("ridge_onehot", "ridge_esmc_300m_permuted"),
                     ("ridge_esmc_300m", "ridge_composition"), ("ridge_esmc_600m", "ridge_composition"),
                     ("ridge_onehot", "ridge_composition"), ("ridge_esmc_300m", "ridge_onehot"), ("ridge_esmc_600m", "ridge_onehot")}


def test_tuned_questions_set_each_tuned_arm_against_its_controls_the_baselines_and_the_language_model() -> None:
    generator = np.random.default_rng(8)
    truth = generator.normal(size=40)
    methods = [f"progen2_small_{arm}" for arm in plan.PREFERENCE_ARMS] + ["ridge_esmc_600m", "ridge_onehot", "claude-x@low"]
    predictions = pd.concat([pd.DataFrame({
        "task_id": "a/one__k256__s0", "split": "a/one", "method": method, "arm": "k256", "seed": 0, "truth": truth,
        "query_id": [f"Q{index + 1:03d}" for index in range(40)], "prediction": truth + generator.normal(size=40)}) for method in methods])
    scores = report.Scores(predictions)

    asked = {(left.method, right.method) for _, left, right in report.tuned_questions(scores, ["claude-x@low"], "ridge_esmc_600m")}

    assert report.tuned_models(scores) == ["progen2_small"]
    for pair in (("progen2_small_dpo", "progen2_small_zero"), ("progen2_small_dpo", "progen2_small_ridge"),
                 ("progen2_small_dpo", "progen2_small_bt"), ("progen2_small_dpo", "progen2_small_dpo_shuffled"),
                 ("progen2_small_dpo_shuffled", "progen2_small_zero"),
                 ("progen2_small_dpo", "progen2_small_dpo_seed2"), ("progen2_small_dpo", "ridge_esmc_600m"),
                 ("progen2_small_zero", "ridge_onehot"), ("progen2_small_dpo", "claude-x@low"), ("progen2_small_bt", "claude-x@low")):
        assert pair in asked
    # A reference that was not scored and a model that was not tuned ask nothing.
    assert not any("zero_shot_esmc_6b" in pair or "progen2_base_dpo" in pair for pair in asked)
    rows = report.paired_rows(scores, ["a/one"], report.tuned_questions(scores, ["claude-x@low"], "ridge_esmc_600m"))
    assert len(rows) == len(asked) and all(row["splits"] == 1 for row in rows)


def tuned_toy(methods: list[tuple[str, str]], queries: int = 40) -> pd.DataFrame:
    """Noisy predictions of one split's queries for each method and arm, the queries shared by every arm."""
    generator = np.random.default_rng(9)
    truth = generator.normal(size=queries)
    return pd.concat([pd.DataFrame({
        "task_id": f"a/one__{arm}__s0", "split": "a/one", "method": method, "arm": arm, "seed": 0, "truth": truth,
        "query_id": [f"Q{index + 1:03d}" for index in range(queries)],
        "prediction": truth + generator.normal(size=queries)}) for method, arm in methods], ignore_index=True)


def test_tuned_blends_average_the_ranks_of_the_tuned_arm_and_each_other_method_that_ran() -> None:
    predictions = tuned_toy([("progen2_small_dpo", "k256"), ("ridge_esmc_600m", "k256"), ("ridge_esmc_600m", "klarge")])

    blends = report.tuned_blends(predictions, "progen2_small", ["ridge_esmc_600m", "zero_shot_esmc_600m"])

    assert len(blends) == 1  # the method that did not run makes no blend
    blend = blends[0].sort_values("query_id")
    assert set(blend["method"]) == {"progen2_small_dpo+ridge_esmc_600m"} and set(blend["arm"]) == {"k256"} and len(blend) == 40
    parts = predictions[predictions["arm"] == "k256"].pivot(index="query_id", columns="method", values="prediction").sort_index()
    np.testing.assert_allclose(blend["prediction"].to_numpy(), parts.rank().mean(axis=1).to_numpy())


def test_tuned_questions_set_a_variant_on_its_own_tasks_against_the_primary_arm_and_a_blend_against_its_parts() -> None:
    predictions = tuned_toy([
        ("progen2_small_dpo", "k256"), ("progen2_small_zero", "k256"), ("ridge_esmc_600m", "k256"),
        ("progen2_small_rows_dpo", "klarge"), ("progen2_small_rows_zero", "klarge"), ("ridge_esmc_600m", "klarge"),
        ("progen2_small_slow_dpo", "k256"), ("claude-x@low", "k256")])
    scores = report.Scores(predictions)
    for blend in report.tuned_blends(predictions, "progen2_small", ["ridge_esmc_600m", "claude-x@low"]):
        scores.add(blend)

    asked = {(left.method, left.arm, right.method, right.arm)
             for _, left, right in report.tuned_questions(scores, ["claude-x@low"], "ridge_esmc_600m")}

    assert report.tuned_models(scores) == ["progen2_small", "progen2_small_slow", "progen2_small_rows"]
    assert report.tuned_arm_of("progen2_small_rows") == "klarge" and report.tuned_arm_of("progen2_small_slow") == "k256"
    for question in (
            # a variant against the primary arm, each on the tasks it ran on
            ("progen2_small_rows_dpo", "klarge", "progen2_small_dpo", "k256"),
            ("progen2_small_rows_zero", "klarge", "progen2_small_zero", "k256"),
            ("progen2_small_slow_dpo", "k256", "progen2_small_dpo", "k256"),
            # a variant that scored no released model of its own is still set against the first model's
            ("progen2_small_slow_dpo", "k256", "progen2_small_zero", "k256"),
            # a variant against the baseline fitted on the rows it was tuned on
            ("progen2_small_rows_dpo", "klarge", "ridge_esmc_600m", "klarge"),
            ("progen2_small_slow_dpo", "k256", "ridge_esmc_600m", "k256"),
            # a language model answered the k256 prompt only
            ("progen2_small_rows_dpo", "klarge", "claude-x@low", "k256"),
            # a blend against each of its two parts
            ("progen2_small_dpo+ridge_esmc_600m", "k256", "progen2_small_dpo", "k256"),
            ("progen2_small_dpo+ridge_esmc_600m", "k256", "ridge_esmc_600m", "k256"),
            ("progen2_small_dpo+claude-x@low", "k256", "claude-x@low", "k256")):
        assert question in asked
    # The primary arm is not set against itself, and no variant is blended.
    assert ("progen2_small_dpo", "k256", "progen2_small_dpo", "k256") not in asked
    assert not any(left.startswith("progen2_small_rows_dpo+") for left, *_ in asked)
    rows = report.paired_rows(scores, ["a/one"], report.tuned_questions(scores, ["claude-x@low"], "ridge_esmc_600m"))
    assert len(rows) == len(asked) and all(row["splits"] == 1 for row in rows)


def test_tuned_questions_set_a_run_against_the_run_it_changes_one_thing_of() -> None:
    predictions = tuned_toy([("progen2_small_dpo", "k256"), ("progen2_small_long20_dpo", "k256"),
                             ("progen2_small_nll20_dpo", "k256"), ("ridge_esmc_600m", "k256")])
    scores = report.Scores(predictions)

    asked = {(question, left.method, right.method) for question, left, right in report.tuned_questions(scores, [], "ridge_esmc_600m")}

    counterpart = "a run of the tuning against the run it differs from in one thing"
    assert (counterpart, "progen2_small_nll20_dpo", "progen2_small_long20_dpo") in asked
    # The fixed recipe itself differs from the primary arm, which every variant is already set against.
    assert not any(question == counterpart and left == "progen2_small_long20_dpo" for question, left, _ in asked)
    assert any(left == "progen2_small_long20_dpo" and right == "progen2_small_dpo" for _, left, right in asked)


def test_tuned_orders_average_the_ranks_of_every_batch_order_and_ask_nothing_until_all_ran() -> None:
    orders = [("progen2_small_dpo", "k256"), ("progen2_small_dpo_seed1", "k256"), ("progen2_small_dpo_seed2", "k256")]
    predictions = tuned_toy([*orders, ("ridge_esmc_600m", "k256"), ("claude-x@low", "k256")])

    averaged = report.tuned_orders(predictions, "progen2_small").sort_values("query_id")

    assert set(averaged["method"]) == {"progen2_small_dpo+orders"} and len(averaged) == 40
    wide = predictions[predictions["method"].isin([method for method, _ in orders])].pivot(index="query_id", columns="method", values="prediction")
    np.testing.assert_allclose(averaged["prediction"].to_numpy(), wide.sort_index().rank().mean(axis=1).to_numpy())
    assert report.tuned_orders(predictions[predictions["method"] != "progen2_small_dpo_seed2"], "progen2_small").empty
    scores = report.Scores(predictions)
    scores.add(averaged)
    asked = {(left.method, right.method) for _, left, right in report.tuned_questions(scores, ["claude-x@low"], "ridge_esmc_600m")}
    assert {("progen2_small_dpo+orders", "progen2_small_dpo"), ("progen2_small_dpo+orders", "ridge_esmc_600m"),
            ("progen2_small_dpo+orders", "claude-x@low")} <= asked


def test_inside_tables_set_a_tuned_model_on_its_held_out_rows_against_the_baselines_fit_on_its_tuning_rows() -> None:
    splits = ["a/one", "a/two", "fgfr2/random"]
    inside = [{"task_id": f"{split}__k256__s0", "split": split, "fit_rows": 205, "held_out_rows": 51,
               "held_out_spearman": {"ridge_onehot": 0.5, "ridge_esmc_600m": 0.6 + 0.1 * index}} for index, split in enumerate(splits)]

    def record(split: str, lift: float, **more: object) -> dict[str, object]:
        curves = [{"learning_rate": 1e-5, "beta": 2.0, "held_out_spearman": [0.30 + lift + 0.01 * epoch for epoch in range(1, 21)]},
                  {"learning_rate": 1e-4, "beta": 2.0, "held_out_spearman": [0.9] + [0.1] * 19}]
        return {"model": ["hugohrban/progen2-small", "43237a0b"], "split": split, "fit_rows": 205, "held_out_rows": 51,
                "held_out_spearman_untuned": 0.2, "failed": {},
                "arms": {"dpo": {"recipe": {"loss": "dpo", "learning_rate": 1e-4, "beta": 2.0}, "epoch": 1, "curves": curves}}, **more}

    records = [record(split, 0.1 * index) for index, split in enumerate(splits)]
    # A run on another arm of the tasks holds out other rows and has no place here.
    records.append(record("a/one", 0.0, arm="klarge", method_prefix="progen2_small_rows"))

    table = report.inside_table(inside, records)
    differences = report.inside_differences(table, {"flip2": ["a/one", "a/two"], "postcutoff": ["fgfr2/random"]})

    assert set(table["method"]) == {"ridge_onehot", "ridge_esmc_600m", "progen2_small_zero", "progen2_small_dpo"}
    tuned = table[table["method"] == "progen2_small_dpo"]
    assert len(tuned) == 3 * 2 * 20 and tuned["chosen"].sum() == 3
    assert tuned[tuned["chosen"]]["held_out_spearman"].tolist() == [0.9, 0.9, 0.9]
    found = differences.set_index(["group", "tuned", "baseline"])
    # After epoch 5 of the fixed setting the two splits score 0.35 and 0.45, against a ridge at 0.6 and 0.7.
    row = found.loc[("flip2", "epoch 5", "ridge_esmc_600m")]
    assert (row["splits"], row["ahead"]) == (2, 0)
    assert (row["tuned_mean"], row["baseline_mean"], row["difference"]) == pytest.approx((0.40, 0.65, -0.25))
    assert row["low"] <= row["difference"] <= row["high"]
    assert found.loc[("flip2", "chosen", "ridge_onehot"), "ahead"] == 2
    assert found.loc[("flip2", "untuned", "ridge_onehot"), "difference"] == pytest.approx(-0.3)
    assert found.loc[("postcutoff", "epoch 20", "ridge_esmc_600m"), "difference"] == pytest.approx(0.7 - 0.8)
    assert set(differences["model"]) == {"progen2_small"}


def test_tuned_name_is_the_variant_a_record_ran_under_and_otherwise_its_checkpoint() -> None:
    assert report.tuned_name({"model": ["hugohrban/progen2-small", "43237a0b"]}) == "progen2_small"
    assert report.tuned_name({"model": ["hugohrban/progen2-small", "43237a0b"], "method_prefix": "progen2_small_rows"}) == "progen2_small_rows"


def test_tuning_table_holds_one_row_an_arm_with_its_chosen_recipe_or_its_failure() -> None:
    record = {
        "model": ["hugohrban/progen2-small", "43237a0b"], "split": "a/one", "fit_rows": 205, "held_out_rows": 51,
        "held_out_spearman_untuned": 0.1, "failed": {"bt": "the loss is nan in epoch 2"},
        "arms": {"dpo": {"recipe": {"loss": "dpo", "learning_rate": 1e-4, "beta": 2.0}, "epoch": 7, "held_out_spearman": 0.5,
                         "curves": [], "seconds": 60.0}},
    }

    table = report.tuning_table([record]).set_index("arm")

    assert list(table.index) == ["dpo", "bt"] and set(table["model"]) == {"progen2_small"}
    chosen = table.loc["dpo"]
    assert (chosen["learning_rate"], chosen["beta"], chosen["epoch"], chosen["held_out_spearman"]) == (1e-4, 2.0, 7, 0.5)
    assert chosen["failed"] == "" and chosen["held_out_spearman_untuned"] == 0.1 and chosen["fit_rows"] == 205
    assert table.loc["bt", "failed"] == "the loss is nan in epoch 2" and np.isnan(table.loc["bt", "epoch"])


def test_trace_table_scores_every_epoch_on_the_traced_queries_and_marks_the_chosen_one() -> None:
    predictions = pd.DataFrame({"task_id": "t", "method": ["m"] * 4 + ["progen2_small_zero"] * 4, "query_id": ["Q0", "Q1", "Q2", "Q3"] * 2,
                                "truth": [1.0, 2.0, 3.0, 4.0] * 2, "prediction": [0.0] * 4 + [-2.0, -3.0, -4.0, -9.0]})
    rising, falling = [0.1, 0.2, 0.3], [0.3, 0.2, 0.1]
    record = {
        "model": ["hugohrban/progen2-small", "43237a0b"], "task_id": "t", "split": "a/one", "traced_queries": 3,
        "arms": {"dpo": {"recipe": {"loss": "dpo", "learning_rate": 1e-5, "beta": 2.0}, "epoch": 2, "held_out_spearman": 0.6, "curves": [
            {"loss": "dpo", "learning_rate": 1e-5, "beta": 0.1, "held_out_spearman": [0.2, 0.3], "query_scores": [rising, rising]},
            {"loss": "dpo", "learning_rate": 1e-5, "beta": 2.0, "held_out_spearman": [0.4, 0.6], "query_scores": [rising, falling]},
        ]}},
    }

    table = report.trace_table([record], predictions)

    assert list(table["epoch"]) == [1, 2, 1, 2] and set(table["queries"]) == {3} and set(table["model"]) == {"progen2_small"}
    # Only the first three queries are scored: the fourth was not traced.
    assert list(table["spearman"]) == [1.0, 1.0, 1.0, -1.0]
    assert list(table["chosen"]) == [False, False, False, True]
    assert list(table["held_out_spearman"]) == [0.2, 0.3, 0.4, 0.6]
    # The mean score of the traced queries after each epoch, beside the same three queries under the untuned model.
    assert list(table["mean_score"]) == pytest.approx([0.2] * 4) and set(table["untuned_mean_score"]) == {-3.0}
    alone = report.trace_table([record], predictions[predictions["method"] == "m"])
    assert alone["untuned_mean_score"].isna().all() and list(alone["spearman"]) == [1.0, 1.0, 1.0, -1.0]


def test_match_groups_set_apart_the_splits_whose_values_repeat() -> None:
    frame = pd.DataFrame({"split": ["a/one"] * 6 + ["a/two"] * 6, "query_id": [f"Q{index}" for index in range(6)] * 2,
                          "truth": [0.11, 0.12, 0.13, 0.14, 0.15, 0.16] + [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]})

    groups = report.match_groups(frame, {"a/one": 2, "a/two": 2}, {"flip2": ["a/one", "a/two"]})

    assert groups == {"flip2": ["a/one", "a/two"], "flip2_continuous": ["a/one"], "flip2_discrete": ["a/two"]}
    # The probe splits, where a route that asked one prompt a task ran its controls, are divided the same way.
    probed = report.match_groups(frame, {"a/one": 2, "a/two": 2}, {"flip2": ["a/one", "a/two"], "probe": ["a/two"]})
    assert probed["probe_continuous"] == [] and probed["probe_discrete"] == ["a/two"]


def test_first_rows_keep_the_first_queries_of_each_split() -> None:
    frame = pd.DataFrame({"split": ["a/one"] * 6 + ["a/two"] * 4, "method": ["m", "n"] * 5,
                          "query_id": ["Q3", "Q3", "Q1", "Q1", "Q2", "Q2", "Q9", "Q9", "Q8", "Q8"]})

    kept = report.first_rows(frame, 2)

    assert sorted(zip(kept["split"], kept["query_id"], kept["method"], strict=True)) == [
        ("a/one", "Q1", "m"), ("a/one", "Q1", "n"), ("a/one", "Q2", "m"), ("a/one", "Q2", "n"),
        ("a/two", "Q8", "m"), ("a/two", "Q8", "n"), ("a/two", "Q9", "m"), ("a/two", "Q9", "n")]
    # The rest of each split is every query the first rows leave: one of the first split's, none of the second's.
    rest = report.rest_rows(frame, 2)
    assert sorted(zip(rest["split"], rest["query_id"], rest["method"], strict=True)) == [("a/one", "Q3", "m"), ("a/one", "Q3", "n")]
    assert len(kept) + len(rest) == len(frame)


def test_match_tables_count_answers_equal_to_the_published_value_beside_a_random_deal() -> None:
    generator = np.random.default_rng(3)
    truth = generator.permutation(60) / 100.0  # 60 distinct values at two decimals
    recalled = truth + generator.uniform(-0.004, 0.004, size=60)
    recalled[:10] += 0.05  # ten answers miss by five units of the last digit: near, not exact
    guessed = generator.permutation(truth)
    frame = pd.DataFrame({"split": ["a/one"] * 60 + ["a/two"] * 60 + ["a/one"] * 60, "arm": "k0_recall",
                          "truth": np.tile(truth, 3), "method": ["recalls"] * 120 + ["guesses"] * 60,
                          "answered": [True] * 179 + [False], "prediction": np.concatenate([recalled, recalled, guessed])})

    per_split, pooled = report.match_tables(frame, {"a/one": 2, "a/two": 2}, {"flip2": ["a/one", "a/two"]}, draws=500)
    pooled = pooled.set_index("method")

    one = per_split[(per_split["method"] == "recalls") & (per_split["split"] == "a/one")].iloc[0]
    assert one[["answered", "exact", "near"]].tolist() == [60, 50, 60]
    # Dealt at random, one answer in 60 lands on its own value, and one in about three lies within ten units of it.
    assert one["exact_dealt"] == pytest.approx(1.0, abs=0.3) and 12 < one["near_dealt"] < 25
    assert pooled.loc["recalls", ["splits", "answered", "exact", "near"]].tolist() == [2, 120, 100, 120]
    assert pooled.loc["recalls", "exact_p"] == pytest.approx(1 / 501) and pooled.loc["recalls", "near_p"] == pytest.approx(1 / 501)
    # A deal of the true values is one more random deal: about one exact match, and no evidence of recall.
    assert pooled.loc["guesses", "answered"] == 59 and pooled.loc["guesses", "exact"] <= 4
    assert pooled.loc["guesses", "exact_p"] > 0.01


def test_effort_table_reads_the_first_prompt_of_each_task() -> None:
    calls = pd.DataFrame({
        "record_id": ["t1__k256__s0__c0", "t1__k256__s0__c1", "t2__k256__s0__c0", "t1__klarge__s0__c0"],
        "arm": ["k256", "k256", "k256", "klarge"], "seed": 0, "model_requested": "claude-mythos-5-2",
        "split": ["a/one", "a/one", "b/two", "a/one"],
        "model_served": "claude-mythos-5-2", "effort": "harness-max", "stop_reason": ["end_turn", "end_turn", "refusal", "end_turn"],
        "answered": [100, 100, 0, 100], "queries": 100, "output_tokens": [900, 5000, 100, 7000],
        "thinking_tokens": [600, 4000, 0, 6000], "elapsed_seconds": [40.0, 90.0, 4.0, 200.0]})

    table = report.effort_table(calls, {"both": ["a/one", "b/two"], "one": ["a/one"]}).set_index(["group", "method"])

    row = table.loc[("both", "claude-mythos-5-2@harness-max")]
    assert row[["prompts", "refused", "answered", "queries"]].tolist() == [2, 1, 100, 200]
    assert row["median_thinking_tokens"] == 300 and row["median_seconds"] == 22.0
    one = table.loc[("one", "claude-mythos-5-2@harness-max")]
    assert one[["prompts", "refused", "answered"]].tolist() == [1, 0, 100] and one["median_thinking_tokens"] == 600


def test_validation_table_joins_each_prompt_s_score_with_what_it_cost(tmp_path: Path) -> None:
    pd.DataFrame({"method": ["m@harness-low", "m@harness-high"], "task_id": ["t1", "t1"], "split": "a/one", "arm": "k256_sequence",
                  "queries": 100, "spearman": [0.5, 0.6]}).to_csv(tmp_path / "metrics.csv", index=False)
    pd.DataFrame({"record_id": ["t1__c0", "t1__c0"], "task_id": ["t1", "t1"], "model_requested": "m", "model_served": "m",
                  "effort": ["harness-low", "harness-high"], "stop_reason": "end_turn", "answered": 100,
                  "output_tokens": [800, 9000], "thinking_tokens": [300, 8000], "elapsed_seconds": [20.0, 95.0]}).to_csv(
                      tmp_path / "calls.csv", index=False)

    table = report.validation_table([tmp_path]).set_index("method")

    assert table.loc["m@harness-high", ["spearman", "thinking_tokens", "seconds"]].tolist() == [0.6, 8000, 95.0]
    assert table.loc["m@harness-low", "answered"] == 100


def test_partly_asked_names_a_series_with_fewer_prompts_than_its_tasks_hold() -> None:
    metrics = pd.DataFrame({"task_id": ["t1", "t1", "t2"], "method": ["m@low", "m@harness", "m@harness"],
                            "arm": ["k256", "k256", "klarge"], "queries": [300, 300, 300], "answered_fraction": [1.0, 1.0, 0.33]})
    calls = pd.DataFrame({
        "record_id": [f"r{index}" for index in range(7)], "model_requested": "m", "model_served": "m",
        "effort": ["low"] * 3 + ["harness"] * 4, "arm": ["k256"] * 6 + ["klarge"],
        "stop_reason": "end_turn", "answered": 100})

    assert report.partly_asked(calls, metrics) == {("m@harness", "klarge")}


def test_audit_reads_a_subagents_model_and_flags_any_tool_beyond_its_own_files(tmp_path: Path) -> None:
    prompts, transcripts = tmp_path / "w", tmp_path / "transcripts"
    transcripts.mkdir()
    (prompts / "p001").mkdir(parents=True)
    (prompts / "p001" / "answer.txt").write_text("Q001\t1\n", encoding="utf-8")
    entries = [{"folder": name, "task_id": f"task_{name}", "chunk": 0, "record_id": f"task_{name}__c0", "prompt_sha256": ""}
               for name in ("p001", "p002")]
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"prompts": str(prompts), "entries": entries}), encoding="utf-8")

    def write_transcript(agent: str, folder: str, calls: list[tuple[str, dict[str, str]]]) -> None:
        lines = [{"type": "user", "timestamp": "2026-11-21T21:30:00Z",
                  "message": {"role": "user", "content": f"Read the file C:\\x\\w\\{folder}\\prompt.txt in full."}}]
        # A reply's blocks are separate entries that repeat the reply's usage, so tokens are counted once per reply.
        usage = {"output_tokens": 800, "output_tokens_details": {"thinking_tokens": 600}}
        lines += [{"type": "assistant", "effort": "high", "timestamp": "2026-11-21T21:30:05Z", "message": {
            "id": "reply_1", "model": "claude-mythos-5-2", "usage": usage, "stop_reason": "tool_use",
            "content": [{"type": "tool_use", "name": name, "input": inputs}]}} for name, inputs in calls]
        # The final reply comes back through the harness's hand-back call, which is not a tool the subagent used.
        lines.append({"type": "assistant", "effort": "high", "timestamp": "2026-11-21T21:31:00Z", "message": {
            "id": "reply_2", "model": "claude-mythos-5-2", "usage": {"output_tokens": 100}, "stop_reason": "tool_use",
            "content": [{"type": "tool_use", "name": "SubagentHandback", "input": {"message": "done"}}]}})
        (transcripts / f"agent-{agent}.jsonl").write_text("\n".join(json.dumps(line) for line in lines), encoding="utf-8")

    write_transcript("a1", "p001", [("Read", {"file_path": "C:\\x\\w\\p001\\prompt.txt"}),
                                    ("Write", {"file_path": "C:/x/w/p001/answer.txt"})])
    write_transcript("a2", "p002", [("Read", {"file_path": "C:\\x\\w\\p002\\prompt.txt"}), ("Bash", {"command": "python fit.py"}),
                                    ("Read", {"file_path": "C:\\x\\w\\p001\\prompt.txt"})])
    (transcripts / "agent-a3.jsonl").write_text(json.dumps({"type": "user", "message": {"role": "user", "content": "Search the web."}}),
                                                encoding="utf-8")

    subagents = subagent_arm.read_transcripts(transcripts)
    assert [found["agent"] for found in subagents] == ["agent-a1", "agent-a2"]
    rows = subagent_arm.audit(index, subagents)

    assert [row["prompt"] for row in rows] == ["w/p001", "w/p002"]
    assert rows[0]["models"] == "claude-mythos-5-2" and rows[0]["tool_calls"] == "Read,Write"
    assert rows[0]["breaches"] == "" and rows[0]["wrote_answer"] and rows[0]["answer_file"] and rows[0]["last_text"] == "done"
    assert rows[0]["efforts"] == "high" and rows[0]["output_tokens"] == 900 and rows[0]["thinking_tokens"] == 600
    assert rows[1]["breaches"] == "Bash; Read C:\\x\\w\\p001\\prompt.txt" and not rows[1]["answer_file"]

    # With the audit, an answer stands only if its subagent kept the protocol, and the record carries the effort.
    (prompts / "p002").mkdir()
    (prompts / "p001" / "prompt.txt").write_text("first", encoding="utf-8")
    (prompts / "p002" / "prompt.txt").write_text("second", encoding="utf-8")
    (prompts / "p002" / "answer.txt").write_text("Q001\t2\n", encoding="utf-8")
    for entry, text in zip(entries, ("first", "second"), strict=True):
        entry["prompt_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    index.write_text(json.dumps({"prompts": str(prompts), "entries": entries}), encoding="utf-8")
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps(subagent_arm.audit(index, subagents)), encoding="utf-8")

    assert subagent_arm.ingest(index, tmp_path / "responses", "claude-mythos-5-2", "high", audit_path) == (1, 1)
    folder = tmp_path / "responses" / "claude-mythos-5-2__harness-high"
    kept, dropped = (json.loads((folder / f"task_{name}__c0.json").read_text(encoding="utf-8")) for name in ("p001", "p002"))
    assert kept["text"] == "Q001\t1\n" and kept["effort"] == "harness-high" and kept["usage"]["output_tokens"] == 900
    assert dropped["text"] == "" and dropped["stop_reason"] == "protocol_breach"

    # A set that is still being asked: a prompt no subagent was given yet gets no record.
    (prompts / "p003").mkdir()
    (prompts / "p003" / "prompt.txt").write_text("third", encoding="utf-8")
    entries.append({"folder": "p003", "task_id": "task_p003", "chunk": 0, "record_id": "task_p003__c0",
                    "prompt_sha256": hashlib.sha256(b"third").hexdigest()})
    index.write_text(json.dumps({"prompts": str(prompts), "entries": entries}), encoding="utf-8")
    assert subagent_arm.ingest(index, tmp_path / "early", "claude-mythos-5-2", "high", audit_path, asked_only=True) == (1, 1)
    assert not (tmp_path / "early" / "claude-mythos-5-2__harness-high" / "task_p003__c0.json").exists()
    assert subagent_arm.ingest(index, tmp_path / "late", "claude-mythos-5-2", "high", audit_path) == (1, 2)

    # A subagent stopped by a failed request, as on a usage limit, leaves no record: the model was never heard.
    stopped = [{"type": "user", "timestamp": "2026-11-21T21:40:00Z",
                "message": {"role": "user", "content": "Read the file C:\\x\\w\\p003\\prompt.txt in full."}},
               {"type": "assistant", "timestamp": "2026-11-21T21:40:02Z", "message": {
                   "id": "error_1", "model": "<synthetic>", "content": [{"type": "text", "text": "You've hit your limit"}]}}]
    (transcripts / "agent-a4.jsonl").write_text("\n".join(json.dumps(line) for line in stopped), encoding="utf-8")
    rows = subagent_arm.audit(index, subagent_arm.read_transcripts(transcripts))
    assert rows[-1]["prompt"] == "w/p003" and rows[-1]["api_error"] == "You've hit your limit" and rows[-1]["models"] == ""
    audit_path.write_text(json.dumps(rows), encoding="utf-8")
    assert subagent_arm.ingest(index, tmp_path / "stopped", "claude-mythos-5-2", "high", audit_path) == (1, 1)
    assert not (tmp_path / "stopped" / "claude-mythos-5-2__harness-high" / "task_p003__c0.json").exists()

    # So does a subagent that is still at work: it has read its prompt and has not handed back.
    at_work = [stopped[0], {"type": "assistant", "effort": "high", "timestamp": "2026-11-21T21:50:02Z", "message": {
        "id": "reply_1", "model": "claude-mythos-5-2", "usage": {"output_tokens": 50}, "stop_reason": "tool_use",
        "content": [{"type": "tool_use", "name": "Read", "input": {"file_path": "C:\\x\\w\\p003\\prompt.txt"}}]}}]
    (transcripts / "agent-a4.jsonl").write_text("\n".join(json.dumps(line) for line in at_work), encoding="utf-8")
    rows = subagent_arm.audit(index, subagent_arm.read_transcripts(transcripts))
    assert rows[-1]["prompt"] == "w/p003" and not rows[-1]["handed_back"] and rows[0]["handed_back"]
    audit_path.write_text(json.dumps(rows), encoding="utf-8")
    assert subagent_arm.ingest(index, tmp_path / "at_work", "claude-mythos-5-2", "high", audit_path) == (1, 1)
    assert not (tmp_path / "at_work" / "claude-mythos-5-2__harness-high" / "task_p003__c0.json").exists()


def test_subagent_prompts_hide_the_task_and_answers_become_records(tmp_path: Path) -> None:
    data_folder, tasks, prompts, responses = (tmp_path / name for name in ("data", "tasks", "prompts", "responses"))
    write_toy_split(data_folder)
    split_info = plan.split_infos(("amylase/by_mutation",))[0]
    frame = flip2.load_split(flip2.split_path(data_folder, split_info))
    layout = load_layout(split_info, frame)
    for arm in (Arm("k256", 64), Arm("k256_cipher", 64, cipher=True), Arm("k16", 16)):
        write_task(build_task(split_info, frame, layout, arm, 0, "test", 120), tasks)

    entries = subagent_arm.export(tasks, prompts, tmp_path / "private" / "index.json", {"k256", "k256_cipher"}, None, {0}, chunks=1)

    assert [entry["folder"] for entry in entries] == ["p001", "p002"]
    assert sorted(path.name for path in prompts.iterdir()) == ["p001", "p002"]
    assert [path.name for path in (prompts / "p001").iterdir()] == ["prompt.txt"]
    assert "amylase" not in (prompts / "p001" / "prompt.txt").read_text(encoding="utf-8")

    (prompts / "p001" / "answer.txt").write_text("Q001\t0.5\nQ002\t-1.25\n", encoding="utf-8")
    assert subagent_arm.ingest(tmp_path / "private" / "index.json", responses, "claude-mythos-5-2") == (1, 1)
    record = json.loads((responses / "claude-mythos-5-2__harness" / f"{entries[0]['record_id']}.json").read_text(encoding="utf-8"))
    assert record["text"].startswith("Q001\t0.5") and record["effort"] == "harness" and record["cost_usd"] == 0.0
    # A prompt that was tried and has no answer file still gets a record, so it scores as unanswered.
    unanswered = json.loads((responses / "claude-mythos-5-2__harness" / f"{entries[1]['record_id']}.json").read_text(encoding="utf-8"))
    assert unanswered["text"] == "" and unanswered["stop_reason"] == "no_answer"

    (prompts / "p002" / "prompt.txt").write_text("changed", encoding="utf-8")
    (prompts / "p002" / "answer.txt").write_text("Q001\t1\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="changed after export"):
        subagent_arm.ingest(tmp_path / "private" / "index.json", responses, "claude-mythos-5-2")

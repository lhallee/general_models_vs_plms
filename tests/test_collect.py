"""The post-cutoff table, and the join of tasks, replies and baselines into scored tables."""

from __future__ import annotations

import json
import sys
import numpy as np
import pandas as pd
import pytest

from pathlib import Path

from general_models_vs_plms import collect, flip2, plan, postcutoff, recognition_probe
from general_models_vs_plms.tasks import Arm, build_task, load_layout, write_task


LETTERS = np.array(list("ACDEFGHIKLMNPQRSTVWY"))


def write_toy_split(folder: Path, rows: int = 500) -> pd.DataFrame:
    generator = np.random.default_rng(2)
    parent = np.array(list("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"))
    weights = generator.normal(size=(len(parent), 20))
    sequences, targets = [], []
    for _ in range(rows):
        variant, target = parent.copy(), 0.0
        for column in generator.choice(len(parent), size=generator.integers(1, 4), replace=False):
            letter = generator.integers(20)
            variant[column] = LETTERS[letter]
            target += weights[column, letter]
        sequences.append("".join(variant))
        targets.append(target)
    frame = pd.DataFrame({"sequence": sequences, "target": targets,
                          "set": np.where(np.arange(rows) < 350, "train", "test"), "validation": False})
    frame = frame.drop_duplicates("sequence").reset_index(drop=True)
    path = folder / "amylase" / "by_mutation.csv.gz"
    path.parent.mkdir(parents=True)
    frame.to_csv(path, index=False)
    return frame


def test_collect_scores_replies_and_baselines_and_fills_unanswered_queries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_folder, tasks, responses, baselines, out = (tmp_path / name for name in ("data", "tasks", "responses", "baselines", "out"))
    write_toy_split(data_folder)
    split_info = plan.split_infos(("amylase/by_mutation",))[0]
    frame = flip2.load_split(flip2.split_path(data_folder, split_info))
    task = build_task(split_info, frame, load_layout(split_info, frame), Arm("k256", 64), 0, "test", 120)
    write_task(task, tasks)
    truth = frame["target"].to_numpy()[task.query_rows]

    model_folder = responses / "claude-fable-5-2__low"
    model_folder.mkdir(parents=True)
    for index in range(len(task.chunks)):
        ids = task.query_ids[index * 100:(index + 1) * 100]
        values = truth[index * 100:(index + 1) * 100]
        answered = [(identifier, value) for identifier, value in zip(ids, values, strict=True) if identifier != "Q003"]
        record = {"record_id": f"{task.task_id}__c{index}", "model_requested": "claude-fable-5-2",
                  "model_served": "claude-fable-5-2", "effort": "low", "stop_reason": "end_turn", "stop_category": None,
                  "text": "\n".join(f"{identifier}\t{value:.4f}" for identifier, value in answered),
                  "usage": {"input_tokens": 10, "output_tokens": 20, "output_tokens_details": {"thinking_tokens": 3}},
                  "cost_usd": 0.01, "elapsed_seconds": 1.5}
        (model_folder / f"{task.task_id}__c{index}.json").write_text(json.dumps(record), encoding="utf-8")
    (responses / "spend__claude-fable-5-2__low.json").write_text("{}", encoding="utf-8")

    baselines.mkdir()
    (baselines / f"{task.task_id}.json").write_text(json.dumps({"ridge_onehot": (-truth).tolist()}), encoding="utf-8")
    (baselines / "reference__amylase-by_mutation.json").write_text(
        json.dumps({"ridge_onehot_full": truth.tolist(), "full_test_spearman": 1.0}), encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["collect", "--data", str(data_folder), "--tasks", str(tasks), "--responses", str(responses),
                                      "--baselines", str(baselines), "--out", str(out)])
    collect.main()

    metrics = pd.read_csv(out / "metrics.csv").set_index("method")
    assert set(metrics.index) == {"claude-fable-5-2@low", "ridge_onehot", "ridge_onehot_full"}
    assert metrics.loc["claude-fable-5-2@low", "answered_fraction"] == pytest.approx(119 / 120)
    assert 0.95 < metrics.loc["claude-fable-5-2@low", "spearman"] < 1.0
    assert metrics.loc["ridge_onehot", "spearman"] == pytest.approx(-1.0)
    assert metrics.loc["ridge_onehot_full", "spearman"] == pytest.approx(1.0)

    predictions = pd.read_csv(out / "predictions.csv.gz")
    unanswered = predictions[(predictions["method"] == "claude-fable-5-2@low") & ~predictions["answered"]]
    assert unanswered["query_id"].tolist() == ["Q003"]
    assert unanswered["prediction"].iloc[0] == pytest.approx(np.median(task.context_values))
    calls = pd.read_csv(out / "calls.csv")
    assert len(calls) == 2 and calls["thinking_tokens"].tolist() == [3, 3] and calls["answered"].sum() == 119


def test_collect_leaves_out_a_task_cut_short_and_keeps_the_subagent_arm_that_asks_one_prompt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_folder, tasks, responses, out = (tmp_path / name for name in ("data", "tasks", "responses", "out"))
    write_toy_split(data_folder)
    split_info = plan.split_infos(("amylase/by_mutation",))[0]
    frame = flip2.load_split(flip2.split_path(data_folder, split_info))
    task = build_task(split_info, frame, load_layout(split_info, frame), Arm("k256", 64), 0, "test", 120)
    write_task(task, tasks)
    truth = frame["target"].to_numpy()[task.query_rows]
    text = "\n".join(f"{identifier}\t{value:.4f}" for identifier, value in zip(task.query_ids[:100], truth[:100], strict=True))
    for name, model, effort in (("claude-opus-5-5__low", "claude-opus-5-5", "low"),
                                ("claude-mythos-5-2__harness-high", "claude-mythos-5-2", "harness-high")):
        (responses / name).mkdir(parents=True)
        record = {"record_id": f"{task.task_id}__c0", "model_requested": model, "model_served": model, "effort": effort,
                  "stop_reason": "end_turn", "stop_category": None, "text": text, "usage": {}, "cost_usd": 0.0,
                  "elapsed_seconds": None}
        (responses / name / f"{task.task_id}__c0.json").write_text(json.dumps(record), encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["collect", "--data", str(data_folder), "--tasks", str(tasks), "--responses", str(responses),
                                      "--out", str(out)])
    collect.main()

    metrics = pd.read_csv(out / "metrics.csv").set_index("method")
    assert list(metrics.index) == ["claude-mythos-5-2@harness-high"]
    assert metrics.loc["claude-mythos-5-2@harness-high", "answered_fraction"] == pytest.approx(100 / 120)
    assert metrics.loc["claude-mythos-5-2@harness-high", "decimals"] == task.decimals
    assert (out / "incomplete_tasks.txt").read_text(encoding="utf-8").split() == ["claude-opus-5-5__low", task.task_id]
    assert len(pd.read_csv(out / "calls.csv")) == 2


def test_added_methods_are_scored_beside_the_rows_already_there_and_replace_none(tmp_path: Path) -> None:
    generator = np.random.default_rng(4)
    truth = generator.normal(size=30)
    queries = [f"Q{index + 1:03d}" for index in range(30)]
    # The scored rows are stored out of query order, as nothing promises an order within a task.
    scored = pd.DataFrame({"task_id": "a-one__k256__s0", "method": "ridge_onehot", "query_id": queries, "row": np.arange(30),
                           "stratum": "1", "truth": truth, "prediction": generator.normal(size=30), "answered": True}).iloc[::-1]
    tables, out = tmp_path / "tables", tmp_path / "out"
    tables.mkdir()
    scored.to_csv(tables / "predictions.csv.gz", index=False)
    pd.DataFrame([{"task_id": "a-one__k256__s0", "split": "a/one", "arm": "k256", "seed": 0, "k": 256, "decimals": 2,
                   "method": "ridge_onehot", "queries": 30, "answered_fraction": 1.0, "spearman": 0.1}]).to_csv(tables / "metrics.csv", index=False)
    (tables / "calls.csv").write_text("record_id\n", encoding="utf-8")
    (tables / "incomplete_tasks.txt").write_text("", encoding="utf-8")

    collect.add_methods(tables, {"a-one__k256__s0": {"progen2_small_dpo": truth.tolist(), "progen2_small_zero": (-truth).tolist()}}, out)

    predictions = pd.read_csv(out / "predictions.csv.gz", dtype={"stratum": str})
    metrics = pd.read_csv(out / "metrics.csv").set_index("method")
    assert len(predictions) == 90 and (out / "calls.csv").exists() and (out / "incomplete_tasks.txt").exists()
    added = predictions[predictions["method"] == "progen2_small_dpo"].sort_values("query_id")
    assert added["prediction"].tolist() == pytest.approx(truth.tolist()) and added["truth"].tolist() == pytest.approx(truth.tolist())
    assert added["row"].tolist() == list(range(30)) and added["answered"].all()
    assert metrics.loc["progen2_small_dpo", "spearman"] == pytest.approx(1.0)
    assert metrics.loc["progen2_small_zero", "spearman"] == pytest.approx(-1.0)
    assert metrics.loc["ridge_onehot", "spearman"] == pytest.approx(0.1)
    keys = metrics.loc["progen2_small_dpo"]
    assert (keys["split"], keys["arm"], int(keys["seed"]), int(keys["k"]), int(keys["queries"])) == ("a/one", "k256", 0, 256, 30)
    with pytest.raises(AssertionError, match="already scored"):
        collect.add_methods(out, {"a-one__k256__s0": {"progen2_small_dpo": truth.tolist()}}, tmp_path / "again")
    with pytest.raises(AssertionError, match="29 predictions for the 30 queries"):
        collect.add_methods(tables, {"a-one__k256__s0": {"progen2_small_bt": truth[:29].tolist()}}, tmp_path / "short")


def test_recognition_question_shows_unnamed_rows_once_per_dataset(tmp_path: Path) -> None:
    data_folder, tasks, out = (tmp_path / name for name in ("data", "tasks", "out"))
    write_toy_split(data_folder)
    split_info = plan.split_infos(("amylase/by_mutation",))[0]
    frame = flip2.load_split(flip2.split_path(data_folder, split_info))
    layout = load_layout(split_info, frame)
    for arm in (Arm("k256", 64), Arm("k256_described", 64, described=True)):
        write_task(build_task(split_info, frame, layout, arm, 0, "test", 120), tasks)

    entries = recognition_probe.export(data_folder, tasks, out, tmp_path / "index.json")

    assert [entry["record_id"] for entry in entries] == ["recognize__amylase"]
    text = (out / "p001" / "prompt.txt").read_text(encoding="utf-8")
    assert text.endswith(recognition_probe.QUESTION)
    assert split_info.description not in text and "amylase" not in text.lower() and "FLIP" not in text
    assert sum(line.count("\t") == 1 for line in text.splitlines()) == recognition_probe.SHOWN_ROWS
    assert json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))["entries"] == entries


def test_variant_table_averages_repeated_changes_and_sets_synonymous_rows_apart() -> None:
    scores_csv = (
        "accession,hgvs_nt,hgvs_splice,hgvs_pro,score\n"
        "a#1,c.1A>G,NA,p.Ala461Thr,1.0\n"
        "a#2,c.2C>T,NA,p.Ala461Thr,3.0\n"
        "a#3,c.3G>A,NA,p.Ala461Ala,1.1\n"
        "a#4,c.4G>A,NA,p.Gly462Val,0.5\n"
        "a#5,c.7G>A,NA,p.Val463Val,0.9\n"
    ).encode()

    parent, every_change = postcutoff.variant_table(scores_csv)
    variants, synonymous = postcutoff.split_synonymous(every_change)

    assert parent == "AGV"
    assert variants[["position", "mutant", "score", "offset"]].values.tolist() == [[461, "T", 2.0, 0], [462, "V", 0.5, 1]]
    assert synonymous["position"].tolist() == [461, 463]


def test_variant_table_refuses_a_gap_in_the_segment_or_a_row_that_is_not_a_substitution() -> None:
    header = "accession,hgvs_nt,hgvs_splice,hgvs_pro,score\n"
    with pytest.raises(ValueError, match="one wild-type residue at every position"):
        postcutoff.variant_table((header + "a,x,NA,p.Ala461Thr,1.0\na,x,NA,p.Gly463Val,1.0\n").encode())
    with pytest.raises(ValueError, match="not a single amino acid substitution"):
        postcutoff.variant_table((header + "a,x,NA,p.Ala461del,1.0\n").encode())
    with pytest.raises(ValueError, match="outside the 20 amino acids"):
        postcutoff.variant_table((header + "a,x,NA,p.Ala461Ter,1.0\n").encode())

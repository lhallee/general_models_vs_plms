"""Preference tuning on a stand-in causal model whose scores can be worked out by hand, with a planted preference."""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pytest
import torch

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from scipy.special import logsumexp
from torch import Tensor, nn

from general_models_vs_plms import baselines, flip2, plan, preference_tuning
from general_models_vs_plms.encoding import one_hot
from general_models_vs_plms.preference_tuning import ARMS, DirectionalScorer, Settings
from general_models_vs_plms.scoring import spearman
from general_models_vs_plms.strata import strata_of
from general_models_vs_plms.tasks import Arm, Task, build_task, load_layout


# The ids of a ProGen2 tokenizer file: three special tokens, the two terminals, then 25 letters with no J.
LETTERS = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
VOCABULARY = {"<|pad|>": 0, "<|bos|>": 1, "<|eos|>": 2, "1": 3, "2": 4, **{letter: 5 + index for index, letter in enumerate(LETTERS)}}
FAST = Settings(learning_rates=(0.03,), betas=(2.0,), epochs=6)


class BigramModel(nn.Module):
    """A causal model whose next-token logits depend on the current token alone, so a score is a sum over a table."""

    def __init__(self, seed: int = 0, width: int = 6) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.table = nn.Parameter(torch.randn(len(VOCABULARY), len(VOCABULARY), generator=generator))  # (v, v)
        self.states = nn.Parameter(torch.randn(len(VOCABULARY), width, generator=generator))  # (v, d)

    def forward(self, input_ids: Tensor, attention_mask: Tensor, use_cache: bool, output_attentions: bool,
                output_hidden_states: bool, return_dict: bool) -> SimpleNamespace:
        # input_ids: (n, t); attention_mask: (n, t)
        assert return_dict and not use_cache and not output_attentions
        assert torch.equal(attention_mask, (input_ids != VOCABULARY["<|pad|>"]).long())
        hidden_states = (self.states[input_ids],) if output_hidden_states else None  # (n, t, d)
        return SimpleNamespace(logits=self.table[input_ids], hidden_states=hidden_states)  # logits: (n, t, v)


class PoisonedModel(BigramModel):
    """A model whose logits are not numbers whenever a gradient is being taken."""

    def forward(self, input_ids: Tensor, attention_mask: Tensor, use_cache: bool, output_attentions: bool,
                output_hidden_states: bool, return_dict: bool) -> SimpleNamespace:
        # input_ids: (n, t); attention_mask: (n, t)
        output = super().forward(input_ids, attention_mask, use_cache, output_attentions, output_hidden_states, return_dict)
        if torch.is_grad_enabled():
            output.logits = output.logits * float("nan")
        return output


def scorer_of(model: nn.Module) -> DirectionalScorer:
    return DirectionalScorer(model, preference_tuning.progen2_alphabet(VOCABULARY), torch.device("cpu"), max_tokens=64)


def reading_sum(table: np.ndarray, reading: str) -> float:
    """By hand: the log-probability of each residue of a reading given the token before it, among the letter tokens."""
    # table: (v, v)
    ids = [VOCABULARY[symbol] for symbol in reading]
    total = 0.0
    for current, following in zip(ids[:-2], ids[1:-1], strict=True):
        letter_logits = table[current, 5:30]  # (25,)
        total += letter_logits[following - 5] - logsumexp(letter_logits)
    return total


def chain_sum(table: np.ndarray, chain: str) -> float:
    """By hand: a chain's residue log-likelihood averaged over its two readings."""
    # table: (v, v)
    return 0.5 * (reading_sum(table, f"1{chain}2") + reading_sum(table, f"2{chain[::-1]}1"))


def write_planted_split(folder: Path, rows: int = 700) -> pd.DataFrame:
    """A split in the FLIP2 file format whose target counts tryptophans less lysines among a row's substitutions."""
    generator = np.random.default_rng(11)
    parent = np.array(list("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"))
    sequences, targets = [], []
    for _ in range(rows):
        variant = parent.copy()
        columns = generator.choice(len(parent), size=generator.integers(1, 5), replace=False)
        variant[columns] = generator.choice(list("WKAG"), size=len(columns))
        sequences.append("".join(variant))
        targets.append(float((variant == "W").sum() - (variant == "K").sum()) + generator.normal(scale=0.05))
    frame = pd.DataFrame({"sequence": sequences, "target": targets}).drop_duplicates("sequence").reset_index(drop=True)
    frame["set"] = np.where(np.arange(len(frame)) < 400, "train", "test")
    frame["validation"] = False
    path = folder / "amylase" / "by_mutation.csv.gz"
    path.parent.mkdir(parents=True)
    frame.to_csv(path, index=False)
    return frame


def planted_task(folder: Path) -> tuple[Task, pd.DataFrame]:
    write_planted_split(folder)
    split_info = plan.split_infos(("amylase/by_mutation",))[0]
    frame = flip2.load_split(flip2.split_path(folder, split_info))
    return build_task(split_info, frame, load_layout(split_info, frame), Arm("k256", 256), 0, "test", 100), frame


def test_alphabet_reads_a_progen2_vocabulary_and_refuses_one_without_an_amino_acid() -> None:
    alphabet = preference_tuning.progen2_alphabet(VOCABULARY)

    assert (alphabet.pad_id, alphabet.first_letter_id, alphabet.letter_count) == (0, 5, 25)
    assert alphabet.encode("1AC2") == [3, 5, 7, 4]
    # The same block of 25 letter ids, with J where tryptophan was.
    without_tryptophan = {("J" if token == "W" else token): index for token, index in VOCABULARY.items()}
    with pytest.raises(AssertionError, match="lacks"):
        preference_tuning.progen2_alphabet(without_tryptophan)


def test_score_is_the_mean_residue_log_likelihood_over_both_readings() -> None:
    model = BigramModel()
    scorer = scorer_of(model)
    table = model.table.detach().numpy()

    alone = scorer.score([["ACDW"]], tokens_per_batch=1000)
    batched = scorer.score([["MKTAYIAKQRW"], ["ACDW"], ["ACDW", "MK"], ["ACDW", ""]], tokens_per_batch=1000)

    assert alone[0] == pytest.approx(chain_sum(table, "ACDW") / 4, abs=1e-5)
    # Padding beside a longer chain changes nothing, and neither does an empty second chain.
    assert batched[1] == pytest.approx(alone[0], abs=1e-6) and batched[3] == pytest.approx(alone[0], abs=1e-6)
    assert batched[0] == pytest.approx(chain_sum(table, "MKTAYIAKQRW") / 11, abs=1e-5)
    # A pair is scored over the residues of both chains.
    assert batched[2] == pytest.approx((chain_sum(table, "ACDW") + chain_sum(table, "MK")) / 6, abs=1e-5)
    # A small token budget splits the rows into several batches and returns them in the order given.
    assert np.allclose(scorer.score([["MKTAYIAKQRW"], ["ACDW"], ["ACDW", "MK"], ["ACDW", ""]], tokens_per_batch=30), batched, atol=1e-6)


def test_states_pool_each_chain_under_both_readings_into_its_slot() -> None:
    model = BigramModel()
    scorer = scorer_of(model)
    states = model.states.detach().numpy()

    features = scorer.states([["AC", "W"], ["ACD", ""]], tokens_per_batch=1000)  # (2, 2 slots x 2 readings x 6)

    mean_ac = states[[VOCABULARY["A"], VOCABULARY["C"]]].mean(axis=0)
    assert features.shape == (2, 24)
    # A mean over a chain's residues is the same under either reading, so the two halves of a slot agree.
    assert np.allclose(features[0, :6], mean_ac, atol=1e-6) and np.allclose(features[0, 6:12], mean_ac, atol=1e-6)
    assert np.allclose(features[0, 12:18], states[VOCABULARY["W"]], atol=1e-6)
    assert np.all(features[1, 12:] == 0.0)


def test_a_chain_longer_than_the_model_reads_is_refused() -> None:
    with pytest.raises(ValueError, match="does not fit"):
        scorer_of(BigramModel()).score([["A" * 63]], tokens_per_batch=1000)


def test_ranked_loss_is_the_plackett_luce_likelihood_of_the_measured_order() -> None:
    ratios = torch.tensor([0.3, -0.1, 0.2], requires_grad=True)
    values = torch.tensor([1.0, 3.0, 2.0])

    order = preference_tuning.order_by_value(values, np.random.default_rng(0))
    loss = preference_tuning.ranked_preference_loss(ratios, order, beta=2.0)
    loss.backward()

    scaled = 2.0 * np.array([-0.1, 0.2, 0.3])
    by_hand = -np.mean([scaled[rank] - logsumexp(scaled[rank:]) for rank in range(3)])
    assert order.tolist() == [1, 2, 0]
    assert float(loss.detach()) == pytest.approx(by_hand, abs=1e-6)
    # Descent raises the ratio of the row measured highest and lowers that of the row measured lowest.
    assert ratios.grad[1] < 0 < ratios.grad[0]


def test_equal_values_are_ordered_at_random() -> None:
    values = torch.tensor([2.0, 1.0, 2.0, 2.0])
    generator = np.random.default_rng(0)

    orders = {tuple(preference_tuning.order_by_value(values, generator).tolist()) for _ in range(50)}

    assert all(order[3] == 1 and sorted(order[:3]) == [0, 2, 3] for order in orders)
    assert len(orders) > 1


def test_bradley_terry_loss_reads_only_pairs_whose_values_differ() -> None:
    scores = torch.tensor([0.1, 0.4, 0.2])

    loss = preference_tuning.bradley_terry_loss(scores, torch.tensor([1.0, 1.0, 5.0]), scale=10.0)
    tied = preference_tuning.bradley_terry_loss(scores, torch.tensor([1.0, 1.0, 1.0]), scale=10.0)

    # The third row is preferred over the first with logit 1 and over the second with logit -2.
    assert float(loss.detach()) == pytest.approx(np.mean([np.log1p(np.exp(-1.0)), np.log1p(np.exp(2.0))]), abs=1e-6)
    assert tied is None


def test_tuning_split_is_drawn_once_from_the_split_name() -> None:
    fit, held = preference_tuning.tuning_split(256, 51, "amylase/by_mutation")

    assert len(fit) == 205 and len(held) == 51
    assert sorted([*fit, *held]) == list(range(256))
    again = preference_tuning.tuning_split(256, 51, "amylase/by_mutation")
    assert np.array_equal(fit, again[0]) and np.array_equal(held, again[1])
    assert not np.array_equal(held, preference_tuning.tuning_split(256, 51, "trpb/by_position")[1])


def test_inside_scores_fit_each_baseline_on_the_tuning_rows_and_score_it_on_the_held_out_rows(tmp_path: Path) -> None:
    task, frame = planted_task(tmp_path)
    split_info = plan.split_infos(("amylase/by_mutation",))[0]
    layout = load_layout(split_info, frame)
    strata = strata_of(split_info, layout)

    found = preference_tuning.inside_scores(task, layout, strata, None, 51, ())

    fit, held = preference_tuning.tuning_split(256, 51, task.split)
    assert (found["task_id"], found["split"], found["fit_rows"], found["held_out_rows"]) == (task.task_id, task.split, 205, 51)
    # The planted target is additive, so a one-hot ridge fit on the 205 rows ranks the other 51.
    assert found["held_out_spearman"]["ridge_onehot"] > 0.8
    rows, values = np.array(task.context_rows), np.array(task.context_values)
    onehot_fit, onehot_held = one_hot(layout, rows[fit], rows[held])
    expected = spearman(values[held], baselines.ridge(onehot_fit, values[fit], onehot_held))
    assert found["held_out_spearman"]["ridge_onehot"] == pytest.approx(expected)
    # No query of the task is read: other queries leave every score as it was.
    moved = preference_tuning.inside_scores(replace(task, query_rows=task.query_rows[::-1]), layout, strata, None, 51, ())
    assert moved["held_out_spearman"] == found["held_out_spearman"]


def test_inside_states_fit_the_hidden_state_ridge_on_the_tuning_rows_and_score_the_held_out_rows(tmp_path: Path) -> None:
    task, frame = planted_task(tmp_path)
    layout = load_layout(plan.split_infos(("amylase/by_mutation",))[0], frame)
    scorer = scorer_of(BigramModel())

    found = preference_tuning.inside_states(scorer, task, layout, Settings())

    fit, held = preference_tuning.tuning_split(256, 51, task.split)
    values = np.array(task.context_values)
    states = scorer.states(preference_tuning.row_chains(layout, task.context_rows), tokens_per_batch=40000)
    fit_states, held_states = baselines.standardize(states[fit], states[held])
    assert found == pytest.approx(spearman(values[held], baselines.ridge(fit_states, values[fit], held_states)))
    # The planted target counts two residues, which a mean of per-residue states holds, so the ridge ranks the rows.
    assert found > 0.5


def test_tuning_learns_a_planted_preference_and_permuted_values_teach_nothing(tmp_path: Path) -> None:
    task, frame = planted_task(tmp_path)
    split_info = plan.split_infos(("amylase/by_mutation",))[0]
    layout = load_layout(split_info, frame)
    scorer = scorer_of(BigramModel())
    truth = frame["target"].to_numpy()[task.query_rows]
    untuned = scorer.score(preference_tuning.row_chains(layout, task.query_rows), FAST.tokens_per_batch)

    methods, record = preference_tuning.task_methods(scorer, "toy", task, layout, ARMS, FAST)

    assert set(methods) == {f"toy_{arm}" for arm in ARMS}
    assert all(len(scores) == 100 and np.isfinite(scores).all() for scores in methods.values())
    assert methods["toy_zero"] == pytest.approx(untuned.tolist())
    score = {arm: spearman(truth, np.array(methods[f"toy_{arm}"])) for arm in ARMS}
    assert abs(score["zero"]) < 0.3
    assert min(score["dpo"], score["bt"], score["dpo_seed1"], score["dpo_seed2"]) > 0.7
    assert abs(score["dpo_shuffled"]) < 0.3
    # The hidden states of this model hold each residue's identity, which a ridge can count.
    assert score["ridge"] > 0.7

    assert (record["fit_rows"], record["held_out_rows"], record["failed"]) == (205, 51, {})
    chosen = record["arms"]["dpo"]
    assert chosen["recipe"] == {"loss": "dpo", "learning_rate": 0.03, "beta": 2.0} and 1 <= chosen["epoch"] <= FAST.epochs
    assert chosen["held_out_spearman"] == max(chosen["curves"][0]["held_out_spearman"])
    assert record["arms"]["bt"]["recipe"]["beta"] is None
    # The other seeds repeat the recipe the reported seed chose, under their own batch order.
    for arm in ("dpo_seed1", "dpo_seed2"):
        assert record["arms"][arm]["recipe"] == chosen["recipe"] and len(record["arms"][arm]["curves"]) == 1
    assert record["arms"]["dpo_seed1"]["curves"][0]["train_loss"] != chosen["curves"][0]["train_loss"]
    assert record["mean_tokens"] == 35.0
    # Every arm starts from the untuned weights and leaves them as it found them.
    assert np.allclose(scorer.score(preference_tuning.row_chains(layout, task.query_rows), FAST.tokens_per_batch), untuned)


def test_selection_keeps_the_best_held_out_epoch_across_the_grid(tmp_path: Path) -> None:
    task, frame = planted_task(tmp_path)
    layout = load_layout(plan.split_infos(("amylase/by_mutation",))[0], frame)
    grid = Settings(learning_rates=(1e-4, 0.03), betas=(0.1, 2.0), epochs=3)

    _, record = preference_tuning.task_methods(scorer_of(BigramModel()), "toy", task, layout, ("dpo",), grid)

    chosen = record["arms"]["dpo"]
    curves = chosen["curves"]
    assert [(curve["learning_rate"], curve["beta"]) for curve in curves] == [(1e-4, 0.1), (1e-4, 2.0), (0.03, 0.1), (0.03, 2.0)]
    best = max(max(curve["held_out_spearman"]) for curve in curves)
    assert chosen["held_out_spearman"] == best
    winner = next(curve for curve in curves if max(curve["held_out_spearman"]) == best)
    assert (chosen["recipe"]["learning_rate"], chosen["recipe"]["beta"]) == (winner["learning_rate"], winner["beta"])
    assert chosen["epoch"] == 1 + int(np.argmax(winner["held_out_spearman"]))


def test_a_trace_keeps_the_leading_queries_scores_after_every_epoch_and_changes_no_choice(tmp_path: Path) -> None:
    task, frame = planted_task(tmp_path)
    layout = load_layout(plan.split_infos(("amylase/by_mutation",))[0], frame)
    grid = Settings(learning_rates=(1e-4, 0.03), betas=(2.0,), epochs=3)

    plain_methods, plain = preference_tuning.task_methods(scorer_of(BigramModel()), "toy", task, layout, ("dpo",), grid)
    methods, record = preference_tuning.task_methods(scorer_of(BigramModel()), "toy", task, layout, ("dpo",), grid, traced=7)

    assert (plain["traced_queries"], record["traced_queries"]) == (0, 7)
    assert all("query_scores" not in curve for curve in plain["arms"]["dpo"]["curves"])
    chosen = record["arms"]["dpo"]
    assert (chosen["recipe"], chosen["epoch"]) == (plain["arms"]["dpo"]["recipe"], plain["arms"]["dpo"]["epoch"])
    assert methods["toy_dpo"] == pytest.approx(plain_methods["toy_dpo"])
    for curve in chosen["curves"]:
        assert np.array(curve["query_scores"]).shape == (3, 7)
    # The scores traced at the chosen epoch of the chosen recipe are the ones the arm reports for those queries.
    kept = next(curve for curve in chosen["curves"] if curve["learning_rate"] == chosen["recipe"]["learning_rate"])
    assert kept["query_scores"][chosen["epoch"] - 1] == pytest.approx(methods["toy_dpo"][:7], abs=1e-6)


def test_fixed_epochs_of_a_traced_run_are_methods_that_nothing_chose(tmp_path: Path) -> None:
    task, frame = planted_task(tmp_path)
    layout = load_layout(plan.split_infos(("amylase/by_mutation",))[0], frame)
    one_setting = Settings(learning_rates=(0.03,), betas=(2.0,), epochs=4)

    methods, record = preference_tuning.task_methods(scorer_of(BigramModel()), "toy", task, layout, ("dpo",), one_setting, traced=100)
    fixed = preference_tuning.fixed_epoch_methods(record, "toy", (2, 4))

    assert set(fixed) == {"toy2_dpo", "toy4_dpo"} and all(len(scores) == 100 for scores in fixed.values())
    curve = record["arms"]["dpo"]["curves"][0]["query_scores"]
    assert fixed["toy2_dpo"] == curve[1] and fixed["toy4_dpo"] == curve[3]
    # The epoch the held-out rows chose is one of the four, and its fixed method is the arm's own prediction.
    chosen = record["arms"]["dpo"]["epoch"]
    assert preference_tuning.fixed_epoch_methods(record, "toy", (chosen,))[f"toy{chosen}_dpo"] == pytest.approx(methods["toy_dpo"], abs=1e-6)
    with pytest.raises(AssertionError, match="epoch 5 was asked for"):
        preference_tuning.fixed_epoch_methods(record, "toy", (5,))
    # A run that traced only its leading queries, or tried two settings, has no fixed recipe to report.
    _, partly = preference_tuning.task_methods(scorer_of(BigramModel()), "toy", task, layout, ("dpo",), one_setting, traced=7)
    with pytest.raises(AssertionError, match="traced 7 of 100"):
        preference_tuning.fixed_epoch_methods(partly, "toy", (2,))
    two = Settings(learning_rates=(1e-4, 0.03), betas=(2.0,), epochs=2)
    _, both = preference_tuning.task_methods(scorer_of(BigramModel()), "toy", task, layout, ("dpo",), two, traced=100)
    with pytest.raises(AssertionError, match="ran 2 settings"):
        preference_tuning.fixed_epoch_methods(both, "toy", (2,))


class ProjectedModel(BigramModel):
    """A bigram model whose logits pass through two linear layers named as an attention block names its projections."""

    def __init__(self, seed: int = 0) -> None:
        super().__init__(seed)
        size = len(VOCABULARY)
        self.qkv_proj = nn.Linear(size, size, bias=False)
        self.out_proj = nn.Linear(size, size, bias=False)
        self.other = nn.Linear(size, size, bias=False)
        with torch.no_grad():
            for layer in (self.qkv_proj, self.out_proj, self.other):
                layer.weight.copy_(torch.eye(size))

    def forward(self, input_ids: Tensor, attention_mask: Tensor, use_cache: bool, output_attentions: bool,
                output_hidden_states: bool, return_dict: bool) -> SimpleNamespace:
        # input_ids: (n, t); attention_mask: (n, t)
        output = super().forward(input_ids, attention_mask, use_cache, output_attentions, output_hidden_states, return_dict)
        output.logits = self.other(self.out_proj(self.qkv_proj(output.logits)))  # (n, t, v)
        return output


def test_a_low_rank_update_starts_at_zero_and_is_all_that_trains() -> None:
    model = ProjectedModel()
    inputs = torch.randn(3, len(VOCABULARY), generator=torch.Generator().manual_seed(1))
    before = model.other(model.out_proj(model.qkv_proj(inputs)))

    trained = preference_tuning.add_low_rank(model, rank=2)

    size = len(VOCABULARY)
    assert trained == 2 * (2 * size + size * 2)
    assert isinstance(model.qkv_proj, preference_tuning.LowRankLinear) and isinstance(model.other, nn.Linear)
    assert {name for name, parameter in model.named_parameters() if parameter.requires_grad} == {
        "qkv_proj.down", "qkv_proj.up", "out_proj.down", "out_proj.up"}
    assert torch.equal(model.other(model.out_proj(model.qkv_proj(inputs))), before)
    # A nonzero second factor moves the layer by `scale` times the product of the two.
    with torch.no_grad():
        model.qkv_proj.up.fill_(0.5)
    expected = inputs + preference_tuning.LOW_RANK_SCALE * (inputs @ model.qkv_proj.down.T) @ model.qkv_proj.up.T
    assert torch.allclose(model.qkv_proj(inputs), expected, atol=1e-6)


def test_low_rank_tuning_learns_the_planted_preference_and_leaves_the_frozen_weights(tmp_path: Path) -> None:
    task, frame = planted_task(tmp_path)
    layout = load_layout(plan.split_infos(("amylase/by_mutation",))[0], frame)
    model = ProjectedModel()
    preference_tuning.add_low_rank(model, rank=4)
    scorer = scorer_of(model)
    frozen = {name: parameter.detach().clone() for name, parameter in model.named_parameters() if not parameter.requires_grad}
    truth = frame["target"].to_numpy()[task.query_rows]

    methods, record = preference_tuning.task_methods(scorer, "toy", task, layout, ("zero", "dpo"), Settings(learning_rates=(0.03,), betas=(2.0,), epochs=10))

    assert abs(spearman(truth, np.array(methods["toy_zero"]))) < 0.3
    assert spearman(truth, np.array(methods["toy_dpo"])) > 0.6
    assert record["arms"]["dpo"]["held_out_spearman"] > 0.6
    # The arm leaves the model as it found it: the frozen weights untouched and the update back at zero.
    assert all(torch.equal(parameter, frozen[name]) for name, parameter in model.named_parameters() if name in frozen)
    assert float(model.qkv_proj.up.abs().sum()) == 0.0 and float(model.out_proj.up.abs().sum()) == 0.0


def test_a_loss_that_is_not_finite_fails_the_arm_and_writes_no_prediction(tmp_path: Path) -> None:
    task, frame = planted_task(tmp_path)
    layout = load_layout(plan.split_infos(("amylase/by_mutation",))[0], frame)

    methods, record = preference_tuning.task_methods(scorer_of(PoisonedModel()), "toy", task, layout, ("zero", "dpo", "dpo_seed1"), FAST)

    assert set(methods) == {"toy_zero"}
    assert "loss is nan" in record["failed"]["dpo"] and record["arms"] == {}
    assert "dpo arm" in record["failed"]["dpo_seed1"]


def test_a_seed_arm_needs_the_recipe_the_reported_seed_chose(tmp_path: Path) -> None:
    task, frame = planted_task(tmp_path)
    layout = load_layout(plan.split_infos(("amylase/by_mutation",))[0], frame)

    with pytest.raises(ValueError, match="dpo arm must run before"):
        preference_tuning.task_methods(scorer_of(BigramModel()), "toy", task, layout, ("dpo_seed1",), FAST)


def test_projection_fits_a_line_through_the_finished_splits_or_falls_back_to_proportion() -> None:
    line = preference_tuning.projected_seconds([(100.0, 250.0), (400.0, 850.0)], [200.0, 50.0])
    # One finished split, or a line that would start below zero, gives seconds in proportion to tokens.
    proportional = preference_tuning.projected_seconds([(100.0, 200.0)], [50.0, 300.0])
    below_zero = preference_tuning.projected_seconds([(100.0, 100.0), (400.0, 900.0)], [200.0])

    assert line == pytest.approx([450.0, 150.0])
    assert proportional == pytest.approx([100.0, 600.0])
    assert below_zero == pytest.approx([400.0])
    assert preference_tuning.mean_tokens([["ACD", "MK"], ["ACDE", ""]]) == pytest.approx((5 + 4 + 6) / 2)


def test_bundle_joins_every_model_folder_by_task(tmp_path: Path) -> None:
    for model, method in (("small", "progen2_small_zero"), ("base", "progen2_base_zero")):
        (tmp_path / model / "records").mkdir(parents=True)
        (tmp_path / model / "amylase-by_mutation__k256__s0.json").write_text(json.dumps({method: [1.0, 2.0]}), encoding="utf-8")
        (tmp_path / model / "records" / "amylase-by_mutation__k256__s0.json").write_text(json.dumps({"model": model}), encoding="utf-8")

    preference_tuning.bundle(tmp_path)

    methods = json.loads((tmp_path / "methods.json").read_text(encoding="utf-8"))
    assert methods == {"amylase-by_mutation__k256__s0": {"progen2_small_zero": [1.0, 2.0], "progen2_base_zero": [1.0, 2.0]}}
    assert [record["model"] for record in json.loads((tmp_path / "records.json").read_text(encoding="utf-8"))] == ["base", "small"]

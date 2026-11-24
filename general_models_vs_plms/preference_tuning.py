"""Preference tuning of ProGen2 on the rows a task shows, and its ranking of the task's queries.

    python -m general_models_vs_plms.preference_tuning --data DIR --tasks DIR --out DIR --model small
        [--splits A,B] [--arms dpo,bt] [--hours H] [--trace N]
        [--variant NAME] [--arm klarge] [--learning-rates R,R] [--betas B,B] [--epochs N] [--low-rank RANK]
    python -m general_models_vs_plms.preference_tuning --model small --check

For the task of seed 0 of each split, under the arm `k256` unless another is named, the module writes one prediction
per query under each method:

- `zero`: the untuned model's score;
- `ridge`: ridge regression on the untuned model's hidden states;
- `dpo`: the score after tuning with the ranked preference loss against the frozen untuned model
  (Widatalla et al. 2024), with `dpo_seed1` and `dpo_seed2` under other batch orders;
- `bt`: the score after tuning with a Bradley-Terry loss on score differences (Hawkins-Hooker et al. 2024);
- `dpo_shuffled`: the ranked loss on the measured values dealt to the wrong rows, which must rank nothing.

A score is the mean log-likelihood per residue, averaged over the two reading directions, as the scorer released
with ProGen2 computes it: a chain is written `1`, its residues, `2`, and reversed for the second reading; the
log-likelihood covers the residues and not the closing terminal; and the logits are restricted to the letter tokens.
A row of two chains is read one chain at a time and scored over the residues of both.

Tuning holds out part of the task's rows. The learning rate, the weight of the preference loss and the epoch are
the ones with the highest Spearman on the held-out rows, and the queries are scored once with those weights. The
module reads no measured value of a query: a task file holds none.

A variant changes one thing about the tuning and is named, so that its methods stand beside the plain ones:
another grid, another arm of the tasks, or a low-rank update of the attention projections with every released
weight frozen. With `--trace` each tuned arm also keeps its scores of the leading queries after every epoch.
"""

from __future__ import annotations

import argparse
import json
import time
import numpy as np
import torch

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch import Tensor, nn
from tqdm import tqdm
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from foundry.layers.cross_entropy import binary_cross_entropy_with_logits, cross_entropy, log_softmax
from . import flip2, plan
from .baselines import PLM_NAMES, SplitFeatures, ridge, standardize, task_baselines, write_methods
from .encoding import AMINO_ACIDS, Layout, sequence_text
from .plm_features import chains_of
from .scoring import spearman
from .strata import strata_of
from .tasks import Task, load_layout, name_code, read_tasks


# Hub repository, pinned revision, and the mean cross-entropy Salesforce's `progen2/likelihood.py` expects of the
# checkpoint on its sanity sequence.
MODELS = {
    "small": ("hugohrban/progen2-small", "43237a0b733c6629226a079266d2985c9fdce9b7", 2.4),
    "base": ("hugohrban/progen2-base", "71228cbfca5960f9fab5775f378bba3673af9f00", 1.9),
}
# Both checkpoints load into the model class of the small repository, whose file runs on current transformers.
CODE_REPOSITORY, CODE_REVISION = MODELS["small"][:2]
CODE_CLASS = "modeling_progen.ProGenForCausalLM"
SANITY_SEQUENCE = (
    "2GFLPFRGADEGLAAREAATLAARGTAARAYREDSWAVPVPRGLLGDLTARVAALGAASPPPADPLAVTLDLHHVTAEVALTTVLDAATLVHGQTRVLSAEDAAEAATAAAAATE"
    "AYLERLQDFVLFMSASVRVWRRGNAAGATGPEWDQWYTVADRDALGSAPTHLAVLGRQADALCHFVLDRVAWGTCGTPLWSGDEDLGNVVATFAGYADRLATAPRDLIM1"
)
SANITY_TOLERANCE = 0.1
N_TERMINAL, C_TERMINAL, PAD_TOKEN = "1", "2", "<|pad|>"
ARMS = plan.PREFERENCE_ARMS
# The tuning seed of each tuned arm. Seed 0 is reported, and the other two give the spread of the primary arm.
ARM_SEEDS = {"dpo": 0, "bt": 0, "dpo_shuffled": 0, "dpo_seed1": 1, "dpo_seed2": 2}
SEED_ARMS = ("dpo_seed1", "dpo_seed2")
# The exit status of a run that left the GPU to another client before it began.
YIELDED = 75
SPLIT_SEED = 0
ADAM_BETAS = (0.9, 0.98)
ADAM_EPSILON = 1e-8
WEIGHT_DECAY = 0.1
# Low-rank tuning: the two projections of an attention block take the update, scaled by this.
LOW_RANK_LAYERS = ("qkv_proj", "out_proj")
LOW_RANK_SCALE = 2.0


@dataclass(frozen=True)
class Alphabet:
    """Token ids of a causal protein language model: its terminals, its padding, and its block of letter tokens."""

    token_ids: dict[str, int]
    pad_id: int
    first_letter_id: int
    letter_count: int

    def encode(self, text: str) -> list[int]:
        """One token per character of a chain written with its terminals."""
        return [self.token_ids[symbol] for symbol in text]


@dataclass(frozen=True)
class Recipe:
    """One point of a tuning grid. `beta` scales the ranked loss's log-ratio and is None for the Bradley-Terry loss."""

    loss: str
    learning_rate: float
    beta: float | None


@dataclass(frozen=True)
class Settings:
    """What the tuned arms share: the grid, the length of a run, and how many rows are held out for selection."""

    learning_rates: tuple[float, ...] = (1e-5, 1e-4)
    betas: tuple[float, ...] = (0.1, 2.0)
    epochs: int = 20
    batch_size: int = 8
    held_out: int = 51
    tokens_per_batch: int = 40000

    def recipes(self, loss: str) -> list[Recipe]:
        """The grid of a loss, lowest learning rate first: the ranked loss crosses it with beta."""
        if loss == "bt":
            return [Recipe("bt", rate, None) for rate in self.learning_rates]
        return [Recipe("dpo", rate, beta) for rate in self.learning_rates for beta in self.betas]


@dataclass(frozen=True)
class Tuned:
    """A tuned arm: its query predictions, the recipe and epoch chosen, and every recipe's curve."""

    predictions: np.ndarray  # (q,)
    recipe: Recipe
    epoch: int
    held_out_spearman: float
    curves: list[dict[str, object]]


def progen2_alphabet(vocabulary: dict[str, int]) -> Alphabet:
    """The alphabet of a ProGen2 tokenizer file, checked for what the score relies on."""
    letters = sorted((index, token) for token, index in vocabulary.items() if len(token) == 1 and token.isalpha() and token.isupper())
    first, last = letters[0][0], letters[-1][0]
    assert last - first + 1 == len(letters), "the letter tokens are not one block of ids"
    missing = [symbol for symbol in (*AMINO_ACIDS, N_TERMINAL, C_TERMINAL, PAD_TOKEN) if symbol not in vocabulary]
    assert not missing, f"the tokenizer lacks {missing}"
    return Alphabet(dict(vocabulary), vocabulary[PAD_TOKEN], first, len(letters))


class DirectionalScorer:
    """A causal protein language model read in both directions, with the score that every arm ranks by."""

    def __init__(self, model: nn.Module, alphabet: Alphabet, device: torch.device, max_tokens: int) -> None:
        self.model = model.to(device).eval()
        self.alphabet = alphabet
        self.device = device
        self.max_tokens = max_tokens

    def token_batch(self, chains: list[str]) -> tuple[Tensor, Tensor]:
        """Each chain's two readings as padded token ids, the forward readings first, and each chain's residue count."""
        longest = max(len(chain) for chain in chains)  # l
        if longest + 2 > self.max_tokens:
            raise ValueError(f"a chain of {longest} residues does not fit the model's {self.max_tokens} positions")
        readings = [N_TERMINAL + chain + C_TERMINAL for chain in chains]
        readings += [reading[::-1] for reading in readings]
        ids = torch.full((len(readings), longest + 2), self.alphabet.pad_id, dtype=torch.long)  # (2 n, l + 2)
        for index, reading in enumerate(readings):
            ids[index, :len(reading)] = torch.tensor(self.alphabet.encode(reading))
        lengths = torch.tensor([len(chain) for chain in chains])  # (n,)
        return ids.to(self.device), lengths.to(self.device)  # (2 n, l + 2), (n,)

    def read(self, ids: Tensor, hidden_states: bool = False) -> tuple[Tensor, Tensor | None]:
        """Logits of a padded batch, and its final hidden states when asked, with nothing left to the model's defaults."""
        # ids: (n, t)
        output = self.model(input_ids=ids, attention_mask=(ids != self.alphabet.pad_id).long(), use_cache=False,
                            output_attentions=False, output_hidden_states=hidden_states, return_dict=True)
        return output.logits, (output.hidden_states[-1] if hidden_states else None)  # (n, t, v), (n, t, d) or None

    def chain_sums(self, chains: list[str]) -> Tensor:
        """The log-likelihood of each chain's residues, summed over them, under each reading."""
        ids, lengths = self.token_batch(chains)  # (2 n, l + 2), (n,)
        logits, _ = self.read(ids)  # (2 n, l + 2, v)
        first, count = self.alphabet.first_letter_id, self.alphabet.letter_count
        # Position t predicts token t + 1, so the first l predictions are the residues and the next is the terminal.
        letter_logits = logits[:, :-1, first:first + count].float()  # (2 n, l + 1, 25)
        log_probabilities = log_softmax(letter_logits, dim=-1)  # (2 n, l + 1, 25)
        targets = (ids[:, 1:] - first).clamp(0, count - 1)  # (2 n, l + 1); a terminal or padding is masked below
        observed = log_probabilities.gather(2, targets[:, :, None]).squeeze(2)  # (2 n, l + 1)
        positions = torch.arange(ids.shape[1] - 1, device=self.device)  # (l + 1,)
        is_residue = positions[None, :] < lengths.repeat(2)[:, None]  # (2 n, l + 1)
        return (observed * is_residue).sum(dim=1).view(2, len(chains))  # (2, n)

    def row_scores(self, rows: list[list[str]]) -> Tensor:
        """The score of each row: mean log-likelihood per residue over its chains, averaged over the two readings."""
        chains = [chain for row in rows for chain in row if chain]
        owners = torch.tensor([index for index, row in enumerate(rows) for chain in row if chain], device=self.device)  # (n,)
        sums = self.chain_sums(chains).mean(dim=0)  # (n,)
        residues = torch.tensor([float(len(chain)) for chain in chains], device=self.device)  # (n,)
        row_sums = torch.zeros(len(rows), device=self.device, dtype=sums.dtype).index_add(0, owners, sums)  # (b,)
        row_residues = torch.zeros(len(rows), device=self.device).index_add(0, owners, residues)  # (b,)
        return row_sums / row_residues  # (b,)

    def row_states(self, rows: list[list[str]], chain_slots: int) -> Tensor:
        """Per row, the mean final hidden state over each chain's residues under each reading, concatenated."""
        chains = [chain for row in rows for chain in row if chain]
        ids, lengths = self.token_batch(chains)  # (2 n, l + 2), (n,)
        _, final = self.read(ids, hidden_states=True)
        assert final is not None
        states = final.float()  # (2 n, l + 2, d)
        positions = torch.arange(ids.shape[1], device=self.device)  # (l + 2,)
        is_residue = (positions[None, :] >= 1) & (positions[None, :] <= lengths.repeat(2)[:, None])  # (2 n, l + 2)
        pooled = (states * is_residue[:, :, None]).sum(dim=1) / lengths.repeat(2)[:, None]  # (2 n, d)
        both = torch.cat([pooled[:len(chains)], pooled[len(chains):]], dim=1)  # (n, 2 d)
        features = torch.zeros(len(rows), chain_slots * both.shape[1], device=self.device)  # (b, c 2 d)
        position = 0
        for index, row in enumerate(rows):
            for slot, chain in enumerate(row):
                if chain:
                    features[index, slot * both.shape[1]:(slot + 1) * both.shape[1]] = both[position]
                    position += 1
        return features  # (b, c 2 d)

    def batches(self, rows: list[list[str]], tokens_per_batch: int) -> list[list[int]]:
        """Row indices grouped longest first, each batch's two readings under the token budget."""
        order = sorted(range(len(rows)), key=lambda index: max(len(chain) for chain in rows[index]), reverse=True)
        batches, current, chain_count, longest = [], [], 0, 0
        for index in order:
            row_chains = sum(1 for chain in rows[index] if chain)
            if current and 2 * (chain_count + row_chains) * (longest + 2) > tokens_per_batch:
                batches.append(current)
                current, chain_count = [], 0
            if not current:
                longest = max(len(chain) for chain in rows[index])
            current.append(index)
            chain_count += row_chains
        return batches + [current]

    @torch.no_grad()
    def score(self, rows: list[list[str]], tokens_per_batch: int) -> np.ndarray:
        """Row scores without gradients, in the order of the rows."""
        scores = np.empty(len(rows))  # (n,)
        for batch in self.batches(rows, tokens_per_batch):
            scores[batch] = self.row_scores([rows[index] for index in batch]).double().cpu().numpy()
        return scores  # (n,)

    @torch.no_grad()
    def states(self, rows: list[list[str]], tokens_per_batch: int) -> np.ndarray:
        """Row hidden-state features without gradients, in the order of the rows."""
        chain_slots = max(len(row) for row in rows)  # c
        features: np.ndarray | None = None
        for batch in self.batches(rows, tokens_per_batch):
            found = self.row_states([rows[index] for index in batch], chain_slots).cpu().numpy()  # (b, c 2 d)
            if features is None:
                features = np.zeros((len(rows), found.shape[1]), dtype=np.float32)  # (n, c 2 d)
            features[batch] = found
        assert features is not None, "no row to embed"
        return features  # (n, c 2 d)

    @torch.no_grad()
    def reference_cross_entropy(self, reading: str) -> float:
        """Mean next-token cross-entropy of a reading written with its terminals, over the whole vocabulary."""
        ids = torch.tensor([self.alphabet.encode(reading)], device=self.device)  # (1, t)
        logits, _ = self.read(ids)  # (1, t, v)
        return float(cross_entropy(logits[0, :-1].float(), ids[0, 1:]))


def no_head_mask(model: nn.Module, head_mask: None, layers: int) -> list[None]:
    """`PreTrainedModel.get_head_mask` for a forward pass that masks no head, which is every pass this module makes."""
    assert head_mask is None, "this module passes no head mask"
    return [None] * layers


def progen2_class() -> type[nn.Module]:
    """The mirror's model class at its pinned revision, with the two things transformers 5 expects of it.

    Its forward pass calls `get_head_mask`, a base-class method that transformers 5 no longer has, and its inner model
    is built without the table of tied weights that transformers 5 reads when a model is constructed. Both are put on
    the mirror's own base class where they are missing: no head is masked, and no weight is tied.
    """
    model_class = get_class_from_dynamic_module(CODE_CLASS, CODE_REPOSITORY, revision=CODE_REVISION)
    base = next(parent for parent in model_class.__mro__ if parent.__name__ == "ProGenPreTrainedModel")
    if not hasattr(base, "get_head_mask"):
        base.get_head_mask = no_head_mask
        print("transformers has no get_head_mask: the mirror's base class is given one that masks no head", flush=True)
    if "all_tied_weights_keys" not in vars(base):
        base.all_tied_weights_keys = {}
    return model_class


def load_progen2(name: str, device: torch.device) -> DirectionalScorer:
    """A pinned ProGen2 checkpoint in float32, checked against the cross-entropy its authors expect of it."""
    repository, revision, expected = MODELS[name]
    model_class = progen2_class()
    # Built by its constructor and filled from the weight file, every name matched. `from_pretrained` builds the model
    # on the meta device, where the plain tensor each attention block keeps (`scale_attn`) is never given a value.
    model = model_class(model_class.config_class.from_pretrained(repository, revision=revision))
    model.load_state_dict(load_file(hf_hub_download(repository, "model.safetensors", revision=revision)), strict=True)
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters()), "the weights are not float32"
    tokenizer_file = Path(hf_hub_download(repository, "tokenizer.json", revision=revision))
    vocabulary = json.loads(tokenizer_file.read_text(encoding="utf-8"))["model"]["vocab"]
    scorer = DirectionalScorer(model, progen2_alphabet(vocabulary), device, max_tokens=int(model.config.n_positions))
    found = scorer.reference_cross_entropy(SANITY_SEQUENCE)
    print(f"progen2-{name}@{revision[:8]}: {sum(parameter.numel() for parameter in model.parameters()):,} parameters, "
          f"sanity cross-entropy {found:.3f} against {expected}", flush=True)
    assert abs(found - expected) < SANITY_TOLERANCE, f"progen2-{name} scores {found:.3f} on the sanity sequence, not {expected}"
    return scorer


class LowRankLinear(nn.Module):
    """A frozen linear layer with a trainable low-rank update added to its output.

    The update is `scale * x A^T B^T`, with A of shape (r, d_in) and B of shape (d_out, r). B starts at zero, so the
    layer starts as the frozen one, and A starts uniform within 1 / sqrt(d_in).
    """

    def __init__(self, layer: nn.Linear, rank: int, scale: float, generator: torch.Generator) -> None:
        super().__init__()
        self.layer = layer
        weight = layer.weight  # (d_out, d_in)
        down = (2 * torch.rand(rank, layer.in_features, generator=generator) - 1) * layer.in_features ** -0.5  # (r, d_in)
        self.down = nn.Parameter(down.to(weight.device, weight.dtype))
        self.up = nn.Parameter(torch.zeros(layer.out_features, rank, device=weight.device, dtype=weight.dtype))  # (d_out, r)
        self.scale = scale

    def forward(self, inputs: Tensor) -> Tensor:
        # inputs: (..., d_in)
        return self.layer(inputs) + self.scale * (inputs @ self.down.T) @ self.up.T  # (..., d_out)


def add_low_rank(model: nn.Module, rank: int, seed: int = 0) -> int:
    """Freeze a model and give each attention projection a trainable low-rank update. Returns the trained weights' count.

    The model's outputs are unchanged until a step is taken, since every update starts at zero.
    """
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    generator = torch.Generator().manual_seed(seed)
    owners = [(module, name) for module in model.modules() for name, child in module.named_children()
              if name in LOW_RANK_LAYERS and isinstance(child, nn.Linear)]
    assert owners, f"the model has no linear layer named {' or '.join(LOW_RANK_LAYERS)}"
    for module, name in owners:
        setattr(module, name, LowRankLinear(getattr(module, name), rank, LOW_RANK_SCALE, generator))
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def order_by_value(values: Tensor, generator: np.random.Generator) -> Tensor:
    """Indices from the highest measured value to the lowest, with equal values in a random order."""
    # values: (b,)
    shuffled = torch.as_tensor(generator.permutation(len(values)), device=values.device)  # (b,)
    return shuffled[torch.argsort(values[shuffled], descending=True, stable=True)]  # (b,)


def ranked_preference_loss(ratios: Tensor, order: Tensor, beta: float) -> Tensor:
    """The ranked preference loss: the Plackett-Luce likelihood of the measured order under `beta` times the ratios.

    A ratio is a row's score under the tuned model less its score under the frozen reference. The row at rank k is
    asked to beat every row ranked below it, so the last row contributes nothing.
    """
    # ratios: (b,); order: (b,), from the highest measured value to the lowest
    scaled = beta * ratios[order]  # (b,)
    chosen = torch.stack([log_softmax(scaled[rank:], dim=0)[0] for rank in range(len(scaled))])  # (b,)
    return -chosen.mean()  # ()


def bradley_terry_loss(scores: Tensor, values: Tensor, scale: float) -> Tensor | None:
    """Binary cross-entropy that the row with the higher measured value has the higher score, over a batch's pairs.

    The logit of a pair is `scale` times the difference of the two scores. None when no two values differ.
    """
    # scores, values: (b,)
    differences = scale * (scores[:, None] - scores[None, :])  # (b, b)
    preferred = values[:, None] > values[None, :]  # (b, b)
    if not bool(preferred.any()):
        return None
    logits = differences[preferred]  # (p,)
    return binary_cross_entropy_with_logits(logits, torch.ones_like(logits))  # ()


@contextmanager
def gradient_precision() -> Iterator[None]:
    """Matrix products at the tensor cores' reduced precision (TF32) for the passes that take a gradient.

    A gradient step tolerates it and runs several times faster. Every pass that scores a row stays outside this block
    and exact, because variants of one parent differ in the third decimal of the score.
    """
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = False


def tuning_split(count: int, held_out: int, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Positions of a task's rows that train and that are held out for selection, drawn once from the split's name."""
    order = np.random.default_rng([SPLIT_SEED, name_code(split), 4]).permutation(count)  # (k,)
    return order[held_out:], order[:held_out]  # (k - held,), (held,)


def tune(
    scorer: DirectionalScorer, pristine: dict[str, Tensor], recipes: list[Recipe], settings: Settings,
    fit_rows: list[list[str]], fit_values: np.ndarray, fit_reference: np.ndarray,
    held_rows: list[list[str]], held_values: np.ndarray, query_rows: list[list[str]], split: str, seed: int, label: str,
    traced: int = 0,
) -> Tuned:
    """Tune from the pristine weights under every recipe, keep the weights of the best held-out epoch, score the queries.

    Every recipe sees the same batches in the same order. Raises FloatingPointError when a loss or a held-out score
    is not finite. With `traced` above zero, each recipe's curve also holds the scores of the first `traced` queries
    after every epoch, so that a reader who has the measured values can see what any other epoch would have scored.
    The choice of weights never reads them.
    """
    # pristine: (...) one tensor per weight name; fit_values, fit_reference: (k_fit,); held_values: (k_held,)
    values = torch.tensor(fit_values, device=scorer.device, dtype=torch.float32)  # (k_fit,)
    reference = torch.tensor(fit_reference, device=scorer.device, dtype=torch.float32)  # (k_fit,)
    scale = float(np.mean([sum(len(chain) for chain in row) for row in fit_rows]))
    best_score, best_recipe, best_epoch, best_state = -np.inf, recipes[0], 0, pristine
    curves = []
    for recipe in recipes:
        scorer.model.load_state_dict(pristine)
        # Every weight that takes a gradient: all of them, or the low-rank updates alone where the model was given them.
        trained = [parameter for parameter in scorer.model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(trained, lr=recipe.learning_rate, betas=ADAM_BETAS, eps=ADAM_EPSILON, weight_decay=WEIGHT_DECAY)
        batch_generator = np.random.default_rng([seed, name_code(split), 6])
        tie_generator = np.random.default_rng([seed, name_code(split), 7])
        held_curve, loss_curve, query_curve = [], [], []
        description = f"{label} {split} lr {recipe.learning_rate:g} beta {recipe.beta}"
        for epoch in tqdm(range(1, settings.epochs + 1), desc=description, mininterval=20):
            order = batch_generator.permutation(len(fit_rows))  # (k_fit,)
            losses = []
            with gradient_precision():
                for start in range(0, len(order), settings.batch_size):
                    batch = torch.as_tensor(order[start:start + settings.batch_size], device=scorer.device)  # (b,)
                    scores = scorer.row_scores([fit_rows[index] for index in batch.tolist()])  # (b,)
                    if recipe.loss == "dpo":
                        ranked = order_by_value(values[batch], tie_generator)  # (b,)
                        loss = ranked_preference_loss(scores - reference[batch], ranked, recipe.beta)
                    else:
                        loss = bradley_terry_loss(scores, values[batch], scale)
                    if loss is None:
                        continue
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError(f"{description}: the loss is {float(loss.detach())} in epoch {epoch}")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    losses.append(float(loss.detach()))
            held_scores = scorer.score(held_rows, settings.tokens_per_batch)  # (k_held,)
            if not np.isfinite(held_scores).all():
                raise FloatingPointError(f"{description}: a held-out score is not finite after epoch {epoch}")
            held = spearman(held_values, held_scores)
            held_curve.append(held)
            loss_curve.append(float(np.mean(losses)) if losses else float("nan"))
            if held > best_score:
                best_score, best_recipe, best_epoch = held, recipe, epoch
                best_state = {name: tensor.detach().clone() for name, tensor in scorer.model.state_dict().items()}
            if traced:
                query_scores = scorer.score(query_rows[:traced], settings.tokens_per_batch)  # (q_traced,)
                query_curve.append([round(float(score), 7) for score in query_scores])
        curves.append({**asdict(recipe), "held_out_spearman": held_curve, "train_loss": loss_curve}
                      | ({"query_scores": query_curve} if traced else {}))

    scorer.model.load_state_dict(best_state)
    predictions = scorer.score(query_rows, settings.tokens_per_batch)  # (q,)
    scorer.model.load_state_dict(pristine)
    return Tuned(predictions, best_recipe, best_epoch, float(best_score), curves)


def row_chains(layout: Layout, rows: list[int]) -> list[list[str]]:
    """Each row's chains as the model reads them."""
    return [chains_of(sequence_text(layout, row)) for row in rows]


def mean_tokens(rows: list[list[str]]) -> float:
    """Tokens of one reading of a row, on average: each chain's residues and its two terminals."""
    return float(np.mean([sum(len(chain) + 2 for chain in row if chain) for row in rows]))


def projected_seconds(done: list[tuple[float, float]], waiting: list[float]) -> list[float]:
    """Seconds each waiting split should take, from the tokens a row and the seconds of the splits already run.

    A line through the finished splits when two or more differ in length and the line rises; else in proportion.
    """
    tokens = np.array([pair[0] for pair in done])  # (n_done,)
    seconds = np.array([pair[1] for pair in done])  # (n_done,)
    slope, intercept = seconds.sum() / tokens.sum(), 0.0
    if len(set(tokens.tolist())) >= 2:
        fitted_slope, fitted_intercept = np.polyfit(tokens, seconds, 1)
        if fitted_slope > 0 and fitted_intercept >= 0:
            slope, intercept = float(fitted_slope), float(fitted_intercept)
    return [intercept + slope * count for count in waiting]


def task_methods(
    scorer: DirectionalScorer, prefix: str, task: Task, layout: Layout, arms: tuple[str, ...], settings: Settings,
    traced: int = 0,
) -> tuple[dict[str, list[float]], dict[str, object]]:
    """Every requested arm's predictions for a task's queries, by method name, and the record of how each was made.

    `traced` is the number of leading queries each tuned arm also scores after every epoch (`tune`).
    """
    context, queries = row_chains(layout, task.context_rows), row_chains(layout, task.query_rows)
    values = np.array(task.context_values)  # (k,)
    pristine = {name: tensor.detach().clone() for name, tensor in scorer.model.state_dict().items()}
    started = time.monotonic()
    context_scores = scorer.score(context, settings.tokens_per_batch)  # (k,)
    fit, held = tuning_split(len(context), settings.held_out, task.split)  # (k_fit,), (k_held,)
    methods: dict[str, np.ndarray] = {}
    arm_records: dict[str, dict[str, object]] = {}
    failed: dict[str, str] = {}
    record: dict[str, object] = {
        "task_id": task.task_id, "split": task.split, "rows": len(context), "queries": len(queries),
        "mean_residues": float(np.mean([sum(len(chain) for chain in row) for row in context])), "mean_tokens": mean_tokens(context),
        "fit_rows": int(len(fit)), "held_out_rows": int(len(held)), "settings": asdict(settings),
        "held_out_spearman_untuned": spearman(values[held], context_scores[held]), "arms": arm_records, "failed": failed,
        "traced_queries": traced,
    }
    if "zero" in arms:
        methods[f"{prefix}_zero"] = scorer.score(queries, settings.tokens_per_batch)  # (q,)
    if "ridge" in arms:
        fit_states, query_states = standardize(scorer.states(context, settings.tokens_per_batch),
                                               scorer.states(queries, settings.tokens_per_batch))  # (k, f), (q, f)
        methods[f"{prefix}_ridge"] = ridge(fit_states, values, query_states)  # (q,)
    record["untuned_seconds"] = time.monotonic() - started

    shuffled = np.random.default_rng([SPLIT_SEED, name_code(task.split), 5]).permutation(values)  # (k,)
    for arm in [arm for arm in arms if arm in ARM_SEEDS]:
        shown = shuffled if arm == "dpo_shuffled" else values  # (k,)
        recipes = settings.recipes("bt" if arm == "bt" else "dpo")
        if arm in SEED_ARMS:
            # Another batch order under the recipe the reported seed chose; its own epoch is chosen on the held-out rows.
            if "dpo" in failed:
                failed[arm] = "the dpo arm, whose recipe this arm repeats, failed"
                continue
            if "dpo" not in arm_records:
                raise ValueError(f"{arm} repeats the recipe the dpo arm chose, so the dpo arm must run before it")
            recipes = [Recipe(**arm_records["dpo"]["recipe"])]
        started = time.monotonic()
        try:
            tuned = tune(scorer, pristine, recipes, settings, [context[index] for index in fit], shown[fit], context_scores[fit],
                         [context[index] for index in held], shown[held], queries, task.split, ARM_SEEDS[arm], f"{prefix}_{arm}",
                         traced)
        except FloatingPointError as error:
            scorer.model.load_state_dict(pristine)
            failed[arm] = str(error)
            continue
        methods[f"{prefix}_{arm}"] = tuned.predictions
        arm_records[arm] = {"recipe": asdict(tuned.recipe), "epoch": tuned.epoch, "held_out_spearman": tuned.held_out_spearman,
                            "curves": tuned.curves, "seconds": time.monotonic() - started}
    return {method: np.asarray(scores, dtype=float).tolist() for method, scores in methods.items()}, record


def inside_scores(
    task: Task, layout: Layout, strata: np.ndarray, features: SplitFeatures | None, held_out: int, models: tuple[str, ...],
) -> dict[str, object]:
    """Each baseline fit on the rows a task's tuning fits on, as its Spearman on the rows the tuning holds out.

    The division is the one `task_methods` makes, so a baseline's score here stands beside a tuned model's held-out
    one: both rank rows drawn from the same part of the split as the rows they learned from. `features` may be None
    when no protein language model is named.
    """
    # strata: (n,), one name per row of the split
    fit, held = tuning_split(len(task.context_rows), held_out, task.split)  # (k_fit,), (k_held,)
    rows, values = np.array(task.context_rows), np.array(task.context_values)  # (k,), (k,)
    inner = replace(task, context_rows=rows[fit].tolist(), context_values=values[fit].tolist(), query_rows=rows[held].tolist())
    predictions = task_baselines(inner, layout, strata, features, models, controls=())
    return {"task_id": task.task_id, "split": task.split, "fit_rows": int(len(fit)), "held_out_rows": int(len(held)),
            "held_out_spearman": {method: spearman(values[held], np.array(scores)) for method, scores in predictions.items()}}


def fixed_epoch_methods(record: dict[str, object], prefix: str, epochs: tuple[int, ...]) -> dict[str, list[float]]:
    """The ranked-loss arm after fixed numbers of epochs, as methods of their own: `<prefix><epochs>_dpo`.

    The scores are the ones a trace of every query kept, so nothing is chosen: not the setting, since the run must
    have had one, and not the epoch. A run that traced only some queries, or more than one setting, is refused.
    """
    assert record["traced_queries"] == record["queries"], f"{record['task_id']} traced {record['traced_queries']} of {record['queries']} queries"
    curves = record["arms"]["dpo"]["curves"]
    assert len(curves) == 1, f"{record['task_id']} ran {len(curves)} settings, and a fixed recipe has one"
    scores = curves[0]["query_scores"]
    assert max(epochs) <= len(scores), f"{record['task_id']} ran {len(scores)} epochs and epoch {max(epochs)} was asked for"
    return {f"{prefix}{epoch}_dpo": scores[epoch - 1] for epoch in epochs}


def inside_states(scorer: DirectionalScorer, task: Task, layout: Layout, settings: Settings) -> float:
    """A ridge on the untuned model's hidden states, fit on a task's tuning rows, as its Spearman on the held-out rows."""
    context = row_chains(layout, task.context_rows)
    values = np.array(task.context_values)  # (k,)
    fit, held = tuning_split(len(context), settings.held_out, task.split)  # (k_fit,), (k_held,)
    states = scorer.states(context, settings.tokens_per_batch)  # (k, f)
    fit_states, held_states = standardize(states[fit], states[held])  # (k_fit, f), (k_held, f)
    return spearman(values[held], ridge(fit_states, values[fit], held_states))


def bundle(out: Path) -> None:
    """Every task's methods under every model as one file, and every record as another, for one transfer."""
    methods: dict[str, dict[str, list[float]]] = {}
    records = []
    for folder in sorted(path for path in out.iterdir() if path.is_dir()):
        for path in sorted(folder.glob("*__s[0-9]*.json")):
            methods.setdefault(path.stem, {}).update(json.loads(path.read_text(encoding="utf-8")))
        records += [json.loads(path.read_text(encoding="utf-8")) for path in sorted((folder / "records").glob("*.json"))]
    (out / "methods.json").write_text(json.dumps(methods), encoding="utf-8")
    (out / "records.json").write_text(json.dumps(records), encoding="utf-8")
    print(f"bundled {len(methods)} tasks and {len(records)} records under {out}", flush=True)


def recorded(folder: Path) -> list[dict[str, object]]:
    """The records of the splits a model has finished, by task."""
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted((folder / "records").glob("*.json"))]


def hold_gpu(seconds: float, yield_file: Path | None) -> None:
    """Take the GPU, compute nothing for a while, and leave if the launcher reports another client beside this one.

    A machine shared with timed runs is entered this way: a run that began in the same moment finds this process gone
    before its own work starts, and one that looks later finds the GPU taken and waits.
    """
    placeholder = torch.empty(64 * 2 ** 20, dtype=torch.uint8, device="cuda")  # (64 MiB,), so the process is listed as a client
    deadline = time.monotonic() + seconds
    while True:
        if yield_file is not None and yield_file.exists():
            print("YIELD: another client took the GPU while this run held it idle", flush=True)
            raise SystemExit(YIELDED)
        if time.monotonic() >= deadline:
            break
        time.sleep(2)
    del placeholder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--check", action="store_true", help="load the checkpoint, check it against its reference, and stop")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--tasks", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--splits", default="", help="splits to run, comma-separated; every split by default")
    parser.add_argument("--arms", default=",".join(ARMS), help="arms to run, comma-separated")
    parser.add_argument("--hours", type=float, default=6.0,
                        help="start no new split once the finished splits and this run have taken this many hours")
    parser.add_argument("--project", action="store_true",
                        help="print the time the waiting splits should take from the finished ones, and exit 3 if it passes --hours")
    parser.add_argument("--hold", type=float, default=0.0, help="seconds to hold the GPU idle before the first pass")
    parser.add_argument("--yield-file", type=Path, help="leave with status 75 if this file appears during the hold")
    parser.add_argument("--trace", type=int, default=0,
                        help="also keep the scores of this many leading queries after every epoch of every tuned arm")
    parser.add_argument("--learning-rates", default="",
                        help="learning rates of the grid, comma-separated, for a model too large for the whole grid; the whole grid by default")
    parser.add_argument("--betas", default="", help="weights of the ranked loss in the grid, comma-separated; both by default")
    parser.add_argument("--epochs", type=int, default=0, help="epochs of a tuning run; 20 by default")
    parser.add_argument("--arm", default="k256", help="the arm of the tasks to tune on, such as klarge for the largest example sets")
    parser.add_argument("--low-rank", type=int, default=0,
                        help="freeze the model and tune a low-rank update of this rank on its attention projections")
    parser.add_argument("--variant", default="",
                        help="a name for a run that changes one thing about the tuning: it goes into the method names and the folder")
    parser.add_argument("--inside", type=Path,
                        help="a folder of stored features: fit each baseline on the rows the tuning fits on, score it on the "
                             "held-out rows, write OUT/inside.json, and stop. No model is loaded and no GPU is used")
    parser.add_argument("--fixed-epochs", default="",
                        help="epochs, comma-separated: with one setting and a trace of every query, also write the ranked-loss "
                             "arm after each of them as a method of its own, with nothing chosen")
    parser.add_argument("--inside-states", action="store_true",
                        help="fit a ridge on the untuned model's hidden states on the rows the tuning fits on, score it on the "
                             "held-out rows, write OUT/inside_MODEL.json, and stop")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda")
    if args.check:
        load_progen2(args.model, device)
        return

    assert args.data is not None and args.tasks is not None and args.out is not None, "--data, --tasks and --out are required"
    arms = tuple(arm for arm in args.arms.split(",") if arm)
    unknown = sorted(set(arms) - set(ARMS))
    if unknown:
        raise SystemExit(f"unknown arms {unknown}; choose from {list(ARMS)}")
    wanted = {split for split in args.splits.split(",") if split}
    infos = {split_info.name: split_info for split_info in plan.ALL_SPLITS}
    unknown = sorted(wanted - set(infos))
    if unknown:
        raise SystemExit(f"unknown splits {unknown}")

    by_split = {task.split: task for task in read_tasks(args.tasks) if task.arm == args.arm and task.seed == 0}
    tasks = [by_split[name] for name in infos if name in by_split and (not wanted or name in wanted)]
    assert tasks, f"no task of arm {args.arm} and seed 0 under {args.tasks}"
    if args.inside is not None:
        models = tuple(name for name in PLM_NAMES if (args.inside / name).is_dir())
        found = []
        for task in tasks:
            split_info = infos[task.split]
            layout = load_layout(split_info, flip2.load_split(flip2.split_path(args.data, split_info)))
            found.append(inside_scores(task, layout, strata_of(split_info, layout), SplitFeatures(args.inside, split_info, models),
                                       Settings().held_out, models))
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "inside.json").write_text(json.dumps(found), encoding="utf-8")
        print(f"INSIDE wrote the held-out scores of the baselines under {list(models)} for {len(found)} splits to {args.out}", flush=True)
        return
    if args.inside_states:
        hold_gpu(args.hold, args.yield_file)
        scorer = load_progen2(args.model, device)
        found = []
        for task in tasks:
            split_info = infos[task.split]
            layout = load_layout(split_info, flip2.load_split(flip2.split_path(args.data, split_info)))
            fit, held = tuning_split(len(task.context_rows), Settings().held_out, task.split)
            score = inside_states(scorer, task, layout, Settings())
            found.append({"task_id": task.task_id, "split": task.split, "fit_rows": int(len(fit)), "held_out_rows": int(len(held)),
                          "held_out_spearman": {f"progen2_{args.model}_ridge": score}})
            print(f"STATES {task.split} held-out {score:.3f}", flush=True)
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / f"inside_{args.model}.json").write_text(json.dumps(found), encoding="utf-8")
        print(f"INSIDE wrote the held-out scores of the ridge on {args.model} hidden states for {len(found)} splits to {args.out}", flush=True)
        return
    name = f"{args.model}_{args.variant}" if args.variant else args.model
    prefix = f"progen2_{name}"
    fixed_epochs = tuple(int(epoch) for epoch in args.fixed_epochs.split(",") if epoch)
    for known in (prefix, *(f"{prefix}{epoch}" for epoch in fixed_epochs)):
        assert known in plan.PREFERENCE_MODELS, f"{known} is not a model the report knows"
    folder = args.out / name
    (folder / "records").mkdir(parents=True, exist_ok=True)
    finished = recorded(folder)
    spent = sum(float(record["seconds"]) for record in finished)
    waiting = [task for task in tasks if not (folder / "records" / f"{task.task_id}.json").exists()]

    def layout_of(task: Task) -> Layout:
        split_info = infos[task.split]
        return load_layout(split_info, flip2.load_split(flip2.split_path(args.data, split_info)))

    if args.project:
        assert finished, "a projection needs a finished split"
        sizes = [mean_tokens(row_chains(layout_of(task), task.context_rows)) for task in waiting]
        estimates = projected_seconds([(float(record["mean_tokens"]), float(record["seconds"])) for record in finished], sizes)
        for task, size, estimate in zip(waiting, sizes, estimates, strict=True):
            print(f"PROJECT {task.split} {size:.0f} tokens a row {estimate / 60:.1f} min", flush=True)
        total = spent + sum(estimates)
        fits = total <= args.hours * 3600
        print(f"PROJECTED {sum(estimates) / 3600:.2f} h for {len(waiting)} splits after {spent / 3600:.2f} h on {len(finished)}: "
              f"{total / 3600:.2f} h, which {'fits' if fits else 'passes'} the budget of {args.hours} h", flush=True)
        raise SystemExit(0 if fits else 3)

    free_bytes, total_bytes = torch.cuda.mem_get_info()
    print(f"{torch.cuda.get_device_name(0)}: {free_bytes / 2 ** 30:.1f} of {total_bytes / 2 ** 30:.1f} GiB free", flush=True)
    hold_gpu(args.hold, args.yield_file)
    scorer = load_progen2(args.model, device)
    if args.low_rank:
        trained = add_low_rank(scorer.model, args.low_rank)
        print(f"low-rank tuning at rank {args.low_rank}: {trained:,} trained weights", flush=True)
    grid = {"learning_rates": tuple(float(rate) for rate in args.learning_rates.split(",") if rate),
            "betas": tuple(float(beta) for beta in args.betas.split(",") if beta), "epochs": args.epochs}
    settings = Settings(**{key: value for key, value in grid.items() if value})
    started = time.monotonic()
    for task in waiting:
        hours = (spent + time.monotonic() - started) / 3600
        if hours > args.hours:
            print(f"STOPPED by the budget of {args.hours} h before {task.split}, after {hours:.2f} h", flush=True)
            break
        split_started = time.monotonic()
        # A trace longer than a task's queries keeps them all.
        methods, record = task_methods(scorer, prefix, task, layout_of(task), arms, settings, min(args.trace, len(task.query_rows)))
        methods |= fixed_epoch_methods(record, prefix, fixed_epochs) if fixed_epochs else {}
        record |= {"model": list(MODELS[args.model][:2]), "method_prefix": prefix, "arm": args.arm, "low_rank": args.low_rank,
                   "fixed_epochs": list(fixed_epochs), "seconds": time.monotonic() - split_started}
        path = folder / f"{task.task_id}.json"
        write_methods(path, methods, add=path.exists())
        (folder / "records" / f"{task.task_id}.json").write_text(json.dumps(record), encoding="utf-8")
        chosen = {arm: (found["recipe"]["learning_rate"], found["recipe"]["beta"], found["epoch"], round(found["held_out_spearman"], 3))
                  for arm, found in record["arms"].items()}
        print(f"SPLIT {task.split} {record['mean_tokens']:.0f} tokens a row {record['seconds']:.0f} s "
              f"untuned held-out {record['held_out_spearman_untuned']:.3f} chosen {chosen} failed {record['failed']}", flush=True)
    bundle(args.out)
    print(f"DONE {len(recorded(folder))} of {len(tasks)} splits, {(spent + time.monotonic() - started) / 3600:.2f} h in all", flush=True)


if __name__ == "__main__":
    main()

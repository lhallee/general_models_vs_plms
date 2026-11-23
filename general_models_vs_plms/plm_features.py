"""Protein language model features for the baselines: mean-pooled embeddings and likelihood scores.

    python -m general_models_vs_plms.plm_features --data DIR --tasks DIR --out DIR --model esmc_600m [--variant shuffled]

For every chain the main tasks touch, one forward pass gives the mean of the final hidden states over the residues
and the sum of the log-probabilities the model assigns to the residues it reads (an unmasked pseudo-log-likelihood).
For each parent chain, one pass per position with that position masked gives the masked marginals from which a
variant's zero-shot score is summed. These rows are computed by this experiment's own forward pass, in BF16, and are
not rows of the canonical embedding store.

Three variants are controls and save the embedding alone. `random_init` keeps a model's architecture and replaces its
weights by a seeded random draw, as `foundry.embedding.canonical.random_init_model` does for the canonical random
store. `shuffled` keeps the weights and shuffles the residues of every chain before the model reads it, each chain in
its own order, which keeps a chain's composition and removes everything else. `permuted` keeps the weights and moves
the residues of every chain by one order shared by all chains of that length, so the model reads a sequence that is
no protein while two variants of one parent still differ at the same places.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import numpy as np
import torch

from pathlib import Path
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

from . import flip2, plan
from .encoding import AMINO_ACIDS, CHAIN_BREAK, parent_text, sequence_text
from .tasks import load_layout, read_tasks


# Hub repository, pinned revision, whether the repository ships its own model code, and whether the weights are held
# in BF16 (the 6B model, whose FP32 weights and activations do not fit one 48 GB card together).
MODELS = {
    "esm2_650m": ("facebook/esm2_t33_650M_UR50D", "08e4846e537177426273712802403f7ba8261b6c", False, False),
    "esmc_300m": ("Synthyra/ESMplusplus_small", "46c5f7d562e47d4c14165b424c71ab7db008e6fb", True, False),
    "esmc_600m": ("Synthyra/ESMplusplus_large", "f813401638b3fddab09748aec1ad2bf537aa4208", True, False),
    "esmc_6b": ("Synthyra/ESMplusplus_6B", "0d579cce3b0f09efa6b3baddf6cc3fd8c9b616c8", True, True),
}
VARIANTS = {"pretrained": "", "random_init": "_random", "shuffled": "_shuffled", "permuted": "_permuted"}
RANDOM_SEED = 0
MASKED_BATCH = 64


def shuffled_chain(chain: str) -> str:
    """A chain's residues in an order drawn from the chain itself, so the same chain always shuffles the same way."""
    seed = int(hashlib.sha256(chain.encode("ascii")).hexdigest()[:16], 16)
    return "".join(np.random.default_rng(seed).permutation(list(chain)))


def permuted_chain(chain: str) -> str:
    """A chain's residues in the one order that every chain of its length shares."""
    order = np.random.default_rng([RANDOM_SEED, len(chain)]).permutation(len(chain))  # (l,)
    return "".join(chain[index] for index in order)


def read_form(chain: str, variant: str) -> str:
    """What the model reads for a chain under a variant: the chain, or a reordering of its residues."""
    if variant == "shuffled":
        return shuffled_chain(chain)
    if variant == "permuted":
        return permuted_chain(chain)
    return chain


class ChainScorer:
    """A masked language model with its tokenizer, on one device, read in BF16."""

    def __init__(self, name: str, device: torch.device, random_init: bool = False) -> None:
        repository, revision, remote_code, half_weights = MODELS[name]
        self.tokenizer = AutoTokenizer.from_pretrained(repository, revision=revision, trust_remote_code=remote_code)
        precision = {"torch_dtype": torch.bfloat16} if half_weights else {}
        model = AutoModelForMaskedLM.from_pretrained(repository, revision=revision, trust_remote_code=remote_code, **precision)
        if random_init:
            model = self.redraw(model)
        self.model = model.to(device).eval()
        self.device = device
        self.residue_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(list(AMINO_ACIDS)), device=device)  # (20,)
        assert len(set(self.residue_ids.tolist())) == 20, "the tokenizer does not give 20 distinct amino acid tokens"
        assert self.tokenizer.mask_token_id is not None, "the tokenizer has no mask token"
        # A model that returns its final state by name is not asked for every layer's state, which the 6B model
        # could not hold for a batch.
        input_ids, attention_mask = self.tokenize(["ACDEFGHIKLMNPQRSTVWY"])  # (1, 22), (1, 22)
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            probe = self.model(input_ids=input_ids, attention_mask=attention_mask)
        self.final_state_by_name = getattr(probe, "last_hidden_state", None) is not None

    @staticmethod
    def redraw(model: torch.nn.Module) -> torch.nn.Module:
        """The same architecture with every weight drawn afresh from the model's own initializer."""
        torch.manual_seed(RANDOM_SEED)
        fresh = type(model)(model.config)
        trained = dict(model.named_parameters())
        # A bias or a normalization gain can sit at its initial value after training; a weight matrix cannot.
        unchanged = [name for name, parameter in fresh.named_parameters()
                     if parameter.ndim >= 2 and torch.equal(parameter, trained[name].to(parameter.dtype))]
        assert not unchanged, f"random initialization left trained weights in place: {unchanged[:5]}"
        return fresh

    def tokenize(self, chains: list[str]) -> tuple[Tensor, Tensor]:
        """Token ids and attention mask, with one token per residue between the two special tokens."""
        batch = self.tokenizer(chains, padding=True, return_tensors="pt")
        input_ids = batch["input_ids"].to(self.device)  # (b, l + 2)
        attention_mask = batch["attention_mask"].to(self.device)  # (b, l + 2)
        lengths = torch.tensor([len(chain) for chain in chains], device=self.device)  # (b,)
        assert input_ids.shape[1] == int(lengths.max()) + 2, "the tokenizer did not give one token per residue"
        assert torch.equal(attention_mask.sum(dim=1), lengths + 2), "a chain's token count is not its length plus two"
        return input_ids, attention_mask  # (b, l + 2), (b, l + 2)

    @torch.inference_mode()
    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> tuple[Tensor, Tensor]:
        """Logits and final hidden states in float32."""
        # input_ids, attention_mask: (b, l + 2)
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            output = self.model(input_ids=input_ids, attention_mask=attention_mask,
                                output_hidden_states=not self.final_state_by_name)
        final = output.last_hidden_state if self.final_state_by_name else output.hidden_states[-1]  # (b, l + 2, d)
        return output.logits.float(), final.float()  # (b, l + 2, v), (b, l + 2, d)

    def embed(self, chains: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Mean final hidden state over residues, and the unmasked pseudo-log-likelihood, per chain."""
        input_ids, attention_mask = self.tokenize(chains)  # (b, l + 2), (b, l + 2)
        logits, hidden_states = self.forward(input_ids, attention_mask)  # (b, l + 2, v), (b, l + 2, d)
        lengths = attention_mask.sum(dim=1) - 2  # (b,)
        positions = torch.arange(input_ids.shape[1], device=self.device)  # (l + 2,)
        is_residue = (positions[None, :] >= 1) & (positions[None, :] <= lengths[:, None])  # (b, l + 2)
        weights = is_residue.float()  # (b, l + 2)
        pooled = (hidden_states * weights[:, :, None]).sum(dim=1) / lengths[:, None]  # (b, d)
        log_probabilities = torch.log_softmax(logits, dim=-1)  # (b, l + 2, v)
        observed = log_probabilities.gather(2, input_ids[:, :, None]).squeeze(2)  # (b, l + 2)
        likelihood = (observed * weights).sum(dim=1)  # (b,)
        return pooled.cpu().numpy(), likelihood.cpu().numpy()  # (b, d), (b,)

    def masked_marginals(self, chain: str) -> np.ndarray:
        """Log-probability of each amino acid at each position of a chain, with that position masked."""
        input_ids, attention_mask = self.tokenize([chain])  # (1, l + 2), (1, l + 2)
        length = len(chain)  # l
        table = np.zeros((length, len(AMINO_ACIDS)), dtype=np.float32)  # (l, 20)
        for start in range(0, length, MASKED_BATCH):
            columns = torch.arange(start, min(start + MASKED_BATCH, length), device=self.device)  # (m,)
            masked = input_ids.repeat(len(columns), 1)  # (m, l + 2)
            masked[torch.arange(len(columns), device=self.device), columns + 1] = self.tokenizer.mask_token_id
            logits, _ = self.forward(masked, attention_mask.repeat(len(columns), 1))  # (m, l + 2, v)
            at_mask = logits[torch.arange(len(columns), device=self.device), columns + 1]  # (m, v)
            residues = torch.log_softmax(at_mask, dim=-1)[:, self.residue_ids]  # (m, 20)
            table[start:start + len(columns)] = residues.cpu().numpy()
        return table  # (l, 20)


def length_batches(chains: list[str], tokens_per_batch: int) -> list[list[str]]:
    """Chains grouped longest first, each batch under the token budget."""
    ordered = sorted(chains, key=len, reverse=True)
    batches, current = [], []
    for chain in ordered:
        if current and (len(current) + 1) * (len(current[0]) + 2) > tokens_per_batch:
            batches.append(current)
            current = []
        current.append(chain)
    if current:
        batches.append(current)
    return batches


def chains_of(text: str) -> list[str]:
    """The chains of a sequence; a pair has two, and its peptide may be empty."""
    return text.split(CHAIN_BREAK)


def needed_rows(tasks_folder: Path) -> dict[str, np.ndarray]:
    """For each split, the rows any task shows or asks about."""
    rows: dict[str, set[int]] = {}
    for task in read_tasks(tasks_folder):
        rows.setdefault(task.split, set()).update(task.context_rows, task.query_rows)
    return {split: np.array(sorted(found)) for split, found in rows.items()}  # each (n_split,)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="pretrained")
    parser.add_argument("--limit", type=int, help="embed at most this many rows a split, for a pilot")
    parser.add_argument("--all-train", action="store_true", help="also embed every training row, for the full-data reference")
    parser.add_argument("--tokens-per-batch", type=int, default=40000)
    args = parser.parse_args()

    device = torch.device("cuda")
    scorer = ChainScorer(args.model, device, random_init=args.variant == "random_init")
    name = args.model + VARIANTS[args.variant]
    folder = args.out / name
    folder.mkdir(parents=True, exist_ok=True)
    infos = {split_info.name: split_info for split_info in plan.ALL_SPLITS}
    summary = {}
    for split, rows in needed_rows(args.tasks).items():
        split_info = infos[split]
        path = folder / f"{split_info.dataset}-{split_info.split}.npz"
        if path.exists():
            continue

        frame = flip2.load_split(flip2.split_path(args.data, split_info))
        if args.all_train:
            rows = np.union1d(rows, np.flatnonzero(frame["part"].to_numpy() == "train"))  # (n,)
        rows = rows[:args.limit] if args.limit else rows  # (n,)
        layout = load_layout(split_info, frame)
        texts = [sequence_text(layout, int(row)) for row in rows]
        unique_chains = sorted({chain for text in texts for chain in chains_of(text) if chain})
        read_as = {chain: read_form(chain, args.variant) for chain in unique_chains}
        embedding_of: dict[str, np.ndarray] = {}
        likelihood_of: dict[str, float] = {}
        for batch in tqdm(length_batches(unique_chains, args.tokens_per_batch), desc=f"{name} {split}", mininterval=20):
            pooled, likelihood = scorer.embed([read_as[chain] for chain in batch])  # (b, d), (b,)
            embedding_of.update(zip(batch, pooled, strict=True))
            likelihood_of.update(zip(batch, likelihood.tolist(), strict=True))

        width = len(next(iter(embedding_of.values())))  # d
        chain_count = max(len(chains_of(text)) for text in texts)  # c
        embeddings = np.zeros((len(rows), chain_count * width), dtype=np.float32)  # (n, c d)
        likelihoods = np.zeros(len(rows), dtype=np.float64)  # (n,)
        for index, text in enumerate(texts):
            for slot, chain in enumerate(chains_of(text)):
                if chain:
                    embeddings[index, slot * width:(slot + 1) * width] = embedding_of[chain]
                    likelihoods[index] += likelihood_of[chain]

        saved = {"rows": rows, "embedding": embeddings}
        if args.variant == "pretrained":
            saved["likelihood"] = likelihoods
            if layout.parent is not None:
                parent_chains = chains_of(parent_text(layout))
                saved["parent_marginals"] = np.concatenate([scorer.masked_marginals(chain) for chain in parent_chains])  # (l_residues, 20)
        np.savez(path, **saved)
        summary[split] = {"rows": int(len(rows)), "unique_chains": len(unique_chains), "width": int(embeddings.shape[1])}

    record = {"model": MODELS[args.model][:2], "variant": args.variant, "weights_in_bf16": MODELS[args.model][3],
              "random_seed": RANDOM_SEED if args.variant == "random_init" else None, "splits": summary}
    (folder / f"summary{'_more' if (folder / 'summary.json').exists() else ''}.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(f"wrote features of {len(summary)} splits to {folder}")


if __name__ == "__main__":
    main()

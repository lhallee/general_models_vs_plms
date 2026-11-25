"""Stronger ESMC adaptation (feature fusion, mutation-site local pooling, ConFit LoRA) and post-cutoff evaluation.

Implements three zero-API-spend improvements over the frozen mean-pooled ridge baselines of `general_models_vs_plms`:

1. **Frozen feature-fusion ridges (`fusion`)**: RidgeCV on standardized mean-pooled embeddings concatenated with the
   zero-shot masked-marginal / pseudo-log-likelihood score (and optionally one-hot sequence indicators), plus a
   multi-scale ESMC-600M + ESMC-6B + zero-shot ridge and a shuffled-label negative control.
2. **Mutation-site local representations (`mutsite`)**: For parented deep mutational scanning splits, extracts the
   final hidden states at the mutated residue positions `M` (`mean_{i in M} h_mut[i]` and
   `mean_{i in M} (h_mut[i] - h_wt[i])`) in one BF16 pass per variant, avoiding 1/L dilution from mean pooling across
   320-606 unchanged residues. Falls back cleanly to mean-pooled embeddings on unparented splits (`rhomax`).
3. **ConFit-style LoRA preference tuning of ESMC (`confit`)**: Attaches rank-8 LoRA adapters (initialized at zero so
   epoch 0 equals pretrained ESMC exactly) to ESMC attention projections, scores each parented variant by its
   simultaneous masked-marginal log-odds `sum_{i in M} [log p_theta(x_i | x_{-M}) - log p_theta(wt_i | x_{-M})]`
   (and unparented splits by mean amino-acid log-probability), and optimizes Bradley-Terry pairwise ranking loss plus
   a KL anchor to frozen ESMC logits on all 256 context rows with no test peeking.

Also provides a materialized runner for the expanded post-June-2026 MaveDB benchmark (`GBA1`, `ARAF` activity,
`ARAF` abundance) and exact 2,000-draw paired bootstrap comparison against existing scored tables.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import numpy as np
import pandas as pd
import torch

from dataclasses import asdict, dataclass
from pathlib import Path
from torch import Tensor, nn
from torch.nn import functional as F
from tqdm import tqdm

from . import flip2, plan
from .baselines import (
    LIKELIHOOD_WEIGHT,
    PLM_NAMES,
    SplitFeatures,
    marginal_scores,
    nearest_neighbor,
    ridge,
    standardize,
    stratum_mean,
)
from .encoding import (
    AMINO_ACIDS,
    CHAIN_BREAK,
    GAP_CODE,
    Layout,
    composition,
    mutation_counts,
    one_hot,
    parent_text,
    sequence_text,
)
from .flip2 import SplitInfo
from .plm_features import ChainScorer, chains_of
from .postcutoff_suite import POSTCUTOFF_ASSAYS, _extract_target_protein, parse_mavedb_missense
from .scoring import bootstrap_spearman, spearman
from .strata import strata_of
from .tasks import Arm, Task, build_task, load_layout, read_tasks


LORA_TARGETS = ("layernorm_qkv.1", "out_proj")
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 0


@dataclass(frozen=True)
class ConFitConfig:
    """Hyperparameters for ConFit-style LoRA preference tuning of ESMC."""

    rank: int = 8
    alpha: float = 16.0
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-5
    weight_decay: float = 0.01
    kl_weight: float = 0.1
    epochs: int = 15
    batch_size: int = 16
    eval_epochs: tuple[int, ...] = (0, 5, 10, 15)
    seed: int = 0
    max_grad_norm: float = 1.0


class LoRALinear(nn.Module):
    """Low-rank adapter wrapping a frozen Linear layer; B is zero-initialized for exact step-0 parity."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        self.base = base
        for param in self.base.parameters():
            param.requires_grad = False
        self.rank = rank
        self.scale = alpha / float(rank)
        self.lora_a = nn.Parameter(torch.empty((rank, base.in_features), dtype=torch.float32))
        self.lora_b = nn.Parameter(torch.zeros((base.out_features, rank), dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x: Tensor) -> Tensor:
        # x: (..., in_features)
        out = self.base(x)  # (..., out_features)
        delta = F.linear(F.linear(x.to(self.lora_a.dtype), self.lora_a), self.lora_b) * self.scale  # (..., out_features)
        return out + delta.to(out.dtype)  # (..., out_features)


def attach_lora(model: nn.Module, rank: int = 8, alpha: float = 16.0, targets: tuple[str, ...] = LORA_TARGETS) -> list[nn.Parameter]:
    """Freeze all base parameters and wrap matching Linear modules with LoRALinear."""
    for param in model.parameters():
        param.requires_grad = False
    replaced = 0
    for full_name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(full_name.endswith(suffix) for suffix in targets):
            continue
        parent_name, _, attr = full_name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        wrapped = LoRALinear(module, rank=rank, alpha=alpha).to(module.weight.device)
        setattr(parent, attr, wrapped)
        replaced += 1
    assert replaced > 0, f"no Linear layers matched LoRA targets {targets}"
    trainable = [param for param in model.parameters() if param.requires_grad]
    return trainable


def detach_lora(model: nn.Module) -> None:
    """Restore original unwrapped Linear modules in place so a base model can be reused cleanly across splits."""
    for full_name, module in list(model.named_modules()):
        if isinstance(module, LoRALinear):
            parent_name, _, attr = full_name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, attr, module.base)


def split_chain_offsets(layout: Layout) -> list[tuple[int, int]]:
    """Start and end column slices of each chain inside a layout (excluding chain-break columns)."""
    slices: list[tuple[int, int]] = []
    start = 0
    for index, label in enumerate(layout.labels):
        if label == CHAIN_BREAK:
            slices.append((start, index))
            start = index + 1
    slices.append((start, len(layout.labels)))
    return slices


def mutated_positions_per_chain(layout: Layout, row: int) -> list[list[tuple[int, str, str]]]:
    """For a parented layout, list of (residue_index_0based, wt_aa, mut_aa) per chain for one row."""
    assert layout.parent is not None, "mutated positions require a parent sequence"
    out: list[list[tuple[int, str, str]]] = []
    for start, end in split_chain_offsets(layout):
        chain_muts: list[tuple[int, str, str]] = []
        res_idx = 0
        for col in range(start, end):
            old_code = int(layout.parent[col])
            new_code = int(layout.codes[row, col])
            if old_code == GAP_CODE:
                continue
            if new_code != old_code and new_code != GAP_CODE:
                chain_muts.append((res_idx, chr(old_code), chr(new_code)))
            res_idx += 1
        out.append(chain_muts)
    return out


def bradley_terry_loss(scores: Tensor, targets: Tensor) -> Tensor:
    """Mean softplus pairwise ranking loss over all strictly ordered pairs (y_i > y_j) in a batch."""
    # scores, targets: (b,)
    diff_y = targets[:, None] - targets[None, :]  # (b, b)
    mask = diff_y > 0  # (b, b)
    if not bool(mask.any()):
        return scores.sum() * 0.0  # ()
    diff_s = scores[:, None] - scores[None, :]  # (b, b)
    return F.softplus(-diff_s[mask]).mean()  # ()


def kl_to_reference(ref_log_probs: Tensor, cur_log_probs: Tensor) -> Tensor:
    """Mean KL(p_ref || p_cur) over rows of 20-AA log-probabilities."""
    # ref_log_probs, cur_log_probs: (m, 20)
    if ref_log_probs.numel() == 0:
        return cur_log_probs.sum() * 0.0  # ()
    ref_probs = ref_log_probs.exp()  # (m, 20)
    return (ref_probs * (ref_log_probs - cur_log_probs)).sum(dim=-1).mean()  # ()


def fusion_baselines_for_task(
    task: Task,
    layout: Layout,
    features: SplitFeatures,
    models: tuple[str, ...] = PLM_NAMES,
    seed: int = 0,
) -> dict[str, list[float]]:
    """Compute frozen feature-fusion ridges and shuffled-label control on a task's context rows."""
    context_rows = np.array(task.context_rows, dtype=np.int64)  # (k,)
    query_rows = np.array(task.query_rows, dtype=np.int64)  # (q,)
    values = np.array(task.context_values, dtype=float)  # (k,)
    fit_onehot, query_onehot = one_hot(layout, context_rows, query_rows)  # (k, f), (q, f)

    preds: dict[str, np.ndarray] = {}
    std_embs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    std_zs: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for name in models:
        fit_emb, query_emb = standardize(
            features.take(name, "embedding", context_rows),
            features.take(name, "embedding", query_rows),
        )  # (k, d), (q, d)
        _, fit_score = features.zero_shot(name, layout, context_rows)  # (k,)
        _, query_score = features.zero_shot(name, layout, query_rows)  # (q,)
        fit_z, query_z = standardize(fit_score[:, None], query_score[:, None])  # (k, 1), (q, 1)
        std_embs[name] = (fit_emb, query_emb)
        std_zs[name] = (fit_z, query_z)

        # Combine standardized embedding + high-leverage zero-shot score
        fit_ez = np.hstack([fit_emb, LIKELIHOOD_WEIGHT * fit_z])  # (k, d + 1)
        query_ez = np.hstack([query_emb, LIKELIHOOD_WEIGHT * query_z])  # (q, d + 1)
        preds[f"ridge_emb_plus_zs_{name}"] = ridge(fit_ez, values, query_ez)

        # Combine one-hot + embedding + zero-shot score
        fit_oez = np.hstack([fit_onehot, fit_emb, LIKELIHOOD_WEIGHT * fit_z])  # (k, f + d + 1)
        query_oez = np.hstack([query_onehot, query_emb, LIKELIHOOD_WEIGHT * query_z])  # (q, f + d + 1)
        preds[f"ridge_onehot_emb_zs_{name}"] = ridge(fit_oez, values, query_oez)

    if "esmc_600m" in models and "esmc_6b" in models:
        fit_600, q_600 = std_embs["esmc_600m"]
        fit_6b, q_6b = std_embs["esmc_6b"]
        fit_z6b, q_z6b = std_zs["esmc_6b"]
        fit_ens = np.hstack([fit_600, fit_6b, LIKELIHOOD_WEIGHT * fit_z6b])  # (k, d1 + d2 + 1)
        q_ens = np.hstack([q_600, q_6b, LIKELIHOOD_WEIGHT * q_z6b])  # (q, d1 + d2 + 1)
        preds["ridge_esmc_600m_6b_zs"] = ridge(fit_ens, values, q_ens)

    if "esmc_600m" in models:
        shuffled_values = np.random.default_rng(seed).permutation(values)  # (k,)
        fit_600, q_600 = std_embs["esmc_600m"]
        fit_z, q_z = std_zs["esmc_600m"]
        preds["ridge_emb_plus_zs_esmc_600m_shuffled"] = ridge(
            np.hstack([fit_600, LIKELIHOOD_WEIGHT * fit_z]),
            shuffled_values,
            np.hstack([q_600, LIKELIHOOD_WEIGHT * q_z]),
        )

    return {method: np.asarray(scores, dtype=float).tolist() for method, scores in preds.items()}


@torch.inference_mode()
def extract_mutsite_features(
    scorer: ChainScorer,
    layout: Layout,
    rows: np.ndarray,
    batch_size: int = 32,
) -> dict[str, np.ndarray]:
    """Extract mean-pooled, mutation-site hidden states, mutation-site delta, and unmasked likelihood for rows.

    For parented splits, `mut_state` averages `h_mut[i]` across mutated sites `i in M`, and `mut_delta` averages
    `h_mut[i] - h_wt[i]` across `i in M` (falling back to mean-pool / zeros when `M` is empty or unparented).
    """
    # rows: (n,)
    unique_rows = np.unique(rows)  # (u,)
    n_rows = len(unique_rows)

    # First compute wild-type hidden states per chain when parent exists
    parent_states: list[np.ndarray] | None = None
    if layout.parent is not None:
        parent_chains = chains_of(parent_text(layout))
        parent_states = []
        for chain in parent_chains:
            ids, mask = scorer.tokenize([chain])  # (1, l + 2)
            _, hidden = scorer.forward(ids, mask)  # (1, l + 2, d)
            parent_states.append(hidden[0, 1 : len(chain) + 1].cpu().numpy())  # (l, d)

    # Determine hidden dim from a single probe
    sample_chains = chains_of(sequence_text(layout, int(unique_rows[0])))
    ids0, mask0 = scorer.tokenize([sample_chains[0]])
    _, h0 = scorer.forward(ids0, mask0)
    hidden_dim = int(h0.shape[-1])

    mean_embs = np.zeros((n_rows, hidden_dim), dtype=np.float32)  # (u, d)
    mut_states = np.zeros((n_rows, hidden_dim), dtype=np.float32)  # (u, d)
    mut_deltas = np.zeros((n_rows, hidden_dim), dtype=np.float32)  # (u, d)
    likelihoods = np.zeros(n_rows, dtype=np.float64)  # (u,)

    # Process chain by chain to keep lengths uniform where possible
    row_chains = [chains_of(sequence_text(layout, int(r))) for r in unique_rows]
    n_chains = len(row_chains[0])
    total_residues = np.zeros(n_rows, dtype=np.int64)
    mut_counts = np.zeros(n_rows, dtype=np.int64)

    for c_idx in range(n_chains):
        seqs = [rc[c_idx] for rc in row_chains]
        non_empty = [i for i, s in enumerate(seqs) if len(s) > 0]
        # Sort by length for minimal padding
        order = sorted(non_empty, key=lambda i: len(seqs[i]))
        for start in range(0, len(order), batch_size):
            batch_indices = order[start : start + batch_size]
            batch_seqs = [seqs[int(i)] for i in batch_indices]
            ids, mask = scorer.tokenize(batch_seqs)  # (b, l_max + 2)
            logits, hidden = scorer.forward(ids, mask)  # (b, l_max + 2, v), (b, l_max + 2, d)
            aa_log_probs = torch.log_softmax(logits[:, :, scorer.residue_ids], dim=-1)  # (b, l_max + 2, 20)
            aa_to_idx = {aa: idx for idx, aa in enumerate(AMINO_ACIDS)}

            for b_pos, u_idx in enumerate(batch_indices):
                u_idx_int = int(u_idx)
                seq = seqs[u_idx_int]
                seq_len = len(seq)
                h_seq = hidden[b_pos, 1 : seq_len + 1].cpu().numpy()  # (l, d)
                mean_embs[u_idx_int] += h_seq.sum(axis=0)
                total_residues[u_idx_int] += seq_len

                seq_aa_ids = torch.tensor([aa_to_idx[ch] for ch in seq], device=aa_log_probs.device)  # (l,)
                lp = aa_log_probs[b_pos, 1 : seq_len + 1, :]  # (l, 20)
                likelihoods[u_idx_int] += float(lp[torch.arange(seq_len, device=lp.device), seq_aa_ids].sum().item())

                if parent_states is not None:
                    row_id = int(unique_rows[u_idx_int])
                    chain_muts = mutated_positions_per_chain(layout, row_id)[c_idx]
                    valid_muts = [pos for pos, _, _ in chain_muts if pos < seq_len and pos < len(parent_states[c_idx])]
                    if valid_muts:
                        wt_h = parent_states[c_idx][valid_muts]  # (m, d)
                        mut_h = h_seq[valid_muts]  # (m, d)
                        mut_states[u_idx_int] += mut_h.sum(axis=0)
                        mut_deltas[u_idx_int] += (mut_h - wt_h).sum(axis=0)
                        mut_counts[u_idx_int] += len(valid_muts)

    mean_embs /= np.maximum(total_residues[:, None], 1)
    likelihoods /= np.maximum(total_residues, 1)

    for u_idx in range(n_rows):
        if mut_counts[u_idx] > 0:
            mut_states[u_idx] /= float(mut_counts[u_idx])
            mut_deltas[u_idx] /= float(mut_counts[u_idx])
        else:
            mut_states[u_idx] = mean_embs[u_idx]
            mut_deltas[u_idx] = 0.0

    extracted: dict[str, np.ndarray] = {
        "rows": unique_rows,
        "mean_emb": mean_embs,
        "mut_state": mut_states,
        "mut_delta": mut_deltas,
        "likelihood": likelihoods,
    }
    if layout.parent is not None:
        parent_chains = chains_of(parent_text(layout))
        extracted["parent_marginals"] = np.concatenate([scorer.masked_marginals(c) for c in parent_chains])
    return extracted  # rows: (u,), mean_emb/mut_state/mut_delta: (u, d), likelihood: (u,)


def stacked_multiview_ridge(
    fit_views: list[np.ndarray],
    query_views: list[np.ndarray],
    fit_zs: np.ndarray,
    query_zs: np.ndarray,
    values: np.ndarray,
    n_folds: int = 5,
    seed: int = 0,
) -> np.ndarray:
    """Block-regularized multi-view stacking via out-of-fold RidgeCV predictions on context rows only."""
    # fit_views: (k, f_v); query_views: (q, f_v); fit_zs: (k,); query_zs: (q,); values: (k,)
    n_ctx = len(values)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_ctx)
    folds = np.array_split(perm, min(n_folds, n_ctx))

    oof_cols: list[np.ndarray] = []
    qry_cols: list[np.ndarray] = []

    for fv, qv in zip(fit_views, query_views, strict=True):
        oof = np.empty(n_ctx, dtype=float)  # (k,)
        for fold_idx in range(len(folds)):
            val_idx = folds[fold_idx]
            tr_idx = np.concatenate([folds[j] for j in range(len(folds)) if j != fold_idx])
            oof[val_idx] = ridge(fv[tr_idx], values[tr_idx], fv[val_idx])
        qry_pred = ridge(fv, values, qv)  # (q,)
        oof_cols.append(oof[:, None])
        qry_cols.append(qry_pred[:, None])

    oof_cols.append(fit_zs[:, None])
    qry_cols.append(query_zs[:, None])

    meta_fit, meta_qry = standardize(np.hstack(oof_cols), np.hstack(qry_cols))  # (k, m), (q, m)
    return ridge(meta_fit, values, meta_qry)  # (q,)


def mutsite_ridges_for_task(
    task: Task,
    layout: Layout,
    arrays: dict[str, np.ndarray],
    model_name: str,
    include_shuffled: bool = False,
    seed: int = 0,
) -> dict[str, list[float]]:
    """Fit mutation-site local ridges + zero-shot fusion + stacked multi-view ridge on a task's context rows."""
    # arrays: (u,) rows, (u, d) embeddings
    pos_map = {int(r): i for i, r in enumerate(arrays["rows"])}
    context_rows = np.array(task.context_rows, dtype=np.int64)
    query_rows = np.array(task.query_rows, dtype=np.int64)
    ctx_idx = [pos_map[int(r)] for r in context_rows]
    qry_idx = [pos_map[int(r)] for r in query_rows]
    values = np.array(task.context_values, dtype=float)

    fit_oh, qry_oh = one_hot(layout, context_rows, query_rows)
    fit_mean, qry_mean = standardize(arrays["mean_emb"][ctx_idx], arrays["mean_emb"][qry_idx])
    fit_ms, qry_ms = standardize(arrays["mut_state"][ctx_idx], arrays["mut_state"][qry_idx])
    fit_md, qry_md = standardize(arrays["mut_delta"][ctx_idx], arrays["mut_delta"][qry_idx])

    if layout.parent is not None and "parent_marginals" in arrays:
        fit_zs = marginal_scores(layout, arrays["parent_marginals"], context_rows)
        qry_zs = marginal_scores(layout, arrays["parent_marginals"], query_rows)
    else:
        fit_zs = arrays["likelihood"][ctx_idx]
        qry_zs = arrays["likelihood"][qry_idx]
    fit_z, qry_z = standardize(fit_zs[:, None], qry_zs[:, None])

    preds: dict[str, np.ndarray] = {}
    if layout.parent is not None:
        fit_local = np.hstack([fit_ms, fit_md])
        qry_local = np.hstack([qry_ms, qry_md])
        stack_fit_views = [fit_mean, fit_local, fit_oh]
        stack_qry_views = [qry_mean, qry_local, qry_oh]
    else:
        fit_local, qry_local = fit_mean, qry_mean
        stack_fit_views = [fit_mean, fit_oh]
        stack_qry_views = [qry_mean, qry_oh]

    preds[f"ridge_mutsite_{model_name}"] = ridge(fit_local, values, qry_local)
    preds[f"ridge_mutsite_plus_zs_{model_name}"] = ridge(
        np.hstack([fit_local, LIKELIHOOD_WEIGHT * fit_z]),
        values,
        np.hstack([qry_local, LIKELIHOOD_WEIGHT * qry_z]),
    )
    preds[f"ridge_mutsite_mean_zs_{model_name}"] = ridge(
        np.hstack([fit_mean, fit_local, LIKELIHOOD_WEIGHT * fit_z]),
        values,
        np.hstack([qry_mean, qry_local, LIKELIHOOD_WEIGHT * qry_z]),
    )
    preds[f"ridge_stacked_{model_name}"] = stacked_multiview_ridge(
        stack_fit_views, stack_qry_views, fit_zs, qry_zs, values, seed=seed
    )
    if include_shuffled:
        shuffled = np.random.default_rng(seed).permutation(values)
        preds[f"ridge_mutsite_plus_zs_{model_name}_shuffled"] = ridge(
            np.hstack([fit_local, LIKELIHOOD_WEIGHT * fit_z]),
            shuffled,
            np.hstack([qry_local, LIKELIHOOD_WEIGHT * qry_z]),
        )
        preds[f"ridge_stacked_{model_name}_shuffled"] = stacked_multiview_ridge(
            stack_fit_views, stack_qry_views, fit_zs, qry_zs, shuffled, seed=seed
        )
    return {k: np.asarray(v, dtype=float).tolist() for k, v in preds.items()}


@dataclass
class EncodedVariant:
    """Pre-tokenized masked variant inputs for fast ConFit scoring and training."""

    chain_input_ids: list[list[int]]
    # Per chain: list of (token_index_1based, wt_aa_index_0to19, mut_aa_index_0to19); empty for unparented
    chain_muts: list[list[tuple[int, int, int]]]
    # Per chain: list of aa_index_0to19 for every residue (used only when unparented)
    chain_aa_indices: list[list[int]]


def encode_variants_for_confit(scorer: ChainScorer, layout: Layout, rows: np.ndarray) -> list[EncodedVariant]:
    """Pre-tokenize variants with all mutated sites masked simultaneously for parented splits."""
    # rows: (n,)
    aa_to_idx = {aa: idx for idx, aa in enumerate(AMINO_ACIDS)}
    mask_id = int(scorer.tokenizer.mask_token_id)
    encoded: list[EncodedVariant] = []

    for row in rows:
        row_int = int(row)
        chains = chains_of(sequence_text(layout, row_int))
        muts_all = mutated_positions_per_chain(layout, row_int) if layout.parent is not None else [[] for _ in chains]
        chain_ids_list: list[list[int]] = []
        chain_muts_list: list[list[tuple[int, int, int]]] = []
        chain_aa_list: list[list[int]] = []

        for chain, muts in zip(chains, muts_all, strict=True):
            if not chain:
                chain_ids_list.append([])
                chain_muts_list.append([])
                chain_aa_list.append([])
                continue
            ids = scorer.tokenizer(chain, add_special_tokens=True)["input_ids"]
            assert len(ids) == len(chain) + 2, "expected 1 token per residue plus 2 special tokens"
            mut_tuples: list[tuple[int, int, int]] = []
            if layout.parent is not None:
                for res_idx, wt_aa, mut_aa in muts:
                    tok_idx = res_idx + 1
                    if tok_idx < len(ids) - 1:
                        ids[tok_idx] = mask_id
                        mut_tuples.append((tok_idx, aa_to_idx[wt_aa], aa_to_idx[mut_aa]))
            chain_ids_list.append(ids)
            chain_muts_list.append(mut_tuples)
            chain_aa_list.append([aa_to_idx[ch] for ch in chain])

        encoded.append(EncodedVariant(chain_ids_list, chain_muts_list, chain_aa_list))
    return encoded


def score_confit_batch(
    scorer: ChainScorer,
    batch: list[EncodedVariant],
    has_parent: bool,
    return_log_probs: bool = False,
) -> tuple[Tensor, Tensor | None]:
    """Compute differentiable ConFit scores (and optionally scored-position 20-AA log-probs) for a batch."""
    device = scorer.device
    pad_id = int(scorer.tokenizer.pad_token_id)
    n_batch = len(batch)
    n_chains = len(batch[0].chain_input_ids)
    total_scores = torch.zeros(n_batch, device=device, dtype=torch.float32)
    total_residues = torch.zeros(n_batch, device=device, dtype=torch.float32)
    collected_lp: list[Tensor] = []

    for c_idx in range(n_chains):
        seqs = [item.chain_input_ids[c_idx] for item in batch]
        non_empty_indices = [i for i, s in enumerate(seqs) if len(s) > 0]
        if not non_empty_indices:
            continue
        max_len = max(len(seqs[i]) for i in non_empty_indices)
        sub_n = len(non_empty_indices)
        input_ids = torch.full((sub_n, max_len), pad_id, dtype=torch.long, device=device)
        attn_mask = torch.zeros((sub_n, max_len), dtype=torch.long, device=device)
        for pos, i in enumerate(non_empty_indices):
            seq = seqs[i]
            input_ids[pos, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
            attn_mask[pos, : len(seq)] = 1

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            out = scorer.model(input_ids=input_ids, attention_mask=attn_mask)
            logits = out.logits[:, :, scorer.residue_ids].float()  # (b, l_max, 20)
        log_probs = torch.log_softmax(logits, dim=-1)  # (b, l_max, 20)

        for pos, i in enumerate(non_empty_indices):
            item = batch[i]
            if has_parent:
                muts = item.chain_muts[c_idx]
                if muts:
                    tok_idx = torch.tensor([m[0] for m in muts], dtype=torch.long, device=device)
                    wt_idx = torch.tensor([m[1] for m in muts], dtype=torch.long, device=device)
                    mut_idx = torch.tensor([m[2] for m in muts], dtype=torch.long, device=device)
                    pos_lp = log_probs[pos, tok_idx, :]  # (m, 20)
                    total_scores[i] = total_scores[i] + (
                        pos_lp[torch.arange(len(muts), device=device), mut_idx]
                        - pos_lp[torch.arange(len(muts), device=device), wt_idx]
                    ).sum()
                    if return_log_probs:
                        collected_lp.append(pos_lp)
            else:
                aa_idx = torch.tensor(item.chain_aa_indices[c_idx], dtype=torch.long, device=device)
                seq_len = len(aa_idx)
                pos_lp = log_probs[pos, 1 : seq_len + 1, :]  # (l, 20)
                total_scores[i] = total_scores[i] + pos_lp[torch.arange(seq_len, device=device), aa_idx].sum()
                total_residues[i] = total_residues[i] + float(seq_len)
                if return_log_probs:
                    collected_lp.append(pos_lp)

    if not has_parent:
        total_scores = total_scores / total_residues.clamp_min(1.0)
    lp_cat = torch.cat(collected_lp, dim=0) if (return_log_probs and collected_lp) else None
    return total_scores, lp_cat  # (b,), (m, 20) or None


@torch.inference_mode()
def predict_confit(
    scorer: ChainScorer,
    encoded_queries: list[EncodedVariant],
    has_parent: bool,
    batch_size: int = 32,
) -> np.ndarray:
    """Score all encoded query variants in eval mode."""
    scorer.model.eval()
    scores_out = np.empty(len(encoded_queries), dtype=np.float64)  # (q,)
    for start in range(0, len(encoded_queries), batch_size):
        batch = encoded_queries[start : start + batch_size]
        s, _ = score_confit_batch(scorer, batch, has_parent=has_parent, return_log_probs=False)
        scores_out[start : start + len(batch)] = s.cpu().numpy()
    return scores_out  # (q,)


def run_confit_split(
    scorer: ChainScorer,
    layout: Layout,
    context_rows: np.ndarray,
    context_values: np.ndarray,
    query_rows: np.ndarray,
    config: ConFitConfig,
    shuffle_labels: bool = False,
) -> dict[str, object]:
    """Run ConFit LoRA preference tuning on one split and return predictions at configured epochs plus timing."""
    # context_rows: (k,); context_values: (k,); query_rows: (q,)
    t0 = time.perf_counter()
    has_parent = layout.parent is not None
    encoded_ctx = encode_variants_for_confit(scorer, layout, context_rows)
    encoded_qry = encode_variants_for_confit(scorer, layout, query_rows)

    values = np.array(context_values, dtype=np.float32)
    if shuffle_labels:
        values = np.random.default_rng(config.seed).permutation(values)

    # Precompute reference log-probs once before attaching LoRA (identical to step 0)
    scorer.model.eval()
    ref_lp_per_item: list[Tensor] = []
    with torch.inference_mode():
        for item in encoded_ctx:
            _, lp = score_confit_batch(scorer, [item], has_parent=has_parent, return_log_probs=True)
            ref_lp_per_item.append(lp.detach() if lp is not None else torch.empty((0, 20), device=scorer.device))

    torch.manual_seed(config.seed)
    trainable = attach_lora(scorer.model, rank=config.rank, alpha=config.alpha)
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate, weight_decay=config.weight_decay)
    n_ctx = len(encoded_ctx)
    steps_per_epoch = max(1, math.ceil(n_ctx / config.batch_size))
    total_steps = config.epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_steps), eta_min=config.min_learning_rate
    )

    preds_by_epoch: dict[int, list[float]] = {}
    epoch_logs: list[dict[str, float]] = []

    try:
        if 0 in config.eval_epochs:
            preds_by_epoch[0] = predict_confit(scorer, encoded_qry, has_parent=has_parent).tolist()

        rng = np.random.default_rng(config.seed)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        for epoch in range(1, config.epochs + 1):
            scorer.model.train()
            order = rng.permutation(n_ctx)
            rank_losses: list[float] = []
            kl_losses: list[float] = []

            for start in range(0, n_ctx, config.batch_size):
                idx = order[start : start + config.batch_size]
                batch_items = [encoded_ctx[int(i)] for i in idx]
                batch_y = torch.tensor(values[idx], dtype=torch.float32, device=scorer.device)
                batch_ref_lp = torch.cat([ref_lp_per_item[int(i)] for i in idx], dim=0)

                optimizer.zero_grad(set_to_none=True)
                scores_t, cur_lp = score_confit_batch(
                    scorer, batch_items, has_parent=has_parent, return_log_probs=(config.kl_weight > 0)
                )
                l_rank = bradley_terry_loss(scores_t, batch_y)
                if config.kl_weight > 0 and cur_lp is not None and batch_ref_lp.numel() > 0:
                    l_kl = kl_to_reference(batch_ref_lp, cur_lp)
                else:
                    l_kl = torch.zeros((), device=scorer.device)
                loss = l_rank + config.kl_weight * l_kl
                assert torch.isfinite(loss), f"non-finite ConFit loss at epoch {epoch}"
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, config.max_grad_norm)
                optimizer.step()
                scheduler.step()

                rank_losses.append(float(l_rank.item()))
                kl_losses.append(float(l_kl.item()))

            epoch_logs.append(
                {
                    "epoch": float(epoch),
                    "rank_loss": float(np.mean(rank_losses)),
                    "kl_loss": float(np.mean(kl_losses)),
                }
            )
            if epoch in config.eval_epochs:
                preds_by_epoch[epoch] = predict_confit(scorer, encoded_qry, has_parent=has_parent).tolist()

        peak_gb = (
            float(torch.cuda.max_memory_allocated() / (1024**3)) if torch.cuda.is_available() else 0.0
        )
    finally:
        detach_lora(scorer.model)
        scorer.model.eval()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed = time.perf_counter() - t0
    return {
        "predictions_by_epoch": {str(k): v for k, v in preds_by_epoch.items()},
        "epoch_logs": epoch_logs,
        "elapsed_seconds": round(elapsed, 2),
        "peak_gpu_gb": round(peak_gb, 3),
        "trainable_params": int(sum(p.numel() for p in trainable)),
    }


def materialize_expanded_postcutoff(raw_dir: Path, data_out: Path, tasks_out: Path, seed: int = 0) -> list[SplitInfo]:
    """Build FLIP2-format `random` and `by_position` splits and `k256` tasks for GBA1, ARAF activity, and ARAF abundance."""
    raw_map = {
        "gba1_gcase_activity": ("gba1_meta.json", "gba1_scores.csv", "gba1_activity"),
        "araf_activity_baseline": ("araf_meta.json", "araf_act_scores.csv", "araf_activity"),
        "araf_abundance_baseline": ("araf_meta.json", "araf_ab_scores.csv", "araf_abundance"),
    }
    assay_by_slug = {a.slug: a for a in POSTCUTOFF_ASSAYS}
    split_infos: list[SplitInfo] = []
    data_out.mkdir(parents=True, exist_ok=True)
    tasks_out.mkdir(parents=True, exist_ok=True)

    for slug, (meta_name, csv_name, dataset_name) in raw_map.items():
        assay = assay_by_slug[slug]
        meta_path = raw_dir / meta_name
        csv_path = raw_dir / csv_name
        assert meta_path.exists() and csv_path.exists(), f"missing raw files for {slug}: {meta_path}, {csv_path}"
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        parent = _extract_target_protein(metadata, assay.target_gene_name)
        missense, synonymous, _ = parse_mavedb_missense(csv_path.read_bytes(), parent, gene_prefix=assay.gene_prefix)
        raw_scores = missense["score"].to_numpy(dtype=float)  # (n,)
        if assay.score_transform == "log2":
            assert (raw_scores > 0).all(), f"{assay.urn}: non-positive scores under log2 transform"
            target = np.log2(raw_scores)  # (n,)
        else:
            target = raw_scores  # (n,)

        sequences = [
            parent[: int(pos) - 1] + mut + parent[int(pos) :]
            for pos, mut in zip(missense["position"], missense["mutant"], strict=True)
        ]
        rng = np.random.default_rng([seed, 2026, len(parent)])
        by_variant = rng.random(len(missense)) < 0.5
        positions = np.sort(missense["position"].unique())
        test_positions = rng.permutation(positions)[: len(positions) // 2]
        by_position = missense["position"].isin(test_positions).to_numpy()

        ds_dir = data_out / dataset_name
        ds_dir.mkdir(parents=True, exist_ok=True)
        for split_name, is_test in (("random", by_variant), ("by_position", by_position)):
            frame = pd.DataFrame(
                {
                    "sequence": sequences,
                    "target": target,
                    "set": np.where(is_test, "test", "train"),
                    "validation": False,
                }
            )
            frame.to_csv(ds_dir / f"{split_name}.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
            split_spec = SplitInfo(
                dataset=dataset_name,
                split=split_name,
                split_type="mutation" if split_name == "random" else "position",
                layout="single_parent",
                description=assay.description,
                published_in=f"MaveDB, as the score set {assay.urn}",
            )
            split_infos.append(split_spec)
            loaded_frame = flip2.load_split(ds_dir / f"{split_name}.csv.gz")
            layout = load_layout(split_spec, loaded_frame)
            for arm in (
                Arm("k256", 256, representation="substitutions"),
                Arm("k256_cipher", 256, cipher=True, representation="substitutions"),
                Arm("k256_flipped", 256, labels="flipped", representation="substitutions"),
                Arm("k0_described", 0, described=True, representation="substitutions"),
            ):
                task = build_task(split_spec, loaded_frame, layout, arm, seed=0, part="test", query_count=plan.MAIN_QUERIES)
                (tasks_out / f"{task.task_id}.json").write_text(json.dumps(asdict(task)), encoding="utf-8")

        (ds_dir / "provenance.json").write_text(
            json.dumps(
                {
                    "slug": slug,
                    "urn": assay.urn,
                    "published_date": assay.published_date,
                    "parent_length": len(parent),
                    "missense_count": int(len(missense)),
                    "synonymous_count": int(len(synonymous)),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return split_infos


def paired_summary(
    truth_by_split: dict[str, np.ndarray],
    preds_by_method: dict[str, dict[str, np.ndarray]],
    baseline_method: str,
    splits: list[str],
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, dict[str, float]]:
    """Mean Spearman and 95% paired bootstrap difference against `baseline_method` across `splits`."""
    # truth_by_split: (q,); preds_by_method: (q,)
    methods = sorted(preds_by_method.keys())
    assert baseline_method in preds_by_method, f"baseline {baseline_method} missing"
    boot_scores = np.zeros((draws, len(methods)), dtype=np.float64)
    point_scores = np.zeros(len(methods), dtype=np.float64)

    for s_idx, split_name in enumerate(splits):
        truth = truth_by_split[split_name]
        pred_mat = np.vstack([preds_by_method[m][split_name] for m in methods])  # (m, q)
        boot_scores += bootstrap_spearman(truth, pred_mat, draws=draws, seed=seed + s_idx)
        for m_idx, m in enumerate(methods):
            point_scores[m_idx] += spearman(truth, preds_by_method[m][split_name])

    boot_scores /= float(len(splits))
    point_scores /= float(len(splits))
    base_idx = methods.index(baseline_method)

    summary: dict[str, dict[str, float]] = {}
    for m_idx, m in enumerate(methods):
        lo, hi = np.quantile(boot_scores[:, m_idx], [0.025, 0.975])
        diff_boot = boot_scores[:, m_idx] - boot_scores[:, base_idx]
        d_lo, d_hi = np.quantile(diff_boot, [0.025, 0.975])
        summary[m] = {
            "mean_spearman": round(float(point_scores[m_idx]), 4),
            "ci_low": round(float(lo), 4),
            "ci_high": round(float(hi), 4),
            "diff_vs_baseline": round(float(point_scores[m_idx] - point_scores[base_idx]), 4),
            "diff_ci_low": round(float(d_lo), 4),
            "diff_ci_high": round(float(d_hi), 4),
            "beats_baseline": bool(d_lo > 0.0),
        }
    return summary


def run_fusion_stage(data_dir: Path, tasks_dir: Path, features_dir: Path, out_dir: Path, splits_filter: set[str] | None) -> None:
    """Run Track 1A frozen feature-fusion baselines on k256 seed 0 tasks."""
    out_dir.mkdir(parents=True, exist_ok=True)
    infos = {s.name: s for s in plan.ALL_SPLITS}
    tasks = [t for t in read_tasks(tasks_dir) if t.arm == "k256" and t.seed == 0]
    if splits_filter:
        tasks = [t for t in tasks if t.split in splits_filter]
    methods_by_task: dict[str, dict[str, list[float]]] = {}
    for task in tqdm(tasks, desc="fusion"):
        split_info = infos[task.split]
        frame = flip2.load_split(flip2.split_path(data_dir, split_info))
        layout = load_layout(split_info, frame)
        feats = SplitFeatures(features_dir, split_info, PLM_NAMES)
        methods_by_task[task.task_id] = fusion_baselines_for_task(task, layout, feats, PLM_NAMES)
    (out_dir / "fusion_methods.json").write_text(json.dumps(methods_by_task), encoding="utf-8")
    print(f"wrote fusion baselines for {len(methods_by_task)} tasks to {out_dir / 'fusion_methods.json'}")


def run_mutsite_stage(
    data_dir: Path,
    tasks_dir: Path,
    out_dir: Path,
    models: tuple[str, ...],
    splits_filter: set[str] | None,
    extra_splits: list[SplitInfo] | None = None,
    include_model_free: bool = False,
) -> None:
    """Run Track 1B / Track 2 mutation-site extraction and ridges."""
    out_dir.mkdir(parents=True, exist_ok=True)
    all_infos = {s.name: s for s in (*plan.ALL_SPLITS, *(extra_splits or ()))}
    tasks = [t for t in read_tasks(tasks_dir) if t.arm == "k256" and t.seed == 0]
    if splits_filter:
        tasks = [t for t in tasks if t.split in splits_filter]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    methods_path = out_dir / "mutsite_methods.json"
    methods_by_task: dict[str, dict[str, list[float]]] = (
        json.loads(methods_path.read_text(encoding="utf-8")) if methods_path.exists() else {}
    )
    timing: dict[str, dict[str, float]] = {}

    if include_model_free:
        for task in tasks:
            split_spec = all_infos[task.split]
            frame = flip2.load_split(flip2.split_path(data_dir, split_spec))
            layout = load_layout(split_spec, frame)
            strata = strata_of(split_spec, layout)
            ctx = np.array(task.context_rows, dtype=np.int64)
            qry = np.array(task.query_rows, dtype=np.int64)
            vals = np.array(task.context_values, dtype=float)
            fit_oh, qry_oh = one_hot(layout, ctx, qry)
            fit_c, qry_c = standardize(composition(layout, ctx), composition(layout, qry))
            mf = {
                "ridge_onehot": ridge(fit_oh, vals, qry_oh).tolist(),
                "ridge_composition": ridge(fit_c, vals, qry_c).tolist(),
                "nearest_neighbor": nearest_neighbor(layout, ctx, vals, qry).tolist(),
                "stratum_mean": stratum_mean(strata, ctx, vals, qry).tolist(),
            }
            if layout.parent is not None:
                mf["mutation_count"] = (-mutation_counts(layout)[qry].astype(float)).tolist()
            methods_by_task.setdefault(task.task_id, {}).update(mf)

    for model_name in models:
        t_model = time.perf_counter()
        scorer = ChainScorer(model_name, device=device)
        feat_dir = out_dir / "features" / model_name
        feat_dir.mkdir(parents=True, exist_ok=True)
        for task in tqdm(tasks, desc=f"mutsite:{model_name}"):
            split_spec = all_infos[task.split]
            frame = flip2.load_split(flip2.split_path(data_dir, split_spec))
            layout = load_layout(split_spec, frame)
            npz_path = feat_dir / f"{split_spec.dataset}-{split_spec.split}.npz"
            rows_needed = np.unique(np.concatenate([np.array(task.context_rows), np.array(task.query_rows)]))
            if npz_path.exists():
                arrays = dict(np.load(npz_path))
            else:
                bs = 16 if model_name == "esmc_6b" else 32
                arrays = extract_mutsite_features(scorer, layout, rows_needed, batch_size=bs)
                np.savez_compressed(npz_path, **arrays)

            # Standard zero-shot + mean-pool ridge + onehot+zs + emb+zs + mutsite ridges
            pos_map = {int(r): i for i, r in enumerate(arrays["rows"])}
            ctx = np.array(task.context_rows, dtype=np.int64)
            qry = np.array(task.query_rows, dtype=np.int64)
            ctx_i = [pos_map[int(r)] for r in ctx]
            qry_i = [pos_map[int(r)] for r in qry]
            vals = np.array(task.context_values, dtype=float)
            fit_oh, qry_oh = one_hot(layout, ctx, qry)
            fit_m, qry_m = standardize(arrays["mean_emb"][ctx_i], arrays["mean_emb"][qry_i])
            if layout.parent is not None and "parent_marginals" in arrays:
                fit_zs = marginal_scores(layout, arrays["parent_marginals"], ctx)
                qry_zs = marginal_scores(layout, arrays["parent_marginals"], qry)
            else:
                fit_zs = arrays["likelihood"][ctx_i]
                qry_zs = arrays["likelihood"][qry_i]
            fit_z, qry_z = standardize(fit_zs[:, None], qry_zs[:, None])

            extra = {
                f"zero_shot_{model_name}": qry_zs.tolist(),
                f"ridge_{model_name}": ridge(fit_m, vals, qry_m).tolist(),
                f"ridge_onehot_plus_{model_name}": ridge(
                    np.hstack([fit_oh, LIKELIHOOD_WEIGHT * fit_z]), vals, np.hstack([qry_oh, LIKELIHOOD_WEIGHT * qry_z])
                ).tolist(),
                f"ridge_emb_plus_zs_{model_name}": ridge(
                    np.hstack([fit_m, LIKELIHOOD_WEIGHT * fit_z]), vals, np.hstack([qry_m, LIKELIHOOD_WEIGHT * qry_z])
                ).tolist(),
            }
            mut_preds = mutsite_ridges_for_task(
                task, layout, arrays, model_name, include_shuffled=(model_name == "esmc_600m")
            )
            methods_by_task.setdefault(task.task_id, {}).update({**extra, **mut_preds})
            methods_path.write_text(json.dumps(methods_by_task), encoding="utf-8")

        timing[model_name] = {"elapsed_seconds": round(time.perf_counter() - t_model, 2)}
        del scorer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    (out_dir / "mutsite_timing.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
    print(f"wrote mutsite baselines for {len(methods_by_task)} tasks to {methods_path}")


def run_confit_stage(
    data_dir: Path,
    tasks_dir: Path,
    out_dir: Path,
    model_name: str,
    splits_filter: set[str] | None,
    config: ConFitConfig,
    run_shuffled: bool = False,
    tag_suffix: str = "",
    extra_splits: list[SplitInfo] | None = None,
) -> None:
    """Run Track 1C / Track 2 ConFit LoRA preference tuning of ESMC across tasks."""
    out_dir.mkdir(parents=True, exist_ok=True)
    all_infos = {s.name: s for s in (*plan.ALL_SPLITS, *(extra_splits or ()))}
    tasks = [t for t in read_tasks(tasks_dir) if t.arm == "k256" and t.seed == 0]
    if splits_filter:
        tasks = [t for t in tasks if t.split in splits_filter]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scorer = ChainScorer(model_name, device=device)

    methods_path = out_dir / f"confit_{model_name}{tag_suffix}_methods.json"
    records_path = out_dir / f"confit_{model_name}{tag_suffix}_records.json"
    methods_by_task: dict[str, dict[str, list[float]]] = (
        json.loads(methods_path.read_text(encoding="utf-8")) if methods_path.exists() else {}
    )
    records_by_task: dict[str, dict[str, object]] = (
        json.loads(records_path.read_text(encoding="utf-8")) if records_path.exists() else {}
    )

    for task in tqdm(tasks, desc=f"confit:{model_name}{tag_suffix}"):
        existing_rec = records_by_task.get(task.task_id)
        if existing_rec is not None and (not run_shuffled or "shuffled_elapsed_seconds" in existing_rec):
            continue
        split_spec = all_infos[task.split]
        frame = flip2.load_split(flip2.split_path(data_dir, split_spec))
        layout = load_layout(split_spec, frame)
        ctx = np.array(task.context_rows, dtype=np.int64)
        qry = np.array(task.query_rows, dtype=np.int64)
        vals = np.array(task.context_values, dtype=float)

        res = run_confit_split(scorer, layout, ctx, vals, qry, config=config, shuffle_labels=False)
        task_methods = methods_by_task.setdefault(task.task_id, {})
        for ep_str, pred_list in res["predictions_by_epoch"].items():
            task_methods[f"confit_{model_name}{tag_suffix}_ep{ep_str}"] = pred_list
        records_by_task[task.task_id] = {
            "split": task.split,
            "elapsed_seconds": res["elapsed_seconds"],
            "peak_gpu_gb": res["peak_gpu_gb"],
            "trainable_params": res["trainable_params"],
            "epoch_logs": res["epoch_logs"],
        }

        if run_shuffled:
            shuf_res = run_confit_split(scorer, layout, ctx, vals, qry, config=config, shuffle_labels=True)
            final_ep = str(config.epochs)
            task_methods[f"confit_{model_name}{tag_suffix}_shuffled"] = shuf_res["predictions_by_epoch"][final_ep]
            records_by_task[task.task_id]["shuffled_elapsed_seconds"] = shuf_res["elapsed_seconds"]

        methods_path.write_text(json.dumps(methods_by_task), encoding="utf-8")
        records_path.write_text(json.dumps(records_by_task, indent=2), encoding="utf-8")

    del scorer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"wrote ConFit outputs for {len(records_by_task)} tasks to {methods_path}")


def rank_average(pred_a: np.ndarray, pred_b: np.ndarray) -> np.ndarray:
    """Average fractional ranks of two prediction arrays of the same length."""
    # pred_a, pred_b: (q,)
    from scipy.stats import rankdata

    ra = rankdata(pred_a) / float(len(pred_a))  # (q,)
    rb = rankdata(pred_b) / float(len(pred_b))  # (q,)
    return 0.5 * (ra + rb)  # (q,)


def evaluate_flip2_and_fgfr2(
    exp_dir: Path,
    ref_predictions_path: Path,
    out_path: Path,
    extra_predictions_paths: list[Path] | None = None,
) -> dict[str, object]:
    """Compare all new methods in `exp_dir` against Claude Mythos 5.2 and frozen ridges with 2,000 paired bootstraps."""
    ref_parts = [pd.read_csv(ref_predictions_path, dtype={"stratum": str})]
    for ep in extra_predictions_paths or []:
        if ep.exists():
            ref_parts.append(pd.read_csv(ep, dtype={"stratum": str}))
    ref = pd.concat(ref_parts, ignore_index=True).drop_duplicates(subset=["task_id", "method", "query_id"], keep="last")

    # Filter k256 seed 0 tasks using canonical task_id suffix (`<dataset>-<split>__k256__s0`)
    k256 = ref[ref["task_id"].astype(str).str.endswith("__k256__s0")].copy()
    k256["split"] = k256["task_id"].astype(str).str.split("__").str[0].str.replace("-", "/", n=1)

    # Load all method JSONs in exp_dir
    new_methods_by_task: dict[str, dict[str, list[float]]] = {}
    for jf in sorted(exp_dir.glob("*_methods.json")):
        loaded = json.loads(jf.read_text(encoding="utf-8"))
        for task_id, mdict in loaded.items():
            new_methods_by_task.setdefault(task_id, {}).update(mdict)

    # Identify Claude/Mythos method names in reference tables
    claude_candidates = [m for m in k256["method"].unique() if "mythos" in str(m).lower() or "fable" in str(m).lower()]
    ref_methods_to_keep = [
        "ridge_onehot",
        "zero_shot_esmc_300m",
        "zero_shot_esmc_600m",
        "zero_shot_esmc_6b",
        "ridge_esmc_300m",
        "ridge_esmc_600m",
        "ridge_esmc_6b",
        "ridge_onehot_plus_esmc_600m",
        "ridge_onehot_plus_esmc_6b",
    ]
    for cand in claude_candidates:
        if cand not in ref_methods_to_keep:
            ref_methods_to_keep.append(cand)

    truth_100: dict[str, np.ndarray] = {}
    truth_300: dict[str, np.ndarray] = {}
    preds_100: dict[str, dict[str, np.ndarray]] = {}
    preds_300: dict[str, dict[str, np.ndarray]] = {}

    for task_id, group in k256.groupby("task_id", sort=False):
        split_name = str(group["split"].iloc[0])
        # Use a full-300-query method for canonical truth_300 ordering
        method_counts = group.groupby("method").size()
        full_method = str(method_counts.idxmax())
        base_q = group[group["method"] == full_method].sort_values("query_id")
        truth_all = base_q["truth"].to_numpy(dtype=float)
        truth_300[split_name] = truth_all
        truth_100[split_name] = truth_all[:100]

        for m in ref_methods_to_keep:
            sub = group[group["method"] == m].sort_values("query_id")
            if len(sub) == 0:
                continue
            vals = sub["prediction"].to_numpy(dtype=float)
            if len(vals) >= 100:
                preds_100.setdefault(m, {})[split_name] = vals[:100]
            if len(vals) == len(truth_all):
                preds_300.setdefault(m, {})[split_name] = vals

        for m, vals_list in new_methods_by_task.get(str(task_id), {}).items():
            # Never overwrite an existing reference baseline from ref_tables
            if m in ref_methods_to_keep and m in preds_100 and split_name in preds_100[m]:
                continue
            vals = np.asarray(vals_list, dtype=float)
            if len(vals) >= 100:
                preds_100.setdefault(m, {})[split_name] = vals[:100]
            if len(vals) == len(truth_all):
                preds_300.setdefault(m, {})[split_name] = vals

    all_splits = sorted(truth_100.keys())
    flip2_splits = [s for s in all_splits if not s.startswith("fgfr2/")]
    fgfr2_splits = [s for s in all_splits if s.startswith("fgfr2/")]

    mythos_col = next((m for m in preds_100 if "mythos" in m.lower() and "@low" in m.lower() and len(preds_100[m]) == len(all_splits)), None)
    if mythos_col is None:
        mythos_col = next((m for m in preds_100 if "mythos" in m.lower() and len(preds_100[m]) == len(all_splits)), None)
    if mythos_col is None:
        mythos_col = next((m for m in preds_100 if "fable" in m.lower() and "@low" in m.lower() and len(preds_100[m]) == len(all_splits)), None)

    ensemble_pairs = [
        ("ensemble_confit600m_mutsite600m", "confit_esmc_600m_ep15", "ridge_mutsite_plus_zs_esmc_600m"),
        ("ensemble_confit600m_stacked600m", "confit_esmc_600m_ep15", "ridge_stacked_esmc_600m"),
        ("ensemble_confit600m_stacked6b", "confit_esmc_600m_ep15", "ridge_stacked_esmc_6b"),
        ("ensemble_mutsite600m_mutsite6b", "ridge_mutsite_plus_zs_esmc_600m", "ridge_mutsite_plus_zs_esmc_6b"),
        ("ensemble_mutsite600m_stacked6b", "ridge_mutsite_plus_zs_esmc_600m", "ridge_stacked_esmc_6b"),
        ("ensemble_stacked600m_stacked6b", "ridge_stacked_esmc_600m", "ridge_stacked_esmc_6b"),
        ("ensemble_confit600m_mutsite6b", "confit_esmc_600m_ep15", "ridge_mutsite_plus_zs_esmc_6b"),
        ("ensemble_confit300m_mutsite600m", "confit_esmc_300m_ep15", "ridge_mutsite_plus_zs_esmc_600m"),
    ]
    if mythos_col:
        ensemble_pairs.extend([
            ("ensemble_mythos_confit600m", mythos_col, "confit_esmc_600m_ep15"),
            ("ensemble_mythos_stacked600m", mythos_col, "ridge_stacked_esmc_600m"),
            ("ensemble_mythos_stacked6b", mythos_col, "ridge_stacked_esmc_6b"),
            ("ensemble_mythos_mutsite600m", mythos_col, "ridge_mutsite_plus_zs_esmc_600m"),
            ("ensemble_mythos_mutsite6b", mythos_col, "ridge_mutsite_plus_zs_esmc_6b"),
        ])

    for ens_name, m_a, m_b in ensemble_pairs:
        if m_a in preds_100 and m_b in preds_100:
            common = sorted(set(preds_100[m_a]) & set(preds_100[m_b]))
            if common:
                preds_100[ens_name] = {s: rank_average(preds_100[m_a][s], preds_100[m_b][s]) for s in common}
        if m_a in preds_300 and m_b in preds_300:
            common3 = sorted(set(preds_300[m_a]) & set(preds_300[m_b]))
            if common3:
                preds_300[ens_name] = {s: rank_average(preds_300[m_a][s], preds_300[m_b][s]) for s in common3}

    def summarize_subset(
        truth_map: dict[str, np.ndarray],
        preds_map: dict[str, dict[str, np.ndarray]],
        subset_splits: list[str],
        baseline_name: str,
    ) -> dict[str, object]:
        # truth_map: (q,); preds_map: (q,)
        usable = {m: d for m, d in preds_map.items() if all(s in d for s in subset_splits)}
        if not subset_splits or baseline_name not in usable:
            return {"splits": subset_splits, "baseline": baseline_name, "methods": {}}
        return {
            "splits": subset_splits,
            "baseline": baseline_name,
            "methods": paired_summary(truth_map, usable, baseline_name, subset_splits),
        }

    # Also compute per-split Spearman for every method present
    per_split_100: dict[str, dict[str, float]] = {}
    for m, d in sorted(preds_100.items()):
        per_split_100[m] = {s: round(spearman(truth_100[s], arr), 4) for s, arr in sorted(d.items())}

    report_obj: dict[str, object] = {
        "mythos_method": mythos_col,
        "flip2_q100_vs_ridge600m": summarize_subset(truth_100, preds_100, flip2_splits, "ridge_esmc_600m"),
        "flip2_q100_vs_mythos": summarize_subset(truth_100, preds_100, flip2_splits, mythos_col) if mythos_col else {},
        "flip2_q300_vs_ridge600m": summarize_subset(truth_300, preds_300, flip2_splits, "ridge_esmc_600m"),
        "fgfr2_q100_vs_ridge_oh6b": summarize_subset(truth_100, preds_100, fgfr2_splits, "ridge_onehot_plus_esmc_6b"),
        "fgfr2_q100_vs_mythos": summarize_subset(truth_100, preds_100, fgfr2_splits, mythos_col) if mythos_col else {},
        "fgfr2_q300_vs_ridge_oh6b": summarize_subset(truth_300, preds_300, fgfr2_splits, "ridge_onehot_plus_esmc_6b"),
        "per_split_q100": per_split_100,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report_obj, indent=2), encoding="utf-8")
    return report_obj


def evaluate_postcutoff_extra(data_dir: Path, tasks_dir: Path, exp_dir: Path, out_path: Path) -> dict[str, object]:
    """Score all methods on the 6 new post-cutoff splits (GBA1, ARAF activity, ARAF abundance) with 2,000 bootstraps."""
    extra_infos = {
        f"{ds}/{sp}": SplitInfo(ds, sp, "mutation" if sp == "random" else "position", "single_parent", "", "")
        for ds in ("gba1_activity", "araf_activity", "araf_abundance")
        for sp in ("random", "by_position")
    }
    tasks = [t for t in read_tasks(tasks_dir) if t.arm == "k256" and t.seed == 0]
    all_methods_by_task: dict[str, dict[str, list[float]]] = {}
    for jf in sorted(exp_dir.glob("*_methods.json")):
        loaded = json.loads(jf.read_text(encoding="utf-8"))
        for task_id, mdict in loaded.items():
            all_methods_by_task.setdefault(task_id, {}).update(mdict)

    truth_300: dict[str, np.ndarray] = {}
    preds_300: dict[str, dict[str, np.ndarray]] = {}
    for task in tasks:
        split_spec = extra_infos[task.split]
        frame = flip2.load_split(flip2.split_path(data_dir, split_spec))
        truth_all = frame["target"].to_numpy(dtype=float)[np.array(task.query_rows, dtype=np.int64)]
        truth_300[task.split] = truth_all
        for m, vals in all_methods_by_task.get(task.task_id, {}).items():
            preds_300.setdefault(m, {})[task.split] = np.asarray(vals, dtype=float)

    splits = sorted(truth_300.keys())
    usable = {m: d for m, d in preds_300.items() if all(s in d for s in splits)}
    base = "ridge_onehot_plus_esmc_6b" if "ridge_onehot_plus_esmc_6b" in usable else "ridge_esmc_600m"
    per_split = {m: {s: round(spearman(truth_300[s], d[s]), 4) for s in splits} for m, d in sorted(usable.items())}
    out_obj: dict[str, object] = {
        "splits": splits,
        "baseline": base,
        "summary": paired_summary(truth_300, usable, base, splits) if base in usable else {},
        "per_split_q300": per_split,
    }
    out_path.write_text(json.dumps(out_obj, indent=2), encoding="utf-8")
    return out_obj


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_fus = sub.add_parser("fusion", help="Track 1A: frozen embedding+zero-shot fusion ridges")
    p_fus.add_argument("--data", type=Path, required=True)
    p_fus.add_argument("--tasks", type=Path, required=True)
    p_fus.add_argument("--features", type=Path, required=True)
    p_fus.add_argument("--out", type=Path, required=True)
    p_fus.add_argument("--splits", default="")

    p_mut = sub.add_parser("mutsite", help="Track 1B/2: mutation-site local embeddings and ridges")
    p_mut.add_argument("--data", type=Path, required=True)
    p_mut.add_argument("--tasks", type=Path, required=True)
    p_mut.add_argument("--out", type=Path, required=True)
    p_mut.add_argument("--models", default="esmc_600m,esmc_6b")
    p_mut.add_argument("--splits", default="")
    p_mut.add_argument("--postcutoff-extra", action="store_true")
    p_mut.add_argument("--include-model-free", action="store_true")

    p_cf = sub.add_parser("confit", help="Track 1C/2: ConFit-style LoRA preference tuning of ESMC")
    p_cf.add_argument("--data", type=Path, required=True)
    p_cf.add_argument("--tasks", type=Path, required=True)
    p_cf.add_argument("--out", type=Path, required=True)
    p_cf.add_argument("--model", required=True, choices=["esmc_300m", "esmc_600m"])
    p_cf.add_argument("--splits", default="")
    p_cf.add_argument("--epochs", type=int, default=15)
    p_cf.add_argument("--rank", type=int, default=8)
    p_cf.add_argument("--lr", type=float, default=1e-4)
    p_cf.add_argument("--kl-weight", type=float, default=0.1)
    p_cf.add_argument("--batch-size", type=int, default=16)
    p_cf.add_argument("--seed", type=int, default=0)
    p_cf.add_argument("--run-shuffled", action="store_true")
    p_cf.add_argument("--tag", default="")
    p_cf.add_argument("--postcutoff-extra", action="store_true")

    p_mat = sub.add_parser("materialize-postcutoff", help="Build GBA1/ARAF FLIP2-format splits and tasks")
    p_mat.add_argument("--raw-dir", type=Path, required=True)
    p_mat.add_argument("--data-out", type=Path, required=True)
    p_mat.add_argument("--tasks-out", type=Path, required=True)

    p_eval = sub.add_parser("evaluate", help="Paired bootstrap evaluation against reference tables")
    p_eval.add_argument("--exp-dir", type=Path, required=True)
    p_eval.add_argument("--ref-predictions", type=Path, required=True)
    p_eval.add_argument("--extra-predictions", type=Path, nargs="*", default=[])
    p_eval.add_argument("--out", type=Path, required=True)

    p_epc = sub.add_parser("evaluate-postcutoff", help="Paired bootstrap evaluation on expanded post-cutoff suite")
    p_epc.add_argument("--data", type=Path, required=True)
    p_epc.add_argument("--tasks", type=Path, required=True)
    p_epc.add_argument("--exp-dir", type=Path, required=True)
    p_epc.add_argument("--out", type=Path, required=True)

    args = parser.parse_args()

    extra_infos = [
        SplitInfo("gba1_activity", "random", "mutation", "single_parent", "", "MaveDB urn:mavedb:00001327-a-1"),
        SplitInfo("gba1_activity", "by_position", "position", "single_parent", "", "MaveDB urn:mavedb:00001327-a-1"),
        SplitInfo("araf_activity", "random", "mutation", "single_parent", "", "MaveDB urn:mavedb:00001314-0-4"),
        SplitInfo("araf_activity", "by_position", "position", "single_parent", "", "MaveDB urn:mavedb:00001314-0-4"),
        SplitInfo("araf_abundance", "random", "mutation", "single_parent", "", "MaveDB urn:mavedb:00001314-0-3"),
        SplitInfo("araf_abundance", "by_position", "position", "single_parent", "", "MaveDB urn:mavedb:00001314-0-3"),
    ]

    if args.command == "materialize-postcutoff":
        infos = materialize_expanded_postcutoff(args.raw_dir, args.data_out, args.tasks_out)
        print(f"materialized {len(infos)} expanded post-cutoff splits into {args.data_out} and {args.tasks_out}")
        return
    if args.command == "evaluate":
        rep = evaluate_flip2_and_fgfr2(args.exp_dir, args.ref_predictions, args.out, args.extra_predictions)
        print(f"wrote evaluation summary to {args.out} (mythos={rep.get('mythos_method')})")
        return
    if args.command == "evaluate-postcutoff":
        rep = evaluate_postcutoff_extra(args.data, args.tasks, args.exp_dir, args.out)
        print(f"wrote post-cutoff evaluation summary ({len(rep['splits'])} splits) to {args.out}")
        return

    splits_filter = {s for s in args.splits.split(",") if s} or None
    if args.command == "fusion":
        run_fusion_stage(args.data, args.tasks, args.features, args.out, splits_filter)
    elif args.command == "mutsite":
        models = tuple(m for m in args.models.split(",") if m)
        run_mutsite_stage(
            args.data,
            args.tasks,
            args.out,
            models,
            splits_filter,
            extra_splits=extra_infos if args.postcutoff_extra else None,
            include_model_free=args.include_model_free,
        )
    elif args.command == "confit":
        eval_eps = tuple(sorted({0, min(5, args.epochs), min(10, args.epochs), args.epochs}))
        cfg = ConFitConfig(
            rank=args.rank,
            learning_rate=args.lr,
            kl_weight=args.kl_weight,
            epochs=args.epochs,
            batch_size=args.batch_size,
            eval_epochs=eval_eps,
            seed=args.seed,
        )
        run_confit_stage(
            args.data,
            args.tasks,
            args.out,
            args.model,
            splits_filter,
            config=cfg,
            run_shuffled=args.run_shuffled,
            tag_suffix=args.tag,
            extra_splits=extra_infos if args.postcutoff_extra else None,
        )


if __name__ == "__main__":
    main()

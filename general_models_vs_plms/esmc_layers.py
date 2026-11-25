"""ESMC intermediate depth-layer (23/27/60), penultimate-layer (-2), final-layer (-1), and ESMFold2 structural stream evaluation.

Uses `foundry.embedding.canonical.build_runtime` and FastPLMs' `TokenTapExecutor` to extract all four BF16 streams in
one forward pass per batch (over all `l + 2` tokens: CLS, residues, EOS, canonical crop 2,046):

- `layer_hidden`: depth layer 23 (`esmc_300m`), 27 (`esmc_600m`), 60 (`esmc_6b`)
- `penult_hidden`: penultimate layer -2 (29 / 35 / 79)
- `final_hidden`: final normalized layer -1 (30 / 36 / 80)
- `structural`: released ESMFold2 width-256 layer-mixture projection (SHA-256 verified)

Reuses existing masked-marginal / unmasked-likelihood zero-shot scores without recomputation and concatenates multi-chain
(`pdz3/single_to_double`) per-chain feature slots cleanly so short peptide chains are never diluted by longer domains.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import numpy as np
import pandas as pd
import torch

from pathlib import Path
from tqdm import tqdm

from . import flip2, plan
from .baselines import LIKELIHOOD_WEIGHT, marginal_scores, ridge, standardize
from .encoding import Layout, one_hot, parent_text, sequence_text
from .esmc_adaptation import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    mutated_positions_per_chain,
    paired_summary,
    rank_average,
    stacked_multiview_ridge,
)
from .flip2 import SplitInfo
from .plm_features import chains_of
from .scoring import spearman
from .tasks import Task, load_layout, read_tasks


MODEL_TO_CANONICAL = {
    "esmc_300m": "esmc_small",
    "esmc_600m": "esmc_large",
    "esmc_6b": "esmc_6b",
}
MODEL_DEPTH_LAYER = {"esmc_300m": 23, "esmc_600m": 27, "esmc_6b": 60}
MODEL_PENULT_LAYER = {"esmc_300m": 29, "esmc_600m": 35, "esmc_6b": 79}
PENULT_HIDDEN = "penult_hidden"


class MultiLayerEsmcExtractor:
    """One-pass BF16 extractor for canonical depth (23/27/60), penultimate (-2), final (-1), and width-256 structural streams."""

    def __init__(self, model_name: str, cache_dir: Path, device: str = "cuda") -> None:
        from fastplms.embeddings import HiddenTap, ResidueVocabulary, TokenTapExecutor
        from fastplms.embeddings.taps import plan_taps
        from foundry.embedding.canonical import FINAL_HIDDEN, LAYER_HIDDEN, STRUCTURAL, build_runtime, canonical_base
        from foundry.embedding.contracts import CANONICAL_MAX_RESIDUES

        assert model_name in MODEL_TO_CANONICAL, f"unsupported model {model_name}"
        base_id = MODEL_TO_CANONICAL[model_name]
        base = canonical_base(base_id)
        assert base.sae_layer == MODEL_DEPTH_LAYER[model_name], (
            f"depth layer mismatch for {model_name}: {base.sae_layer} vs {MODEL_DEPTH_LAYER[model_name]}"
        )
        runtime = build_runtime(base, cache_dir, device=device, fetch=True)
        by_name = {tap.name: tap for tap in runtime.taps}
        penult_tap = HiddenTap(PENULT_HIDDEN, -2, dtype=torch.bfloat16)
        taps = [by_name[LAYER_HIDDEN], penult_tap, by_name[FINAL_HIDDEN], by_name[STRUCTURAL]]
        tap_plan = plan_taps(taps, int(runtime.model.embedding_tap_state_count))

        self.model_name = model_name
        self.base_id = base_id
        self.hidden_width = int(base.hidden_width)
        self.struct_width = 256
        self.device = torch.device(device)
        self.vocabulary = ResidueVocabulary(runtime.model.tokenizer)
        self.executor = TokenTapExecutor(
            runtime.model, tap_plan, vocabulary=self.vocabulary, max_residues=CANONICAL_MAX_RESIDUES, dtype=None
        )
        self.evidence = runtime.evidence

    @torch.no_grad()
    def run_chains(self, chains: list[str]) -> dict[str, list[np.ndarray]]:
        """Extract per-token `(l+2, w)` float32 arrays for depth, penultimate, final, and structural streams."""
        from foundry.embedding.contracts import normalize_sequence

        normed = [normalize_sequence(seq) for seq in chains]
        digests = [hashlib.sha256(seq.encode("ascii")).hexdigest() for seq in normed]
        with torch.autocast(self.device.type, enabled=False):
            packed = self.executor.run_batch(normed, digests)
            packed.wait()

        row_counts = [int(c) for c in packed.rows]
        starts = np.cumsum([0, *row_counts[:-1]])
        stream_out: dict[str, list[np.ndarray]] = {}
        for name in ("layer_hidden", PENULT_HIDDEN, "final_hidden", "structural"):
            values = packed.streams[name]["values"].float().cpu().numpy()  # (n_total, w)
            stream_out[name] = [
                values[int(s) : int(s + c)] for s, c in zip(starts, row_counts, strict=True)
            ]
        return stream_out  # per stream: b arrays of (l_i + 2, w)


def pool_mean_and_var(tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and population variance across all `l + 2` token rows (CLS, residues, EOS)."""
    # tokens: (l + 2, w)
    mean = tokens.mean(axis=0).astype(np.float32)  # (w,)
    var = tokens.var(axis=0).astype(np.float32)  # (w,)
    return mean, np.concatenate([mean, var], axis=0)  # (w,), (2w,)


def extract_multilayer_split(
    extractor: MultiLayerEsmcExtractor,
    layout: Layout,
    rows: np.ndarray,
    batch_size: int = 32,
) -> dict[str, np.ndarray]:
    """Extract pooled mean/mean_var and mutated-site state/delta across depth, penultimate, final, and structural streams.

    Multi-chain rows (e.g. `pdz3/single_to_double`) concatenate each chain slot rather than averaging across chains,
    preserving short peptide chains alongside longer domains.
    """
    # rows: (n,)
    unique_rows = np.unique(rows)  # (u,)
    n_rows = len(unique_rows)
    row_chains = [chains_of(sequence_text(layout, int(r))) for r in unique_rows]
    n_chains = len(row_chains[0])

    # Extract wild-type per-chain token tensors once if parent exists
    parent_tokens: dict[str, list[np.ndarray]] | None = None
    if layout.parent is not None:
        parent_chains = chains_of(parent_text(layout))
        parent_tokens = extractor.run_chains(parent_chains)

    stream_specs = (
        ("depth", "layer_hidden", extractor.hidden_width),
        ("penult", PENULT_HIDDEN, extractor.hidden_width),
        ("final", "final_hidden", extractor.hidden_width),
        ("struct", "structural", extractor.struct_width),
    )

    out: dict[str, np.ndarray] = {"rows": unique_rows}
    for prefix, _, width in stream_specs:
        out[f"{prefix}_mean"] = np.zeros((n_rows, n_chains * width), dtype=np.float32)
        out[f"{prefix}_mv"] = np.zeros((n_rows, n_chains * 2 * width), dtype=np.float32)
        out[f"{prefix}_mut_state"] = np.zeros((n_rows, n_chains * width), dtype=np.float32)
        out[f"{prefix}_mut_delta"] = np.zeros((n_rows, n_chains * width), dtype=np.float32)

    for c_idx in range(n_chains):
        seqs = [rc[c_idx] for rc in row_chains]
        unique_seqs = sorted({s for s in seqs if len(s) > 0})
        # Cache unique chain strings per split to avoid duplicate forward passes on repeated chains (e.g., PDZ3 domain)
        seq_to_streams: dict[str, dict[str, np.ndarray]] = {}
        order = sorted(unique_seqs, key=len)
        for start in range(0, len(order), batch_size):
            batch_seqs = order[start : start + batch_size]
            batch_out = extractor.run_chains(batch_seqs)
            for idx_b, seq_str in enumerate(batch_seqs):
                seq_to_streams[seq_str] = {k: v[idx_b] for k, v in batch_out.items()}

        for u_idx, seq_str in enumerate(seqs):
            if not seq_str:
                continue
            row_id = int(unique_rows[u_idx])
            chain_muts = mutated_positions_per_chain(layout, row_id)[c_idx] if layout.parent is not None else []
            per_stream = seq_to_streams[seq_str]
            for prefix, stream_key, width in stream_specs:
                toks = per_stream[stream_key]  # (l + 2, w), row 0=CLS, rows 1..l=residues, row l+1=EOS
                mean_v, mv_v = pool_mean_and_var(toks)
                out[f"{prefix}_mean"][u_idx, c_idx * width : (c_idx + 1) * width] = mean_v
                out[f"{prefix}_mv"][u_idx, c_idx * 2 * width : (c_idx + 1) * 2 * width] = mv_v

                if parent_tokens is not None:
                    wt_toks = parent_tokens[stream_key][c_idx]  # (l_wt + 2, w)
                    # Residue 0-based index `pos` sits at token row `pos + 1` (after CLS)
                    valid_toks = [pos + 1 for pos, _, _ in chain_muts if pos + 1 < len(toks) - 1 and pos + 1 < len(wt_toks) - 1]
                    if valid_toks:
                        mut_h = toks[valid_toks]  # (m, w)
                        wt_h = wt_toks[valid_toks]  # (m, w)
                        out[f"{prefix}_mut_state"][u_idx, c_idx * width : (c_idx + 1) * width] = mut_h.mean(axis=0)
                        out[f"{prefix}_mut_delta"][u_idx, c_idx * width : (c_idx + 1) * width] = (mut_h - wt_h).mean(axis=0)
                    else:
                        out[f"{prefix}_mut_state"][u_idx, c_idx * width : (c_idx + 1) * width] = mean_v
                        out[f"{prefix}_mut_delta"][u_idx, c_idx * width : (c_idx + 1) * width] = 0.0
                else:
                    out[f"{prefix}_mut_state"][u_idx, c_idx * width : (c_idx + 1) * width] = mean_v
                    out[f"{prefix}_mut_delta"][u_idx, c_idx * width : (c_idx + 1) * width] = 0.0

    return out  # rows: (u,), pooled and mutsite arrays: (u, c * w) or (u, c * 2w)


def load_zero_shot_scores(
    zs_npz_path: Path,
    layout: Layout,
    context_rows: np.ndarray,
    query_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Load precomputed masked-marginal (parented) or unmasked likelihood (unparented) zero-shot scores."""
    # context_rows: (k,); query_rows: (q,)
    zs_arrays = np.load(zs_npz_path)
    if layout.parent is not None and "parent_marginals" in zs_arrays:
        fit_zs = marginal_scores(layout, zs_arrays["parent_marginals"], context_rows)  # (k,)
        qry_zs = marginal_scores(layout, zs_arrays["parent_marginals"], query_rows)  # (q,)
        return fit_zs, qry_zs  # (k,), (q,)
    pos_map = {int(r): i for i, r in enumerate(zs_arrays["rows"])}
    fit_zs = zs_arrays["likelihood"][[pos_map[int(r)] for r in context_rows]]  # (k,)
    qry_zs = zs_arrays["likelihood"][[pos_map[int(r)] for r in query_rows]]  # (q,)
    return fit_zs, qry_zs  # (k,), (q,)


def multilayer_ridges_for_task(
    task: Task,
    layout: Layout,
    arrays: dict[str, np.ndarray],
    fit_zs: np.ndarray,
    qry_zs: np.ndarray,
    model_name: str,
    include_shuffled: bool = False,
    seed: int = 0,
) -> dict[str, list[float]]:
    """Fit all depth, penultimate, final, and structural stream ridge and stacked models on context rows."""
    # arrays: (u,) rows, (u, f) features; fit_zs: (k,); qry_zs: (q,)
    pos_map = {int(r): i for i, r in enumerate(arrays["rows"])}
    ctx_rows = np.array(task.context_rows, dtype=np.int64)
    qry_rows = np.array(task.query_rows, dtype=np.int64)
    ctx_i = [pos_map[int(r)] for r in ctx_rows]
    qry_i = [pos_map[int(r)] for r in qry_rows]
    values = np.array(task.context_values, dtype=float)

    fit_oh, qry_oh = one_hot(layout, ctx_rows, qry_rows)
    fit_z, qry_z = standardize(fit_zs[:, None], qry_zs[:, None])

    def std_pair(key: str) -> tuple[np.ndarray, np.ndarray]:
        return standardize(arrays[key][ctx_i], arrays[key][qry_i])  # (k, f), (q, f)

    # Pooled features
    f_final_m, q_final_m = std_pair("final_mean")
    f_pen_m, q_pen_m = std_pair("penult_mean")
    f_pen_mv, q_pen_mv = std_pair("penult_mv")
    f_dep_m, q_dep_m = std_pair("depth_mean")
    f_dep_mv, q_dep_mv = std_pair("depth_mv")
    f_str_mv, q_str_mv = std_pair("struct_mv")

    f_dep_str_mv = np.hstack([f_dep_mv, f_str_mv])
    q_dep_str_mv = np.hstack([q_dep_mv, q_str_mv])
    f_pen_str_mv = np.hstack([f_pen_mv, f_str_mv])
    q_pen_str_mv = np.hstack([q_pen_mv, q_str_mv])

    preds: dict[str, np.ndarray] = {
        f"ridge_final_mean_l2_{model_name}": ridge(f_final_m, values, q_final_m),
        f"ridge_penult_mean_{model_name}": ridge(f_pen_m, values, q_pen_m),
        f"ridge_penult_mv_{model_name}": ridge(f_pen_mv, values, q_pen_mv),
        f"ridge_depth_mean_{model_name}": ridge(f_dep_m, values, q_dep_m),
        f"ridge_depth_mv_{model_name}": ridge(f_dep_mv, values, q_dep_mv),
        f"ridge_struct_mv_{model_name}": ridge(f_str_mv, values, q_str_mv),
        f"ridge_depth_struct_mv_{model_name}": ridge(f_dep_str_mv, values, q_dep_str_mv),
        f"ridge_penult_struct_mv_{model_name}": ridge(f_pen_str_mv, values, q_pen_str_mv),
        f"ridge_depth_struct_mv_zs_{model_name}": ridge(
            np.hstack([f_dep_str_mv, LIKELIHOOD_WEIGHT * fit_z]),
            values,
            np.hstack([q_dep_str_mv, LIKELIHOOD_WEIGHT * qry_z]),
        ),
        f"ridge_penult_mv_zs_{model_name}": ridge(
            np.hstack([f_pen_mv, LIKELIHOOD_WEIGHT * fit_z]),
            values,
            np.hstack([q_pen_mv, LIKELIHOOD_WEIGHT * qry_z]),
        ),
        f"ridge_penult_struct_mv_zs_{model_name}": ridge(
            np.hstack([f_pen_str_mv, LIKELIHOOD_WEIGHT * fit_z]),
            values,
            np.hstack([q_pen_str_mv, LIKELIHOOD_WEIGHT * qry_z]),
        ),
    }

    # Mutated-site local representations (fallback to mean+var on unparented rhomax)
    if layout.parent is not None:
        f_fin_ms, q_fin_ms = std_pair("final_mut_state")
        f_fin_md, q_fin_md = std_pair("final_mut_delta")
        f_pen_ms, q_pen_ms = std_pair("penult_mut_state")
        f_pen_md, q_pen_md = std_pair("penult_mut_delta")
        f_dep_ms, q_dep_ms = std_pair("depth_mut_state")
        f_dep_md, q_dep_md = std_pair("depth_mut_delta")
        f_str_ms, q_str_ms = std_pair("struct_mut_state")
        f_str_md, q_str_md = std_pair("struct_mut_delta")

        f_fin_loc, q_fin_loc = np.hstack([f_fin_ms, f_fin_md]), np.hstack([q_fin_ms, q_fin_md])
        f_pen_loc, q_pen_loc = np.hstack([f_pen_ms, f_pen_md]), np.hstack([q_pen_ms, q_pen_md])
        f_dep_loc, q_dep_loc = np.hstack([f_dep_ms, f_dep_md]), np.hstack([q_dep_ms, q_dep_md])
        f_str_loc, q_str_loc = np.hstack([f_str_ms, f_str_md]), np.hstack([q_str_ms, q_str_md])

        f_dep_str_loc = np.hstack([f_dep_loc, f_str_loc])
        q_dep_str_loc = np.hstack([q_dep_loc, q_str_loc])
        f_pen_str_loc = np.hstack([f_pen_loc, f_str_loc])
        q_pen_str_loc = np.hstack([q_pen_loc, q_str_loc])

        fin_stack_f, fin_stack_q = [f_final_m, f_fin_loc, fit_oh], [q_final_m, q_fin_loc, qry_oh]
        pen_stack_f, pen_stack_q = [f_pen_mv, f_pen_loc, fit_oh], [q_pen_mv, q_pen_loc, qry_oh]
        pen_str_stack_f = [f_pen_mv, f_str_mv, f_pen_loc, f_str_loc, fit_oh]
        pen_str_stack_q = [q_pen_mv, q_str_mv, q_pen_loc, q_str_loc, qry_oh]
        dep_stack_f, dep_stack_q = [f_dep_mv, f_dep_loc, fit_oh], [q_dep_mv, q_dep_loc, qry_oh]
        dep_str_stack_f = [f_dep_mv, f_str_mv, f_dep_loc, f_str_loc, fit_oh]
        dep_str_stack_q = [q_dep_mv, q_str_mv, q_dep_loc, q_str_loc, qry_oh]
    else:
        f_fin_loc, q_fin_loc = f_final_m, q_final_m
        f_pen_loc, q_pen_loc = f_pen_mv, q_pen_mv
        f_dep_loc, q_dep_loc = f_dep_mv, q_dep_mv
        f_dep_str_loc, q_dep_str_loc = f_dep_str_mv, q_dep_str_mv
        f_pen_str_loc, q_pen_str_loc = f_pen_str_mv, q_pen_str_mv

        fin_stack_f, fin_stack_q = [f_final_m, fit_oh], [q_final_m, qry_oh]
        pen_stack_f, pen_stack_q = [f_pen_mv, fit_oh], [q_pen_mv, qry_oh]
        pen_str_stack_f, pen_str_stack_q = [f_pen_mv, f_str_mv, fit_oh], [q_pen_mv, q_str_mv, qry_oh]
        dep_stack_f, dep_stack_q = [f_dep_mv, fit_oh], [q_dep_mv, qry_oh]
        dep_str_stack_f, dep_str_stack_q = [f_dep_mv, f_str_mv, fit_oh], [q_dep_mv, q_str_mv, qry_oh]

    preds[f"ridge_final_mutsite_zs_{model_name}"] = ridge(
        np.hstack([f_fin_loc, LIKELIHOOD_WEIGHT * fit_z]), values, np.hstack([q_fin_loc, LIKELIHOOD_WEIGHT * qry_z])
    )
    preds[f"ridge_final_stacked_{model_name}"] = stacked_multiview_ridge(
        fin_stack_f, fin_stack_q, fit_zs, qry_zs, values, seed=seed
    )

    preds[f"ridge_penult_mutsite_{model_name}"] = ridge(f_pen_loc, values, q_pen_loc)
    preds[f"ridge_penult_mutsite_zs_{model_name}"] = ridge(
        np.hstack([f_pen_loc, LIKELIHOOD_WEIGHT * fit_z]), values, np.hstack([q_pen_loc, LIKELIHOOD_WEIGHT * qry_z])
    )
    preds[f"ridge_penult_stacked_{model_name}"] = stacked_multiview_ridge(
        pen_stack_f, pen_stack_q, fit_zs, qry_zs, values, seed=seed
    )
    preds[f"ridge_penult_struct_mutsite_zs_{model_name}"] = ridge(
        np.hstack([f_pen_str_loc, LIKELIHOOD_WEIGHT * fit_z]),
        values,
        np.hstack([q_pen_str_loc, LIKELIHOOD_WEIGHT * qry_z]),
    )
    preds[f"ridge_penult_struct_stacked_{model_name}"] = stacked_multiview_ridge(
        pen_str_stack_f, pen_str_stack_q, fit_zs, qry_zs, values, seed=seed
    )

    preds[f"ridge_depth_mutsite_{model_name}"] = ridge(f_dep_loc, values, q_dep_loc)
    preds[f"ridge_depth_mutsite_zs_{model_name}"] = ridge(
        np.hstack([f_dep_loc, LIKELIHOOD_WEIGHT * fit_z]), values, np.hstack([q_dep_loc, LIKELIHOOD_WEIGHT * qry_z])
    )
    preds[f"ridge_depth_stacked_{model_name}"] = stacked_multiview_ridge(
        dep_stack_f, dep_stack_q, fit_zs, qry_zs, values, seed=seed
    )
    preds[f"ridge_depth_struct_mutsite_{model_name}"] = ridge(f_dep_str_loc, values, q_dep_str_loc)
    preds[f"ridge_depth_struct_mutsite_zs_{model_name}"] = ridge(
        np.hstack([f_dep_str_loc, LIKELIHOOD_WEIGHT * fit_z]),
        values,
        np.hstack([q_dep_str_loc, LIKELIHOOD_WEIGHT * qry_z]),
    )
    preds[f"ridge_depth_struct_stacked_{model_name}"] = stacked_multiview_ridge(
        dep_str_stack_f, dep_str_stack_q, fit_zs, qry_zs, values, seed=seed
    )

    if include_shuffled:
        shuffled = np.random.default_rng(seed).permutation(values)
        preds[f"ridge_depth_struct_mv_{model_name}_shuffled"] = ridge(f_dep_str_mv, shuffled, q_dep_str_mv)
        preds[f"ridge_depth_struct_stacked_{model_name}_shuffled"] = stacked_multiview_ridge(
            dep_str_stack_f, dep_str_stack_q, fit_zs, qry_zs, shuffled, seed=seed
        )
        preds[f"ridge_penult_stacked_{model_name}_shuffled"] = stacked_multiview_ridge(
            pen_stack_f, pen_stack_q, fit_zs, qry_zs, shuffled, seed=seed
        )

    return {k: np.asarray(v, dtype=float).tolist() for k, v in preds.items()}


def run_extract_and_fit(
    data_dir: Path,
    tasks_dir: Path,
    zs_features_dir: Path,
    cache_dir: Path,
    out_dir: Path,
    models: tuple[str, ...],
    splits_filter: set[str] | None,
    postcutoff_extra: bool = False,
) -> None:
    """Extract canonical depth, penultimate, final, and structural streams and fit all ridges."""
    out_dir.mkdir(parents=True, exist_ok=True)
    extra_infos = (
        [
            SplitInfo("gba1_activity", "random", "mutation", "single_parent", "", "MaveDB urn:mavedb:00001327-a-1"),
            SplitInfo("gba1_activity", "by_position", "position", "single_parent", "", "MaveDB urn:mavedb:00001327-a-1"),
            SplitInfo("araf_activity", "random", "mutation", "single_parent", "", "MaveDB urn:mavedb:00001314-0-4"),
            SplitInfo("araf_activity", "by_position", "position", "single_parent", "", "MaveDB urn:mavedb:00001314-0-4"),
            SplitInfo("araf_abundance", "random", "mutation", "single_parent", "", "MaveDB urn:mavedb:00001314-0-3"),
            SplitInfo("araf_abundance", "by_position", "position", "single_parent", "", "MaveDB urn:mavedb:00001314-0-3"),
        ]
        if postcutoff_extra
        else []
    )
    all_infos = {s.name: s for s in (*plan.ALL_SPLITS, *extra_infos)}
    tasks = [t for t in read_tasks(tasks_dir) if t.arm == "k256" and t.seed == 0]
    if splits_filter:
        tasks = [t for t in tasks if t.split in splits_filter]

    methods_path = out_dir / "layers_methods.json"
    timing_path = out_dir / "layers_timing.json"
    methods_by_task: dict[str, dict[str, list[float]]] = (
        json.loads(methods_path.read_text(encoding="utf-8")) if methods_path.exists() else {}
    )
    timing: dict[str, dict[str, object]] = (
        json.loads(timing_path.read_text(encoding="utf-8")) if timing_path.exists() else {}
    )

    for model_name in models:
        t0 = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        extractor = MultiLayerEsmcExtractor(model_name, cache_dir=cache_dir, device="cuda" if torch.cuda.is_available() else "cpu")
        feat_dir = out_dir / "features" / model_name
        feat_dir.mkdir(parents=True, exist_ok=True)
        parity_checks: dict[str, dict[str, float]] = {}

        for task in tqdm(tasks, desc=f"layers:{model_name}"):
            split_spec = all_infos[task.split]
            frame = flip2.load_split(flip2.split_path(data_dir, split_spec))
            layout = load_layout(split_spec, frame)
            npz_path = feat_dir / f"{split_spec.dataset}-{split_spec.split}.npz"
            rows_needed = np.unique(np.concatenate([np.array(task.context_rows), np.array(task.query_rows)]))

            if npz_path.exists():
                arrays = dict(np.load(npz_path))
            else:
                bs = 16 if model_name == "esmc_6b" else 32
                arrays = extract_multilayer_split(extractor, layout, rows_needed, batch_size=bs)
                np.savez_compressed(npz_path, **arrays)

            # Verify non-degenerate separation between depth, penultimate, and final layers
            d_vs_f = float(np.linalg.norm(arrays["depth_mean"] - arrays["final_mean"]) / (np.linalg.norm(arrays["final_mean"]) + 1e-8))
            p_vs_f = float(np.linalg.norm(arrays["penult_mean"] - arrays["final_mean"]) / (np.linalg.norm(arrays["final_mean"]) + 1e-8))
            assert d_vs_f > 1e-3 and p_vs_f > 1e-3, f"degenerate layer taps for {model_name} on {task.split}: d={d_vs_f}, p={p_vs_f}"
            parity_checks[task.split] = {"rel_diff_depth_vs_final": round(d_vs_f, 4), "rel_diff_penult_vs_final": round(p_vs_f, 4)}

            zs_path = zs_features_dir / model_name / f"{split_spec.dataset}-{split_spec.split}.npz"
            assert zs_path.exists(), f"missing zero-shot feature file {zs_path}"
            fit_zs, qry_zs = load_zero_shot_scores(
                zs_path, layout, np.array(task.context_rows, dtype=np.int64), np.array(task.query_rows, dtype=np.int64)
            )
            preds = multilayer_ridges_for_task(
                task, layout, arrays, fit_zs, qry_zs, model_name, include_shuffled=(model_name == "esmc_600m")
            )
            methods_by_task.setdefault(task.task_id, {}).update(preds)
            methods_path.write_text(json.dumps(methods_by_task), encoding="utf-8")

        peak_gb = float(torch.cuda.max_memory_allocated() / (1024**3)) if torch.cuda.is_available() else 0.0
        timing[model_name] = {
            "depth_layer": MODEL_DEPTH_LAYER[model_name],
            "penult_layer": MODEL_PENULT_LAYER[model_name],
            "elapsed_seconds": round(time.perf_counter() - t0, 2),
            "peak_gpu_gb": round(peak_gb, 3),
            "splits": len(tasks),
            "layer_separation": parity_checks,
        }
        timing_path.write_text(json.dumps(timing, indent=2), encoding="utf-8")
        del extractor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"wrote multi-layer ESMC methods for {len(methods_by_task)} tasks to {methods_path}")


def evaluate_layers_flip2(
    layers_dir: Path,
    ref_predictions_path: Path,
    extra_predictions_paths: list[Path],
    out_path: Path,
) -> dict[str, object]:
    """Compare all depth+struct, penultimate, and final ESMC methods against Mythos 5.2 and old ridges (2,000 bootstraps)."""
    ref_parts = [pd.read_csv(ref_predictions_path, dtype={"stratum": str})]
    for ep in extra_predictions_paths:
        if ep.exists():
            ref_parts.append(pd.read_csv(ep, dtype={"stratum": str}))
    ref = pd.concat(ref_parts, ignore_index=True).drop_duplicates(subset=["task_id", "method", "query_id"], keep="last")
    k256 = ref[ref["task_id"].astype(str).str.endswith("__k256__s0")].copy()
    k256["split"] = k256["task_id"].astype(str).str.split("__").str[0].str.replace("-", "/", n=1)

    new_by_task: dict[str, dict[str, list[float]]] = json.loads(
        (layers_dir / "layers_methods.json").read_text(encoding="utf-8")
    )
    ref_methods = [
        "ridge_onehot",
        "zero_shot_esmc_300m",
        "zero_shot_esmc_600m",
        "zero_shot_esmc_6b",
        "ridge_esmc_300m",
        "ridge_esmc_600m",
        "ridge_esmc_6b",
        "ridge_onehot_plus_esmc_600m",
        "ridge_onehot_plus_esmc_6b",
        "claude-fable-5-2@low",
        "claude-fable-5-2@high",
        "claude-mythos-5-2@harness-low",
    ]

    truth_100: dict[str, np.ndarray] = {}
    truth_300: dict[str, np.ndarray] = {}
    preds_100: dict[str, dict[str, np.ndarray]] = {}
    preds_300: dict[str, dict[str, np.ndarray]] = {}

    for task_id, group in k256.groupby("task_id", sort=False):
        split_name = str(group["split"].iloc[0])
        full_m = str(group.groupby("method").size().idxmax())
        base_q = group[group["method"] == full_m].sort_values("query_id")
        t_all = base_q["truth"].to_numpy(dtype=float)
        truth_300[split_name] = t_all
        truth_100[split_name] = t_all[:100]

        for m in ref_methods:
            sub = group[group["method"] == m].sort_values("query_id")
            if len(sub) == 0:
                continue
            vals = sub["prediction"].to_numpy(dtype=float)
            if len(vals) >= 100:
                preds_100.setdefault(m, {})[split_name] = vals[:100]
            if len(vals) == len(t_all):
                preds_300.setdefault(m, {})[split_name] = vals

        for m, vals_list in new_by_task.get(str(task_id), {}).items():
            vals = np.asarray(vals_list, dtype=float)
            if len(vals) >= 100:
                preds_100.setdefault(m, {})[split_name] = vals[:100]
            if len(vals) == len(t_all):
                preds_300.setdefault(m, {})[split_name] = vals

    # Multi-scale pure-pLM ensembles + Mythos hybrids
    ensemble_specs = [
        ("ens_final_mutsite_600m_6b", ("ridge_final_mutsite_zs_esmc_600m", "ridge_final_mutsite_zs_esmc_6b")),
        ("ens_penult_mutsite_600m_6b", ("ridge_penult_mutsite_zs_esmc_600m", "ridge_penult_mutsite_zs_esmc_6b")),
        ("ens_penult_stacked_600m_6b", ("ridge_penult_stacked_esmc_600m", "ridge_penult_stacked_esmc_6b")),
        ("ens_penult_struct_stacked_600m_6b", ("ridge_penult_struct_stacked_esmc_600m", "ridge_penult_struct_stacked_esmc_6b")),
        ("ens_depth_mutsite_600m_6b", ("ridge_depth_mutsite_zs_esmc_600m", "ridge_depth_mutsite_zs_esmc_6b")),
        ("ens_depth_struct_mv_600m_6b", ("ridge_depth_struct_mv_esmc_600m", "ridge_depth_struct_mv_esmc_6b")),
        ("ens_depth_struct_mutsite_600m_6b", ("ridge_depth_struct_mutsite_zs_esmc_600m", "ridge_depth_struct_mutsite_zs_esmc_6b")),
        ("ens_depth_struct_stacked_600m_6b", ("ridge_depth_struct_stacked_esmc_600m", "ridge_depth_struct_stacked_esmc_6b")),
        ("ens_depth_struct_stacked_all3", ("ridge_depth_struct_stacked_esmc_300m", "ridge_depth_struct_stacked_esmc_600m", "ridge_depth_struct_stacked_esmc_6b")),
        ("ens_mythos_depth_struct_stacked_600m", ("claude-mythos-5-2@harness-low", "ridge_depth_struct_stacked_esmc_600m")),
        ("ens_mythos_depth_struct_stacked_6b", ("claude-mythos-5-2@harness-low", "ridge_depth_struct_stacked_esmc_6b")),
        ("ens_mythos_depth_struct_mutsite_600m", ("claude-mythos-5-2@harness-low", "ridge_depth_struct_mutsite_zs_esmc_600m")),
        ("ens_mythos_penult_stacked_600m", ("claude-mythos-5-2@harness-low", "ridge_penult_stacked_esmc_600m")),
        ("ens_mythos_penult_mutsite_600m", ("claude-mythos-5-2@harness-low", "ridge_penult_mutsite_zs_esmc_600m")),
    ]
    from scipy.stats import rankdata

    for ens_name, members in ensemble_specs:
        for p_map in (preds_100, preds_300):
            if all(m in p_map for m in members):
                common = sorted(set.intersection(*(set(p_map[m]) for m in members)))
                if common:
                    if len(members) == 2:
                        p_map[ens_name] = {s: rank_average(p_map[members[0]][s], p_map[members[1]][s]) for s in common}
                    else:
                        p_map[ens_name] = {
                            s: sum(rankdata(p_map[m][s]) / float(len(p_map[m][s])) for m in members) / float(len(members))
                            for s in common
                        }

    all_splits = sorted(truth_100.keys())
    flip2_splits = [s for s in all_splits if not s.startswith("fgfr2/")]
    fgfr2_splits = [s for s in all_splits if s.startswith("fgfr2/")]

    def summarize(t_map: dict[str, np.ndarray], p_map: dict[str, dict[str, np.ndarray]], splits: list[str], base: str) -> dict[str, object]:
        # t_map: (q,); p_map: (q,)
        usable = {m: d for m, d in p_map.items() if all(s in d for s in splits)}
        if not splits or base not in usable:
            return {"splits": splits, "baseline": base, "methods": {}}
        return {"splits": splits, "baseline": base, "methods": paired_summary(t_map, usable, base, splits, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED)}

    per_split_100 = {m: {s: round(spearman(truth_100[s], arr), 4) for s, arr in sorted(d.items())} for m, d in sorted(preds_100.items())}
    report = {
        "flip2_q100_vs_mythos": summarize(truth_100, preds_100, flip2_splits, "claude-mythos-5-2@harness-low"),
        "flip2_q100_vs_ridge600m": summarize(truth_100, preds_100, flip2_splits, "ridge_esmc_600m"),
        "flip2_q300_vs_ridge600m": summarize(truth_300, preds_300, flip2_splits, "ridge_esmc_600m"),
        "fgfr2_q100_vs_mythos": summarize(truth_100, preds_100, fgfr2_splits, "claude-mythos-5-2@harness-low"),
        "fgfr2_q300_vs_ridge_oh6b": summarize(truth_300, preds_300, fgfr2_splits, "ridge_onehot_plus_esmc_6b"),
        "per_split_q100": per_split_100,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def evaluate_layers_postcutoff(
    data_dir: Path,
    tasks_dir: Path,
    layers_dir: Path,
    prior_postcutoff_dir: Path,
    out_path: Path,
) -> dict[str, object]:
    """Score all depth+struct, penultimate, and final methods on the 6 post-cutoff GBA1/ARAF splits (q300)."""
    extra_infos = {
        f"{ds}/{sp}": SplitInfo(ds, sp, "mutation" if sp == "random" else "position", "single_parent", "", "")
        for ds in ("gba1_activity", "araf_activity", "araf_abundance")
        for sp in ("random", "by_position")
    }
    tasks = [t for t in read_tasks(tasks_dir) if t.arm == "k256" and t.seed == 0]
    merged: dict[str, dict[str, list[float]]] = {}
    for jf in (prior_postcutoff_dir / "mutsite_methods.json", layers_dir / "layers_methods.json"):
        if jf.exists():
            for tid, md in json.loads(jf.read_text(encoding="utf-8")).items():
                merged.setdefault(tid, {}).update(md)

    truth_300: dict[str, np.ndarray] = {}
    preds_300: dict[str, dict[str, np.ndarray]] = {}
    for task in tasks:
        split_spec = extra_infos[task.split]
        frame = flip2.load_split(flip2.split_path(data_dir, split_spec))
        truth_300[task.split] = frame["target"].to_numpy(dtype=float)[np.array(task.query_rows, dtype=np.int64)]
        for m, vals in merged.get(task.task_id, {}).items():
            preds_300.setdefault(m, {})[task.split] = np.asarray(vals, dtype=float)

    for ens_name, m_a, m_b in (
        ("ens_depth_struct_stacked_600m_6b", "ridge_depth_struct_stacked_esmc_600m", "ridge_depth_struct_stacked_esmc_6b"),
        ("ens_penult_stacked_600m_6b", "ridge_penult_stacked_esmc_600m", "ridge_penult_stacked_esmc_6b"),
        ("ens_final_stacked_600m_6b", "ridge_final_stacked_esmc_600m", "ridge_final_stacked_esmc_6b"),
    ):
        if m_a in preds_300 and m_b in preds_300:
            common = sorted(set(preds_300[m_a]) & set(preds_300[m_b]))
            if common:
                preds_300[ens_name] = {s: rank_average(preds_300[m_a][s], preds_300[m_b][s]) for s in common}

    splits = sorted(truth_300.keys())
    usable = {m: d for m, d in preds_300.items() if all(s in d for s in splits)}
    base = "ridge_onehot_plus_esmc_6b" if "ridge_onehot_plus_esmc_6b" in usable else "ridge_final_mean_l2_esmc_600m"
    per_split = {m: {s: round(spearman(truth_300[s], d[s]), 4) for s in splits} for m, d in sorted(usable.items())}
    report = {
        "splits": splits,
        "baseline": base,
        "summary": paired_summary(truth_300, usable, base, splits, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED) if base in usable else {},
        "per_split_q300": per_split,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Extract depth/penult/final/structural features and fit ridges")
    p_run.add_argument("--data", type=Path, required=True)
    p_run.add_argument("--tasks", type=Path, required=True)
    p_run.add_argument("--zs-features", type=Path, required=True)
    p_run.add_argument("--cache", type=Path, required=True)
    p_run.add_argument("--out", type=Path, required=True)
    p_run.add_argument("--models", default="esmc_300m,esmc_600m,esmc_6b")
    p_run.add_argument("--splits", default="")
    p_run.add_argument("--postcutoff-extra", action="store_true")

    p_ef = sub.add_parser("evaluate-flip2", help="Paired bootstrap evaluation on FLIP2 + FGFR2")
    p_ef.add_argument("--layers-dir", type=Path, required=True)
    p_ef.add_argument("--ref-predictions", type=Path, required=True)
    p_ef.add_argument("--extra-predictions", type=Path, nargs="*", default=[])
    p_ef.add_argument("--out", type=Path, required=True)

    p_ep = sub.add_parser("evaluate-postcutoff", help="Paired bootstrap evaluation on GBA1/ARAF post-cutoff suite")
    p_ep.add_argument("--data", type=Path, required=True)
    p_ep.add_argument("--tasks", type=Path, required=True)
    p_ep.add_argument("--layers-dir", type=Path, required=True)
    p_ep.add_argument("--prior-postcutoff-dir", type=Path, required=True)
    p_ep.add_argument("--out", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "run":
        models = tuple(m for m in args.models.split(",") if m)
        splits = {s for s in args.splits.split(",") if s} or None
        run_extract_and_fit(
            args.data, args.tasks, args.zs_features, args.cache, args.out, models, splits, args.postcutoff_extra
        )
    elif args.command == "evaluate-flip2":
        evaluate_layers_flip2(args.layers_dir, args.ref_predictions, args.extra_predictions, args.out)
        print(f"wrote FLIP2/FGFR2 layer comparison to {args.out}")
    elif args.command == "evaluate-postcutoff":
        evaluate_layers_postcutoff(args.data, args.tasks, args.layers_dir, args.prior_postcutoff_dir, args.out)
        print(f"wrote post-cutoff layer comparison to {args.out}")


if __name__ == "__main__":
    main()

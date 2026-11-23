"""The baselines on a toy split whose target is additive in its substitutions, with stand-in model features."""

from __future__ import annotations

import json
import numpy as np

from pathlib import Path

from general_models_vs_plms import baselines, flip2, plan
from general_models_vs_plms.encoding import AMINO_ACIDS, Layout, build_layout, composition, one_hot, random_vectors
from general_models_vs_plms.scoring import spearman
from general_models_vs_plms.strata import strata_of
from general_models_vs_plms.tasks import Arm, build_task, load_layout
from test_collect import write_toy_split


def write_toy_features(folder: Path, layout: Layout, rows: np.ndarray) -> None:
    """Features in the files `plm_features` writes: the embedding is a random projection of the one-hot rows."""
    # rows: (n,)
    generator = np.random.default_rng(3)
    encoded, _ = one_hot(layout, rows, rows[:1])  # (n, f)
    projection = generator.normal(size=(encoded.shape[1], 24))  # (f, 24)
    marginals = generator.normal(size=(len(layout.parent), len(AMINO_ACIDS)))  # (l, 20)
    for name in baselines.PLM_NAMES:
        (folder / name).mkdir(parents=True)
        np.savez(folder / name / "amylase-by_mutation.npz", rows=rows, embedding=encoded @ projection,
                 likelihood=generator.normal(size=len(rows)), parent_marginals=marginals)
    # A control's file holds the embedding alone; here it is noise, so its ridge has nothing to learn.
    for name in baselines.CONTROL_EMBEDDINGS:
        (folder / name).mkdir(parents=True)
        np.savez(folder / name / "amylase-by_mutation.npz", rows=rows, embedding=generator.normal(size=(len(rows), 24)))


def test_baselines_cover_every_method_and_ridge_learns_an_additive_target(tmp_path: Path) -> None:
    frame = write_toy_split(tmp_path / "data")
    split_info = plan.split_infos(("amylase/by_mutation",))[0]
    frame = flip2.load_split(flip2.split_path(tmp_path / "data", split_info))
    layout = load_layout(split_info, frame)
    write_toy_features(tmp_path / "features", layout, np.arange(len(frame)))
    features = baselines.SplitFeatures(tmp_path / "features", split_info, baselines.PLM_NAMES + baselines.CONTROL_EMBEDDINGS)
    task = build_task(split_info, frame, layout, Arm("k256", 256), 0, "test", 100)
    truth = frame["target"].to_numpy()[task.query_rows]
    strata = strata_of(split_info, layout)

    predictions = baselines.task_baselines(task, layout, strata, features)

    expected = {"ridge_onehot", "nearest_neighbor", "stratum_mean", "mutation_count", "ridge_composition", "ridge_random_vector"}
    expected |= {f"ridge_{name}" for name in baselines.CONTROL_EMBEDDINGS}
    for name in baselines.PLM_NAMES:
        expected |= {f"ridge_{name}", f"zero_shot_{name}", f"likelihood_{name}", f"ridge_onehot_plus_{name}"}
    assert set(predictions) == expected
    assert all(len(values) == 100 and np.isfinite(values).all() for values in predictions.values())
    # 256 rows of one to three substitutions cannot pin down 660 additive effects, so the bar is a clear signal.
    assert spearman(truth, np.array(predictions["ridge_onehot"])) > 0.3
    assert spearman(truth, np.array(predictions["ridge_esm2_650m"])) > 0.1
    # A vector drawn per sequence says nothing about a sequence the ridge has not seen: three standard errors of zero.
    assert abs(spearman(truth, np.array(predictions["ridge_random_vector"]))) < 0.3

    added = baselines.task_baselines(task, layout, strata, features, models=("esmc_6b",), controls=(), model_free=False)

    assert set(added) == {"ridge_esmc_6b", "zero_shot_esmc_6b", "likelihood_esmc_6b", "ridge_onehot_plus_esmc_6b"}
    assert added["ridge_esmc_6b"] == predictions["ridge_esmc_6b"]

    full = baselines.full_train_plm_reference(frame, layout, np.array(task.query_rows), features)
    reference = baselines.full_train_reference(split_info, frame, layout, np.array(task.query_rows))

    assert set(full) == {f"{prefix}_{name}_full" for prefix in ("ridge", "ridge_onehot_plus") for name in baselines.PLM_NAMES}
    assert spearman(truth, np.array(full["ridge_onehot_plus_esmc_600m_full"])) > spearman(truth, np.array(predictions["ridge_onehot"])) - 0.05
    assert spearman(truth, np.array(reference["ridge_onehot_full"])) >= spearman(truth, np.array(predictions["ridge_onehot"])) - 0.02


def test_added_methods_join_the_file_and_replace_nothing_else(tmp_path: Path) -> None:
    path = tmp_path / "task.json"
    baselines.write_methods(path, {"ridge_onehot": [1.0, 2.0]}, add=False)

    baselines.write_methods(path, {"ridge_esmc_6b": [3.0, 4.0]}, add=True)

    assert json.loads(path.read_text(encoding="utf-8")) == {"ridge_onehot": [1.0, 2.0], "ridge_esmc_6b": [3.0, 4.0]}


def test_composition_counts_residues_per_chain_and_ignores_order() -> None:
    layout = build_layout(["AAC:AC", "ACA:CA", "CCC:A"], "pair")

    fractions = composition(layout, np.arange(3))  # (3, 40)

    alanine, cysteine = AMINO_ACIDS.index("A"), AMINO_ACIDS.index("C")
    assert fractions.shape == (3, 40)
    assert np.allclose(fractions[0], fractions[1])
    assert np.isclose(fractions[0, alanine], 2 / 3) and np.isclose(fractions[0, cysteine], 1 / 3)
    assert np.isclose(fractions[2, 20 + alanine], 1.0) and np.isclose(fractions[2, cysteine], 1.0)
    assert np.allclose(fractions[:, :20].sum(axis=1), 1.0) and np.allclose(fractions[:, 20:].sum(axis=1), 1.0)


def test_random_vectors_follow_the_sequence_and_nothing_else() -> None:
    layout = build_layout(["ACDE", "ACDF", "ACDE"], "single_parent")

    vectors = random_vectors(layout, np.arange(3), 960)  # (3, 960)

    assert np.array_equal(vectors[0], vectors[2])
    assert abs(np.corrcoef(vectors[0], vectors[1])[0, 1]) < 0.15
    assert abs(vectors.mean()) < 0.1 and abs(vectors.std() - 1.0) < 0.1

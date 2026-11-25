"""Unit tests for ESMC multi-layer + structural stream pooling and feature slot layout."""

from __future__ import annotations

import numpy as np

from general_models_vs_plms.esmc_layers import MODEL_DEPTH_LAYER, MODEL_PENULT_LAYER, pool_mean_and_var


def test_canonical_layer_indices_match_spec() -> None:
    assert MODEL_DEPTH_LAYER == {"esmc_300m": 23, "esmc_600m": 27, "esmc_6b": 60}
    assert MODEL_PENULT_LAYER == {"esmc_300m": 29, "esmc_600m": 35, "esmc_6b": 79}


def test_pool_mean_and_var_shapes_and_values() -> None:
    tokens = np.array([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]], dtype=np.float32)  # (3, 2)
    mean, mv = pool_mean_and_var(tokens)
    assert mean.shape == (2,)
    assert mv.shape == (4,)
    assert np.allclose(mean, [3.0, 6.0])
    assert np.allclose(mv[:2], [3.0, 6.0])
    assert np.allclose(mv[2:], [8.0 / 3.0, 32.0 / 3.0])

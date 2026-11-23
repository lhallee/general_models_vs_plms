"""The sequence reorderings and the untrained model that `plm_features` uses as controls."""

from __future__ import annotations

import torch

from transformers import EsmConfig, EsmForMaskedLM

from general_models_vs_plms import plm_features


CHAIN = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVV"


def test_shuffled_chain_keeps_composition_and_is_stable() -> None:
    shuffled = plm_features.shuffled_chain(CHAIN)

    assert sorted(shuffled) == sorted(CHAIN) and shuffled != CHAIN
    assert shuffled == plm_features.shuffled_chain(CHAIN)
    # One substitution gives the variant an order of its own, so the two no longer line up.
    variant = CHAIN[:10] + "W" + CHAIN[11:]
    differing = sum(left != right for left, right in zip(shuffled, plm_features.shuffled_chain(variant), strict=True))
    assert differing > 10


def test_permuted_chain_moves_a_substitution_to_one_shared_place() -> None:
    variant = CHAIN[:10] + "W" + CHAIN[11:]
    permuted, permuted_variant = plm_features.permuted_chain(CHAIN), plm_features.permuted_chain(variant)

    assert sorted(permuted) == sorted(CHAIN) and permuted != CHAIN
    differing = [index for index, (left, right) in enumerate(zip(permuted, permuted_variant, strict=True)) if left != right]
    assert len(differing) == 1 and permuted_variant[differing[0]] == "W"
    assert plm_features.read_form(CHAIN, "pretrained") == CHAIN
    assert plm_features.read_form(CHAIN, "permuted") == permuted and plm_features.read_form(CHAIN, "shuffled") != permuted


def test_redraw_gives_the_same_untrained_weights_every_time_and_none_of_the_trained() -> None:
    config = EsmConfig(vocab_size=33, hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32,
                       max_position_embeddings=64, pad_token_id=1, mask_token_id=32)
    trained = EsmForMaskedLM(config)
    with torch.no_grad():
        for parameter in trained.parameters():
            parameter.add_(1.0)

    first = plm_features.ChainScorer.redraw(trained)
    second = plm_features.ChainScorer.redraw(trained)

    assert type(first) is type(trained)
    matrices = {name: parameter for name, parameter in first.named_parameters() if parameter.ndim >= 2}
    assert matrices
    for name, parameter in second.named_parameters():
        if name in matrices:
            assert torch.equal(parameter, matrices[name])
            assert not torch.equal(parameter, dict(trained.named_parameters())[name])

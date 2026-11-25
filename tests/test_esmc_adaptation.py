"""Unit tests for ESMC adaptation (LoRA parity, Bradley-Terry/KL losses, mutated positions, paired bootstrap)."""

from __future__ import annotations

import numpy as np
import torch

from torch import nn

from general_models_vs_plms.encoding import build_layout
from general_models_vs_plms.esmc_adaptation import (
    LoRALinear,
    attach_lora,
    bradley_terry_loss,
    detach_lora,
    kl_to_reference,
    mutated_positions_per_chain,
    paired_summary,
)


def test_lora_attach_zero_init_parity_and_detach() -> None:
    class TinyBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layernorm_qkv = nn.Sequential(nn.LayerNorm(8), nn.Linear(8, 24))
            self.out_proj = nn.Linear(24, 8)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.out_proj(self.layernorm_qkv(x))

    model = TinyBlock()
    x = torch.randn(3, 8)
    y_before = model(x).detach()

    trainable = attach_lora(model, rank=2, alpha=4.0)
    assert len(trainable) == 4  # 2 Linear layers x (lora_a, lora_b)
    assert isinstance(model.layernorm_qkv[1], LoRALinear)
    assert isinstance(model.out_proj, LoRALinear)
    y_after_init = model(x).detach()
    assert torch.allclose(y_before, y_after_init, atol=1e-6)

    # One gradient step moves only LoRA parameters
    loss = model(x).pow(2).mean()
    loss.backward()
    for p in trainable:
        assert p.grad is not None

    detach_lora(model)
    assert isinstance(model.layernorm_qkv[1], nn.Linear)
    assert isinstance(model.out_proj, nn.Linear)
    assert torch.allclose(y_before, model(x).detach(), atol=1e-6)


def test_bradley_terry_and_kl_losses() -> None:
    targets = torch.tensor([1.0, 2.0, 3.0])
    good = torch.tensor([1.0, 5.0, 9.0], requires_grad=True)
    bad = torch.tensor([9.0, 5.0, 1.0], requires_grad=True)
    loss_good = bradley_terry_loss(good, targets)
    loss_bad = bradley_terry_loss(bad, targets)
    assert float(loss_good) < float(loss_bad)
    loss_good.backward()
    assert good.grad is not None and torch.isfinite(good.grad).all()

    lp = torch.log_softmax(torch.randn(5, 20), dim=-1)
    assert float(kl_to_reference(lp, lp)) < 1e-6
    shifted = torch.log_softmax(torch.randn(5, 20), dim=-1)
    assert float(kl_to_reference(lp, shifted)) > 0.0


def test_mutated_positions_per_chain_two_chains() -> None:
    # Pair of chains separated by ":"
    seqs = ["ACDE:FGHI", "ACDE:FGHI", "WCDE:FGHI", "ACDE:FGYI", "WCDE:FGYI"]
    layout = build_layout(seqs, "pair")
    assert layout.parent is not None
    assert mutated_positions_per_chain(layout, 0) == [[], []]
    assert mutated_positions_per_chain(layout, 2) == [[(0, "A", "W")], []]
    assert mutated_positions_per_chain(layout, 3) == [[], [(2, "H", "Y")]]
    assert mutated_positions_per_chain(layout, 4) == [[(0, "A", "W")], [(2, "H", "Y")]]


def test_paired_summary_detects_clear_improvement() -> None:
    rng = np.random.default_rng(7)
    truth = {f"s{i}": rng.normal(size=100) for i in range(4)}
    base = {s: truth[s] * 0.2 + rng.normal(scale=1.0, size=100) for s in truth}
    strong = {s: truth[s] * 0.9 + rng.normal(scale=0.3, size=100) for s in truth}
    summary = paired_summary(truth, {"base": base, "strong": strong}, "base", list(truth.keys()), draws=200)
    assert summary["strong"]["beats_baseline"] is True
    assert summary["strong"]["diff_ci_low"] > 0.0
    assert summary["base"]["diff_vs_baseline"] == 0.0

from __future__ import annotations

from concept_erasure import LeaceEraser
import torch
from torch import Tensor

from .alignment import align
from .artifacts import WhiteningTransform
from .config import OTConfig


def normalized_mahalanobis_loss(
    current: Tensor,  # (n_samples, hidden_dim)
    target: Tensor,  # (n_samples, hidden_dim)
    whitening: WhiteningTransform,
    *,
    epsilon: float = 1e-8,
    reduction: str = "mean",
) -> Tensor:
    difference = whitening(current - target).square().sum(dim=-1)  # (n_samples,)
    scale = whitening(target).square().sum(dim=-1).clamp_min(epsilon)  # (n_samples,)
    values = difference / scale
    if reduction == "none":
        return values
    if reduction == "sum":
        return values.sum()
    if reduction == "mean":
        return values.mean()
    raise ValueError(f"Unknown reduction: {reduction}")


def carot_alignment_loss(
    source_hidden_states: Tensor,
    target_hidden_states: Tensor,
    eraser: LeaceEraser,
    whitening: WhiteningTransform,
    config: OTConfig | None = None,
    *,
    reduction: str = "mean",
) -> Tensor:
    """Training loss from the paper; gradients flow only through target states.

    Args:
        source_hidden_states: Hidden states from the source language.
            shape = ``(source_tokens, hidden)``.
        target_hidden_states: Hidden states from the target language.
            shape = ``(target_tokens, hidden)``.
        eraser: Fitted LEACE eraser for the target layer.
        whitening: Fitted whitening transform for the target layer.
        config: Optional hyperparameters for unbalanced Sinkhorn alignment.
        reduction: How to reduce the per-sample loss values.
            choices = ["none", "mean", "sum"]
            "mean" is used in the paper.

    Returns:
        Scalar loss value; gradients flow only through ``target_hidden_states``.
    """

    config = config or OTConfig()
    with torch.no_grad():
        alignment_target = align(
            source_hidden_states.detach(),
            target_hidden_states.detach(),
            eraser,
            whitening,
            config,
        ).semantic_states
    current_semantic = eraser(target_hidden_states)
    return normalized_mahalanobis_loss(
        current_semantic,
        alignment_target,
        whitening,
        epsilon=config.numerical_epsilon,
        reduction=reduction,
    )

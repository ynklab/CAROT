from __future__ import annotations

from dataclasses import dataclass

from concept_erasure import LeaceEraser
import ot
import torch
from torch import Tensor, nn

from .artifacts import LayerArtifacts, WhiteningTransform
from .config import OTConfig


@dataclass
class AlignmentOutput:
    """CAROT outputs for one parallel sentence pair."""

    hidden_states: Tensor
    semantic_states: Tensor
    transport_plan: Tensor
    received_mass: Tensor
    interpolation: Tensor


def squared_mahalanobis_cost(
    source: Tensor, target: Tensor, whitening: WhiteningTransform
) -> Tensor:
    """Return a ``(n_target, n_source)`` squared-distance matrix."""
    source_white = whitening(source)
    target_white = whitening(target)
    return torch.cdist(target_white.float(), source_white.float(), p=2).square()


def _unbalanced_sinkhorn(cost: Tensor, config: OTConfig) -> Tensor:
    n_target, n_source = cost.shape
    target_mass = torch.full(
        (n_target,), 1.0 / n_target, device=cost.device, dtype=cost.dtype
    )
    source_mass = torch.full(
        (n_source,), 1.0 / n_source, device=cost.device, dtype=cost.dtype
    )
    return ot.sinkhorn_unbalanced(
        target_mass,
        source_mass,
        cost,
        reg=config.regularization,
        reg_m=config.marginal_relaxation,
        numItermax=config.max_iterations,
        stopThr=config.stop_threshold,
    )


def filter_transport_plan(plan: Tensor, threshold: float, epsilon: float) -> Tensor:
    """Remove diffuse rows and columns using the perplexity filter."""
    if threshold >= 1.0:
        return plan
    row_distribution = plan / plan.sum(dim=-1, keepdim=True).clamp_min(epsilon)
    col_distribution = plan / plan.sum(dim=0, keepdim=True).clamp_min(epsilon)
    row_ppl = torch.exp(
        -(row_distribution * row_distribution.clamp_min(epsilon).log()).sum(dim=-1)
    )  # (n_target,)
    col_ppl = torch.exp(
        -(col_distribution * col_distribution.clamp_min(epsilon).log()).sum(dim=0)
    )  # (n_source,)
    row_keep = row_ppl <= threshold * plan.shape[1]
    col_keep = col_ppl <= threshold * plan.shape[0]
    return plan * row_keep[:, None] * col_keep[None, :]


def _align_single(
    source_hidden_states: Tensor,
    target_hidden_states: Tensor,
    eraser: LeaceEraser,
    whitening: WhiteningTransform,
    config: OTConfig,
) -> AlignmentOutput:
    source_semantic = eraser(source_hidden_states)
    target_semantic = eraser(target_hidden_states)
    target_language = target_hidden_states - target_semantic

    n_source = source_hidden_states.shape[0]
    n_target = target_hidden_states.shape[0]
    if n_source == 1 or n_target == 1:
        plan = torch.full(
            (n_target, n_source),
            1.0 / (n_target if n_source == 1 else n_source),
            device=target_hidden_states.device,
            dtype=target_hidden_states.dtype,
        )
        aligned_semantic = (
            source_semantic.expand(n_target, -1)
            if n_source == 1
            else source_semantic.mean(dim=0, keepdim=True)
        )
        received_mass = plan.sum(dim=-1)
        interpolation = torch.ones_like(received_mass)
        return AlignmentOutput(
            hidden_states=(aligned_semantic + target_language).to(target_hidden_states.dtype),
            semantic_states=aligned_semantic.to(target_hidden_states.dtype),
            transport_plan=plan,
            received_mass=received_mass,
            interpolation=interpolation,
        )

    cost = squared_mahalanobis_cost(source_semantic, target_semantic, whitening)
    # Min-max normalization to stabilize the Sinkhorn algorithm
    cost = (cost - cost.min()) / (cost.max() - cost.min()).clamp_min(
        config.numerical_epsilon
    )
    # Compute the unbalanced Sinkhorn transport plan and filter out diffuse rows/columns
    plan = _unbalanced_sinkhorn(cost, config).to(
        device=target_hidden_states.device, dtype=target_hidden_states.dtype
    )
    plan = filter_transport_plan(
        plan, config.perplexity_threshold, config.numerical_epsilon
    )

    # Compute the aligned hidden states as a convex combination of transported source semantics and target semantics
    row_mass = plan.sum(dim=-1)  # (n_target,) Received mass for each target token
    nominal_mass = 1.0 / target_hidden_states.shape[0]
    interpolation = (row_mass / (nominal_mass + config.numerical_epsilon)).clamp(max=1.0)
    normalized_plan = plan / row_mass[:, None].clamp_min(config.numerical_epsilon)
    transported = normalized_plan @ source_semantic  # (n_target, hidden_dim) Transported source semantics
    aligned_semantic = (
        interpolation[:, None] * transported
        + (1.0 - interpolation[:, None]) * target_semantic
    )
    # Add the target language component back to the aligned semantics to get the final aligned hidden states
    aligned_hidden = aligned_semantic + target_language
    return AlignmentOutput(
        hidden_states=aligned_hidden.to(target_hidden_states.dtype),
        semantic_states=aligned_semantic.to(target_hidden_states.dtype),
        transport_plan=plan,
        received_mass=row_mass,
        interpolation=interpolation,
    )


def align(
    source_hidden_states: Tensor,
    target_hidden_states: Tensor,
    eraser: LeaceEraser,
    whitening: WhiteningTransform,
    config: OTConfig | None = None,
) -> AlignmentOutput:
    """Apply CAROT to one parallel sentence pair.

    Args:
        source_hidden_states: ``(source_tokens, hidden)``.
        target_hidden_states: ``(target_tokens, hidden)``.
    """
    config = config or OTConfig()
    if source_hidden_states.ndim != 2 or target_hidden_states.ndim != 2:
        raise ValueError("Source and target states must both have shape (tokens, hidden)")
    if source_hidden_states.shape[-1] != target_hidden_states.shape[-1]:
        raise ValueError("Source and target hidden sizes differ")
    if source_hidden_states.shape[0] == 0 or target_hidden_states.shape[0] == 0:
        raise ValueError("Source and target must each contain at least one token")
    return _align_single(
        source_hidden_states,
        target_hidden_states,
        eraser,
        whitening,
        config,
    )


class CAROT(nn.Module):
    """Layer-specific CAROT transform."""

    def __init__(self, artifacts: LayerArtifacts, config: OTConfig | None = None):
        super().__init__()
        self.register_buffer("eraser_proj_left", artifacts.eraser.proj_left.detach().clone())
        self.register_buffer("eraser_proj_right", artifacts.eraser.proj_right.detach().clone())
        self.register_buffer("eraser_bias", artifacts.eraser.bias.detach().clone())
        self.whitening = artifacts.whitening
        self.metadata = artifacts.metadata
        self.config = config or OTConfig()

    @property
    def eraser(self) -> LeaceEraser:
        """Reconstruct the compact eraser from registered buffers.

        Registering the tensors makes the usual ``module.to(device, dtype)``
        operation move both LEACE and whitening parameters together.
        """
        return LeaceEraser(
            proj_left=self.eraser_proj_left,
            proj_right=self.eraser_proj_right,
            bias=self.eraser_bias,
        )

    def forward(
        self,
        source_hidden_states: Tensor,
        target_hidden_states: Tensor,
    ) -> Tensor:
        return self.align(source_hidden_states, target_hidden_states).hidden_states

    def align(
        self,
        source_hidden_states: Tensor,
        target_hidden_states: Tensor,
    ) -> AlignmentOutput:
        return align(
            source_hidden_states,
            target_hidden_states,
            self.eraser,
            self.whitening,
            self.config,
        )

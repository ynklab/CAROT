from __future__ import annotations

from typing import Any

from concept_erasure import LeaceEraser
import torch
import torch.nn.functional as F
from torch import Tensor

from .artifacts import LayerArtifacts, WhiteningTransform


def pool_sentence_embeddings(
    hidden_states: Tensor,
    attention_mask: Tensor,
    *,
    method: str = "position_weighted",
) -> Tensor:
    """Aggregate ``[batch, tokens, hidden]`` states into sentence embeddings.

    The caller should clear BOS/EOS/padding positions in ``attention_mask`` first.
    method = "position_weighted" is always used in the paper.
    """
    if hidden_states.ndim != 3 or attention_mask.shape != hidden_states.shape[:2]:
        raise ValueError("Expected hidden states (batch, tokens, hidden) and mask (batch, tokens)")
    mask = attention_mask.to(device=hidden_states.device, dtype=hidden_states.dtype)
    if torch.any(mask.sum(dim=1) == 0):
        raise ValueError("Every sentence must contain at least one selected token")
    if method == "mean":
        weights = mask / mask.sum(dim=1, keepdim=True)
    elif method == "position_weighted":
        positions = mask.cumsum(dim=1) * mask
        weights = positions / positions.sum(dim=1, keepdim=True)
    else:
        raise ValueError(f"Unknown pooling method: {method}")
    return (hidden_states * weights.unsqueeze(-1)).sum(dim=1)


def fit_layer_artifacts(
    sentence_embeddings: Tensor,
    language_labels: Tensor,
    *,
    eigenvalue_relative_threshold: float = 1e-5,
    metadata: dict[str, Any] | None = None,
) -> LayerArtifacts:
    """Fit one layer's LEACE eraser and semantic covariance whitening.

    Args:
        sentence_embeddings: Hidden states of LLMs.
            shape = (num_samples, hidden_dim)
        language_labels: Language labels for each sample.
            shape = (num_samples,)
        eigenvalue_relative_threshold: Eigenvalues below this fraction of the maximum
            eigenvalue are discarded when computing the whitening transform.
        metadata: Optional dictionary of additional information to store with the artifacts.

    Returns:
        LayerArtifacts containing the fitted LEACE eraser, whitening transform, and metadata.
    """

    if sentence_embeddings.ndim != 2:
        raise ValueError("sentence_embeddings must have shape (samples, hidden)")
    if language_labels.ndim != 1 or language_labels.shape[0] != sentence_embeddings.shape[0]:
        raise ValueError("language_labels must have shape (samples,)")
    labels = language_labels.long()
    num_languages = int(labels.max().item()) + 1
    one_hot = F.one_hot(labels, num_classes=num_languages).float().to(sentence_embeddings.device)
    # one_hot.shape = (samples, languages)

    # Fit the LEACE eraser to remove language information from the embeddings
    eraser = LeaceEraser.fit(sentence_embeddings.float(), one_hot)

    # Compute the covariance matrix of the language-erased embeddings and fit the whitening transform
    semantic = eraser(sentence_embeddings.float())
    covariance = torch.cov(semantic.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    keep = eigenvalues > eigenvalues.max() * eigenvalue_relative_threshold
    if not torch.any(keep):
        raise ValueError("Semantic covariance has no eigenvalues above the threshold")
    whitening = WhiteningTransform(eigenvalues[keep], eigenvectors[:, keep])

    return LayerArtifacts(eraser, whitening, metadata or {})

"""Public API for CAROT."""

from .alignment import AlignmentOutput, CAROT, align
from .artifacts import LayerArtifacts, WhiteningTransform
from .config import OTConfig
from .fitting import fit_layer_artifacts, pool_sentence_embeddings
from .losses import carot_alignment_loss, normalized_mahalanobis_loss

__all__ = [
    "AlignmentOutput",
    "CAROT",
    "LayerArtifacts",
    "OTConfig",
    "WhiteningTransform",
    "align",
    "carot_alignment_loss",
    "fit_layer_artifacts",
    "normalized_mahalanobis_loss",
    "pool_sentence_embeddings",
]

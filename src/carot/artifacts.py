from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from concept_erasure import LeaceEraser
import torch
from torch import Tensor, nn


class WhiteningTransform(nn.Module):
    """Low-rank inverse square root of the semantic covariance matrix."""

    def __init__(self, eigenvalues: Tensor, eigenvectors: Tensor):
        super().__init__()
        if eigenvalues.ndim != 1 or eigenvectors.ndim != 2:
            raise ValueError("Expected eigenvalues [k] and eigenvectors [d, k]")
        if eigenvectors.shape[1] != eigenvalues.shape[0]:
            raise ValueError("The eigensystem dimensions do not match")
        if torch.any(eigenvalues <= 0):
            raise ValueError("Whitening eigenvalues must be positive")
        self.register_buffer("eigenvalues", eigenvalues.detach().clone())
        self.register_buffer("eigenvectors", eigenvectors.detach().clone())

    @property
    def matrix(self) -> Tensor:
        return self.eigenvectors / self.eigenvalues.sqrt().unsqueeze(0)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return hidden_states @ self.matrix.to(
            device=hidden_states.device, dtype=hidden_states.dtype
        )


@dataclass
class LayerArtifacts:
    """Model- and layer-specific parameters fitted before CAROT is applied."""

    eraser: LeaceEraser
    whitening: WhiteningTransform
    metadata: dict[str, Any]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        eraser_state = {
            "proj_left": self.eraser.proj_left.detach().clone(),
            "proj_right": self.eraser.proj_right.detach().clone(),
            "bias": self.eraser.bias.detach().clone(),
        }
        torch.save(
            {
                "format_version": 1,
                "eraser": eraser_state,
                "whitening": self.whitening.state_dict(),
                "metadata": self.metadata,
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        map_location: str | torch.device = "cpu",
    ) -> "LayerArtifacts":
        payload = torch.load(path, map_location=map_location, weights_only=True)
        if payload.get("format_version") != 1:
            raise ValueError("Unsupported CAROT artifact format")
        eraser_state = payload["eraser"]
        whitening_state = payload["whitening"]
        return cls(
            eraser=LeaceEraser(
                proj_left=eraser_state["proj_left"],
                proj_right=eraser_state["proj_right"],
                bias=eraser_state["bias"],
            ),
            whitening=WhiteningTransform(
                eigenvalues=whitening_state["eigenvalues"],
                eigenvectors=whitening_state["eigenvectors"],
            ),
            metadata=payload.get("metadata", {}),
        )

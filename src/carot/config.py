from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class OTConfig:
    """Hyperparameters for CAROT's unbalanced Sinkhorn alignment."""

    regularization: float = 0.05
    marginal_relaxation: float = 0.5
    perplexity_threshold: float = 0.25
    max_iterations: int = 1000
    stop_threshold: float = 1e-6
    numerical_epsilon: float = 1e-8

    def __post_init__(self) -> None:
        if self.regularization <= 0:
            raise ValueError("regularization must be positive")
        if self.marginal_relaxation <= 0:
            raise ValueError("marginal_relaxation must be positive")
        if self.perplexity_threshold <= 0:
            raise ValueError("perplexity_threshold must be positive")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")

    def to_dict(self) -> dict:
        return asdict(self)


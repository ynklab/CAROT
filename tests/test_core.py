import torch
from concept_erasure import LeaceEraser

from carot import OTConfig, WhiteningTransform, align, carot_alignment_loss
from carot.alignment import filter_transport_plan, squared_mahalanobis_cost
from carot.fitting import pool_sentence_embeddings
from carot.losses import normalized_mahalanobis_loss


def identity_eraser(hidden_size: int) -> LeaceEraser:
    empty = torch.zeros(hidden_size, 0)
    return LeaceEraser(empty, empty.T, torch.zeros(hidden_size))


def test_position_weighted_pooling_ignores_masked_tokens():
    hidden = torch.tensor([[[1.0], [3.0], [100.0]]])
    mask = torch.tensor([[1, 1, 0]])
    pooled = pool_sentence_embeddings(hidden, mask)
    assert torch.allclose(pooled, torch.tensor([[(1.0 + 2.0 * 3.0) / 3.0]]))


def test_leace_eraser_identity_with_empty_concept_rank():
    hidden = torch.randn(4, 3)
    assert torch.equal(identity_eraser(3)(hidden), hidden)


def test_squared_mahalanobis_cost_shape_and_values():
    whitening = WhiteningTransform(torch.ones(2), torch.eye(2))
    source = torch.tensor([[0.0, 0.0], [2.0, 0.0]])
    target = torch.tensor([[1.0, 0.0]])
    cost = squared_mahalanobis_cost(source, target, whitening)
    assert cost.shape == (1, 2)
    assert torch.allclose(cost, torch.ones(1, 2))


def test_filter_removes_diffuse_row():
    plan = torch.tensor([[0.25, 0.25], [0.49, 0.01]])
    filtered = filter_transport_plan(plan, threshold=0.75, epsilon=1e-8)
    assert torch.equal(filtered[0], torch.zeros(2))


def test_normalized_loss_is_zero_at_target():
    whitening = WhiteningTransform(torch.ones(3), torch.eye(3))
    target = torch.randn(5, 3)
    loss = normalized_mahalanobis_loss(target, target, whitening)
    assert loss.item() == 0.0


def test_single_source_token_does_not_require_sinkhorn():
    source = torch.tensor([[1.0, 2.0]])
    target = torch.zeros(3, 2)
    output = align(
        source,
        target,
        identity_eraser(2),
        WhiteningTransform(torch.ones(2), torch.eye(2)),
        OTConfig(perplexity_threshold=0.25),
    )
    assert torch.equal(output.semantic_states, source.expand(3, -1))
    assert torch.equal(output.interpolation, torch.ones(3))


def test_alignment_rejects_batched_states():
    source = torch.ones(2, 1, 2)
    target = torch.zeros(2, 1, 2)
    try:
        align(
            source,
            target,
            identity_eraser(2),
            WhiteningTransform(torch.ones(2), torch.eye(2)),
        )
    except ValueError as error:
        assert "(tokens, hidden)" in str(error)
    else:
        raise AssertionError("align accepted batched hidden states")


def test_alignment_loss_backpropagates_only_through_target():
    source = torch.tensor([[1.0, 1.0]], requires_grad=True)
    target = torch.zeros(2, 2, requires_grad=True)
    loss = carot_alignment_loss(
        source,
        target,
        identity_eraser(2),
        WhiteningTransform(torch.ones(2), torch.eye(2)),
    )
    loss.backward()
    assert target.grad is not None
    assert source.grad is None

import pytest
import torch

from cfmusic.conditioning.schema import build_condition_batch
from cfmusic.losses.centered_content import centered_content_loss
from cfmusic.training.transport_trainer import TransportLossModule
from cfmusic.transport.conditional_flow import ConditionalFlow


def test_permits_class_mean_displacement_but_not_residual_destruction():
    source = torch.tensor([[[2., 3.], [4., 5.]]])
    means = torch.tensor([[[1., 0.], [0., 1.]], [[5., 2.], [3., 4.]]])
    src, tgt = torch.tensor([0]), torch.tensor([1])
    ideal = source + means[1] - means[0]
    assert centered_content_loss(ideal, source, src, tgt, means) == 0
    assert centered_content_loss(means[1:].clone(), source, src, tgt, means) > 0
    assert centered_content_loss(source, source, src, tgt, means) > 0


def test_gradient_only_on_prediction():
    source = torch.ones(2, 3, 4, requires_grad=True)
    generated = torch.zeros(2, 3, 4, requires_grad=True)
    means = torch.zeros(2, 3, 4)
    loss = centered_content_loss(generated, source, torch.tensor([0, 1]), torch.tensor([1, 0]), means)
    loss.backward()
    assert source.grad is None
    assert generated.grad is not None and generated.grad.abs().sum() > 0
    assert means.grad is None


def test_rejects_learnable_means_and_invalid_labels():
    a, ids = torch.zeros(2, 3, 4), torch.tensor([0, 1])
    with pytest.raises(ValueError):
        centered_content_loss(a, a, ids, ids, torch.zeros(2, 3, 4, requires_grad=True))
    with pytest.raises(ValueError):
        centered_content_loss(a, a, ids, torch.tensor([0, 2]), torch.zeros(2, 3, 4))


class TinyField(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(.1))
        self.offset = torch.nn.Parameter(torch.tensor(.2))

    def forward(self, state, time, condition):
        mask = torch.ones_like(condition.style_id) if condition.condition_mask is None else condition.condition_mask
        label = (condition.style_id * mask).to(state)[:, None, None]
        return self.scale * state + self.offset * label


def loss_arguments():
    return dict(condition_contrast_weight=0., condition_contrast_margin=0., condition_contrast_samples=None,
                roundtrip_weight=0., roundtrip_steps=1, roundtrip_samples=None, roundtrip_cosine_weight=.1,
                endpoint_weight=0., endpoint_steps=1, endpoint_samples_per_style=1,
                exogeneity_hsic_weight=0., exogeneity_prior_weight=0., exogeneity_cross_mmd_weight=0.,
                exogeneity_cross_swd_weight=0., exogeneity_steps=1, exogeneity_samples_per_style=1,
                global_step=0, factorial_conditioning=False, factorial_active_axis=None)


def test_disabled_auxiliary_preserves_primary_loss_and_rng():
    flow = ConditionalFlow(TinyField())
    x = torch.randn(4, 2, 3)
    condition = build_condition_batch({"dataset_id": torch.zeros(4, dtype=torch.long),
                                      "genre_id": torch.tensor([0, 0, 1, 1])}, torch.device("cpu"), task="genre")
    torch.manual_seed(15)
    direct = flow.training_loss(x, condition)
    expected_rng = torch.get_rng_state()
    torch.manual_seed(15)
    actual = TransportLossModule(flow)(x, condition, None, None, **loss_arguments())
    torch.testing.assert_close(actual["loss"], direct["loss"])
    assert torch.equal(expected_rng, torch.get_rng_state())


def test_enabled_auxiliary_backpropagates_without_extra_parameters():
    flow = ConditionalFlow(TinyField(), classifier_free_guidance=True, condition_dropout=.1)
    objective = TransportLossModule(flow, content_means=torch.zeros(2, 2, 3))
    x = torch.randn(4, 2, 3)
    condition = build_condition_batch({"dataset_id": torch.zeros(4, dtype=torch.long),
                                      "genre_id": torch.tensor([0, 0, 1, 1])}, torch.device("cpu"), task="genre")
    result = objective(x, condition, None, None, **loss_arguments(), content_weight=.5, content_steps=2)
    assert result["content_loss"] > 0
    torch.testing.assert_close(result["loss"], result["cfm_loss"] + .5 * result["content_loss"])
    result["loss"].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in flow.parameters())
    assert sum(p.numel() for p in objective.parameters()) == sum(p.numel() for p in flow.parameters())
    assert "content_means" not in objective.state_dict()

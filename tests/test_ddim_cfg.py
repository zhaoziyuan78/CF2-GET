import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from cfmusic.conditioning.schema import ConditionBatch, build_condition_batch
from cfmusic.transport.conditional_ddim import ConditionalDDIM
from cfmusic.transport.factory import create_transport


class RecordingDenoiser(nn.Module):
    """Null branch predicts 0.01 x; the conditional branch adds a style offset."""

    def __init__(self) -> None:
        super().__init__()
        self.masks: list[torch.Tensor] = []

    def forward(self, state: torch.Tensor, time: torch.Tensor, condition: ConditionBatch) -> torch.Tensor:
        mask = (
            torch.ones(state.shape[0]) if condition.condition_mask is None else condition.condition_mask
        )
        self.masks.append(mask.clone())
        offset = 1e-3 * (condition.style_id.to(state) + 1) * mask.to(state)
        return 0.01 * state + offset[:, None, None]


def condition(count: int) -> ConditionBatch:
    return build_condition_batch(
        {"dataset_id": torch.zeros(count, dtype=torch.long), "style_id": torch.arange(count) % 6},
        torch.device("cpu"),
        task="genre",
    )


def model(**options) -> ConditionalDDIM:
    settings = dict(train_timesteps=100, classifier_free_guidance=True, condition_dropout=0.1,
                    prediction_guidance_scale=4.0)
    settings.update(options)
    return ConditionalDDIM(RecordingDenoiser(), **settings)  # type: ignore[arg-type]


def test_guided_epsilon_uses_null_and_conditional_branches() -> None:
    state, labels = torch.randn(4, 2, 3), condition(4)
    guided, evaluations = model()._guided_predict(state, 50, labels, 4.0)
    offset = 1e-3 * (labels.style_id.float() + 1)
    assert evaluations == 2
    torch.testing.assert_close(guided, 0.01 * state + 4.0 * offset[:, None, None])
    plain, evaluations = model(classifier_free_guidance=False)._guided_predict(state, 50, labels, 4.0)
    assert evaluations == 1
    torch.testing.assert_close(plain, 0.01 * state + offset[:, None, None])


def test_training_applies_condition_dropout() -> None:
    transport = model(condition_dropout=0.5)
    losses = transport.training_loss(torch.randn(512, 2, 3), condition(512))
    assert 0.35 < float(losses["condition_dropout_fraction"]) < 0.65
    mask = transport.denoiser.masks[0]
    assert mask.min() == 0 and mask.max() == 1
    unguided = model(classifier_free_guidance=False, condition_dropout=0.5)
    assert float(unguided.training_loss(torch.randn(8, 2, 3), condition(8))["condition_dropout_fraction"]) == 0


def test_counterfactual_guides_only_the_target_prediction() -> None:
    transport = model()
    output = transport.counterfactual(torch.randn(3, 2, 3), condition(3), condition(3), num_steps=4)
    masks = transport.denoiser.masks
    assert len(masks) == 12
    assert all(bool(mask.eq(1).all()) for mask in masks[:8])  # abduction and reconstruction
    assert all(mask.tolist() == [0, 0, 0, 1, 1, 1] for mask in masks[8:])  # target CFG
    assert output.inverse_nfe == 4 and output.forward_nfe == 4 + 8


def test_guided_ddim_requires_condition_dropout() -> None:
    with pytest.raises(ValueError, match="condition_dropout"):
        model(condition_dropout=0.0)


def test_factory_builds_guided_ddim_and_rejects_repulsion() -> None:
    from hydra import compose, initialize_config_dir

    from cfmusic.config import CONFIG_DIR

    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="config", overrides=["experiment=xmidi_ddim_cfg"]).transport
    transport = create_transport(cfg)
    assert isinstance(transport, ConditionalDDIM)
    assert transport.classifier_free_guidance and transport.condition_dropout == 0.1
    assert (transport.abduction_guidance_scale, transport.reconstruction_guidance_scale,
            transport.prediction_guidance_scale) == (1.0, 1.0, 4.0)
    assert transport.max_timestep == 955
    rejected = OmegaConf.merge(cfg, {"source_repulsion_scale": 0.5})
    with pytest.raises(ValueError, match="plain classifier-free guidance"):
        create_transport(rejected)

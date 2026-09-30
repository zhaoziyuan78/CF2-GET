from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from cfmusic.latent.compatibility import expected_transport_input_dim

CONFIGS = Path(__file__).parents[1] / "configs"


@pytest.mark.parametrize(
    "experiment",
    [
        "xmidi_otcfm_hsic",
        "xmidi_otcfm_no_hsic",
        "xmidi_ddim",
        "xmidi_ddim_cfg",
        "xmidi_latent_midi_vae",
        "xmidi_midi_vae",
    ],
)
def test_transports_match_the_16bar_chunk_codec(experiment: str) -> None:
    with initialize_config_dir(config_dir=str(CONFIGS), version_base=None):
        config = compose(config_name="config", overrides=[f"experiment={experiment}"])

    assert str(config.data.name) == "xmidi"
    assert int(config.data.segmentation.num_bars) == 16
    assert str(config.codec.type) == "bar_chunk_vae"
    assert int(config.codec.latent_tokens) == 128
    assert int(config.codec.latent_dim) == 512
    assert int(config.tokenizer.max_sequence_length) == 5120
    assert expected_transport_input_dim(config.transport) == 512
    if config.transport.type != "midi_vae":
        assert int(config.transport.model.latent_tokens) == 128

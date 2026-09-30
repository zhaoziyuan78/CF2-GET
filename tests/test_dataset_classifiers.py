
import miditoolkit
import pytest
import torch
from hydra import compose, initialize_config_dir

from cfmusic.config import CONFIG_DIR
from cfmusic.data.midi_io import _validate_midi
from cfmusic.evaluation.style_effect import TokenStyleClassifier


@pytest.mark.parametrize("dataset,enabled", [("xmidi", False)])
def test_dataset_evaluation_defaults(dataset: str, enabled: bool) -> None:
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="config", overrides=[f"data={dataset}"])
    assert bool(cfg.evaluation.classifier.enabled) == enabled
    assert bool(cfg.evaluation.clamp3.enabled) != enabled


def test_zero_duration_drum_onsets_are_preserved() -> None:
    midi = miditoolkit.MidiFile(ticks_per_beat=480)
    drum = miditoolkit.Instrument(program=0, is_drum=True)
    drum.notes = [miditoolkit.Note(80, 36, 120, 120)]
    midi.instruments = [drum]
    _validate_midi(midi)
    assert (drum.notes[0].start, drum.notes[0].end, drum.notes[0].velocity) == (120, 121, 80)
    drum.is_drum = False
    drum.notes[0].end = 120
    with pytest.raises(ValueError, match="Invalid note interval"):
        _validate_midi(midi)


def test_classifier_padding_uses_beat_pad_not_pattern_zero() -> None:
    model = TokenStyleClassifier(593, 4, d_model=16, layers=1, heads=4, max_length=32, pad_id=429)
    tokens = torch.tensor([[428, 0, 427, 429]])
    model(tokens, tokens.ne(429)).sum().backward()
    assert model.embedding.weight.grad is not None
    assert model.embedding.weight.grad[0].abs().sum() > 0
    assert model.embedding.weight.grad[429].abs().sum() == 0



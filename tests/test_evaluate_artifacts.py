import json
from pathlib import Path

import miditoolkit
import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

from cfmusic.commands.evaluate import artifact_midi_validity
from cfmusic.conditioning.schema import CONDITION_SCHEMA_VERSION


@pytest.mark.parametrize("valid_generated", [True, False])
def test_clamp3_evaluation_reaches_report(
    tmp_path: Path, monkeypatch, valid_generated: bool
) -> None:
    from cfmusic.commands import build_report, evaluate

    paths = {name: tmp_path / name for name in ("artifacts_dir", "processed_dir", "reports_dir")}
    artifact = paths["artifacts_dir"] / "test_run/xmidi/example"
    artifact.mkdir(parents=True)
    card_dir = paths["processed_dir"] / "xmidi"
    card_dir.mkdir(parents=True)
    (card_dir / "dataset_card.json").write_text(
        json.dumps({"style_vocabulary": ["classical", "rock"]})
    )
    _write_midi(artifact / "source.mid", with_note=True)
    _write_midi(artifact / "counterfactual.mid", with_note=valid_generated)
    _write_midi(artifact / "vae_reconstruction.mid", with_note=True)
    (artifact / "counterfactual_metadata.json").write_text(
        json.dumps(
            {
                "condition_schema_version": CONDITION_SCHEMA_VERSION,
                "sample_id": "song",
                "segment_id": "song-0",
                "source_style_id": 0,
                "target_style_id": 1,
                "source_style": "classical",
                "target_style": "rock",
            }
        )
    )
    torch.save(torch.zeros(1, 4), artifact / "abducted_noise.pt")
    monkeypatch.setattr(evaluate, "prepare_config", lambda _: paths)
    monkeypatch.setattr(build_report, "prepare_config", lambda _: paths)
    monkeypatch.setattr(evaluate, "embedding_protocol", lambda _: {"model": "clamp3-c2"})

    def embeddings(**kwargs):
        return {
            key: np.eye(2)[int(key.startswith("generated") or key.startswith("style-0001"))]
            for key in [*kwargs["midi_files"], *kwargs["texts"]]
        }

    monkeypatch.setattr(evaluate, "extract_clamp3_embeddings", embeddings)
    cfg = OmegaConf.create(
        {
            "data": {"name": "xmidi"},
            "task": "genre",
            "experiment": {"name": "test_run"},
            "evaluation": {
                "classifier": {"enabled": False},
                "clamp3": {
                    "enabled": True,
                    "repository": str(tmp_path),
                    "cache_dir": str(tmp_path),
                    "style_template": "This is {style} music.",
                },
            },
            "report": {"experiments": ["test_run"]},
            "seed": 0,
        }
    )
    evaluate.main.__wrapped__(cfg)
    build_report.main.__wrapped__(cfg)
    frame = pd.read_csv(paths["reports_dir"] / "per_transition_results.csv")
    assert frame["clamp3_target_style_success"].iloc[0] == float(valid_generated)
    assert frame["clamp3_source_pseudolabel_success"].iloc[0] == 1.0
    assert frame["clamp3_transport_distance_valid"].iloc[0] == float(valid_generated)
    if valid_generated:
        assert frame["clamp3_transport_distance"].iloc[0] == pytest.approx(2**0.5)
    assert (paths["reports_dir"] / "main_results.md").is_file()
    protocol = json.loads((artifact.parent / "evaluation/evaluation_protocol.json").read_text())
    assert protocol["clamp3_enabled"] is True
    assert protocol["clamp3_model"] == {"model": "clamp3-c2"}


@pytest.mark.parametrize("duplicate", [False, True])
def test_incomplete_or_duplicate_manifest_is_not_reported(tmp_path, monkeypatch, duplicate):
    from cfmusic.commands import evaluate

    root = tmp_path / "run/xmidi"
    root.mkdir(parents=True)
    (root / "counterfactual_metadata.json").write_text("{}")
    (root / "generation_manifest.json").write_text(
        json.dumps(
            {
                "planned_transitions": 2,
                "metadata_files": [
                    "counterfactual_metadata.json",
                    "counterfactual_metadata.json" if duplicate else "missing.json",
                ],
            }
        )
    )
    monkeypatch.setattr(evaluate, "prepare_config", lambda _: {"artifacts_dir": tmp_path})
    cfg = OmegaConf.create(
        {"data": {"name": "xmidi"}, "task": "genre", "experiment": {"name": "run"}}
    )
    with pytest.raises(ValueError, match="partial evaluation"):
        evaluate.main.__wrapped__(cfg)


def _write_midi(path: Path, *, with_note: bool) -> None:
    midi = miditoolkit.MidiFile(ticks_per_beat=480)
    instrument = miditoolkit.Instrument(program=0)
    if with_note:
        instrument.notes.append(miditoolkit.Note(velocity=80, pitch=60, start=0, end=480))
    midi.instruments.append(instrument)
    midi.dump(str(path))


def test_empty_generated_midi_is_recorded_instead_of_raising(tmp_path: Path) -> None:
    source = tmp_path / "source.mid"
    generated = tmp_path / "counterfactual.mid"
    _write_midi(source, with_note=True)
    _write_midi(generated, with_note=False)

    metrics = artifact_midi_validity(source, generated)

    assert metrics["source_midi_valid"] == 1.0
    assert metrics["generated_midi_valid"] == 0.0
    assert "MIDI has no notes" in str(metrics["generated_midi_error"])

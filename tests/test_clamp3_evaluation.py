import json
from pathlib import Path

import miditoolkit
import mido
import numpy as np
import pytest

from cfmusic.evaluation.clamp3 import clamp3_style_metrics, midi_to_mtf, style_prompt


def test_mtf_excludes_text_and_labels(tmp_path: Path) -> None:
    midi = mido.MidiFile()
    track = mido.MidiTrack()
    midi.tracks.append(track)
    for kind in ("text", "track_name", "lyrics", "marker"):
        field = "name" if kind == "track_name" else "text"
        track.append(mido.MetaMessage(kind, **{field: "secret_genre_label"}))
    track.append(mido.Message("note_on", note=60, velocity=90))
    track.append(mido.Message("note_off", note=60, time=480))
    midi.save(tmp_path / "secret_genre_label.mid")
    midi_to_mtf(tmp_path / "secret_genre_label.mid", tmp_path / "input.mtf")
    text = (tmp_path / "input.mtf").read_text()
    assert "secret_genre_label" not in text
    assert "note_on" in text


def test_clamp3_worker_rejects_unversioned_and_changed_caches(tmp_path: Path, monkeypatch) -> None:
    from cfmusic.commands import clamp3_batched_worker as worker

    monkeypatch.setattr(worker, "embedding_protocol", lambda _: {"schema": "clamp3-c2"})
    source = tmp_path / "input.mtf"
    source.write_text("ticks_per_beat 480")
    entries = [{"key": "music", "mtf_path": str(source)}]
    output = tmp_path / "output"
    worker._cache_protocol(entries, tmp_path, output)
    worker._cache_protocol(entries, tmp_path, output)
    source.write_text("ticks_per_beat 960\nnote_on 0 60 80 0")
    with pytest.raises(ValueError, match="Incompatible"):
        worker._cache_protocol(entries, tmp_path, output)
    unversioned = tmp_path / "old-cache"
    unversioned.mkdir()
    np.save(unversioned / "music.npy", np.ones(768))
    with pytest.raises(ValueError, match="Unversioned"):
        worker._cache_protocol(entries, tmp_path, unversioned)


@pytest.mark.parametrize("shape", [(512,), (2, 768)])
def test_clamp3_rejects_wrong_embedding_shapes(tmp_path: Path, shape) -> None:
    from cfmusic.evaluation.clamp3 import _normalized_feature

    path = tmp_path / "embedding.npy"
    np.save(path, np.ones(shape))
    with pytest.raises(ValueError, match="invalid embedding"):
        _normalized_feature(path)


def test_clamp3_rejects_unidentified_model_protocol(tmp_path: Path) -> None:
    from cfmusic.evaluation.clamp3 import read_embedding_protocol

    path = tmp_path / "embedding_protocol.json"
    path.write_text(json.dumps({"embedding_model": {"schema": "old-model"}}))
    with pytest.raises(ValueError, match="Not a versioned"):
        read_embedding_protocol(path)


def test_gpu_worker_respects_amd_allocation(monkeypatch) -> None:
    from cfmusic.evaluation.clamp3 import gpu_worker_environment

    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "3,5")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    env = gpu_worker_environment(1, 2)
    assert env["HIP_VISIBLE_DEVICES"] == env["CUDA_VISIBLE_DEVICES"] == "5"
    with pytest.raises(ValueError, match="allocation"):
        gpu_worker_environment(1, 3)


def test_clamp3_style_prompt_and_similarity_metrics() -> None:
    prompts = [style_prompt("classical"), style_prompt("rock")]
    metrics = clamp3_style_metrics(
        np.array([0.0, 1.0]),
        style_embeddings=[np.array([1.0, 0.0]), np.array([0.0, 1.0])],
        source_style_id=0,
        target_style_id=1,
    )

    assert prompts == [
        "This is a piece of classical music.",
        "This is a piece of rock music.",
    ]
    assert metrics["clamp3_target_style_success"] == 1.0
    assert metrics["clamp3_target_minus_source"] == 1.0


def test_clamp3_logit_bias_changes_only_calibrated_decision() -> None:
    metrics = clamp3_style_metrics(
        np.array([0.6, 0.5]),
        style_embeddings=[np.array([1.0, 0.0]), np.array([0.0, 1.0])],
        source_style_id=0,
        target_style_id=1,
        logit_bias=[0.0, 0.2],
    )

    assert metrics["clamp3_target_similarity"] == 0.5
    assert metrics["clamp3_target_style_success"] == 1.0
    assert np.isclose(metrics["clamp3_calibrated_target_minus_source"], 0.1)


def test_midi_to_mtf_writes_clamp3_performance_format(tmp_path: Path) -> None:
    midi_path = tmp_path / "piece.mid"
    mtf_path = tmp_path / "piece.mtf"
    midi = miditoolkit.MidiFile(ticks_per_beat=480)
    instrument = miditoolkit.Instrument(program=0)
    instrument.notes.append(miditoolkit.Note(velocity=80, pitch=60, start=0, end=480))
    midi.instruments.append(instrument)
    midi.dump(str(midi_path))

    midi_to_mtf(midi_path, mtf_path)
    contents = mtf_path.read_text(encoding="utf-8")
    assert contents.startswith("ticks_per_beat 480\n")
    assert "note_on" in contents

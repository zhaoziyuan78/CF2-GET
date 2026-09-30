"""16-bar data preparation rules, temporal audit merge, and the end-to-end pipeline script."""

import importlib
import json
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import miditoolkit
import numpy as np
import pandas as pd
import pytest
from hydra import compose, initialize_config_dir

from cfmusic.commands.build_report import merge_temporal_audit
from cfmusic.commands.evaluate_temporal import KEYS, temporal_table
from cfmusic.config import CONFIG_DIR

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def prepare(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    return importlib.import_module("prepare_xmidi_16bar")


def midi(notes: list[tuple[int, int, int]], *, drum: bool = False) -> miditoolkit.MidiFile:
    value = miditoolkit.MidiFile(ticks_per_beat=4)
    instrument = miditoolkit.Instrument(program=0, is_drum=drum)
    instrument.notes = [miditoolkit.Note(80, pitch, start, end) for pitch, start, end in notes]
    value.instruments = [instrument]
    return value


def test_quality_windows_count_onsets_sustain_and_silence(prepare) -> None:
    segment = SimpleNamespace(start_bar=0, num_bars=16, index=0)
    # 64 beats; one sustained note per beat for the first 60 beats, then a 4-beat rest.
    dense = midi([(60, 4 * beat, 4 * beat + 4) for beat in range(60)])
    row = prepare.quality_windows(dense, [segment])[0]
    assert row["segment_note_onsets"] == 60 and row["too_few_notes"]
    assert row["active_bars"] == 15 and not row["too_few_active_bars"]
    assert row["maximum_silent_beats"] == 4 and not row["long_silence"]
    # A whole-segment sustained note is active everywhere but has one onset.
    sustained = prepare.quality_windows(midi([(60, 0, 256)]), [segment])[0]
    assert sustained["active_beat_fraction"] == 1.0 and sustained["segment_note_onsets"] == 1
    # Out-of-range pitches are ignored; drums count only their onset beat.
    ignored = prepare.quality_windows(midi([(10, 0, 256)]), [segment])[0]
    assert ignored["active_beat_fraction"] == 0.0 and ignored["long_silence"]
    drums = prepare.quality_windows(midi([(36, 0, 256)], drum=True), [segment])[0]
    assert drums["active_beat_fraction"] == 1 / 64


def test_q30_filter_is_per_genre_and_train_only(prepare) -> None:
    rows = []
    for style in range(6):
        for position in range(10):
            for split in ("train", "test"):
                rows.append(dict(split=split, style_id=style, sample_id=f"{split}{style}{position}",
                                 genre_margin=float(position + style)))
    index = pd.DataFrame(rows)
    filtered, counts = prepare.filter_train(index)
    train = filtered.loc[filtered.split.eq("train")]
    assert filtered.loc[filtered.split.eq("test")].shape[0] == 60  # held-out splits unchanged
    for style in range(6):
        margins = train.loc[train.style_id.eq(style), "genre_margin"]
        assert margins.min() >= np.quantile(np.arange(10) + style, 0.3)
    assert counts.retained_segments.tolist() == [7] * 6


def test_segments_and_labels_use_the_config_layout(prepare, tmp_path: Path) -> None:
    layout = prepare.Layout(tmp_path)
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="config", overrides=[f"paths.data_root={tmp_path}"])
    assert Path(cfg.data.latent_index) == layout.filtered / "index.parquet"
    assert Path(cfg.counterfactual.sampling_weights_file) == layout.work / "source_labels.parquet"


def write_generation(root: Path, rows: list[dict], *, budget: bool = True) -> None:
    files = []
    for row in rows:
        directory = root / f"{row['sample_id']}"
        directory.mkdir(parents=True)
        for name, pitch in (("source.mid", 60), ("counterfactual.mid", 60 + row["shift"])):
            midi([(pitch, 4 * beat, 4 * beat + 4) for beat in range(64)]).dump(str(directory / name))
        metadata = {key: row[key] for key in KEYS}
        if budget:
            metadata["decode_completion"] = {"counterfactual": {"complete": row["shift"] == 0,
                                                                "budget_exhausted": False}}
        (directory / "counterfactual_metadata.json").write_text(json.dumps(metadata))
        files.append(str(directory.relative_to(root) / "counterfactual_metadata.json"))
    (root / "generation_manifest.json").write_text(
        json.dumps(dict(planned_transitions=len(files), metadata_files=files)))


def test_temporal_audit_is_merged_into_the_evaluated_transitions(tmp_path: Path) -> None:
    rows = [dict(dataset="xmidi", sample_id=f"s{i}", segment_id=f"s{i}:bar0:n16",
                 source_style_id=0, target_style_id=1, shift=shift)
            for i, shift in enumerate((0, 1))]
    write_generation(tmp_path, rows)
    table = temporal_table(tmp_path, beats=64)
    assert table.beat_chroma_64beats.round(6).tolist() == [1.0, 0.0]
    assert table.decode_complete.tolist() == [1.0, 0.0]
    path = tmp_path / "evaluation/temporal_results.csv"
    path.parent.mkdir()
    table.to_csv(path, index=False)
    frame = pd.DataFrame([{key: row[key] for key in KEYS} | {"mse": 0.1} for row in rows])
    frame = frame.astype({key: "string" for key in KEYS[:3]})
    merged = merge_temporal_audit(frame, path)
    assert merged.beat_chroma_64beats.round(6).tolist() == [1.0, 0.0]
    with pytest.raises(ValueError, match="does not match"):
        merge_temporal_audit(frame.iloc[:1], path)


def test_pipeline_commands_compose_and_keep_the_published_protocol() -> None:
    output = subprocess.check_output(["bash", "scripts/run_xmidi_16bar.sh", "--dry-run"],
                                     cwd=ROOT, text=True)
    stages = [line[1:-1] for line in output.splitlines() if line.startswith("[") and line.endswith("]")]
    assert stages[:9] == ["download", "prepare", "codec_8bar", "segments", "codec_16bar", "cache",
                          "label_prepare", "label_workers", "label_publish"]
    assert stages[-1] == "report" and len(stages) == 9 + 6 * 5 + 1  # six methods, five stages each
    lines = output.splitlines()
    for name in stages:
        arguments = shlex.split(lines[lines.index(f"[{name}]") + 1])
        if "-m" not in arguments or arguments[arguments.index("-m") + 1] == "cfmusic.commands.download":
            continue
        overrides = arguments[arguments.index("-m") + 2:]
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(config_name="config", overrides=overrides)
        if name.startswith("generate_otcfm"):
            assert cfg.transport.prediction_guidance_scale == 4.0  # target CFG=4
        if name == "codec_16bar":
            assert cfg.codec.training.batch_size * 8 == 64 and cfg.codec.training.max_steps == 1000
        if name == "codec_8bar":
            assert cfg.codec.training.batch_size * 8 == 128
    assert "cfmusic.commands.generate_midi_vae" in output and "flowedit" not in output


def test_prepare_script_help_lists_every_phase() -> None:
    text = subprocess.check_output([sys.executable, "scripts/prepare_xmidi_16bar.py", "--help"],
                                   cwd=ROOT, text=True)
    for phase in ("segments", "label-prepare", "label-worker", "label-publish"):
        assert phase in text

"""Whole-segment temporal content (beat chroma) and decode-completion audit of a generation.

Writes ``<artifacts>/<experiment>/<dataset>/evaluation/temporal_results.csv``; build_report
merges it into the per-transition table. The chroma covers every beat of the segment
(64 beats for 16 bars), unlike the note-endpoint duration, which ignores trailing rests.
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import miditoolkit
import pandas as pd
from omegaconf import DictConfig

from cfmusic.config import CONFIG_DIR, prepare_config
from cfmusic.evaluation.temporal_content import beat_chroma, beat_chroma_cosine
from cfmusic.progress import track

KEYS = ["dataset", "sample_id", "segment_id", "source_style_id", "target_style_id"]


def temporal_table(root: Path, beats: int) -> pd.DataFrame:
    """One row per planned transition of the generation under ``root``."""

    manifest = json.loads((root / "generation_manifest.json").read_text(encoding="utf-8"))
    column = f"beat_chroma_{beats}beats"
    rows = []
    for relative in track(manifest["metadata_files"], description="Temporal content", unit="pair"):
        path = root / relative
        item = json.loads(path.read_text(encoding="utf-8"))
        source = miditoolkit.MidiFile(str(path.parent / "source.mid"))
        try:
            output = miditoolkit.MidiFile(str(path.parent / "counterfactual.mid"))
            chroma = beat_chroma_cosine(
                beat_chroma(source, beats=beats), beat_chroma(output, beats=beats)
            )
        except (OSError, ValueError, EOFError):
            chroma = 0.0  # an unreadable generation keeps its place in the denominator
        row = {key: item[key] for key in KEYS}
        row[column] = chroma
        # Direct piano-roll models have no token EOS; their decode columns stay empty.
        if "decode_completion" in item:
            completion = item["decode_completion"]["counterfactual"]
            row.update(
                decode_complete=float(completion["complete"]),
                decode_budget_exhausted=float(completion["budget_exhausted"]),
            )
        rows.append(row)
    table = pd.DataFrame(rows)
    if len(table) != int(manifest["planned_transitions"]) or table[KEYS].duplicated().any():
        raise ValueError("Temporal audit does not cover the planned transitions exactly once")
    return table


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    paths = prepare_config(cfg)
    root = paths["artifacts_dir"] / str(cfg.experiment.name) / str(cfg.data.name)
    beats = 4 * int(cfg.data.segmentation.num_bars)
    column = f"beat_chroma_{beats}beats"
    table = temporal_table(root, beats)
    output = root / "evaluation" / "temporal_results.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    summary = {column: float(table[column].mean())}
    if "decode_complete" in table:
        summary["decode_complete"] = float(table.decode_complete.mean())
    print(f"Temporal audit: {output}; {json.dumps(summary)}")


if __name__ == "__main__":
    main()

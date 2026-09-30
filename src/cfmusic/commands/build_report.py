"""Aggregate experiment artifacts into reproducible result tables and figures."""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from cfmusic.config import CONFIG_DIR, prepare_config
from cfmusic.progress import track
from cfmusic.reporting.tables import write_result_table


def load_evaluation_results(path: Path) -> pd.DataFrame:
    """Recover legacy dataset identity from the evaluation's own recorded protocol.

    Older direct MIDI-VAE artifacts omitted this field. Never guess it from the
    current Hydra configuration or a directory name, and never rewrite metrics.
    All other cohort keys still have to pass the strict paired validation below.
    """
    frame = pd.read_csv(path, dtype={key: "string" for key in ("dataset", "sample_id", "segment_id")})
    frame.attrs["source_path"] = str(path)
    protocol_path = path.with_name("evaluation_protocol.json")
    if not protocol_path.is_file():
        return frame
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    dataset = protocol.get("dataset") if isinstance(protocol, dict) else None
    if not isinstance(dataset, str) or not dataset.strip():
        raise ValueError(f"Missing dataset identity in evaluation protocol: {protocol_path}")
    if "dataset" not in frame:
        frame["dataset"] = dataset
        print(f"Recovered dataset={dataset!r} from {protocol_path} for {len(frame)} rows")
    else:
        missing = frame["dataset"].isna() | frame["dataset"].str.strip().eq("")
        if (frame.loc[~missing, "dataset"] != dataset).any():
            raise ValueError(f"Dataset identity conflicts with {protocol_path}: {path}")
        if missing.any():
            frame.loc[missing, "dataset"] = dataset
            print(f"Recovered dataset={dataset!r} from {protocol_path} for {int(missing.sum())} rows")
    return frame


def merge_temporal_audit(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Attach cfmusic.commands.evaluate_temporal columns, requiring the same transitions."""

    if not path.is_file():
        return frame
    keys = ["dataset", "sample_id", "segment_id", "source_style_id", "target_style_id"]
    audit = pd.read_csv(path, dtype=dict.fromkeys(keys[:3], "string"))
    merged = frame.merge(audit, on=keys, how="left", validate="one_to_one", indicator=True)
    if len(audit) != len(frame) or not merged["_merge"].eq("both").all():
        raise ValueError(f"Temporal audit does not match the evaluated transitions: {path}")
    merged = merged.drop(columns="_merge")
    merged.attrs.update(frame.attrs)
    return merged


def validate_paired_cohorts(frames: list[pd.DataFrame]) -> None:
    """A fair comparison needs the same segments AND intervention conditions."""
    keys = ["dataset", "sample_id", "segment_id", "source_style_id", "target_style_id"]
    reference = None
    reference_origin = None
    for index, frame in enumerate(frames):
        origin = frame.attrs.get("source_path", f"frame {index}")
        missing = [key for key in keys if key not in frame]
        if missing:
            raise ValueError(f"Incomplete source/target identities in {origin}: missing columns {missing}")
        if frame.empty:
            raise ValueError(f"Empty source/target cohort in {origin}")
        invalid = frame[keys].isna().any(axis=1)
        for key in keys:
            invalid |= frame[key].astype(str).str.strip().eq("")
        if invalid.any():
            raise ValueError(f"Incomplete source/target identities in {origin}: {int(invalid.sum())} null/blank rows")
        duplicate = frame.duplicated(keys, keep=False)
        if duplicate.any():
            raise ValueError(f"Duplicate source/target identities in {origin}: {int(duplicate.sum())} rows")
        current = set(frame[keys].itertuples(index=False, name=None))
        if reference is not None and current != reference:
            raise ValueError(
                "Models did not evaluate the same source/target cohort; refusing the comparison: "
                f"{origin} vs {reference_origin} "
                f"({len(reference - current)} missing, {len(current - reference)} extra identities)"
            )
        reference = current
        reference_origin = origin


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    paths = prepare_config(cfg)
    experiments = [str(value) for value in cfg.report.experiments] or [str(cfg.experiment.name)]
    frames: list[pd.DataFrame] = []
    leakage_rows: list[dict[str, object]] = []
    for experiment in track(
        experiments,
        description="Collect experiment results",
        total=len(experiments),
        unit="experiment",
    ):
        found = False
        for path in (paths["artifacts_dir"] / experiment).rglob("per_transition_results.csv"):
            frame = load_evaluation_results(path)
            if frame.empty:
                raise ValueError(f"Empty evaluation for {experiment}")
            frame = merge_temporal_audit(frame, path.parent / "temporal_results.csv")
            found = True
            frame["experiment"] = experiment
            frames.append(frame)
        if not found:
            raise FileNotFoundError(f"Missing requested evaluation: {experiment}")
        for path in (paths["artifacts_dir"] / experiment).rglob("noise_leakage.json"):
            leakage_rows.append(
                {"experiment": experiment, **json.loads(path.read_text(encoding="utf-8"))}
            )
    if not frames:
        raise FileNotFoundError(f"No evaluated experiment results found for {experiments}")
    if bool(cfg.report.get("require_paired_cohort", False)):
        validate_paired_cohorts(frames)
    all_results = pd.concat(frames, ignore_index=True)
    numeric = [
        column
        for column in all_results.select_dtypes("number").columns
        if not column.endswith("_id") and column not in {"seed", "rank", "global_step"}
    ]
    main_results = all_results.groupby("experiment", as_index=False)[numeric].mean()
    reports = paths["reports_dir"]
    reports.mkdir(parents=True, exist_ok=True)
    counts = all_results.groupby("experiment").size()
    main_results["transitions"] = main_results.experiment.map(counts)
    if {"source_style_id", "target_style_id"}.issubset(all_results):
        pairs = all_results.drop_duplicates(["experiment", "source_style_id", "target_style_id"])
        main_results["directed_pairs"] = main_results.experiment.map(
            pairs.groupby("experiment").size()
        )
        per_pair = all_results.groupby(
            ["experiment", "source_style_id", "target_style_id"], as_index=False
        )[numeric].mean()
        per_pair.to_csv(reports / "per_style_pair_results.csv", index=False)
    write_result_table(main_results, reports / "all_metrics")
    preferred = [
        "experiment",
        "transitions",
        "directed_pairs",
        "generated_midi_valid",
        "clamp3_target_style_success",
        "clamp3_transport_distance",
        "clamp3_transport_distance_valid",
        "classifier_target_style_success",
        "pitch_class_histogram_cosine",
        "melody_contour_cosine",
        "beat_chroma_64beats",
        "generated_duration_beats",
        "decode_complete",
        "mse",
        "inverse_nfe",
        "forward_nfe",
    ]
    write_result_table(
        main_results[[c for c in preferred if c in main_results]], reports / "main_results"
    )
    all_results.to_csv(reports / "per_transition_results.csv", index=False)
    leakage_frame = pd.json_normalize(leakage_rows, sep=".")
    leakage_frame.to_csv(reports / "noise_leakage.csv", index=False)
    if not leakage_frame.empty:
        leakage_columns = [
            c
            for c in leakage_frame
            if c
            in {
                "experiment",
                "applicable",
                "independent_groups",
                "probe_test",
                "chance_balanced_accuracy",
            }
            or c.endswith(".balanced_accuracy")
            or c.endswith(".balanced_accuracy_ci95")
        ]
        write_result_table(leakage_frame[leakage_columns], reports / "leakage_summary")
        main_path = reports / "main_results.md"
        main_path.write_text(
            main_path.read_text(encoding="utf-8")
            + "\n\n[Independent factual leakage and Gaussian controls](leakage_summary.md) · "
            + "[All metrics](all_metrics.csv) · [Per-style-pair metrics](per_style_pair_results.csv)\n",
            encoding="utf-8",
        )
    for name, columns in {
        "inversion_vs_nfe": ["experiment", "inverse_nfe", "mse"],
        "style_content_pareto": [
            "experiment",
            "classifier_target_minus_source",
            "clamp3_target_minus_source",
            "pitch_class_histogram_cosine",
            "melody_contour_cosine",
        ],
        "counterfactual_consistency": ["experiment", "mse", "mae", "cosine"],
        "cross_seed_ambiguity": ["experiment"],
    }.items():
        available = [column for column in columns if column in all_results]
        all_results[available].to_csv(reports / f"{name}.csv", index=False)
    (reports / "dataset_cards").mkdir(parents=True, exist_ok=True)
    for experiment in experiments:
        for card in (paths["artifacts_dir"] / experiment).glob("*/evaluation/dataset_card.json"):
            name = f"{experiment}_{card.parent.parent.name}.json"
            (reports / "dataset_cards" / name).write_text(
                card.read_text(encoding="utf-8"), encoding="utf-8"
            )
    cards = list(paths["processed_dir"].glob("*/dataset_card.json"))
    for card in track(cards, description="Collect dataset cards", total=len(cards), unit="card"):
        (
            reports / "dataset_cards" / card.name.replace("dataset_card", card.parent.name)
        ).write_text(card.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Report written to {reports}")


if __name__ == "__main__":
    main()

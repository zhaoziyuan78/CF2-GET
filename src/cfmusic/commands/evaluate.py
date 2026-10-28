"""Evaluate generated artifacts without paired counterfactual references."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from cfmusic.config import CONFIG_DIR, dataset_asset_path, prepare_config
from cfmusic.data.midi_io import validate_midi
from cfmusic.evaluation.clamp3 import (
    clamp3_style_metrics,
    clamp3_transport_distance,
    embedding_protocol,
    extract_clamp3_embeddings,
    style_prompt,
)
from cfmusic.evaluation.content_preservation import (
    descriptor_preservation,
    midi_quality_metrics,
)
from cfmusic.evaluation.embedding_style import EmbeddingStyleEvaluator
from cfmusic.evaluation.noise_leakage import leakage_metrics, train_temporal_probe
from cfmusic.evaluation.token_style import TokenStyleEvaluator, classifier_style_metrics
from cfmusic.memory import peak_memory_gib, reset_peak_memory
from cfmusic.progress import track


def artifact_midi_validity(source: Path, generated: Path) -> dict[str, object]:
    """Return explicit validity metrics without aborting the evaluation run."""

    source_result = validate_midi(source)
    generated_result = validate_midi(generated)
    metrics: dict[str, object] = {
        "source_midi_valid": float(source_result.valid),
        "generated_midi_valid": float(generated_result.valid),
    }
    if source_result.reason is not None:
        metrics["source_midi_error"] = source_result.reason
    if generated_result.reason is not None:
        metrics["generated_midi_error"] = generated_result.reason
    return metrics


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    paths = prepare_config(cfg)
    data_name = str(cfg.data.name)
    task = str(cfg.data.get("task", cfg.task))
    artifact_root = paths["artifacts_dir"] / str(cfg.experiment.name) / data_name
    generation_manifest = artifact_root / "generation_manifest.json"
    if generation_manifest.exists():
        generation_index = json.loads(generation_manifest.read_text(encoding="utf-8"))
        metadata_files = [artifact_root / str(p) for p in generation_index["metadata_files"]]
        if (
            len(metadata_files) != int(generation_index["planned_transitions"])
            or any(not path.is_file() for path in metadata_files)
            or len(set(metadata_files)) != len(metadata_files)
        ):
            raise ValueError("Incomplete generation manifest; refusing a partial evaluation")
    else:
        metadata_files = sorted(artifact_root.rglob("counterfactual_metadata.json"))
    if not metadata_files:
        raise FileNotFoundError(f"No generated artifacts found under {artifact_root}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reset_peak_memory(device)
    records = [json.loads(path.read_text(encoding="utf-8")) for path in metadata_files]
    clamp_cfg = cfg.evaluation.clamp3
    classifier: TokenStyleEvaluator | None = None
    classifier_checkpoint: Path | None = None
    if bool(cfg.evaluation.get("classifier", {}).get("enabled", False)):
        classifier_checkpoint = (
            Path(str(cfg.evaluator_checkpoint)).expanduser().resolve()
            if cfg.evaluator_checkpoint is not None
            else paths["checkpoints_dir"] / "evaluators" / data_name / task / "transformer.pt"
        )
        card = json.loads(
            dataset_asset_path(cfg, paths, "dataset_card.json").read_text(encoding="utf-8")
        )
        classifier = TokenStyleEvaluator(
            classifier_checkpoint,
            dataset=data_name,
            task=task,
            style_vocabulary=card["style_vocabulary"],
            device=device,
        )
    clamp_embeddings: dict[str, np.ndarray] = {}
    style_embeddings: list[np.ndarray] = []
    style_logit_bias: list[float] | None = None
    clamp_protocol: dict[str, object] | None = None
    source_clamp_keys: list[str | None] = [None] * len(records)
    vae_clamp_keys: list[str | None] = [None] * len(records)
    embedding_classifier = None
    embedding_classifier_cfg = cfg.evaluation.get("embedding_classifier", {})
    if bool(embedding_classifier_cfg.get("enabled", False)) and not bool(clamp_cfg.enabled):
        raise ValueError("Embedding classifier requires CLaMP3 extraction")
    if bool(clamp_cfg.enabled):
        clamp_protocol = embedding_protocol(Path(str(clamp_cfg.repository)).expanduser().resolve())
        card = json.loads(
            dataset_asset_path(cfg, paths, "dataset_card.json").read_text(encoding="utf-8")
        )
        factorial_axes = {
            str(record["factorial_intervention"])
            for record in records
            if record.get("factorial_intervention") is not None
        }
        if len(factorial_axes) > 1:
            raise ValueError("One evaluation run cannot mix factorial intervention axes")
        vocabulary_key = (
            f"{next(iter(factorial_axes))}_vocabulary" if factorial_axes else "style_vocabulary"
        )
        style_labels = [str(value) for value in card[vocabulary_key]]
        if bool(embedding_classifier_cfg.get("enabled", False)):
            if classifier is not None:
                raise ValueError("Select one primary trained evaluator, token or embedding")
            embedding_classifier = EmbeddingStyleEvaluator(
                Path(str(embedding_classifier_cfg.checkpoint)),
                dataset=data_name,
                task=task,
                vocabulary=style_labels,
                embedding_model=clamp_protocol,
            )
        midi_files: dict[str, Path] = {}
        for index, (metadata_path, _metadata) in enumerate(
            zip(metadata_files, records, strict=True)
        ):
            generated = metadata_path.parent / "counterfactual.mid"
            if validate_midi(generated).valid:
                midi_files[f"generated-{index:08d}"] = generated
            source = metadata_path.parent / "source.mid"
            if validate_midi(source).valid:
                source_key = (
                    "source-"
                    + hashlib.sha256(str(_metadata["segment_id"]).encode()).hexdigest()[:16]
                )
                midi_files.setdefault(source_key, source)
                source_clamp_keys[index] = source_key
            vae_reconstruction = metadata_path.parent / "vae_reconstruction.mid"
            if vae_reconstruction.is_file() and validate_midi(vae_reconstruction).valid:
                vae_key = (
                    "vae-reconstruction-"
                    + hashlib.sha256(str(_metadata["segment_id"]).encode()).hexdigest()[:16]
                )
                midi_files.setdefault(vae_key, vae_reconstruction)
                vae_clamp_keys[index] = vae_key
        relabel_config_value = clamp_cfg.get("relabel_config")
        membership = {label: [label] for label in style_labels}
        midi_view_policy = "all"
        if relabel_config_value is not None:
            relabel_config_path = Path(str(relabel_config_value)).expanduser().resolve()
            relabel_config = json.loads(relabel_config_path.read_text(encoding="utf-8"))
            if relabel_config.get("embedding_model") != clamp_protocol:
                raise ValueError(
                    "Relabel prompts/calibration belong to a different embedding model"
                )
            if relabel_config.get("genre_vocabulary") != style_labels:
                raise ValueError("Relabel vocabulary does not match the evaluation dataset card")
            templates = [str(value) for value in relabel_config["prompt_templates"]]
            bias_mapping = relabel_config["prompt_logit_bias"]
            style_logit_bias = [float(bias_mapping[label]) for label in style_labels]
            if relabel_config.get("aggregation") == "max_member_prototypes":
                membership = relabel_config["genre_membership"]
                if set(membership) != set(style_labels) or any(
                    not members for members in membership.values()
                ):
                    raise ValueError("Genre family vocabulary is incomplete")
            if (
                relabel_config.get("feature_protocol")
                == "cfmusic.clamp3-c2.three-512-patch-views.raw-fp32.v1"
            ):
                midi_view_policy = "three"
        else:
            configured_templates = clamp_cfg.get("style_templates")
            templates = (
                [str(value) for value in configured_templates]
                if configured_templates is not None
                else [str(clamp_cfg.style_template)]
            )
        prompt_labels = list(
            dict.fromkeys(member for label in style_labels for member in membership[label])
        )
        prompt_ids = {label: index for index, label in enumerate(prompt_labels)}
        prompts = {
            f"style-{style_id:04d}-{template_id:02d}": style_prompt(
                str(clamp_cfg.get("style_name_mapping", {}).get(label, label)), template
            )
            for style_id, label in enumerate(prompt_labels)
            for template_id, template in enumerate(templates)
        }
        clamp_embeddings = extract_clamp3_embeddings(
            repository=Path(str(clamp_cfg.repository)).expanduser().resolve(),
            midi_files=midi_files,
            texts=prompts,
            python_executable=(
                str(clamp_cfg.python_executable)
                if clamp_cfg.get("python_executable") is not None
                else None
            ),
            cache_dir=Path(str(clamp_cfg.cache_dir)).expanduser().resolve(),
            batch_size=int(clamp_cfg.get("batch_size", 4)),
            device=str(clamp_cfg.get("device", "auto")),
            midi_view_policy=midi_view_policy,
            preserve_metadata_timing=bool(clamp_cfg.get("preserve_metadata_timing", False)),
        )
        style_embeddings = []
        for label in style_labels:
            members = []
            for member in membership[label]:
                style_id = prompt_ids[member]
                embedding = np.stack(
                    [
                        clamp_embeddings[f"style-{style_id:04d}-{template_id:02d}"]
                        for template_id in range(len(templates))
                    ]
                ).mean(axis=0)
                members.append(embedding / np.linalg.norm(embedding))
            style_embeddings.append(np.stack(members))
    rows: list[dict[str, object]] = []
    noise_by_sample: dict[str, tuple[torch.Tensor, int]] = {}
    invalid_generated_midis = 0
    evaluation_progress = track(
        metadata_files,
        description="Evaluate counterfactuals",
        total=len(metadata_files),
        unit="transition",
    )
    for artifact_index, (metadata_path, metadata) in enumerate(
        zip(evaluation_progress, records, strict=True)
    ):
        directory = metadata_path.parent
        source, generated = directory / "source.mid", directory / "counterfactual.mid"
        metrics = artifact_midi_validity(source, generated)
        source_is_valid = bool(metrics["source_midi_valid"])
        generated_is_valid = bool(metrics["generated_midi_valid"])
        if not generated_is_valid:
            invalid_generated_midis += 1
        if generated_is_valid:
            metrics.update(midi_quality_metrics(generated))
        if source_is_valid and generated_is_valid:
            metrics.update(descriptor_preservation(source, generated))
        if classifier is not None:
            num_bars = int(cfg.data.segmentation.num_bars)
            source_id, target_id = (
                int(metadata["source_style_id"]),
                int(metadata["target_style_id"]),
            )
            if generated_is_valid:
                metrics.update(
                    classifier_style_metrics(
                        classifier.probabilities(generated, num_bars=num_bars),
                        source_style_id=source_id,
                        target_style_id=target_id,
                    )
                )
            else:
                # Invalid outputs count as failures, not silently excluded successes.
                metrics["classifier_target_style_success"] = 0.0
            if source_is_valid:
                metrics["classifier_source_accuracy"] = float(
                    int(classifier.probabilities(source, num_bars=num_bars).argmax()) == source_id
                )
            vae_path = directory / "vae_reconstruction.mid"
            if vae_path.is_file():
                metrics["classifier_vae_style_retention"] = (
                    float(
                        int(classifier.probabilities(vae_path, num_bars=num_bars).argmax())
                        == source_id
                    )
                    if validate_midi(vae_path).valid
                    else 0.0
                )
        if generated_is_valid and style_embeddings:
            metrics.update(
                clamp3_style_metrics(
                    clamp_embeddings[f"generated-{artifact_index:08d}"],
                    style_embeddings=style_embeddings,
                    source_style_id=int(metadata["source_style_id"]),
                    target_style_id=int(metadata["target_style_id"]),
                    logit_bias=style_logit_bias,
                )
            )
        elif not generated_is_valid and style_embeddings:
            metrics["clamp3_target_style_success"] = 0.0
        source_embedding_key = source_clamp_keys[artifact_index]
        if style_embeddings:
            metrics["clamp3_transport_distance_valid"] = float(
                generated_is_valid and source_embedding_key is not None
            )
            if generated_is_valid and source_embedding_key is not None:
                metrics["clamp3_transport_distance"] = clamp3_transport_distance(
                    clamp_embeddings[source_embedding_key],
                    clamp_embeddings[f"generated-{artifact_index:08d}"],
                    style_embeddings=style_embeddings,
                    source_style_id=int(metadata["source_style_id"]),
                    target_style_id=int(metadata["target_style_id"]),
                )
        vae_embedding_key = vae_clamp_keys[artifact_index]
        source_style_id = int(metadata["source_style_id"])
        if embedding_classifier is not None:
            if generated_is_valid:
                metrics.update(
                    classifier_style_metrics(
                        embedding_classifier.probabilities(
                            clamp_embeddings[f"generated-{artifact_index:08d}"]
                        ),
                        source_style_id=source_style_id,
                        target_style_id=int(metadata["target_style_id"]),
                    )
                )
            else:
                metrics["classifier_target_style_success"] = 0.0
            if source_embedding_key is not None:
                metrics["classifier_source_accuracy"] = float(
                    embedding_classifier.probabilities(
                        clamp_embeddings[source_embedding_key]
                    ).argmax()
                    == source_style_id
                )
            if (directory / "vae_reconstruction.mid").is_file():
                metrics["classifier_vae_style_retention"] = float(
                    vae_embedding_key is not None
                    and embedding_classifier.probabilities(
                        clamp_embeddings[vae_embedding_key]
                    ).argmax()
                    == source_style_id
                )
        if source_embedding_key is not None and source_embedding_key in clamp_embeddings:
            source_style_metrics = clamp3_style_metrics(
                clamp_embeddings[source_embedding_key],
                style_embeddings=style_embeddings,
                source_style_id=source_style_id,
                target_style_id=source_style_id,
                logit_bias=style_logit_bias,
            )
            metrics["clamp3_source_pseudolabel_success"] = source_style_metrics[
                "clamp3_target_style_success"
            ]
            metrics["clamp3_source_pseudolabel_similarity"] = source_style_metrics[
                "clamp3_target_similarity"
            ]
        if vae_embedding_key is not None and vae_embedding_key in clamp_embeddings:
            vae_style_metrics = clamp3_style_metrics(
                clamp_embeddings[vae_embedding_key],
                style_embeddings=style_embeddings,
                source_style_id=source_style_id,
                target_style_id=source_style_id,
                logit_bias=style_logit_bias,
            )
            metrics["clamp3_vae_reconstruction_pseudolabel_retention"] = vae_style_metrics[
                "clamp3_target_style_success"
            ]
            metrics["clamp3_vae_reconstruction_pseudolabel_similarity"] = vae_style_metrics[
                "clamp3_target_similarity"
            ]
        row: dict[str, object] = {**metadata, **metadata.get("latent_roundtrip", {}), **metrics}
        rows.append(row)
        sample_id = str(metadata["sample_id"])
        if (
            bool(cfg.evaluation.get("artifact_noise_leakage", True))
            and sample_id not in noise_by_sample
            and (directory / "abducted_noise.pt").is_file()
            and metadata.get("representation_kind", "gaussian_noise") == "gaussian_noise"
        ):
            noise = torch.load(
                directory / "abducted_noise.pt", map_location="cpu", weights_only=True
            )[0]
            noise_by_sample[sample_id] = (noise, int(metadata["source_style_id"]))
        (directory / "evaluation_metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        evaluation_progress.set_postfix(
            completed=len(rows), invalid=invalid_generated_midis, refresh=False
        )
    output_dir = artifact_root / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    card_path = dataset_asset_path(cfg, paths, "dataset_card.json")
    (output_dir / "dataset_card.json").write_text(
        card_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (output_dir / "evaluation_protocol.json").write_text(
        json.dumps(
            {
                "dataset": data_name,
                "dataset_card": str(card_path),
                "classifier_enabled": classifier is not None or embedding_classifier is not None,
                "classifier_kind": "clamp3_embedding"
                if embedding_classifier is not None
                else "token"
                if classifier is not None
                else None,
                "classifier_checkpoint": str(classifier_checkpoint)
                if classifier_checkpoint
                else None,
                "clamp3_enabled": bool(clamp_cfg.enabled),
                "clamp3_model": clamp_protocol,
                "clamp3_preserve_metadata_timing": bool(
                    clamp_cfg.get("preserve_metadata_timing", False)
                ),
                "embedding_classifier": embedding_classifier.metadata
                if embedding_classifier is not None
                else None,
                "embedding_classifier_checkpoint": str(embedding_classifier_cfg.get("checkpoint"))
                if embedding_classifier is not None
                else None,
                "embedding_classifier_score_semantics": "uncalibrated softmax of supervised decision scores"
                if embedding_classifier is not None
                else None,
                "transitions": len(rows),
                "split": generation_index.get("split") if generation_manifest.exists() else None,
                "transport_distance": "signed projection onto normalized (target-source) direction; normalized mean-member class vectors",
                "transport_distance_invalid_policy": "missing, with separate valid fraction; invalid outputs count as success failures",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "per_transition_results.csv", index=False)
    numeric = frame.select_dtypes(include=[np.number]).mean().to_dict()
    numeric["invalid_generated_midis"] = invalid_generated_midis
    numeric["peak_gpu_memory_gib"] = peak_memory_gib(device)
    (output_dir / "aggregate.json").write_text(json.dumps(numeric, indent=2), encoding="utf-8")
    values = list(noise_by_sample.values())
    if len(values) >= 8 and len({label for _, label in values}) >= 2:
        noises = torch.stack([noise for noise, _ in values])
        labels = torch.tensor([label for _, label in values])
        minimum_class = min(int((labels == label).sum()) for label in torch.unique(labels))
        if minimum_class >= 2:
            leakage = leakage_metrics(noises, labels, seed=int(cfg.seed))
            leakage.update(
                train_temporal_probe(
                    noises,
                    labels,
                    seed=int(cfg.seed),
                    log_dir=output_dir / "temporal_probe_training",
                )
            )
            (output_dir / "noise_leakage.json").write_text(
                json.dumps(leakage, indent=2), encoding="utf-8"
            )
    print(json.dumps(numeric, indent=2))


if __name__ == "__main__":
    main()

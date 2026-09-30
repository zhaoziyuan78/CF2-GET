"""Large group-disjoint factual-latent leakage audit, without decoding any MIDI."""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from omegaconf import DictConfig
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cfmusic.conditioning.schema import build_condition_batch, validate_condition_checkpoint
from cfmusic.config import CONFIG_DIR, dataset_asset_path, prepare_config
from cfmusic.distributed import cleanup_distributed, distributed_barrier, initialize_distributed
from cfmusic.latent.compatibility import validate_transport_cache_provenance
from cfmusic.latent.dataset import LatentDataset
from cfmusic.memory import autocast_context, sdpa_kernel_context
from cfmusic.progress import track
from cfmusic.training.checkpointing import checkpoint_model_state
from cfmusic.transport.conditional_ddim import ConditionalDDIM
from cfmusic.transport.conditional_flow import ConditionalFlow
from cfmusic.transport.factory import create_transport, validate_guidance_checkpoint


def heldout_probe_audit(
    features: np.ndarray, controls: np.ndarray, labels: np.ndarray, *, seed: int = 0
) -> dict[str, object]:
    """One source group per row; fit scaler/probes on probe-train only."""
    if min(np.bincount(labels)) < 10:
        raise ValueError("Leakage audit requires at least ten independent groups per class")
    indices = np.arange(len(labels))
    train, remaining = train_test_split(indices, test_size=0.4, random_state=seed, stratify=labels)
    validation, test = train_test_split(
        remaining, test_size=0.5, random_state=seed, stratify=labels[remaining]
    )
    results: dict[str, object] = {
        "probe_train": len(train),
        "probe_validation": len(validation),
        "probe_test": len(test),
        "chance_balanced_accuracy": 1 / len(np.unique(labels)),
    }
    for kind, values in (("abducted", features), ("gaussian_control", controls)):
        candidates = []
        for regularization in (0.01, 0.1, 1.0, 10.0):
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=regularization, class_weight="balanced", max_iter=1000, random_state=seed
                ),
            )
            model.fit(values[train], labels[train])
            candidates.append(
                (
                    balanced_accuracy_score(labels[validation], model.predict(values[validation])),
                    regularization,
                    model,
                )
            )
        _, regularization, linear = max(candidates, key=lambda item: item[0])
        mlp = make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(128, 64),
                max_iter=250,
                early_stopping=True,
                validation_fraction=max(0.1, (len(np.unique(labels)) + 1) / len(train)),
                n_iter_no_change=15,
                random_state=seed,
            ),
        )
        mlp.fit(values[train], labels[train])
        for probe, model in (("linear", linear), ("mlp", mlp)):
            prediction = model.predict(values[test])
            balanced = float(balanced_accuracy_score(labels[test], prediction))
            rng = np.random.default_rng(seed)
            by_class = [np.flatnonzero(labels[test] == label) for label in np.unique(labels)]
            bootstrap = [
                np.mean(
                    [
                        np.mean(prediction[take] == labels[test][take])
                        for group in by_class
                        for take in [rng.choice(group, len(group), replace=True)]
                    ]
                )
                for _ in range(500)
            ]
            results[f"{kind}_{probe}"] = {
                "balanced_accuracy": balanced,
                "macro_f1": float(f1_score(labels[test], prediction, average="macro")),
                "balanced_accuracy_ci95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
                "linear_C": regularization if probe == "linear" else None,
            }
    results["interpretation"] = (
        "Held-out source-label predictability, with an independent Gaussian negative control; not causal identification."
    )
    return results


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    context = initialize_distributed()
    try:
        paths = prepare_config(cfg)
        output = (
            paths["artifacts_dir"] / str(cfg.experiment.name) / str(cfg.data.name) / "evaluation"
        )
        output.mkdir(parents=True, exist_ok=True)
        if str(cfg.transport.type) not in {"cfm", "ddim"}:
            if context.is_main:
                (output / "noise_leakage.json").write_text(
                    json.dumps(
                        {
                            "applicable": False,
                            "reason": "This baseline has no shared Gaussian abduction; its factual/style code is not noise.",
                        },
                        indent=2,
                    )
                )
            return
        split = str(cfg.leakage.split)
        if split not in {"validation", "test"}:
            raise ValueError("Leakage factual inputs must use validation or test")
        dataset = LatentDataset(
            paths["latent_dir"] / str(cfg.data.name),
            index_path=Path(str(cfg.data.latent_index)),
            split=split,
        )
        groups = pd.read_parquet(
            dataset_asset_path(cfg, paths, "manifest.parquet"), columns=["sample_id", "group_id"]
        ).drop_duplicates("sample_id")
        eligible = dataset.frame.reset_index(names="latent_row").merge(
            groups, on="sample_id", validate="many_to_one"
        )
        eligible = eligible.drop_duplicates("group_id")
        rng = np.random.default_rng(int(cfg.seed))
        selected = (
            pd.concat(
                [
                    part.iloc[rng.permutation(len(part))[: int(cfg.leakage.samples_per_style)]]
                    for _, part in eligible.groupby("style_id", sort=True)
                ]
            )
            .sort_values("latent_row")
            .reset_index(drop=True)
        )
        if selected.groupby("style_id").size().min() < 10 or selected.style_id.nunique() != int(
            cfg.data.num_genres
        ):
            raise ValueError("Insufficient independent groups for the requested leakage audit")
        payload = torch.load(
            str(cfg.transport_checkpoint), map_location="cpu", weights_only=False, mmap=True
        )
        validate_transport_cache_provenance(
            payload, [dataset.metadata],
            allow_label_overlay_change=bool(
                cfg.counterfactual.get("allow_label_overlay_change", False)
            ),
        )
        validate_condition_checkpoint(payload, task="genre", factorial=False)
        validate_guidance_checkpoint(payload, cfg.transport, exact_training_match=False)
        model = create_transport(cfg.transport).to(context.device)
        model.load_state_dict(
            checkpoint_model_state(payload, weights=str(cfg.counterfactual.transport_weights))
        )
        model.eval()
        if isinstance(model, (ConditionalFlow, ConditionalDDIM)) and model.abduction_guidance_scale != 1:
            raise ValueError("Factual noise must be abducted without CFG")
        dimension = int(cfg.codec.latent_tokens) * int(cfg.codec.latent_dim)
        projection_dim = int(cfg.leakage.projection_dim)
        projection_rng = torch.Generator(device=context.device).manual_seed(int(cfg.seed) + 319)
        projection = (
            torch.randn(dimension, projection_dim, device=context.device, generator=projection_rng)
            / dimension**0.5
        )
        local = selected.iloc[context.rank :: context.world_size]
        features, controls, row_ids = [], [], []
        torch.manual_seed(int(cfg.seed) + context.rank + 419)
        for start in track(
            range(0, len(local), int(cfg.leakage.batch_size)),
            description="Factual noise (no decode)",
            unit="batch",
        ):
            part = local.iloc[start : start + int(cfg.leakage.batch_size)]
            tensors = [dataset[int(index)]["latent"] for index in part.latent_row]
            if not all(isinstance(value, torch.Tensor) for value in tensors):
                raise TypeError("Leakage inputs must be cached latent tensors")
            batch = torch.stack([value for value in tensors if isinstance(value, torch.Tensor)]).to(
                context.device
            )
            labels = torch.tensor(part.style_id.to_numpy(), device=context.device)
            condition = build_condition_batch(
                {"dataset_id": torch.zeros_like(labels), "genre_id": labels},
                context.device,
                task="genre",
            )
            with (
                torch.inference_mode(),
                sdpa_kernel_context(
                    context.device, str(cfg.transport.training.get("sdpa_backend", "math"))
                ),
                autocast_context(context.device, str(cfg.transport.training.precision)),
            ):
                noise = model.abduct(
                    batch, condition, num_steps=int(cfg.transport.solver.num_steps)
                )
            features.append((noise.float().flatten(1) @ projection).cpu().numpy())
            controls.append((torch.randn_like(noise).float().flatten(1) @ projection).cpu().numpy())
            row_ids.extend(part.index.tolist())
        local_result = {
            "rows": row_ids,
            "features": np.concatenate(features) if features else np.empty((0, projection_dim)),
            "controls": np.concatenate(controls) if controls else np.empty((0, projection_dim)),
        }
        gathered: list[object] = [local_result]
        if context.world_size > 1:
            gathered = [None] * context.world_size
            dist.all_gather_object(gathered, local_result)
        if context.is_main:
            complete = np.empty((len(selected), projection_dim), dtype=np.float32)
            gaussian = np.empty_like(complete)
            seen = []
            for rank_result in gathered:
                assert isinstance(rank_result, dict)
                complete[rank_result["rows"]] = rank_result["features"]
                gaussian[rank_result["rows"]] = rank_result["controls"]
                seen.extend(rank_result["rows"])
            if sorted(seen) != list(range(len(selected))):
                raise ValueError("Incomplete distributed leakage extraction")
            summary = heldout_probe_audit(
                complete, gaussian, selected.style_id.to_numpy(), seed=int(cfg.seed)
            )
            summary.update(
                applicable=True,
                factual_split=split,
                independent_groups=len(selected),
                class_counts=selected.style_id.value_counts().sort_index().to_dict(),
                projection_dim=projection_dim,
                no_midi_decoding=True,
                abduction_guidance_scale=1.0,
                checkpoint=str(Path(str(cfg.transport_checkpoint)).resolve()),
            )
            selected[["sample_id", "group_id", "style_id"]].to_parquet(
                output / "leakage_sources.parquet", index=False
            )
            np.savez_compressed(
                output / "leakage_projected_features.npz",
                features=complete,
                gaussian_control=gaussian,
                labels=selected.style_id.to_numpy(),
            )
            (output / "noise_leakage.json").write_text(json.dumps(summary, indent=2))
        distributed_barrier(context)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()

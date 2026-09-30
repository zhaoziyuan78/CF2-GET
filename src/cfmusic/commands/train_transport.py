"""Train shared conditional CFM or DDIM on frozen normalized latents."""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from cfmusic.conditioning.schema import (
    condition_schema_provenance,
    validate_condition_checkpoint,
)
from cfmusic.config import CONFIG_DIR, config_mapping, prepare_config
from cfmusic.data.samplers import (
    BalancedStyleShardBatchSampler,
    ShardBatchSampler,
)
from cfmusic.distributed import (
    DistributedContext,
    cleanup_distributed,
    initialize_distributed,
)
from cfmusic.latent.combined import CombinedLatentDataset
from cfmusic.latent.compatibility import (
    serialize_cache_metadata,
    validate_latent_dataset,
    validate_transport_cache_provenance,
)
from cfmusic.latent.dataset import LatentDataset
from cfmusic.models.probes import DynamicNoiseProjector
from cfmusic.paths import save_run_context
from cfmusic.reproducibility import seed_everything
from cfmusic.training.checkpointing import checkpoint_model_state, resolve_resume_checkpoint
from cfmusic.training.optim import create_adamw
from cfmusic.training.schedules import warmup_cosine_scheduler
from cfmusic.training.transport_trainer import (
    heldout_condition_batch,
    inverse_frequency_weights,
    train_transport_steps,
)
from cfmusic.transport.factory import create_transport, validate_guidance_checkpoint


def _train(cfg: DictConfig, context: DistributedContext) -> None:
    paths = prepare_config(cfg)
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    if context.is_main:
        save_run_context(cfg, run_dir)
    seed_everything(int(cfg.seed))
    data_name = str(cfg.data.name)
    training = cfg.transport.training
    latent_root = paths["latent_dir"] / data_name
    latent_index_value = cfg.data.get("latent_index")
    latent_index = (
        Path(str(latent_index_value)).expanduser().resolve()
        if latent_index_value is not None
        else None
    )
    dataset: Dataset[dict[str, torch.Tensor | str | int]]
    if "datasets" in cfg.data:
        if latent_index is not None:
            raise ValueError("data.latent_index is only supported for a single latent dataset")
        dataset_names = [str(name) for name in cfg.data.datasets]
        combined = CombinedLatentDataset(
            [paths["latent_dir"] / name for name in dataset_names], split="train"
        )
        dataset = combined
        latent_datasets = combined.datasets
    else:
        dataset = LatentDataset(
            latent_root,
            split="train",
            index_path=latent_index,
            shard_cache_size=int(training.get("shard_cache_size", 1)),
        )
        dataset_names = [data_name]
        latent_datasets = [dataset]
    for name, latent_dataset in zip(dataset_names, latent_datasets, strict=True):
        validate_latent_dataset(
            latent_dataset,
            codec_cfg=cfg.codec,
            transport_cfg=cfg.transport,
            dataset_name=name,
        )
    cache_metadata = [latent_dataset.metadata for latent_dataset in latent_datasets]
    factorial = bool(cfg.experiment.get("factorial", False))
    task = str(cfg.data.get("task", cfg.task))
    active_axis = str(cfg.counterfactual.get("factorial_intervention", "genre"))
    if active_axis == "joint":
        active_axis = "genre"
    label_values: dict[str, list[int]] = {}
    label_columns = ("genre_id", "emotion_id") if factorial else ("style_id",)
    for output_column in label_columns:
        source_column = (
            output_column
            if factorial
            else (
                f"{task}_id"
                if all(f"{task}_id" in latent.frame for latent in latent_datasets)
                else "style_id"
            )
        )
        label_values[output_column] = sorted(
            {
                int(value)
                for latent_dataset in latent_datasets
                if source_column in latent_dataset.frame
                for value in latent_dataset.frame[source_column].dropna().astype(int).tolist()
            }
        )
    balance_column = (
        f"{task}_id"
        if not factorial and all(f"{task}_id" in latent.frame for latent in latent_datasets)
        else "style_id"
    )
    all_style_labels = [
        int(value)
        for latent_dataset in latent_datasets
        for value in latent_dataset.frame[balance_column].astype(int).tolist()
    ]
    class_balance_exponent = float(training.get("class_balance_exponent", 0.0))
    style_loss_weights = (
        inverse_frequency_weights(all_style_labels, exponent=class_balance_exponent)
        if class_balance_exponent > 0 and not factorial
        else None
    )
    condition_objective = cfg.transport.get("condition_objective", {})
    condition_contrast_weight = (
        float(condition_objective.get("weight", 0.0))
        if bool(condition_objective.get("enabled", False))
        else 0.0
    )
    roundtrip_config = cfg.transport.get("roundtrip", {})
    roundtrip_weight = (
        float(roundtrip_config.get("weight", 0.0))
        if bool(roundtrip_config.get("enabled", False))
        else 0.0
    )
    if roundtrip_weight > 0 and str(cfg.transport.type) != "cfm":
        raise ValueError("The unified round-trip objective currently requires transport.type=cfm")
    endpoint_config = cfg.transport.get("endpoint_matching", {})
    endpoint_weight = (
        float(endpoint_config.get("weight", 0.0))
        if bool(endpoint_config.get("enabled", False))
        else 0.0
    )
    content_config = cfg.transport.get("content_preservation", {})
    content_weight = float(content_config.get("weight", 0.0)) if content_config.get("enabled", False) else 0.0
    content_means = None
    if content_weight > 0:
        if factorial or task != "genre" or str(cfg.transport.type) != "cfm" or latent_index is None:
            raise ValueError("Content preservation requires a single genre-only CFM label overlay")
        means_path = Path(str(content_config["means_file"])).resolve()
        receipt = json.loads(Path(str(content_config["means_receipt"])).read_text())
        stat = means_path.stat()
        if receipt["means"] != {"path": str(means_path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}:
            raise ValueError("Fixed content means differ from their receipt")
        if receipt["latent_cache_metadata"] != cache_metadata[0]:
            raise ValueError("Fixed content means were estimated from a different latent dataset")
        payload = torch.load(means_path, map_location="cpu", weights_only=True)
        stat = latent_index.stat()
        expected_index = {"path": str(latent_index), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        if (payload["spec"] != receipt["spec"] or payload["spec"]["index"] != expected_index
                or payload["spec"]["split"] != "train" or not payload["spec"]["normalized"]):
            raise ValueError("Content means require matching normalized training-only inputs")
        content_means = payload["means"]
        if (tuple(content_means.shape) != (int(cfg.transport.conditioning.num_styles), int(cfg.codec.latent_tokens), int(cfg.codec.latent_dim))
                or content_means.requires_grad or not torch.isfinite(content_means).all()):
            raise ValueError("Invalid fixed content means")
    exogeneity_config = cfg.transport.get("exogeneity", {})
    exogeneity_weights = {
        "hsic": float(exogeneity_config.get("hsic_weight", 0.0)),
        "prior": float(exogeneity_config.get("prior_weight", 0.0)),
        "cross_mmd": float(exogeneity_config.get("cross_class_mmd_weight", 0.0)),
        "cross_swd": float(exogeneity_config.get("cross_class_swd_weight", 0.0)),
    }
    exogeneity_enabled = bool(exogeneity_config.get("enabled", False)) and any(
        value > 0 for value in exogeneity_weights.values()
    )
    if not exogeneity_enabled:
        exogeneity_weights = {name: 0.0 for name in exogeneity_weights}
    noise_projector = (
        DynamicNoiseProjector(
            int(cfg.codec.latent_tokens) * int(cfg.codec.latent_dim),
            int(exogeneity_config.get("projection_dim", 128)),
            num_views=int(exogeneity_config.get("projection_views", 3)),
            seed=int(exogeneity_config.get("projection_seed", 2026)),
            refresh_interval=int(exogeneity_config.get("projection_refresh_interval", 1)),
            block_tokens=int(exogeneity_config.get("block_tokens", 8)),
            block_channels=int(exogeneity_config.get("block_channels", 32)),
            force_fp32=bool(exogeneity_config.get("projection_fp32", False)),
        )
        if exogeneity_enabled
        else None
    )
    checkpoint_subdir = str(cfg.experiment.get("checkpoint_subdir", "transport_stage1"))
    if Path(checkpoint_subdir).name != checkpoint_subdir or checkpoint_subdir in {"", ".", ".."}:
        raise ValueError(f"Invalid transport checkpoint subdirectory: {checkpoint_subdir!r}")
    checkpoint_dir = paths["checkpoints_dir"] / str(cfg.experiment.name) / checkpoint_subdir
    validation_batch: dict[str, torch.Tensor] | None = None
    if context.is_main and isinstance(dataset, LatentDataset):
        validation_dataset = LatentDataset(
            latent_root,
            split="validation",
            index_path=latent_index,
            shard_cache_size=max(1, len(label_values.get("style_id", []))),
        )
        validate_latent_dataset(
            validation_dataset,
            codec_cfg=cfg.codec,
            transport_cfg=cfg.transport,
            dataset_name=data_name,
        )
        validation_batch = heldout_condition_batch(
            validation_dataset,
            samples_per_style=int(training.get("validation_samples_per_style", 16)),
            task=task,
            factorial=factorial,
            active_axis=active_axis,
        )
    if bool(cfg.experiment.get("shuffled_labels", False)) and isinstance(dataset, LatentDataset):
        generator = torch.Generator().manual_seed(int(cfg.seed))
        original = torch.as_tensor(dataset.frame[balance_column].astype(int).to_numpy())
        shuffled = original[torch.randperm(len(original), generator=generator)]
        dataset.frame[balance_column] = shuffled.numpy()
        if balance_column != "style_id":
            dataset.frame["style_id"] = shuffled.numpy()
        dataset.refresh_index()
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if context.is_main:
            (checkpoint_dir / "label_shuffle.json").write_text(
                json.dumps({"seed": int(cfg.seed), "permutation": shuffled.tolist()}),
                encoding="utf-8",
            )
    shard_ids = getattr(dataset, "shard_ids", None)
    if not isinstance(shard_ids, list):
        raise TypeError("Latent datasets must expose shard_ids for locality-aware training")
    sampling = cfg.transport.get("sampling", {})
    batch_sampler: BalancedStyleShardBatchSampler | ShardBatchSampler
    if bool(sampling.get("balance_by_style", False)):
        if not isinstance(dataset, LatentDataset):
            raise ValueError("Style-balanced shard sampling requires one latent dataset")
        if not bool(sampling.get("unique_song_per_batch", True)):
            raise ValueError("Balanced shard sampling requires unique_song_per_batch=true")
        confidence_weighted = bool(sampling.get("confidence_weighted", False))
        if confidence_weighted and "label_confidence_weight" not in dataset.frame:
            raise ValueError(
                "confidence_weighted=true requires label_confidence_weight in the latent overlay"
            )
        if not confidence_weighted:
            dataset._sample_weights = None
        classes_per_batch = int(sampling.get("classes_per_batch", len(label_values["style_id"])))
        samples_per_class = int(sampling.get("samples_per_class", 64))
        configured_batch = int(training.batch_size)
        if classes_per_batch * samples_per_class != configured_batch:
            raise ValueError(
                "Balanced sampler classes_per_batch * samples_per_class must equal "
                f"training.batch_size ({configured_batch})"
            )
        batch_sampler = BalancedStyleShardBatchSampler(
            dataset.frame[balance_column].astype(int).tolist(),
            shard_ids,
            dataset.frame["sample_id"].astype(str).tolist(),
            classes_per_batch=classes_per_batch,
            samples_per_class=samples_per_class,
            rank=context.rank,
            world_size=context.world_size,
            seed=int(cfg.seed),
        )
    else:
        batch_sampler = ShardBatchSampler(
            shard_ids,
            batch_size=int(training.batch_size),
            sample_ids=(
                dataset.frame["sample_id"].astype(str).tolist()
                if isinstance(dataset, LatentDataset)
                else None
            ),
            rank=context.rank,
            world_size=context.world_size,
            seed=int(cfg.seed),
        )
    workers = int(training.get("dataloader_workers", 2))
    loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=workers,
        pin_memory=context.device.type == "cuda",
        persistent_workers=workers > 0,
        prefetch_factor=int(training.get("prefetch_factor", 2)) if workers > 0 else None,
    )
    gradient_accumulation = int(training.gradient_accumulation)
    optimizer_steps_per_epoch = (len(loader) + gradient_accumulation - 1) // gradient_accumulation
    max_steps = int(training.max_steps)
    if training.get("max_epochs") is not None:
        epoch_limited_steps = int(training.max_epochs) * optimizer_steps_per_epoch
        max_steps = min(
            max_steps,
            max(int(training.get("min_steps", 1)), epoch_limited_steps),
        )
    device = context.device
    transport = create_transport(cfg.transport).to(device)
    optimizer = create_adamw(
        transport,
        learning_rate=float(training.learning_rate),
        weight_decay=float(training.weight_decay),
    )
    scheduler = warmup_cosine_scheduler(
        optimizer, warmup_steps=min(int(training.warmup_steps), max_steps), max_steps=max_steps
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    explicit_resume = Path(str(cfg.resume_from)) if cfg.resume_from is not None else None
    resume_checkpoint = resolve_resume_checkpoint(
        checkpoint_dir,
        resume=bool(cfg.get("resume", False)),
        resume_from=explicit_resume,
        announce=context.is_main,
    )
    if resume_checkpoint is not None:
        resumed = torch.load(resume_checkpoint, map_location="cpu", weights_only=False, mmap=True)
        if not isinstance(resumed, dict):
            raise TypeError("Transport checkpoint must contain a mapping")
        validate_transport_cache_provenance(resumed, cache_metadata)
        validate_condition_checkpoint(resumed, task=task, factorial=factorial)
        validate_guidance_checkpoint(resumed, cfg.transport, exact_training_match=True)
        del resumed
    initialization_provenance = {}
    initial_value = cfg.get("initialize_from")
    if initial_value is not None:
        initial_path = Path(str(initial_value)).resolve()
        initial_weights = str(cfg.get("initialize_weights", "ema"))
        stat = initial_path.stat()
        initialization_provenance = {
            "initialization_checkpoint": str(initial_path), "initialization_weights": initial_weights,
            "initialization_checkpoint_bytes": str(stat.st_size),
            "initialization_checkpoint_mtime_ns": str(stat.st_mtime_ns),
        }
        if resume_checkpoint is None:
            initialized = torch.load(initial_path, map_location="cpu", weights_only=False, mmap=True)
            validate_transport_cache_provenance(initialized, cache_metadata)
            validate_condition_checkpoint(initialized, task=task, factorial=factorial)
            validate_guidance_checkpoint(initialized, cfg.transport, exact_training_match=False)
            transport.load_state_dict(checkpoint_model_state(initialized, weights=initial_weights), strict=True)
            del initialized
            if context.is_main:
                print(f"Weights-only initialization from {initial_path} ({initial_weights}); fresh optimizer/schedule/EMA")
    if context.is_main:
        global_batch = (
            int(training.batch_size) * int(training.gradient_accumulation) * context.world_size
        )
        print(
            f"Transport training plan: {len(dataset)} samples, {len(loader)} "
            f"micro-batches/rank/epoch, world_size={context.world_size}, "
            f"global_effective_batch={global_batch}, optimizer_steps={max_steps} "
            f"(hard_cap={int(training.max_steps)}), "
            f"sdpa_backend={training.get('sdpa_backend', 'math')!s}"
        )
        if style_loss_weights is not None:
            formatted = ", ".join(
                f"{style}:{weight:.3f}" for style, weight in sorted(style_loss_weights.items())
            )
            print(f"Transport class-balanced loss weights (mean=1): {formatted}")
        if bool(sampling.get("balance_by_style", False)):
            print(
                "Transport batches: style-balanced, unique-song, shard-local; "
                f"classes={int(sampling.get('classes_per_batch', 0))}, "
                f"samples/class={int(sampling.get('samples_per_class', 0))}, "
                f"confidence_weighted={bool(sampling.get('confidence_weighted', False))}"
            )
        if bool(cfg.transport.get("classifier_free_guidance", False)):
            legacy_scale = float(cfg.transport.get("guidance_scale", 1.0))
            print(
                "Classifier-free guidance: "
                f"condition_dropout={float(cfg.transport.get('condition_dropout', 0.0)):g}, "
                f"abduction={float(cfg.transport.get('abduction_guidance_scale', legacy_scale)):g}, "
                f"reconstruction={float(cfg.transport.get('reconstruction_guidance_scale', legacy_scale)):g}, "
                f"prediction={float(cfg.transport.get('prediction_guidance_scale', legacy_scale)):g}, "
                f"source_repulsion={float(cfg.transport.get('source_repulsion_scale', 0.0)):g}"
            )
        if content_weight > 0:
            print(
                f"Cross-style centered content: weight={content_weight:g}, "
                f"warmup={content_config.get('warmup_steps', 0)}, ramp={content_config.get('ramp_steps', 0)}, "
                f"interval={content_config.get('interval', 8)}, offset={content_config.get('offset', 2)}, "
                f"steps={content_config.get('solver_steps', 4)}, target_CFG={content_config.get('guidance_scale', 4)}, "
                f"samples/class/rank={content_config.get('samples_per_style', 1)}; fixed train-only means"
            )
        if roundtrip_weight > 0:
            print(
                "Unified CFM round-trip objective: "
                f"weight={roundtrip_weight:g}, "
                f"warmup={int(roundtrip_config.get('warmup_steps', 0))}, "
                f"ramp={int(roundtrip_config.get('ramp_steps', 0))}, "
                f"interval={int(roundtrip_config.get('interval', 1))}, "
                f"solver_steps={int(roundtrip_config.get('inverse_steps', 2))}, "
                f"samples/rank={int(roundtrip_config.get('samples_per_batch', 0)) or 'all'}"
            )
        if endpoint_weight > 0:
            print(
                "Sparse endpoint matching: "
                f"weight={endpoint_weight:g}, "
                f"interval={int(endpoint_config.get('interval', 1))}, "
                f"offset={int(endpoint_config.get('offset', 0))}, "
                f"steps={int(endpoint_config.get('solver_steps', 4))}, "
                f"samples/style/rank={int(endpoint_config.get('samples_per_style', 4))}"
            )
        if exogeneity_enabled:
            print(
                "Sparse unguided exogeneity: "
                f"weights={exogeneity_weights}, "
                f"interval={int(exogeneity_config.get('interval', 8))}, "
                f"offset={int(exogeneity_config.get('offset', 0))}, "
                f"steps={int(exogeneity_config.get('inverse_steps', 4))}"
            )
        if condition_contrast_weight > 0:
            print(
                "Transport condition contrast: "
                f"weight={condition_contrast_weight:g}, "
                f"margin={float(condition_objective.get('margin', 0.0)):g}, "
                f"samples/rank={int(condition_objective.get('samples_per_batch', 0))}"
            )
            if validation_batch is not None:
                print(
                    f"Held-out condition probe: {validation_batch['latent'].shape[0]} "
                    f"balanced validation latents, evaluated every {int(training.checkpoint_interval)} steps"
                )
    train_transport_steps(
        transport,
        loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        max_steps=max_steps,
        gradient_accumulation=gradient_accumulation,
        gradient_clip_norm=float(training.gradient_clip_norm),
        precision=str(training.precision),
        sdpa_backend=str(training.get("sdpa_backend", "math")),
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=int(training.checkpoint_interval),
        config=config_mapping(cfg.transport),
        provenance={
            "latent_cache_metadata_json": serialize_cache_metadata(cache_metadata),
            **condition_schema_provenance(task=task, factorial=factorial),
            **initialization_provenance,
        },
        factorial_conditioning=factorial,
        condition_task=task,
        factorial_active_axis=active_axis,
        condition_contrast_weight=condition_contrast_weight,
        condition_contrast_margin=float(condition_objective.get("margin", 0.0)),
        condition_contrast_samples=int(condition_objective.get("samples_per_batch", 0)) or None,
        condition_vocabularies=label_values,
        style_loss_weights=style_loss_weights,
        validation_batch=validation_batch,
        validation_seed=int(cfg.seed) + 2026,
        validation_solver_steps=int(training.get("validation_solver_steps", 8)),
        roundtrip_weight=roundtrip_weight,
        roundtrip_warmup_steps=int(roundtrip_config.get("warmup_steps", 0)),
        roundtrip_ramp_steps=int(roundtrip_config.get("ramp_steps", 0)),
        roundtrip_interval=int(roundtrip_config.get("interval", 1)),
        roundtrip_offset=int(roundtrip_config.get("offset", 0)),
        roundtrip_inverse_steps=int(roundtrip_config.get("inverse_steps", 2)),
        roundtrip_samples_per_batch=(int(roundtrip_config.get("samples_per_batch", 0)) or None),
        roundtrip_cosine_weight=float(roundtrip_config.get("cosine_weight", 0.1)),
        endpoint_weight=endpoint_weight,
        endpoint_warmup_steps=int(endpoint_config.get("warmup_steps", 0)),
        endpoint_ramp_steps=int(endpoint_config.get("ramp_steps", 0)),
        endpoint_interval=int(endpoint_config.get("interval", 1)),
        endpoint_offset=int(endpoint_config.get("offset", 0)),
        endpoint_solver_steps=int(endpoint_config.get("solver_steps", 4)),
        endpoint_samples_per_style=int(endpoint_config.get("samples_per_style", 4)),
        exogeneity_hsic_weight=exogeneity_weights["hsic"],
        exogeneity_prior_weight=exogeneity_weights["prior"],
        exogeneity_cross_mmd_weight=exogeneity_weights["cross_mmd"],
        exogeneity_cross_swd_weight=exogeneity_weights["cross_swd"],
        exogeneity_warmup_steps=int(exogeneity_config.get("warmup_steps", 0)),
        exogeneity_ramp_steps=int(exogeneity_config.get("ramp_steps", 0)),
        exogeneity_interval=int(exogeneity_config.get("interval", 8)),
        exogeneity_offset=int(exogeneity_config.get("offset", 0)),
        exogeneity_inverse_steps=int(exogeneity_config.get("inverse_steps", 4)),
        exogeneity_samples_per_style=int(exogeneity_config.get("samples_per_style", 4)),
        noise_projector=noise_projector,
        content_means=content_means,
        content_weight=content_weight,
        content_warmup_steps=int(content_config.get("warmup_steps", 0)),
        content_ramp_steps=int(content_config.get("ramp_steps", 0)),
        content_interval=int(content_config.get("interval", 8)),
        content_offset=int(content_config.get("offset", 2)),
        content_solver_steps=int(content_config.get("solver_steps", 4)),
        content_samples_per_style=int(content_config.get("samples_per_style", 1)),
        content_guidance_scale=float(content_config.get("guidance_scale", 4.0)),
        ema_decay=float(training.get("ema_decay", 0.9999)),
        ema_update_interval=int(training.get("ema_update_interval", 10)),
        log_interval=int(training.get("log_interval", 10)),
        find_unused_parameters=bool(cfg.transport.get("independent_per_style", False)),
        resume_from=resume_checkpoint,
        distributed=context,
    )


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    context = initialize_distributed()
    try:
        _train(cfg, context)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()

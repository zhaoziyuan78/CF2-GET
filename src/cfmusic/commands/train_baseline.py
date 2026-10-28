"""Train the MIDI-VAE or Latent-MIDI-VAE baseline with epoch-boundary early stopping.

Each epoch ends with a validation pass; training stops after ``patience`` epochs without
improvement (after the minimum epoch/step budget) and keeps the best checkpoint.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import hydra
import torch
import torch.distributed as dist
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import DataLoader

from cfmusic.baselines.factory import baseline_model, roll_dataset
from cfmusic.baselines.midi_vae import MIDIVAE
from cfmusic.config import CONFIG_DIR, config_mapping, prepare_config, seed_everything
from cfmusic.data.samplers import (
    ResumeBatchSampler,
    ShardBatchSampler,
)
from cfmusic.distributed import (
    cleanup_distributed,
    decorrelate_worker_rng,
    distributed_barrier,
    distributed_model,
    initialize_distributed,
    maybe_no_sync,
    set_data_epoch,
)
from cfmusic.latent.dataset import LatentDataset
from cfmusic.logging import MetricLogger
from cfmusic.memory import autocast_context, peak_memory_gib
from cfmusic.progress import progress_bar
from cfmusic.training.checkpointing import (
    load_checkpoint,
    resolve_resume_checkpoint,
    save_checkpoint,
)
from cfmusic.training.epoch_early_stopping import EpochEarlyStopping
from cfmusic.training.schedules import warmup_cosine_scheduler
from cfmusic.training.state import TrainState


def runtime_accumulation(training: Any, runtime: Any, world_size: int) -> int:
    """Keep the reference global batch when explicitly migrating to fewer GPUs."""
    reference = int(runtime.get("baseline_reference_world_size", world_size))
    numerator = int(training.gradient_accumulation) * reference
    if reference <= 0 or world_size <= 0 or numerator % world_size:
        raise ValueError("Reference global batch must divide the actual world size exactly")
    return numerator // world_size


def restore_data_cursor(state: TrainState, extra: dict, world_size: int) -> None:
    # Epoch-end checkpoints are saved before the loop increments the epoch.
    # Do not repeat that epoch (and its early-stopping observation) on a new node.
    if extra.get("epoch_complete", False):
        state.epoch += 1
        state.batch_in_epoch = 0
    elif state.world_size != world_size:
        state.batch_in_epoch = 0
    state.world_size = world_size


def _train(cfg: DictConfig) -> None:
    context = initialize_distributed()
    try:
        paths = prepare_config(cfg)
        seed_everything(int(cfg.seed))
        training = cfg.transport.training
        if str(cfg.transport.type) not in ("midi_vae", "latent_midi_vae"):
            raise ValueError("Epoch early stopping only supports the two VAE baselines")
        stop_cfg = training.early_stopping
        stop = EpochEarlyStopping(**config_mapping(stop_cfg))
        if stop.min_steps < int(training.warmup_steps):
            raise ValueError("Early stopping must not begin during learning-rate warmup")
        checkpoint_dir = paths["checkpoints_dir"] / str(cfg.experiment.name) / "transport"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        index = Path(str(cfg.data.latent_index))
        train_latents = LatentDataset(
            paths["latent_dir"] / str(cfg.data.name),
            index_path=index,
            split="train",
            shard_cache_size=int(training.get("shard_cache_size", 4)),
        )
        model = baseline_model(cfg.transport).to(context.device)
        direct = isinstance(model, MIDIVAE)
        dataset: Any = roll_dataset(train_latents, cfg, paths) if direct else train_latents
        sampler = ShardBatchSampler(
            train_latents.shard_ids,
            batch_size=int(training.batch_size),
            sample_ids=train_latents.frame.sample_id.astype(str).tolist(),
            rank=context.rank,
            world_size=context.world_size,
            seed=int(cfg.seed),
        )
        workers = int(training.dataloader_workers)
        resume_sampler = ResumeBatchSampler(sampler)
        loader = DataLoader(
            dataset,
            batch_sampler=resume_sampler,
            num_workers=workers,
            pin_memory=context.device.type == "cuda",
            persistent_workers=workers > 0,
        )
        generator_parameters = list(model.parameters())
        optimizer = torch.optim.AdamW(
            generator_parameters,
            lr=float(training.learning_rate),
            weight_decay=float(training.weight_decay),
        )
        max_steps = int(training.max_steps)
        accumulation = runtime_accumulation(training, cfg.get("runtime", {}), context.world_size)
        if max_steps <= 0 or accumulation <= 0:
            raise ValueError("Training budget/accumulation must be positive")
        scheduler = warmup_cosine_scheduler(
            optimizer, warmup_steps=min(int(training.warmup_steps), max_steps), max_steps=max_steps
        )
        scaler = torch.GradScaler(
            "cuda", enabled=context.device.type == "cuda" and str(training.precision) == "fp16"
        )
        state = TrainState()
        resume = resolve_resume_checkpoint(
            checkpoint_dir,
            resume=bool(cfg.resume),
            resume_from=Path(str(cfg.resume_from)) if cfg.resume_from is not None else None,
            announce=context.is_main,
        )
        best = float("inf")
        saved_extra = {}
        if resume:
            saved = torch.load(resume, map_location="cpu", weights_only=False, mmap=True)
            state = load_checkpoint(
                resume, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler
            )
            best = float(saved.get("extra_state", {}).get("best_validation", best))
            stop.load_state_dict(saved["extra_state"]["early_stopping"])
            saved_extra = saved["extra_state"]
            previous_runtime = saved_extra.get("runtime", {})
            effective_batch = int(training.batch_size) * accumulation * context.world_size
            if previous_runtime and previous_runtime["effective_global_batch"] != effective_batch:
                raise ValueError("Effective global batch changed on migrated baseline resume")
            del saved
        if state.world_size != context.world_size and context.is_main and resume:
            print(
                "World size changed; advancing a completed epoch, or restarting only "
                "the incomplete epoch's data cursor.",
                flush=True,
            )
        restore_data_cursor(state, saved_extra, context.world_size)
        runtime = dict(
            world_size=context.world_size,
            per_rank_batch=int(training.batch_size),
            gradient_accumulation=accumulation,
            effective_global_batch=int(training.batch_size) * accumulation * context.world_size,
        )
        if context.is_main:
            print(f"Baseline runtime batching: {runtime}", flush=True)
        if resume and context.is_main:
            print(
                f"Restored training state: step={state.global_step}, epoch={state.epoch}, "
                f"batch_in_epoch={state.batch_in_epoch}, world_size={state.world_size}",
                flush=True,
            )
        wrapped = distributed_model(model, context)
        decorrelate_worker_rng(context)
        validation_loader = None
        if context.is_main:
            validation = LatentDataset(
                paths["latent_dir"] / str(cfg.data.name), index_path=index, split="validation"
            )
            # The preprocessed split is group-disjoint from training; keep one segment per song.
            selected = (
                validation.frame.drop_duplicates("sample_id")
                .groupby("style_id", sort=True)
                .head(int(training.get("validation_samples_per_style", 16)))
            )
            validation.frame = selected.reset_index(drop=True)
            validation.refresh_index()
            validation_data = roll_dataset(validation, cfg, paths) if direct else validation
            validation_loader = DataLoader(
                validation_data, batch_size=min(int(training.batch_size), 32), num_workers=0
            )
        logger = (
            MetricLogger(checkpoint_dir, append=resume is not None) if context.is_main else None
        )
        progress = progress_bar(
            description=f"Train {cfg.transport.name}",
            total=max_steps,
            initial=state.global_step,
            unit="step",
        )
        def move(batch: dict[str, Any]) -> dict[str, Tensor]:
            return {
                k: v.to(context.device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }

        def losses(
            module: Any, batch: dict[str, Tensor], *, phase: str = "generator"
        ) -> dict[str, Tensor]:
            if direct:
                return module(batch)
            return module(batch["latent"], batch["style_id"].long())

        optimizer.zero_grad(set_to_none=True)
        while state.global_step < max_steps and not stop.stopped:
            set_data_epoch(loader, state.epoch)
            # Keep absolute epoch indices: the shortened loader length must not
            # alter accumulation boundaries or the next checkpoint's cursor.
            epoch_batches = len(sampler)
            if epoch_batches == 0:
                raise RuntimeError("Empty baseline data loader")
            start_batch = state.batch_in_epoch
            resume_sampler.set_start_batch(start_batch)
            if start_batch == epoch_batches:
                state.epoch += 1
                state.batch_in_epoch = 0
                continue
            if context.is_main:
                print(
                    f"Epoch {state.epoch}: skipping {start_batch}/{epoch_batches} completed "
                    f"batches at the index level (no dataset reads); waiting for first "
                    f"unseen batch with {workers} loader workers per rank.",
                    flush=True,
                )
            saw_batch = False
            for batch_index, raw in enumerate(loader, start=start_batch):
                if not saw_batch and context.is_main:
                    print(
                        f"First unseen batch ready: epoch={state.epoch}, "
                        f"batch={batch_index + 1}/{epoch_batches}; "
                        f"continuing from optimizer step {state.global_step}.",
                        flush=True,
                    )
                saw_batch = True
                batch = move(raw)
                synchronize = (
                    batch_index + 1
                ) % accumulation == 0 or batch_index + 1 == epoch_batches
                actual_accumulation = min(
                    accumulation, epoch_batches - (batch_index // accumulation) * accumulation
                )
                model.train()
                with (
                    maybe_no_sync(wrapped, synchronize=synchronize),
                    autocast_context(context.device, str(training.precision)),
                ):
                    report = losses(wrapped, batch)
                    scaler.scale(report["loss"] / actual_accumulation).backward()
                if not synchronize:
                    continue
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(
                    generator_parameters, float(training.gradient_clip_norm)
                )
                if not torch.isfinite(norm):
                    raise FloatingPointError("Non-finite baseline gradients")
                scaler.step(optimizer)
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                scheduler.step()
                state.global_step += 1
                state.batch_in_epoch = batch_index + 1
                progress.update(1)
                if logger and (
                    state.global_step == 1 or state.global_step % int(training.log_interval) == 0
                ):
                    logger.log(
                        {
                            "step": state.global_step,
                            **{k: float(v.detach()) for k, v in report.items()},
                            "peak_gpu_memory_gib": peak_memory_gib(context.device),
                            "generator_learning_rate": optimizer.param_groups[0]["lr"],
                            "generator_gradient_norm": float(norm),
                        }
                    )
                epoch_end = batch_index + 1 == epoch_batches
                if (
                    state.global_step % int(training.checkpoint_interval) == 0
                    or state.global_step == max_steps
                    or epoch_end
                ):
                    distributed_barrier(context)
                    validation_loss = 0.0
                    validation_totals = {}
                    count = 0
                    if context.is_main:
                        model.eval()
                        total, count = 0.0, 0
                        validation_totals: dict[str, float] = {}
                        # Keep validation RNG isolated from the training stream.
                        devices = [context.local_rank] if context.device.type == "cuda" else []
                        with torch.random.fork_rng(devices=devices), torch.inference_mode():
                            torch.manual_seed(int(cfg.seed) + 913)
                            assert validation_loader is not None
                            for raw_validation in validation_loader:
                                validation_batch = move(raw_validation)
                                with autocast_context(context.device, str(training.precision)):
                                    values = losses(model, validation_batch)
                                    value = values["loss"]
                                size = len(validation_batch["style_id"])
                                total += float(value) * size
                                count += size
                                for key, metric in values.items():
                                    validation_totals[key] = (
                                        validation_totals.get(key, 0.0) + float(metric) * size
                                    )
                        validation_loss = total / max(1, count)
                        if count == 0 or not torch.isfinite(torch.tensor(validation_loss)):
                            raise FloatingPointError("Empty or non-finite baseline validation")
                    # Every rank observes the same complete validation loss. No rank
                    # exits alone while peers are waiting in backward or a barrier.
                    shared_loss = torch.tensor(
                        validation_loss, dtype=torch.float64, device=context.device
                    )
                    if context.world_size > 1:
                        dist.broadcast(shared_loss, src=0)
                    validation_loss = float(shared_loss)
                    if not math.isfinite(validation_loss):
                        raise FloatingPointError("Non-finite broadcast validation loss")
                    if epoch_end:
                        stop.observe(
                            validation_loss,
                            completed_epochs=state.epoch + 1,
                            global_step=state.global_step,
                        )
                    improved = validation_loss < best
                    best = min(best, validation_loss)
                    if context.is_main:
                        extra = {
                            "runtime": runtime,
                            "early_stopping": stop.state_dict(),
                            "stop_reason": "validation_plateau"
                            if stop.stopped
                            else ("max_steps" if state.global_step == max_steps else None),
                            "epoch_complete": epoch_end,
                            "best_validation": best,
                            "validation_loss": validation_loss,
                            "checkpoint_selection": "validation_loss",
                        }
                        # Commit selected weights before the resume cursor that refers to them.
                        for filename in ["best.pt", "last.pt"] if improved else ["last.pt"]:
                            save_checkpoint(
                                checkpoint_dir / filename,
                                model=model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                scaler=scaler,
                                state=state,
                                ema=None,
                                config=config_mapping(cfg.transport),
                                extra_state=extra,
                            )
                        if logger:
                            logger.log(
                                {
                                    "step": state.global_step,
                                    "validation_loss": validation_loss,
                                    "completed_epochs": state.epoch + int(epoch_end),
                                    "epoch_complete": epoch_end,
                                    "early_stopping_bad_epochs": stop.bad_epochs,
                                    "early_stopping_triggered": stop.stopped,
                                    **{
                                        f"validation_{k}": value / count
                                        for k, value in validation_totals.items()
                                    },
                                }
                            )
                    distributed_barrier(context)
                if state.global_step >= max_steps or stop.stopped:
                    break
            if not saw_batch:
                raise RuntimeError("Empty baseline data loader")
            if state.global_step < max_steps and not stop.stopped:
                state.epoch += 1
                state.batch_in_epoch = 0
        if context.is_main:
            reason = "validation_plateau" if stop.stopped else "max_steps"
            if not (checkpoint_dir / "best.pt").is_file():
                raise RuntimeError("Completed VAE training has no validation-selected checkpoint")
            completion = dict(
                reason=reason,
                global_step=state.global_step,
                epoch=state.epoch,
                batch_in_epoch=state.batch_in_epoch,
                early_stopping=stop.state_dict(),
                best_validation=best,
                evaluation_checkpoint=str(checkpoint_dir / "best.pt"),
            )
            temporary = checkpoint_dir / "training_complete.json.tmp"
            temporary.write_text(json.dumps(completion, indent=2, allow_nan=False) + "\n")
            temporary.replace(checkpoint_dir / "training_complete.json")
            print(
                f"VAE training complete: {reason}, step={state.global_step}; evaluate best.pt",
                flush=True,
            )
        distributed_barrier(context)
        if logger:
            logger.close()
        progress.close()
    finally:
        cleanup_distributed()


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    _train(cfg)


if __name__ == "__main__":
    main()

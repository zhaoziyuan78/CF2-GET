"""Exercise the actual v3 training loop on tiny CPU models, including resume."""

import json
from pathlib import Path
from typing import ClassVar

import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

from cfmusic.commands import train_baseline
from cfmusic.distributed import DistributedContext


class SmallDataset(torch.utils.data.Dataset):
    reads: ClassVar[list] = []

    def __init__(self, *args, split="train", **kwargs):
        self.split = split
        self.frame = pd.DataFrame(
            dict(
                sample_id=[f"{split}-{i}" for i in range(8)],
                segment_id=[f"{split}-{i}" for i in range(8)],
                style_id=[0, 1] * 4,
            )
        )
        self.metadata = {}
        self.shard_ids = ["one"] * 8

    def refresh_index(self):
        pass

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        self.reads.append((self.split, index))
        return dict(
            latent=torch.ones(2, 3) * (index + 1) / 10,
            style_id=torch.tensor(int(self.frame.iloc[index].style_id)),
        )


class NoLog:
    def __init__(self, *args, **kwargs):
        pass

    def log(self, *args, **kwargs):
        pass

    def update(self, *args):
        pass

    def close(self):
        pass


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    card = tmp_path / "card.json"
    card.write_text(json.dumps(dict(genre_vocabulary=["a", "b"])))
    for module in (train_baseline,):
        monkeypatch.setattr(
            module,
            "initialize_distributed",
            lambda: DistributedContext(0, 0, 1, torch.device("cpu")),
        )
        monkeypatch.setattr(module, "cleanup_distributed", lambda: None)
        monkeypatch.setattr(
            module,
            "prepare_config",
            lambda cfg: dict(checkpoints_dir=Path(cfg.output), latent_dir=tmp_path),
        )
        monkeypatch.setattr(module, "save_run_context", lambda *args: None)
        monkeypatch.setattr(module, "LatentDataset", SmallDataset)
        monkeypatch.setattr(module, "validate_latent_dataset", lambda *args, **kwargs: None)
        monkeypatch.setattr(module, "validate_transport_cache_provenance", lambda *args: None)
        monkeypatch.setattr(module, "MetricLogger", NoLog)
        monkeypatch.setattr(module, "progress_bar", lambda **kwargs: NoLog())
    return OmegaConf.create(
        dict(
            output=str(tmp_path),
            seed=0,
            resume=True,
            resume_from=None,
            codec={},
            experiment=dict(name="latent_midi_vae"),
            data=dict(name="xmidi", task="genre", latent_index="unused", dataset_card=str(card)),
            transport=dict(
                name="Latent-MIDI-VAE",
                type="latent_midi_vae",
                model=dict(latent_tokens=2, latent_dim=3, hidden_dim=4, code_dim=4, layers=1),
                conditioning=dict(num_styles=2),
                loss=dict(beta=0.001, style_weight=0.1),
                training=dict(
                    batch_size=2,
                    gradient_accumulation=1,
                    learning_rate=0.0,
                    weight_decay=0.0,
                    max_steps=40,
                    warmup_steps=0,
                    gradient_clip_norm=1.0,
                    precision="fp32",
                    checkpoint_interval=1,
                    dataloader_workers=0,
                    log_interval=1,
                    validation_samples_per_style=2,
                    early_stopping=dict(
                        patience=2,
                        min_epochs=1,
                        min_steps=0,
                        relative_delta=0.001,
                        absolute_delta=0.0001,
                    ),
                ),
            ),
        )
    )


def checkpoint(cfg, name="last.pt"):
    return Path(cfg.output) / cfg.experiment.name / "transport" / name


def test_real_loop_stops_only_at_epoch_end_and_resume_stays_stopped(fixture):
    train_baseline._train(fixture)
    saved = torch.load(checkpoint(fixture), weights_only=False)
    assert saved["train_state"]["global_step"] == 12
    assert saved["train_state"]["batch_in_epoch"] == 4
    assert saved["extra_state"]["early_stopping"]["last_epoch"] == 3
    assert saved["extra_state"]["stop_reason"] == "validation_plateau"
    best = torch.load(checkpoint(fixture, "best.pt"), weights_only=False)
    assert best["train_state"]["global_step"] == 1
    SmallDataset.reads.clear()
    train_baseline._train(fixture)
    assert not SmallDataset.reads  # No previously completed MIDI/latent reads on resume.
    done = json.loads(checkpoint(fixture, "training_complete.json").read_text())
    assert done["reason"] == "validation_plateau"
    assert done["global_step"] == 12


def test_interrupted_mid_epoch_keeps_patience_and_exact_unseen_cursor(fixture, monkeypatch):
    save = train_baseline.save_checkpoint

    def interrupted(path, **kwargs):
        save(path, **kwargs)
        if path.name == "last.pt" and kwargs["state"].global_step == 5:
            raise RuntimeError("test interruption")

    monkeypatch.setattr(train_baseline, "save_checkpoint", interrupted)
    with pytest.raises(RuntimeError, match="test interruption"):
        train_baseline._train(fixture)
    saved = torch.load(checkpoint(fixture), weights_only=False)
    assert saved["train_state"]["batch_in_epoch"] == 1
    assert saved["extra_state"]["early_stopping"]["last_epoch"] == 1
    monkeypatch.setattr(train_baseline, "save_checkpoint", save)
    SmallDataset.reads.clear()
    train_baseline._train(fixture)
    saved = torch.load(checkpoint(fixture), weights_only=False)
    assert saved["train_state"]["global_step"] == 12
    assert len([r for r in SmallDataset.reads if r[0] == "train"]) == 14


@pytest.mark.parametrize("direct", [False, True])
def test_training_reaches_max_steps_when_early_stopping_does_not_trigger(
    fixture, monkeypatch, direct
):
    if direct:

        class Rolls(torch.utils.data.Dataset):
            def __init__(self, latents):
                self.latents = latents

            def __len__(self):
                return len(self.latents)

            def __getitem__(self, index):
                source = self.latents[index]
                return dict(
                    style_id=source["style_id"],
                    pitch=torch.full((16, 1), 60 + index),
                    velocity=torch.full((16, 1), 0.75),
                    program=torch.zeros(1, dtype=torch.long),
                )

        fixture.experiment.name = "midi_vae"
        fixture.transport.type = "midi_vae"
        fixture.transport.model.voices = 1
        fixture.transport.model.num_bars = 1
        for module in (train_baseline,):
            monkeypatch.setattr(module, "roll_dataset", lambda latents, cfg, paths: Rolls(latents))
    fixture.transport.training.max_steps = 5
    fixture.transport.training.learning_rate = 0.001
    fixture.transport.training.early_stopping.min_epochs = 100
    train_baseline._train(fixture)
    updated = torch.load(checkpoint(fixture), weights_only=False)
    assert updated["extra_state"]["stop_reason"] == "max_steps"
    assert not updated["extra_state"]["early_stopping"]["stopped"]


def test_new_entry_rejects_removed_baselines(fixture):
    fixture.transport.type = "music_cyclegan"
    with pytest.raises(ValueError, match="only supports"):
        train_baseline._train(fixture)


def _ddp_worker(rank, cfg, rendezvous):
    import torch.distributed as dist

    torch.set_num_threads(1)
    patch = pytest.MonkeyPatch()
    patch.setattr(torch.cuda, "is_available", lambda: False)
    patch.setattr(torch.cuda, "is_initialized", lambda: False)
    for name, value in {
        "prepare_config": lambda cfg: dict(
            checkpoints_dir=Path(cfg.output), latent_dir=Path(cfg.output)
        ),
        "save_run_context": lambda *args: None,
        "LatentDataset": SmallDataset,
        "validate_latent_dataset": lambda *args, **kwargs: None,
        "validate_transport_cache_provenance": lambda *args: None,
        "MetricLogger": NoLog,
        "progress_bar": lambda **kwargs: NoLog(),
        "cleanup_distributed": lambda: None,
    }.items():
        patch.setattr(train_baseline, name, value)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    train_baseline.initialize_distributed = lambda: DistributedContext(
        rank, rank, 2, torch.device("cpu")
    )
    try:
        train_baseline._train(cfg)
    finally:
        dist.destroy_process_group()
        patch.undo()


def test_two_cpu_ranks_stop_together_at_complete_epoch(fixture):
    # Spawn fresh CPU-only ranks; no GPU or production data is touched.
    torch.multiprocessing.start_processes(
        _ddp_worker,
        args=(fixture, str(Path(fixture.output) / "gloo_init")),
        nprocs=2,
        join=True,
        start_method="spawn",
    )
    saved = torch.load(checkpoint(fixture), weights_only=False)
    assert saved["train_state"]["world_size"] == 2
    assert saved["train_state"]["global_step"] == 6
    assert saved["train_state"]["batch_in_epoch"] == 2
    assert saved["extra_state"]["early_stopping"]["last_epoch"] == 3
    assert saved["extra_state"]["stop_reason"] == "validation_plateau"

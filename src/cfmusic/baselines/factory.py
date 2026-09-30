"""Build the MIDI-VAE baselines and their direct piano-roll training data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

from cfmusic.baselines.midi_vae import MIDIVAE, MIDIRollDataset
from cfmusic.baselines.models import LatentMIDIVAE
from cfmusic.config import dataset_asset_path
from cfmusic.latent.dataset import LatentDataset
from cfmusic.transport.factory import create_transport


def baseline_model(cfg: DictConfig) -> MIDIVAE | LatentMIDIVAE:
    if str(cfg.type) == "midi_vae":
        return MIDIVAE(
            num_styles=int(cfg.conditioning.num_styles),
            hidden_dim=int(cfg.model.hidden_dim),
            code_dim=int(cfg.model.code_dim),
            voices=int(cfg.model.voices),
            num_bars=int(cfg.model.num_bars),
            beta=float(cfg.loss.beta),
            style_weight=float(cfg.loss.style_weight),
        )
    model = create_transport(cfg)
    if not isinstance(model, LatentMIDIVAE):
        raise ValueError("Baseline training only accepts MIDI-VAE / Latent-MIDI-VAE")
    return model


def roll_dataset(
    latents: LatentDataset, cfg: DictConfig, paths: dict[str, Path]
) -> MIDIRollDataset:
    manifest = pd.read_parquet(
        dataset_asset_path(cfg, paths, "manifest.parquet"),
        columns=["segment_id", "source_midi_path", "start_bar", "num_bars"],
    )
    frame = latents.frame[["segment_id", "style_id", "sample_id"]].merge(
        manifest, on="segment_id", how="left", validate="one_to_one", sort=False
    )
    if frame.source_midi_path.isna().any() or len(frame) != len(latents):
        raise ValueError("Direct baseline cannot resolve the common evaluation/training segments")
    return MIDIRollDataset(frame, voices=int(cfg.transport.model.voices))

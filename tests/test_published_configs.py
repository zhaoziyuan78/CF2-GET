"""The release configs compose exactly the training configs of the published models."""

import json
from pathlib import Path

import pandas as pd
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from cfmusic.config import CONFIG_DIR, resolve_label_configuration

ROOT = Path(__file__).resolve().parents[1]
PUBLISHED = json.loads((ROOT / "tests/fixtures/published_training_configs.json").read_text())
GENRES = ["classical", "jazz", "rock", "pop", "country", "latin"]
EMOTIONS = ["angry", "exciting", "fear", "funny", "happy", "lazy", "magnificent", "quiet",
            "romantic", "sad", "warm"]


@pytest.fixture
def overlay(tmp_path: Path) -> Path:
    """A six-genre label overlay, as published by scripts/prepare_xmidi_16bar.py."""
    card = tmp_path / "dataset_card.json"
    card.write_text(json.dumps(dict(style_vocabulary=GENRES, genre_vocabulary=GENRES,
                                    emotion_vocabulary=EMOTIONS)))
    index = tmp_path / "index.parquet"
    pd.DataFrame({"segment_id": ["a:bar0:n16"]}).to_parquet(index)
    index.with_suffix(".metadata.json").write_text(json.dumps(dict(
        dataset_card=str(card), manifest=str(tmp_path / "manifest.parquet"), genre_vocabulary=GENRES)))
    return index


def composed(experiment: str, overlay: Path) -> dict:
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="config", overrides=[
            f"experiment={experiment}", f"data.latent_index={overlay}", f"paths.project_root={ROOT}"])
    if experiment.startswith("xmidi_codec"):
        section = OmegaConf.to_container(cfg.codec, resolve=True)
        section["tokenizer"] = OmegaConf.to_container(cfg.tokenizer, resolve=True)
        return section
    resolve_label_configuration(cfg, {"processed_dir": ROOT / "data/processed"})
    return OmegaConf.to_container(cfg.transport, resolve=True)


@pytest.mark.parametrize("experiment", sorted(PUBLISHED))
def test_release_recipe_equals_published_training_config(experiment: str, overlay: Path) -> None:
    release = composed(experiment, overlay)
    expected = json.loads(json.dumps(PUBLISHED[experiment]))
    if experiment == "xmidi_codec_16bar":
        # The release fine-tunes directly from its own stage-1 checkpoint (same warm start).
        assert release["training"].pop("initialize_from").endswith(
            "checkpoints/xmidi_codec_8bar/codec/xmidi/last.pt")
        expected["training"].pop("initialize_from")
    if experiment == "xmidi_ddim":
        # Inference-only sampling-grid floor, added after the published DDIM was trained.
        assert release["diffusion"].pop("min_alpha_cumprod") == 4.6e-3
    assert release == expected


def test_default_test_protocol_is_the_paired_2000_source_cohort() -> None:
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="config")
    assert cfg.experiment.name == "xmidi_otcfm_hsic"
    counterfactual = cfg.counterfactual
    assert (counterfactual.split, counterfactual.max_total_sources, counterfactual.targets_per_source) == (
        "test", 2000, 1)
    assert counterfactual.source_sampling_policy == counterfactual.target_policy == "proportional"
    assert counterfactual.require_all_transitions and counterfactual.unique_sources
    assert counterfactual.transport_weights == "ema" and counterfactual.full_decode_budget
    assert cfg.report.require_paired_cohort

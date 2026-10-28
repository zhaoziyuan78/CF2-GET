"""Hydra configuration helpers shared by command entry points."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from omegaconf import DictConfig, OmegaConf, open_dict

from cfmusic.paths import ensure_paths, resolved_paths

_PACKAGE_CONFIG_DIR = Path(__file__).resolve().parent / "configs"
_PROJECT_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
CONFIG_DIR = str(_PACKAGE_CONFIG_DIR if _PACKAGE_CONFIG_DIR.is_dir() else _PROJECT_CONFIG_DIR)


def prepare_config(cfg: DictConfig) -> dict[str, Path]:
    """Resolve, create, and print all configured directories."""
    paths = resolved_paths(cfg)
    resolve_label_configuration(cfg, paths)
    OmegaConf.resolve(cfg)
    ensure_paths(paths)
    if int(os.environ.get("RANK", "0")) == 0:
        print("Resolved paths:")
        for name, path in paths.items():
            print(f"  {name}: {path}")
    return paths


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def dataset_asset_path(cfg: DictConfig, paths: dict[str, Path], name: str) -> Path:
    """Use an explicit label-overlay asset without replacing the base dataset files."""
    key = "dataset_card" if name == "dataset_card.json" else "manifest"
    value = cfg.data.get(key)
    return (
        Path(str(value)).expanduser().resolve()
        if value is not None
        else paths["processed_dir"] / str(cfg.data.name) / name
    )


def resolve_label_configuration(cfg: DictConfig, paths: dict[str, Path]) -> None:
    """Resolve overlay vocabulary for every consumer; keep original checkpoint shapes intact."""
    index_value = cfg.data.get("latent_index")
    overlay = None
    if index_value is not None:
        index = Path(str(index_value)).expanduser().resolve()
        sidecar = index.with_suffix(".metadata.json")
        if sidecar.is_file():
            overlay = json.loads(sidecar.read_text(encoding="utf-8"))
            if overlay.get("dataset_card") and overlay.get("manifest"):
                with open_dict(cfg.data):
                    for key in ("dataset_card", "manifest"):
                        configured = cfg.data.get(key)
                        if (
                            configured is not None
                            and Path(str(configured)).resolve() != Path(overlay[key]).resolve()
                        ):
                            raise ValueError(
                                f"Configured {key} conflicts with latent label overlay"
                            )
                        cfg.data[key] = overlay[key]
                    cfg.data.auto_conditioning_size = True
    if not bool(cfg.data.get("auto_conditioning_size", False)):
        return
    card = json.loads(
        dataset_asset_path(cfg, paths, "dataset_card.json").read_text(encoding="utf-8")
    )
    styles = card["style_vocabulary"]
    genres = card.get("genre_vocabulary", styles)
    if str(cfg.data.get("task", cfg.get("task", "genre"))) == "genre" and styles != genres:
        raise ValueError(
            "Genre-task style vocabulary and genre vocabulary must have identical order"
        )
    if not styles or len(styles) != len(set(styles)) or len(genres) != len(set(genres)):
        raise ValueError("Dataset vocabulary must contain unique non-empty classes")
    if overlay is not None and overlay.get("genre_vocabulary") != genres:
        raise ValueError("Dataset card and latent overlay use different genre vocabularies")
    with open_dict(cfg.data):
        cfg.data.num_styles = len(styles)
        cfg.data.num_genres = len(genres)
    if "transport" in cfg and "conditioning" in cfg.transport:
        cfg.transport.conditioning.num_styles = len(styles)
        cfg.transport.conditioning.num_genres = len(genres)
        if card.get("emotion_vocabulary"):
            cfg.transport.conditioning.num_emotions = len(card["emotion_vocabulary"])
        sampling = cfg.transport.get("sampling")
        if sampling is not None and bool(sampling.get("balance_by_style", False)):
            if sampling.get("classes_per_batch") is None:
                sampling.classes_per_batch = len(styles)
            if cfg.transport.training.get("batch_size") is None:
                cfg.transport.training.batch_size = int(sampling.classes_per_batch) * int(
                    sampling.samples_per_class
                )
    if overlay and overlay.get("relabel_config") and "evaluation" in cfg:
        value = cfg.evaluation.clamp3.get("relabel_config")
        if (
            value is not None
            and Path(str(value)).resolve() != Path(overlay["relabel_config"]).resolve()
        ):
            raise ValueError("Evaluation calibration differs from label overlay")
        cfg.evaluation.clamp3.relabel_config = overlay["relabel_config"]


def config_mapping(cfg: DictConfig) -> dict[str, object]:
    """Convert a DictConfig to a string-keyed mapping for checkpoint metadata."""
    value = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(value, dict):
        raise TypeError("Expected a mapping configuration")
    return {str(key): item for key, item in value.items()}

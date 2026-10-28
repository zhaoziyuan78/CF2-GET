"""Path resolution helpers."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from omegaconf import DictConfig


def resolve_path(path: str | Path) -> Path:
    """Resolve an absolute or cwd-relative path without requiring it to exist."""
    candidate = Path(os.path.expandvars(str(path))).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (Path.cwd() / candidate).resolve()


def resolved_paths(cfg: DictConfig) -> dict[str, Path]:
    """Return all configured path values as absolute Paths."""
    return {str(key): resolve_path(value) for key, value in cfg.paths.items()}


def ensure_paths(paths: Mapping[str, Path]) -> None:
    """Create configured output directories."""
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

"""Construct configured dataset source instances."""

from __future__ import annotations

from omegaconf import DictConfig

from cfmusic.download.base import DatasetSource
from cfmusic.download.google_drive import GoogleDriveSource


def create_source(dataset: str, cfg: DictConfig) -> DatasetSource:
    source = cfg.source
    if source.kind == "google_drive":
        return GoogleDriveSource(dataset, source.file_id, source.archive_name)
    raise ValueError(f"Unsupported dataset source kind: {source.kind}")

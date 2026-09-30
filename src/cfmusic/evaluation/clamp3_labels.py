"""Six-genre CLaMP3 segment labels: prompt prototypes and cosine genre scores."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from cfmusic.evaluation.clamp3 import extract_clamp3_embeddings

GENRES = ("classical", "jazz", "rock", "pop", "country", "latin")
TEMPLATES = (
    "This is a piece of {style} music.",
    "The musical genre is {style}.",
    "A {style} instrumental track.",
    "This composition has a {style} style.",
    "Music best described as {style}.",
    "An example of {style} music.",
)
FEATURE_PROTOCOL = "cfmusic.clamp3-c2.three-512-patch-views.raw-fp32.v1"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms > 0)


def open_array(path: Path, shape: tuple[int, ...], dtype: Any, fill: int = 0) -> Any:
    """Open (or create) a resumable memory-mapped score/status array."""

    if path.exists():
        array = np.load(path, mmap_mode="r+")
        if array.shape != shape or array.dtype != np.dtype(dtype):
            raise ValueError(f"Incompatible embedding array: {path}")
        return array
    array = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    array[:] = fill
    array.flush()
    return array


def gpu_patch_forward(self: Any, input_patches: torch.Tensor, input_masks: torch.Tensor) -> Any:
    # Exact official one-hot + Linear computation, without the obsolete GPU->CPU->GPU cast.
    patches = torch.nn.functional.one_hot(input_patches, num_classes=128)
    patches = patches.reshape(len(input_patches), -1, 64 * 128).to(
        dtype=self.patch_embedding.weight.dtype
    )
    return self.base(inputs_embeds=self.patch_embedding(patches), attention_mask=input_masks)


def prompt_embeddings(repository: Path, genres=GENRES, templates=TEMPLATES) -> np.ndarray:
    """CLaMP3 text embeddings, shaped [genre, template, 768]."""

    texts = {
        f"g-{g:02d}-p-{p:02d}": template.format(style=genre)
        for g, genre in enumerate(genres)
        for p, template in enumerate(templates)
    }
    embeddings = extract_clamp3_embeddings(
        repository=repository, midi_files={}, texts=texts, batch_size=64, device="cuda"
    )
    return np.stack(
        [
            [embeddings[f"g-{g:02d}-p-{p:02d}"] for p in range(len(templates))]
            for g in range(len(genres))
        ]
    )


def singleton_taxonomy(embedding_model: dict, label_source: str) -> dict:
    """One CLaMP3 prototype per genre (the mean of its template embeddings), zero logit bias."""

    return dict(
        label_source=label_source,
        embedding_model=embedding_model,
        feature_protocol=FEATURE_PROTOCOL,
        genre_vocabulary=list(GENRES),
        candidate_genres=list(GENRES),
        genre_membership={genre: [genre] for genre in GENRES},
        prompt_templates=list(TEMPLATES),
        prompt_logit_bias=dict.fromkeys(GENRES, 0.0),
        aggregation="max_member_prototypes",
        selection="fixed six singleton genres; no taxonomy search",
    )


def family_scores(features: np.ndarray, prototypes: np.ndarray, taxonomy: dict) -> np.ndarray:
    """Cosine similarity of pooled segment features to every genre prototype."""

    members = normalize(features) @ prototypes.T
    vocabulary = taxonomy["candidate_genres"]
    return np.stack(
        [
            members[:, [vocabulary.index(member) for member in taxonomy["genre_membership"][genre]]]
            .max(axis=1)
            + float(taxonomy["prompt_logit_bias"][genre])
            for genre in taxonomy["genre_vocabulary"]
        ],
        axis=1,
    ).astype(np.float32)

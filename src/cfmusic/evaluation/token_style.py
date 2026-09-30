"""Apply an independently trained, dataset-specific classifier to generated MIDI."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from omegaconf import OmegaConf

from cfmusic.data.midi_io import load_midi
from cfmusic.evaluation.style_effect import TokenStyleClassifier
from cfmusic.memory import autocast_context
from cfmusic.tokenization.factory import tokenizer_from_config


class TokenStyleEvaluator:
    def __init__(
        self,
        checkpoint_path: Path,
        *,
        dataset: str,
        task: str,
        style_vocabulary: Sequence[str],
        device: torch.device,
    ) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
        metadata = checkpoint.get("metadata", {})
        if (
            metadata.get("schema_version") != "cfmusic.token-evaluator.v1"
            or metadata.get("dataset") != dataset
            or metadata.get("task") != task
            or metadata.get("style_vocabulary") != list(style_vocabulary)
            or metadata.get("trained_on_generated_samples") is not False
            or metadata.get("training_split") != "train"
        ):
            raise ValueError(
                "Evaluator metadata/dataset/class ordering mismatch; train a dataset-specific "
                "classifier with train_evaluator before evaluating generated music"
            )
        config = OmegaConf.create(checkpoint["config"])
        tokenizer_config = OmegaConf.create(checkpoint["tokenizer"])
        self.tokenizer = tokenizer_from_config(tokenizer_config)
        self.model = TokenStyleClassifier(
            len(self.tokenizer.vocabulary),
            len(style_vocabulary),
            d_model=int(config.d_model),
            layers=int(config.layers),
            heads=int(config.heads),
            dropout=float(config.dropout),
            max_length=int(tokenizer_config.max_sequence_length),
            pad_id=self.tokenizer.vocabulary.pad_id,
        )
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.model.to(device).eval()
        self.device = device
        self.precision = str(config.inference.precision)
        self._cache: dict[tuple[int, int, int, int, int], NDArray[np.float64]] = {}

    def probabilities(self, path: Path, *, num_bars: int) -> NDArray[np.float64]:
        stat = path.stat()
        # Generated transitions share hard-linked factual and reconstruction MIDI.
        # Reuse those predictions without hashing or reading each file again.
        key = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, num_bars)
        if key not in self._cache:
            ids = self.tokenizer.encode(load_midi(path), num_bars=num_bars)
            tokens = torch.tensor([ids], dtype=torch.long, device=self.device)
            mask = tokens.ne(self.tokenizer.vocabulary.pad_id)
            with torch.inference_mode(), autocast_context(self.device, self.precision):
                probabilities = self.model(tokens, mask).float().softmax(-1)[0]
            values = probabilities.cpu().numpy().astype(np.float64)
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite classifier prediction for {path}")
            self._cache[key] = values
        return self._cache[key]


def classifier_style_metrics(
    probabilities: NDArray[np.float64], *, source_style_id: int, target_style_id: int
) -> dict[str, float]:
    if not (
        0 <= source_style_id < len(probabilities) and 0 <= target_style_id < len(probabilities)
    ):
        raise ValueError("Artifact style id is outside the classifier vocabulary")
    return {
        "classifier_target_style_success": float(int(probabilities.argmax()) == target_style_id),
        "classifier_target_probability": float(probabilities[target_style_id]),
        "classifier_source_probability": float(probabilities[source_style_id]),
        "classifier_target_minus_source": float(
            probabilities[target_style_id] - probabilities[source_style_id]
        ),
    }

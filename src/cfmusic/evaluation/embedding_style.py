"""Frozen, real-data-only supervised heads on CLaMP3 embeddings."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np


class EmbeddingStyleEvaluator:
    """Load a locally trained qualified head; never fit on generated artifacts."""

    def __init__(
        self,
        checkpoint: Path,
        *,
        dataset: str,
        task: str,
        vocabulary: list[str],
        embedding_model: dict[str, object],
    ) -> None:
        payload = joblib.load(checkpoint)
        if payload.get("schema") != "cfmusic.asap.clamp-classifier.v1":
            raise ValueError("Unsupported CLaMP3 classifier checkpoint")
        if payload.get("dataset") != dataset or payload.get("task") != task:
            raise ValueError("Classifier dataset/task mismatch")
        if payload.get("composer_vocabulary") != vocabulary:
            raise ValueError("Classifier vocabulary mismatch")
        if (
            payload.get("training_split") != "train"
            or payload.get("trained_on_generated_samples") is not False
        ):
            raise ValueError("Classifier must be trained only on real training data")
        if payload.get("embedding_model") != embedding_model:
            raise ValueError("Classifier belongs to another CLaMP3 model/protocol")
        if not payload.get("qualified", False):
            raise ValueError("Classifier has not passed held-out qualification")
        self.model = payload["model"]
        self.kind = payload["feature_kind"]
        self.prototypes = np.asarray(payload["text_prototypes"])
        self.metadata = {k: v for k, v in payload.items() if k not in {"model", "text_prototypes"}}
        if list(self.model.classes_) != list(range(len(vocabulary))):
            raise ValueError("Classifier class indices do not match dataset vocabulary")

    def probabilities(self, embedding: np.ndarray) -> np.ndarray:
        vector = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        norms = np.linalg.norm(vector, axis=1, keepdims=True)
        if vector.shape != (1, 768) or not np.isfinite(vector).all() or np.any(norms <= 0):
            raise ValueError("Invalid music embedding")
        vector = vector / norms
        if self.kind == "text":
            features = vector @ self.prototypes.T
        elif self.kind == "hybrid":
            features = np.concatenate([vector, vector @ self.prototypes.T], axis=1)
        elif self.kind == "clamp":
            features = vector
        else:
            raise ValueError("Unknown classifier features")
        scores = np.asarray(
            self.model.decision_function(features)[0]
            if hasattr(self.model, "decision_function")
            else np.log(np.clip(self.model.predict_proba(features)[0], 1e-12, 1)),
            dtype=float,
        )
        scores = np.exp(scores - scores.max())
        # Softmax of decision scores is NOT claimed to be calibrated confidence.
        return scores / scores.sum()

from pathlib import Path

import joblib
import mido
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from cfmusic.evaluation.clamp3 import midi_to_mtf_text
from cfmusic.evaluation.embedding_style import EmbeddingStyleEvaluator


def classifier_payload():
    x = np.eye(768, dtype=np.float32)[:6]
    model = LogisticRegression().fit(x, np.arange(6))
    return {
        "schema": "cfmusic.asap.clamp-classifier.v1",
        "model": model,
        "feature_kind": "clamp",
        "text_prototypes": x,
        "dataset": "asap",
        "task": "style",
        "composer_vocabulary": list("abcdef"),
        "embedding_model": {"schema": "test"},
        "training_split": "train",
        "trained_on_generated_samples": False,
        "qualified": True,
    }


def test_supervised_head_preserves_class_order_and_normalizes(tmp_path: Path):
    path = tmp_path / "head.joblib"
    joblib.dump(classifier_payload(), path)
    evaluator = EmbeddingStyleEvaluator(
        path,
        dataset="asap",
        task="style",
        vocabulary=list("abcdef"),
        embedding_model={"schema": "test"},
    )
    vector = np.eye(768)[3] * 5
    assert evaluator.probabilities(vector).argmax() == 3
    assert evaluator.probabilities(vector).sum() == pytest.approx(1)


@pytest.mark.parametrize(
    "field,value",
    [
        ("qualified", False),
        ("dataset", "xmidi"),
        ("trained_on_generated_samples", True),
        ("embedding_model", {}),
    ],
)
def test_unqualified_or_incompatible_heads_cannot_evaluate(tmp_path, field, value):
    payload = classifier_payload()
    payload[field] = value
    path = tmp_path / "head.joblib"
    joblib.dump(payload, path)
    with pytest.raises(ValueError):
        EmbeddingStyleEvaluator(
            path,
            dataset="asap",
            task="style",
            vocabulary=list("abcdef"),
            embedding_model={"schema": "test"},
        )


def test_opt_in_sanitization_preserves_removed_delta_times(tmp_path):
    midi = mido.MidiFile()
    midi.tracks.append(
        mido.MidiTrack(
            [
                mido.MetaMessage("text", text="composer", time=30),
                mido.Message("note_on", note=60, velocity=80, time=20),
                mido.Message("note_off", note=60, time=100),
            ]
        )
    )
    path = tmp_path / "music.mid"
    midi.save(path)
    text = midi_to_mtf_text(path, preserve_metadata_timing=True)
    assert "composer" not in text
    assert "note_on 50" in text
    assert "note_on 20" in midi_to_mtf_text(path)  # legacy experiments remain unchanged

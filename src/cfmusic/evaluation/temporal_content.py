"""Fixed-beat pitched-content diagnostics; no external embeddings or alignment search."""

from __future__ import annotations

import miditoolkit
import numpy as np


def beat_chroma(midi: miditoolkit.MidiFile, *, beats: float = 32., bin_beats: float = .5) -> np.ndarray:
    """Duration-overlap chroma on an absolute beat grid, excluding percussion."""
    bins = round(beats / bin_beats) if bin_beats > 0 else 0
    if beats <= 0 or bin_beats <= 0 or bins <= 0 or not np.isclose(bins * bin_beats, beats):
        raise ValueError("Positive beats must be an integer multiple of bin_beats")
    if midi.ticks_per_beat <= 0:
        raise ValueError("Invalid MIDI beat resolution")
    result = np.zeros((bins, 12), dtype=np.float64)
    left = np.arange(bins) * bin_beats
    right = left + bin_beats
    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            start, end = note.start / midi.ticks_per_beat, note.end / midi.ticks_per_beat
            if end <= start:
                continue
            result[:, note.pitch % 12] += np.maximum(0., np.minimum(right, end) - np.maximum(left, start))
    return result


def beat_chroma_cosine(source: np.ndarray, generated: np.ndarray) -> float:
    """Average per-bin cosine on union-active bins; silence cannot inflate scores."""
    if source.shape != generated.shape or source.ndim != 2 or source.shape[1] != 12:
        raise ValueError("Expected aligned [beats, 12] chroma arrays")
    if not np.isfinite(source).all() or not np.isfinite(generated).all():
        raise ValueError("Non-finite chroma")
    if (source < 0).any() or (generated < 0).any():
        raise ValueError("Chroma must be nonnegative")
    a, b = np.linalg.norm(source, axis=1), np.linalg.norm(generated, axis=1)
    active = (a > 0) | (b > 0)
    if not active.any():
        return 0.
    scores = np.divide((source * generated).sum(axis=1), a * b,
                       out=np.zeros_like(a), where=(a * b) > 0)
    return float(np.clip(scores[active], 0., 1.).mean())

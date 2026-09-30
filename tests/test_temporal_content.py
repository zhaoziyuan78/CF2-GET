import miditoolkit
import numpy as np
import pytest

from cfmusic.evaluation.temporal_content import beat_chroma, beat_chroma_cosine


def sequence(pitches, *, ticks=480, drum=False, program=0, velocity=80):
    midi = miditoolkit.MidiFile(ticks_per_beat=ticks)
    track = miditoolkit.Instrument(program=program, is_drum=drum)
    track.notes = [miditoolkit.Note(velocity, p, i*ticks, (i+1)*ticks) for i, p in enumerate(pitches)]
    midi.instruments = [track]
    return midi


def test_self_octave_instrument_velocity_and_resolution_invariance():
    a = beat_chroma(sequence([60, 64, 67]))
    b = beat_chroma(sequence([72, 76, 79], ticks=960, program=24, velocity=110))
    assert beat_chroma_cosine(a, a) == pytest.approx(1.)
    assert beat_chroma_cosine(a, b) == pytest.approx(1.)


def test_fixed_time_order_not_global_histogram():
    a = beat_chroma(sequence([60, 64, 67]))
    b = beat_chroma(sequence([67, 60, 64]))
    np.testing.assert_allclose(a.sum(axis=0), b.sum(axis=0))
    assert beat_chroma_cosine(a, b) == 0.


def test_no_silence_reward_or_drum_pitch_contamination():
    a = beat_chroma(sequence([60]))
    drums = beat_chroma(sequence([60], drum=True))
    assert beat_chroma_cosine(drums, drums) == 0.
    assert beat_chroma_cosine(a, drums) == 0.
    assert beat_chroma_cosine(a, a) == pytest.approx(1.)


def test_note_segmentation_does_not_change_sustain():
    midi = sequence([60])
    midi.instruments[0].notes = [miditoolkit.Note(80, 60, 0, 240), miditoolkit.Note(80, 60, 240, 480)]
    np.testing.assert_allclose(beat_chroma(midi), beat_chroma(sequence([60])))


def test_overlap_duration_and_window_clipping():
    midi = sequence([60])
    midi.instruments[0].notes[0].start = 120
    midi.instruments[0].notes[0].end = 20000
    value = beat_chroma(midi, beats=1.)
    assert value[:, 0].tolist() == [.25, .5]
    assert value.sum() == .75


def test_validation():
    with pytest.raises(ValueError):
        beat_chroma(sequence([60]), bin_beats=0)
    with pytest.raises(ValueError):
        beat_chroma_cosine(np.zeros((2, 12)), np.zeros((3, 12)))
    with pytest.raises(ValueError):
        beat_chroma_cosine(np.ones((2, 12)), -np.ones((2, 12)))

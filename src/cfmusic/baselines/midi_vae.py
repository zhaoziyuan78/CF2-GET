"""Direct symbolic MIDI-VAE: pitch/velocity/instrument GRUs and style-coordinate swap.

PyTorch adaptation of Brunner et al. (ISMIR 2018), not the original Keras code.
Uses four monophonic voices, 1/16-note cells, full 128 pitches, and the common
eight-bar evaluation windows. Drums/excess voices are outside this baseline's
representation, and are reported as such rather than silently copied to output.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import miditoolkit
import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from cfmusic.baselines.models import gaussian_kl, swap_style
from cfmusic.data.midi_io import load_midi

ROLL_SCHEMA = "midi-vae-four-voices-pitch129-velocity-onset-program129-16th.v1"


def midi_to_rolls(
    midi: miditoolkit.MidiFile, *, start_bar: int, num_bars: int, voices: int = 4
) -> dict[str, Tensor]:
    steps = num_bars * 16
    start, end = (
        start_bar * 4 * midi.ticks_per_beat,
        (start_bar + num_bars) * 4 * midi.ticks_per_beat,
    )
    tracks = [
        (i, [n for n in i.notes if start <= n.start < end])
        for i in midi.instruments
        if not i.is_drum
    ]
    tracks = sorted((pair for pair in tracks if pair[1]), key=lambda pair: -len(pair[1]))
    allocation = [(index, 0) for index in range(min(voices, len(tracks)))]
    depth = 1
    while tracks and len(allocation) < voices:
        for index in range(len(tracks)):
            if len(allocation) < voices:
                allocation.append((index, depth))
        depth += 1
    pitch = np.full((steps, voices), 128, dtype=np.int64)
    velocity = np.zeros((steps, voices), dtype=np.float32)
    programs = np.full(voices, 128, dtype=np.int64)
    for track_index in sorted({i for i, _ in allocation}):
        instrument, notes = tracks[track_index]
        cells: list[list[tuple[int, int, int]]] = [[] for _ in range(steps)]
        for note_id, note in enumerate(notes):
            first = min(steps - 1, max(0, round((note.start - start) * 4 / midi.ticks_per_beat)))
            last = min(steps, max(first + 1, round((note.end - start) * 4 / midi.ticks_per_beat)))
            for step in range(first, last):
                cells[step].append((note.pitch, note.velocity, note_id))
        cells = [sorted(cell, reverse=True) for cell in cells]
        for voice, (index, level) in enumerate(allocation):
            if index != track_index:
                continue
            programs[voice] = instrument.program
            previous = -1
            for step, cell in enumerate(cells):
                if level >= len(cell):
                    previous = -1
                    continue
                note_pitch, loudness, note_id = cell[level]
                pitch[step, voice] = note_pitch
                velocity[step, voice] = 0.5 + loudness / 254 if note_id != previous else 0.0
                previous = note_id
    return {
        "pitch": torch.from_numpy(pitch),
        "velocity": torch.from_numpy(velocity),
        "program": torch.from_numpy(programs),
    }


def rolls_to_midi(
    pitch: Tensor, velocity: Tensor, programs: Tensor, *, tempo: float = 120
) -> miditoolkit.MidiFile:
    p, v, instruments = pitch.cpu().numpy(), velocity.cpu().numpy(), programs.cpu().numpy()
    midi = miditoolkit.MidiFile(ticks_per_beat=480)
    midi.tempo_changes = [miditoolkit.TempoChange(tempo, 0)]
    midi.time_signature_changes = [miditoolkit.TimeSignature(4, 4, 0)]
    for voice in range(p.shape[1]):
        program = int(instruments[voice])
        if program == 128:
            continue
        instrument = miditoolkit.Instrument(program=program)
        active: tuple[int, int, int] | None = None
        for step in range(len(p) + 1):
            current = int(p[step, voice]) if step < len(p) else 128
            onset = step < len(p) and float(v[step, voice]) >= 0.5
            if active is not None and (current != active[0] or onset):
                instrument.notes.append(
                    miditoolkit.Note(active[2], active[0], active[1] * 120, step * 120)
                )
                active = None
            if current < 128 and active is None:
                value = float(v[step, voice])
                loudness = max(1, min(127, round((value - 0.5) * 254))) if onset else 80
                active = (current, step, loudness)
        if instrument.notes:
            midi.instruments.append(instrument)
    return midi


class MIDIRollDataset(Dataset[dict[str, Tensor]]):
    def __init__(self, frame: pd.DataFrame, *, voices: int = 4) -> None:
        self.frame, self.voices = frame.reset_index(drop=True), voices
        self.cache: OrderedDict[str, miditoolkit.MidiFile] = OrderedDict()

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        row = self.frame.iloc[index]
        path = str(row.source_midi_path)
        if path not in self.cache:
            self.cache[path] = load_midi(Path(path))
            if len(self.cache) > 4:
                self.cache.popitem(last=False)
        self.cache.move_to_end(path)
        result = midi_to_rolls(
            self.cache[path],
            start_bar=int(row.start_bar),
            num_bars=int(row.num_bars),
            voices=self.voices,
        )
        result["style_id"] = torch.tensor(int(row.style_id))
        return result


class MIDIVAE(nn.Module):
    representation_kind = "style_content_posterior_not_gaussian_noise"

    def __init__(
        self,
        *,
        num_styles: int,
        hidden_dim: int = 256,
        code_dim: int = 256,
        voices: int = 4,
        num_bars: int = 8,
        beta: float = 0.001,
        style_weight: float = 0.1,
    ) -> None:
        super().__init__()
        if code_dim <= num_styles:
            raise ValueError("MIDI-VAE needs non-style posterior coordinates")
        self.num_styles, self.voices, self.steps = num_styles, voices, num_bars * 16
        self.beta, self.style_weight = beta, style_weight
        self.pitch_embedding = nn.Embedding(129, 32)
        self.program_embedding = nn.Embedding(129, 16)
        self.pitch_encoder = nn.GRU(32, hidden_dim, 2, batch_first=True)
        self.velocity_encoder = nn.GRU(1, hidden_dim, batch_first=True)
        self.program_encoder = nn.GRU(16, hidden_dim, batch_first=True)
        self.posterior = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, code_dim * 2)
        )
        self.decoder_seed = nn.Linear(code_dim, hidden_dim)
        self.position = nn.Parameter(torch.randn(1, self.steps * voices, hidden_dim) * 0.02)
        self.pitch_decoder = nn.GRU(hidden_dim, hidden_dim, 2, batch_first=True)
        self.velocity_decoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.program_decoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.pitch_head = nn.Linear(hidden_dim, 129)
        self.velocity_head = nn.Linear(hidden_dim, 1)
        self.program_head = nn.Linear(hidden_dim, 129)

    def encode(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        pitch = batch["pitch"].flatten(1)
        _, p = self.pitch_encoder(self.pitch_embedding(pitch))
        _, v = self.velocity_encoder(batch["velocity"].flatten(1)[:, :, None])
        _, i = self.program_encoder(self.program_embedding(batch["program"]))
        mean, logvar = self.posterior(torch.cat([p[-1], v[-1], i[-1]], -1)).chunk(2, -1)
        return mean, logvar.clamp(-12, 8)

    def decode(self, code: Tensor) -> dict[str, Tensor]:
        inputs = torch.tanh(self.decoder_seed(code))[:, None] + self.position
        p, _ = self.pitch_decoder(inputs)
        v, _ = self.velocity_decoder(inputs)
        i, _ = self.program_decoder(inputs[:, : self.voices])
        return {
            "pitch_logits": self.pitch_head(p).reshape(-1, self.steps, self.voices, 129),
            "velocity": self.velocity_head(v).sigmoid().reshape(-1, self.steps, self.voices),
            "program_logits": self.program_head(i),
        }

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        mean, logvar = self.encode(batch)
        code = mean + torch.randn_like(mean) * (0.5 * logvar).exp() if self.training else mean
        prediction = self.decode(code)
        pitch = F.cross_entropy(prediction["pitch_logits"].flatten(0, 2), batch["pitch"].flatten())
        velocity = F.mse_loss(prediction["velocity"], batch["velocity"])
        program = F.cross_entropy(
            prediction["program_logits"].flatten(0, 1), batch["program"].flatten()
        )
        style = F.cross_entropy(mean[:, : self.num_styles], batch["style_id"])
        kl = gaussian_kl(mean, logvar)
        return {
            "loss": pitch + program + 0.1 * velocity + self.style_weight * style + self.beta * kl,
            "pitch_loss": pitch,
            "program_loss": program,
            "velocity_loss": velocity,
            "style_loss": style,
            "kl_loss": kl,
        }

    def transfer(self, batch: dict[str, Tensor], target: Tensor) -> dict[str, Tensor]:
        mean, _ = self.encode(batch)
        return self.decode(swap_style(mean, batch["style_id"], target, self.num_styles))

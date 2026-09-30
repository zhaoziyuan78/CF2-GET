"""CLaMP3 C2 MIDI/text evaluation using the official model and global pooling."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import mido
import numpy as np

from cfmusic.progress import track

CLAMP3_WEIGHT_FILENAMES = (
    "weights_clamp3_c2_h_size_768_t_model_FacebookAI_xlm-roberta-base_t_length_128_"
    "a_size_768_a_layers_12_a_length_128_s_size_768_s_layers_12_p_size_64_p_length_512.pth",
)
EMBEDDING_SCHEMA = "cfmusic.clamp3-c2.global.v1"


def gpu_worker_environment(rank: int, num_gpus: int) -> dict[str, str]:
    """Respect an existing GPU allocation and keep HIP/CUDA masks consistent on AMD."""
    if not 0 <= rank < num_gpus:
        raise ValueError("GPU rank must be within a positive worker count")
    environment = os.environ.copy()
    visible = environment.get("HIP_VISIBLE_DEVICES", environment.get("CUDA_VISIBLE_DEVICES"))
    devices = [value.strip() for value in visible.split(",") if value.strip()] if visible else []
    if visible is not None and len(devices) < num_gpus:
        raise ValueError(f"Requested {num_gpus} GPUs but allocation exposes {devices}")
    selected = devices[rank] if devices else str(rank)
    environment["HIP_VISIBLE_DEVICES"] = selected
    environment["CUDA_VISIBLE_DEVICES"] = selected
    return environment


def embedding_protocol(repository: Path) -> dict[str, object]:
    """Identify installed weights and pooling without rehashing a multi-GB file."""
    installation_path = repository / "cfmusic_installation.json"
    if not installation_path.is_file():
        raise FileNotFoundError(
            "CLaMP3 is not configured; run uv run python scripts/setup_clamp3.py"
        )
    installation = json.loads(installation_path.read_text(encoding="utf-8"))
    if (
        installation.get("model") != "sander-wood/clamp3"
        or installation.get("variant") != "c2"
        or installation.get("weight_filename") != CLAMP3_WEIGHT_FILENAMES[0]
    ):
        raise ValueError("This project requires the official CLaMP3 C2 symbolic release")
    weight = repository / "code" / CLAMP3_WEIGHT_FILENAMES[0]
    stat = weight.stat()
    if stat.st_size != installation["weight_bytes"]:
        raise ValueError(f"Incomplete CLaMP3 checkpoint: {weight}")
    return {
        "schema": EMBEDDING_SCHEMA,
        "model": installation["model"],
        "variant": "c2",
        "model_revision": installation["model_revision"],
        "source_revision": installation["source_revision"],
        "text_revision": installation["text_revision"],
        "weight_filename": weight.name,
        "weight_bytes": stat.st_size,
        "weight_mtime_ns": stat.st_mtime_ns,
        "pooling": "official_global_then_length_weighted_mean_then_l2",
        "dimension": 768,
        "dtype": "float32",
        "mtf": "m3_compatible_no_text_or_sysex",
    }


def read_embedding_protocol(path: Path) -> dict[str, object]:
    """Read provenance from an embedding sidecar, rejecting unidentified old vectors."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    model = payload.get("embedding_model")
    if not isinstance(model, dict) or model.get("schema") != EMBEDDING_SCHEMA:
        raise ValueError(f"Not a versioned CLaMP3 C2 embedding cache: {path}")
    return model


def style_prompt(label: str, template: str = "This is a piece of {style} music.") -> str:
    if "{style}" not in template:
        raise ValueError("CLaMP 3 style template must contain {style}")
    return template.format(style=label.replace("_", " "))


def midi_to_mtf(source: Path, destination: Path, *, preserve_metadata_timing: bool = False) -> None:
    """Convert MIDI to M3-compatible MTF, excluding textual metadata and SysEx."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        midi_to_mtf_text(source, preserve_metadata_timing=preserve_metadata_timing),
        encoding="utf-8",
    )


def midi_to_mtf_text(source: Path, *, preserve_metadata_timing: bool = False) -> str:
    """In-memory form for large jobs: never persist redundant intermediate MTF files."""

    midi = mido.MidiFile(source)
    lines = [f"ticks_per_beat {midi.ticks_per_beat}"]
    ignored_meta = {
        "text",
        "copyright",
        "track_name",
        "instrument_name",
        "lyrics",
        "marker",
        "cue_marker",
        "device_name",
        "sequencer_specific",
    }
    pending = 0
    for message in midi.merged_track:
        if preserve_metadata_timing:
            pending += message.time
        if (message.is_meta and message.type in ignored_meta) or message.type == "sysex":
            continue
        if preserve_metadata_timing:
            message = message.copy(time=pending)
            pending = 0
        encoded = " ".join(str(value) for value in message.dict().values())
        lines.append(encoded.encode("unicode_escape").decode("utf-8"))
    return "\n".join(lines)


def _normalized_feature(path: Path) -> np.ndarray:
    feature = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(feature)
    if feature.shape != (768,) or not np.isfinite(norm) or norm <= 0:
        raise ValueError(f"CLaMP 3 produced an invalid embedding: {path}")
    return feature / norm


def extract_clamp3_embeddings(
    *,
    repository: Path,
    midi_files: Mapping[str, Path],
    texts: Mapping[str, str],
    python_executable: str | None = None,
    cache_dir: Path | None = None,
    batch_size: int = 4,
    device: str = "auto",
    midi_view_policy: str = "all",
    preserve_metadata_timing: bool = False,
) -> dict[str, np.ndarray]:
    """Extract MIDI/text jointly in an offline worker using the official CLaMP3 model."""
    repository = repository.resolve()
    embedding_protocol(repository)
    keys = [*midi_files, *texts]
    if len(keys) != len(set(keys)) or any(
        not key or key in {".", ".."} or "/" in key or "\\" in key for key in keys
    ):
        raise ValueError("Embedding keys must be unique plain filenames")
    if not keys:
        return {}

    with tempfile.TemporaryDirectory(prefix="cfmusic-clamp3-") as temporary:
        root = Path(temporary)
        inputs = root / "inputs"
        outputs = root / "outputs"
        inputs.mkdir()
        for key, source in track(
            midi_files.items(),
            description="Convert MIDI to CLaMP 3 MTF",
            total=len(midi_files),
            unit="midi",
        ):
            midi_to_mtf(
                source, inputs / f"{key}.mtf", preserve_metadata_timing=preserve_metadata_timing
            )
        for key, value in texts.items():
            (inputs / f"{key}.txt").write_text(value, encoding="utf-8")
        environment = os.environ.copy()
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            environment["HF_HOME"] = str(cache_dir)
        subprocess.run(
            [
                python_executable or sys.executable,
                "-m",
                "cfmusic.commands.clamp3_batched_worker",
                "--repository",
                str(repository),
                "--input-dir",
                str(inputs),
                "--output-dir",
                str(outputs),
                "--batch-size",
                str(batch_size),
                "--device",
                device,
                "--midi-view-policy",
                midi_view_policy,
            ],
            cwd=repository / "code",
            env=environment,
            check=True,
        )
        expected = [*midi_files, *texts]
        missing = [key for key in expected if not (outputs / f"{key}.npy").is_file()]
        if missing:
            raise RuntimeError(f"CLaMP 3 did not produce embeddings for: {missing[:10]}")
        return {key: _normalized_feature(outputs / f"{key}.npy") for key in expected}


def clamp3_transport_distance(
    source_embedding: np.ndarray,
    generated_embedding: np.ndarray,
    *,
    style_embeddings: Sequence[np.ndarray],
    source_style_id: int,
    target_style_id: int,
) -> float:
    """Signed projection (z_cf-z_source) onto the fixed unit (c_target-c_source).

    Max-member classification has no single class vector. For this metric only,
    a class is the normalized mean of its normalized member prototypes. The
    direction is fixed before observing the generated MIDI, not re-selected to
    maximize its displacement. These are CLaMP vectors, not BEAT VAE latents.
    """

    def unit(value: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(value)
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("Transport distance requires finite non-zero embeddings")
        return value / norm

    def center(index: int) -> np.ndarray:
        members = np.atleast_2d(np.asarray(style_embeddings[index], dtype=np.float64))
        return unit(np.stack([unit(member) for member in members]).mean(0))

    if not (
        0 <= source_style_id < len(style_embeddings)
        and 0 <= target_style_id < len(style_embeddings)
    ):
        raise ValueError("Transport distance style id is outside the active vocabulary")
    if source_style_id == target_style_id:
        return 0.0
    direction = unit(center(target_style_id) - center(source_style_id))
    return float((unit(generated_embedding) - unit(source_embedding)) @ direction)


def clamp3_style_metrics(
    music_embedding: np.ndarray,
    *,
    style_embeddings: Sequence[np.ndarray],
    source_style_id: int,
    target_style_id: int,
    logit_bias: Sequence[float] | None = None,
) -> dict[str, float]:
    similarities = np.asarray(
        [
            float(np.max(np.asarray(style_embedding) @ music_embedding))
            for style_embedding in style_embeddings
        ]
    )
    if not (0 <= target_style_id < len(similarities) and 0 <= source_style_id < len(similarities)):
        raise ValueError("Artifact style id is outside the CLaMP 3 text vocabulary")
    decision_scores = similarities
    if logit_bias is not None:
        bias = np.asarray(logit_bias, dtype=np.float64)
        if bias.shape != similarities.shape or not np.isfinite(bias).all():
            raise ValueError("CLaMP 3 logit bias must contain one finite value per style")
        decision_scores = similarities + bias
    metrics = {
        "clamp3_target_similarity": float(similarities[target_style_id]),
        "clamp3_source_similarity": float(similarities[source_style_id]),
        "clamp3_target_minus_source": float(
            similarities[target_style_id] - similarities[source_style_id]
        ),
        "clamp3_target_style_success": float(int(decision_scores.argmax()) == target_style_id),
    }
    if logit_bias is not None:
        metrics["clamp3_calibrated_target_minus_source"] = float(
            decision_scores[target_style_id] - decision_scores[source_style_id]
        )
    return metrics

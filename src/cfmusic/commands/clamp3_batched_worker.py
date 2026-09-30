"""Resumable CLaMP3 C2 MIDI/text embedding worker (ROCm, CUDA, or CPU)."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer, BertConfig

from cfmusic.evaluation.clamp3 import (
    CLAMP3_WEIGHT_FILENAMES,
    _normalized_feature,
    embedding_protocol,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--file-list", type=Path)
    inputs.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--midi-view-policy", choices=["all", "three"], default="all")
    return parser.parse_args()


def _segments(patches: torch.Tensor, *, patch_length: int) -> tuple[list[torch.Tensor], list[int]]:
    """Official overlapping final chunk and length weights, for text or MIDI."""
    if patches.ndim not in (1, 2) or len(patches) == 0 or patch_length <= 0:
        raise ValueError(f"Expected non-empty token/patch input, got {patches.shape}")
    chunks = [patches[i : i + patch_length] for i in range(0, len(patches), patch_length)]
    chunks[-1] = patches[-patch_length:]
    full, remainder = divmod(len(patches), patch_length)
    return chunks, [patch_length] * full + ([remainder] if remainder else [])


def _load_official_module(repository: Path) -> Any:
    """Import the pinned model/patchilizer without allocating pretrained weights."""
    modules: dict[str, Any] = {}
    previous_config = sys.modules.get("config")
    try:
        for name in ("config", "utils"):
            spec = importlib.util.spec_from_file_location(
                f"_cfmusic_clamp3_{name}", repository / "code" / f"{name}.py"
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"Missing official CLaMP3 module: {name}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            modules[name] = module
            if name == "config":
                sys.modules["config"] = module
    finally:
        if previous_config is None:
            sys.modules.pop("config", None)
        else:
            sys.modules["config"] = previous_config
    return modules["utils"]


def _load_official_model(repository: Path, device: str = "auto") -> tuple[Any, Any, Any]:
    """Load all official C2 weights strictly; base weights would only be overwritten."""
    embedding_protocol(repository)
    installation = json.loads((repository / "cfmusic_installation.json").read_text())
    text_dir = installation["text_model_dir"]
    official = _load_official_module(repository)
    text_config = AutoConfig.from_pretrained(text_dir, local_files_only=True)
    # This isolated module gets a config-only text constructor. All model forward
    # methods remain official; strict loading supplies fine-tuned text weights.
    official.AutoModel = SimpleNamespace(
        from_pretrained=lambda _name: AutoModel.from_config(
            text_config, attn_implementation="eager"
        )
    )

    def bert_config(length: int) -> BertConfig:
        return BertConfig(
            vocab_size=1,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            max_position_embeddings=length,
            attn_implementation="eager",
        )

    model = official.CLaMP3Model(
        audio_config=bert_config(128),
        symbolic_config=bert_config(512),
        text_model_name=text_dir,
        hidden_size=768,
        load_m3=False,
    )
    checkpoint = torch.load(
        repository / "code" / CLAMP3_WEIGHT_FILENAMES[0],
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    target = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == "auto"
        else torch.device(device)
    )
    model.to(target).eval().requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(text_dir, local_files_only=True)
    print(
        f"Loaded official CLaMP3 C2 on {target}; {sum(p.numel() for p in model.parameters()):,} parameters"
    )
    return model, official.M3Patchilizer(), tokenizer


def _entry_path(entry: dict[str, str]) -> Path:
    return Path(entry.get("mtf_path") or entry["text_path"])


def _cache_protocol(
    entries: list[dict[str, str]], repository: Path, output_dir: Path, midi_view_policy: str = "all"
) -> None:
    """Reject cache reuse across different models, inputs, or pooling methods."""
    keys = [entry["key"] for entry in entries]
    if len(set(keys)) != len(keys) or any(
        not key or key in {".", ".."} or "/" in key or "\\" in key for key in keys
    ):
        raise ValueError("Embedding keys must be unique plain filenames")
    inputs = {}
    for entry in entries:
        path = _entry_path(entry).resolve()
        stat = path.stat()
        inputs[entry["key"]] = [str(path), stat.st_size, stat.st_mtime_ns]
    protocol = {
        "embedding_model": embedding_protocol(repository),
        "inputs": inputs,
        "midi_view_policy": midi_view_policy,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    sidecar = output_dir / "embedding_protocol.json"
    if sidecar.exists():
        if json.loads(sidecar.read_text()) != protocol:
            raise ValueError(
                f"Incompatible CLaMP3 embedding cache; use a new output directory: {output_dir}"
            )
    else:
        if any(output_dir.glob("*.npy")):
            raise ValueError(
                f"Unversioned embedding cache; use a new output directory: {output_dir}"
            )
        temporary = sidecar.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(protocol, indent=2) + "\n")
        temporary.replace(sidecar)


def _encode_file_batch(
    entries: list[dict[str, str]],
    *,
    model: Any,
    patchilizer: Any,
    tokenizer: Any,
    batch_size: int,
    output_dir: Path,
    midi_view_policy: str = "all",
) -> None:
    feature_sums: dict[str, np.ndarray] = {}
    weight_sums: dict[str, int] = defaultdict(int)
    for modality in ("mtf_path", "text_path"):
        length = 512 if modality == "mtf_path" else 128
        pad_id = patchilizer.pad_token_id if modality == "mtf_path" else tokenizer.pad_token_id
        pending: list[tuple[str, torch.Tensor, int]] = []

        def flush(
            pending: list[tuple[str, torch.Tensor, int]] = pending,
            length: int = length,
            modality: str = modality,
            pad_id: int = pad_id,
        ) -> None:
            if not pending:
                return
            shape = (len(pending), length, 64) if modality == "mtf_path" else (len(pending), length)
            inputs = torch.full(shape, int(pad_id), dtype=torch.long)
            masks = torch.zeros((len(pending), length), dtype=torch.long)
            for index, (_, segment, _) in enumerate(pending):
                inputs[index, : len(segment)] = segment
                masks[index, : len(segment)] = 1
            with torch.inference_mode():
                if modality == "mtf_path":
                    features = model.get_symbolic_features(
                        inputs.to(model.device), masks.to(model.device), get_global=True
                    )
                else:
                    features = model.get_text_features(
                        inputs.to(model.device), masks.to(model.device), get_global=True
                    )
            # CLaMP3 projects raw pooled states. Do not normalize individual chunks!
            for (key, _, weight), feature in zip(
                pending, features.float().cpu().numpy(), strict=True
            ):
                if key not in feature_sums:
                    feature_sums[key] = np.zeros(768, dtype=np.float64)
                feature_sums[key] += feature.astype(np.float64) * weight
                weight_sums[key] += weight
            pending.clear()

        for entry in entries:
            if modality not in entry:
                continue
            text = Path(entry[modality]).read_text(encoding="utf-8")
            if modality == "mtf_path":
                tokens = torch.tensor(
                    patchilizer.encode(text, add_special_patches=True), dtype=torch.long
                )
            else:
                # Upstream uses an unordered set. Stable first-seen ordering keeps
                # equivalent line deduplication deterministic for multi-line prompts.
                text = tokenizer.sep_token.join(
                    dict.fromkeys(line for line in text.splitlines() if line)
                )
                tokens = tokenizer(text, return_tensors="pt")["input_ids"].squeeze(0)
            if modality == "mtf_path" and midi_view_policy == "three":
                starts = sorted({0, max(0, (len(tokens) - 512) // 2), max(0, len(tokens) - 512)})
                segments = [tokens[start : start + 512] for start in starts]
                weights = [len(segment) for segment in segments]
            else:
                segments, weights = _segments(tokens, patch_length=length)
            for segment, weight in zip(segments, weights, strict=True):
                pending.append((entry["key"], segment, weight))
                if len(pending) == batch_size:
                    flush()
        flush()
    output_dir.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        key = entry["key"]
        feature = feature_sums[key] / weight_sums[key]
        norm = float(np.linalg.norm(feature))
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError(f"CLaMP3 produced an invalid feature: {key}")
        destination = output_dir / f"{key}.npy"
        temporary = destination.with_suffix(".npy.tmp")
        with temporary.open("wb") as stream:
            np.save(stream, (feature / norm).astype(np.float32), allow_pickle=False)
        temporary.replace(destination)


def main() -> None:
    args = _parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.file_list is not None:
        entries = [
            json.loads(line) for line in args.file_list.read_text().splitlines() if line.strip()
        ]
    else:
        entries = [
            {
                "key": path.stem,
                "mtf_path" if path.suffix == ".mtf" else "text_path": str(path.resolve()),
            }
            for path in sorted(args.input_dir.iterdir())
            if path.suffix in {".mtf", ".txt"}
        ]
    repository = args.repository.resolve()
    _cache_protocol(entries, repository, args.output_dir, args.midi_view_policy)
    pending = []
    for entry in entries:
        path = args.output_dir / f"{entry['key']}.npy"
        if path.is_file():
            _normalized_feature(path)
        else:
            pending.append(entry)
    if not pending:
        print(f"All {len(entries)} CLaMP3 embeddings validated and available")
        return
    model, patchilizer, tokenizer = _load_official_model(repository, args.device)
    for start in tqdm(range(0, len(pending), args.batch_size), desc="CLaMP3 files"):
        _encode_file_batch(
            pending[start : start + args.batch_size],
            model=model,
            patchilizer=patchilizer,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            midi_view_policy=args.midi_view_policy,
        )


if __name__ == "__main__":
    main()

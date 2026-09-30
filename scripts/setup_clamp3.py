"""Install the pinned official CLaMP3 C2 release without CUDA or redundant base weights."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from huggingface_hub import hf_hub_download

SOURCE_REVISION = "9016d2b0c8d12d1aa79c2e0ab201e6822bdc83a8"
MODEL_REVISION = "355625cc1c6f73726bbcd0eb9276ac7152d56426"
TEXT_REVISION = "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089"
WEIGHT_FILENAME = (
    "weights_clamp3_c2_h_size_768_t_model_FacebookAI_xlm-roberta-base_t_length_128_"
    "a_size_768_a_layers_12_a_length_128_s_size_768_s_layers_12_p_size_64_p_length_512.pth"
)
WEIGHT_BYTES = 2_518_324_732


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--repository", type=Path, default=root / "external" / "clamp3")
    parser.add_argument("--model-dir", type=Path, default=root / "checkpoints/external/clamp3")
    args = parser.parse_args()
    repository = args.repository.resolve()
    model_dir = args.model_dir.resolve()
    if not repository.exists():
        subprocess.run(
            [
                "git",
                "clone",
                "--no-checkout",
                "https://github.com/sanderwood/clamp3.git",
                str(repository),
            ],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(repository), "checkout", "--detach", SOURCE_REVISION], check=True
        )
    revision = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != SOURCE_REVISION:
        raise ValueError(f"Expected official source {SOURCE_REVISION}, found {revision}")
    model_dir.mkdir(parents=True, exist_ok=True)
    weight = Path(
        hf_hub_download(
            "sander-wood/clamp3", WEIGHT_FILENAME, revision=MODEL_REVISION, local_dir=model_dir
        )
    )
    if weight.stat().st_size != WEIGHT_BYTES:
        raise ValueError(f"Incomplete CLaMP3 checkpoint: {weight}")
    text_dir = model_dir / "xlm-roberta-base"
    for filename in (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "sentencepiece.bpe.model",
    ):
        hf_hub_download(
            "FacebookAI/xlm-roberta-base", filename, revision=TEXT_REVISION, local_dir=text_dir
        )
    link = repository / "code" / WEIGHT_FILENAME
    if not link.exists():
        link.symlink_to(weight)
    elif link.resolve() != weight.resolve():
        raise FileExistsError(f"Refusing to replace a different checkpoint: {link}")
    installation = {
        "model": "sander-wood/clamp3",
        "variant": "c2",
        "model_revision": MODEL_REVISION,
        "source_revision": SOURCE_REVISION,
        "weight_filename": WEIGHT_FILENAME,
        "weight_bytes": WEIGHT_BYTES,
        "text_revision": TEXT_REVISION,
        "text_model_dir": str(text_dir),
        "embedding_dimension": 768,
        "inference": "cfmusic.commands.clamp3_batched_worker",
    }
    destination = repository / "cfmusic_installation.json"
    destination.write_text(json.dumps(installation, indent=2) + "\n", encoding="utf-8")
    print(f"CLaMP3 C2 installed: {destination}")
    print(
        "Full CLaMP3 checkpoint includes text and symbolic weights; no base M3/XLM-R weights needed."
    )


if __name__ == "__main__":
    main()

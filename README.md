# CF²-GET

CF²-GET (Conditional Flow for Counterfactual Symbolic Music Genre Transfer) treats genre transfer as counterfactual inference. A 16-bar MIDI passage is tokenized with BEAT and encoded into a latent by a frozen Transformer VAE. A single genre-conditioned flow, trained with optimal-transport flow matching, a round-trip loss and an HSIC independence term, is run backward under the source genre to recover the passage's exogenous noise (abduction). The genre is then set to the target (intervention), and the same flow is run forward from that noise and decoded back to MIDI (prediction).

## Installation

```bash
# Python >= 3.11; install the PyTorch build that matches your CUDA or ROCm driver first
pip install torch
pip install -e .
python scripts/setup_clamp3.py
```

## Usage

```bash
# Full XMIDI pipeline: data, codec, latent cache, CLaMP3 labels, all models and evaluation
ACCEPT_XMIDI_LICENSE=true bash scripts/run_xmidi_16bar.sh

# CF²-GET only, on 4 GPUs
METHODS=otcfm_hsic GPUS=4 ACCEPT_XMIDI_LICENSE=true bash scripts/run_xmidi_16bar.sh

# Print every command without running it
bash scripts/run_xmidi_16bar.sh --dry-run
```

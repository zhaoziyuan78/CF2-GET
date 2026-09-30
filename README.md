# CF2-GET: CONDITIONAL FLOW FOR COUNTERFACTUAL SYMBOLIC MUSIC GENRE TRANSFER

CF2-GET performs **counterfactual genre transfer** of symbolic music. A MIDI segment is
encoded into a continuous latent, the style-independent exogenous noise is *abducted* by inverting
a genre-conditional flow, and the same noise is decoded again under a different genre:
$$
z = E_\psi(x),\qquad u = F_\theta^{-1}(z;\,s),\qquad
z^{cf} = F_\theta(u;\,s'),\qquad x^{cf} = D_\psi(z^{cf}).
$$

This repository contains the complete pipeline for the XMIDI experiments: data preparation,
the BEAT tokenizer, CLaMP3 genre labels, the OT-CFM + HSIC transport and all baselines, the paired
test protocol, and the evaluation.

## Method in brief

| Component | Setting |
|---|---|
| Representation | Deterministic BEAT tokenizer (593 tokens, 4 steps/beat), velocity and all 128 programs kept |
| Codec | Stage 1: 8-bar Transformer VAE (12+12 layers, d=768, 64×512 latent). Stage 2: 16-bar chunk VAE (two 8-bar chunks sharing the stage-1 weights, 128×512 latent), fine-tuned for 1,000 steps |
| Data | XMIDI, 16-bar windows with hop 8 and a content-quality filter; six CLaMP3 genres (classical, jazz, rock, pop, country, latin); training windows below their genre's 30th margin percentile are dropped (validation/test untouched) |
| Transport | Latent DiT vector field (10 layers, d=768, AdaLN) trained with conditional flow matching; noise is coupled to data by Hungarian OT inside each genre minibatch |
| Regularizers | Sparse round-trip loss; warm-up HSIC between 4-step abductions and genre labels |
| Inference | 32-step Heun solver for abduction, reconstruction and target prediction |

Baselines trained or run by the same pipeline: OT-CFM without HSIC (ablation), DDIM Inversion
(conditional DDIM on the same DiT backbone with deterministic inversion), Latent-MIDI-VAE
(style-coordinate swap on the frozen codec latents), and a direct 16-bar piano-roll MIDI-VAE.

## Installation

Python 3.11+ is required. Install the PyTorch build that matches your GPU driver first, then the
package:

```bash
# e.g. CUDA:  pip install torch --index-url https://download.pytorch.org/whl/cu124
# e.g. ROCm:  pip install torch --index-url https://download.pytorch.org/whl/rocm6.2
pip install -e ".[dev]"
```

CLaMP3 (official C2 release, pinned revisions) is needed for genre labels and for evaluation:

```bash
python scripts/setup_clamp3.py        # -> external/clamp3 and checkpoints/external/clamp3
```

The published results were produced on 8× AMD Instinct MI210 (ROCm 7.2, PyTorch 2.11). All
training commands support one GPU or multi-GPU DDP through `torchrun`; batch sizes in the configs
are per GPU, and the transport recipe assumes 8 GPUs (global batch 768).

## Data

XMIDI is downloaded from the authors' Google Drive release
([project page](https://github.com/xmusic-project/XMIDI_Dataset)). 

## Reproducing the full experiment

```bash
bash scripts/run_xmidi_16bar.sh --dry-run          # print every command
ACCEPT_XMIDI_LICENSE=true bash scripts/run_xmidi_16bar.sh
```

The script runs the stages below in order. Every stage is resumable: it is skipped when its output
exists and it previously completed with exactly the same command (state in
`reports/pipeline_state/`). Paths default to the repository and can be moved with
`CFMUSIC_DATA_ROOT` (data, cache, labels) and `CFMUSIC_OUTPUT_ROOT` (checkpoints, artifacts,
reports, runs). `METHODS` selects the models, `GPUS` the number of GPUs.

| Stage | Command | Output |
|---|---|---|
| download, prepare | `cfmusic.commands.download`, `cfmusic.commands.prepare data=xmidi` | 8-bar/hop-4 manifest; exact duplicates grouped and song splits leakage-safe (seed 42) |
| codec_8bar | `train_codec experiment=xmidi_codec_8bar` | stage-1 codec (8 epochs, global batch 128) |
| segments | `scripts/prepare_xmidi_16bar.py segments` | 16-bar windows: ≥64 note onsets, ≥12 active bars, ≥60% active beats, ≤8 silent beats, ≤5,120 BEAT tokens (never truncated) |
| codec_16bar | `train_codec experiment=xmidi_codec_16bar` | stage-2 chunk VAE initialized from stage 1 |
| cache | `cache_latents experiment=xmidi_codec_16bar` | fp16 posterior means with per-token/channel normalization |
| label_* | `scripts/prepare_xmidi_16bar.py label-prepare / label-worker / label-publish` | CLaMP3 six-genre labels, q30 training overlay, train-song genre distribution |
| train_\<method\> | `train_transport` or `train_baseline` | 50k-step transports (EMA); VAEs with validation early stopping (best checkpoint) |
| generate / evaluate / temporal / leakage | `generate_counterfactuals` (`generate_midi_vae`), `evaluate`, `evaluate_temporal`, `evaluate_factual_leakage` | paired test artifacts and metrics |
| report | `build_report` | `reports/main_results.md`, `all_metrics.csv`, per-pair and leakage tables |

Individual stages can also be run directly with Hydra overrides, for example:

```bash
torchrun --standalone --nproc_per_node=8 -m cfmusic.commands.train_transport \
    experiment=xmidi_otcfm_hsic experiment.name=otcfm_hsic \
    paths.processed_dir=data/processed_16bar paths.latent_dir=data/latents_16bar resume=true
torchrun --standalone --nproc_per_node=8 -m cfmusic.commands.generate_counterfactuals \
    experiment=xmidi_otcfm_hsic experiment.name=otcfm_hsic_test \
    paths.processed_dir=data/processed_16bar paths.latent_dir=data/latents_16bar \
    codec_checkpoint=checkpoints/xmidi_codec_16bar/codec/xmidi/last.pt \
    transport_checkpoint=checkpoints/otcfm_hsic/transport/last.pt \
    transport.prediction_guidance_scale=4.0
```

## Evaluation

All methods are evaluated on the same test sources and targets; failed generations stay in
every denominator.

* **Target style**: CLaMP3 C2 success (the generated MIDI's highest-scoring genre is the target)
  and transport distance (signed projection of the CLaMP3 embedding change, generated minus
  source, onto the unit source-to-target genre direction).
* **Content preservation**: pitch-class histogram cosine, melody contour cosine, and 64-beat
  chroma cosine over the whole segment.
* **Structure**: MIDI validity, duration, and decode completion (EOS within the token budget with
  all 16 bars).
* **Abduction**: latent round-trip MSE and linear/MLP probes of genre from the abducted noise on
  1,200 independent test songs, with Gaussian controls.

If you use this code, please cite the paper (see `CITATION.cff`).

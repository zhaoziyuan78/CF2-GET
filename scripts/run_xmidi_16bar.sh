#!/usr/bin/env bash
# End-to-end XMIDI 16-bar pipeline:
#   data -> 8-bar codec -> 16-bar windows -> 16-bar chunk VAE -> latent cache -> CLaMP3 labels
#   -> OT-CFM+HSIC and baselines -> paired test (2,000 sources) -> evaluation -> report.
#
# Finished stages are skipped on reruns (markers and logs in $CFMUSIC_OUTPUT_ROOT/logs).
# Run `bash scripts/run_xmidi_16bar.sh --dry-run` to print the commands.
# Environment variables (defaults in brackets):
#   GPUS [8]                          GPUs per torchrun job (the recipe assumes 8)
#   CFMUSIC_DATA_ROOT [<repo>/data]   raw/processed data, latent cache, labels
#   CFMUSIC_OUTPUT_ROOT [<repo>]      checkpoints/, artifacts/, reports/, runs/, logs/
#   CLAMP3_REPOSITORY [<repo>/external/clamp3]   installed by scripts/setup_clamp3.py
#   ACCEPT_XMIDI_LICENSE [false]      set to true after reviewing the XMIDI terms
#   METHODS [otcfm_hsic otcfm_no_hsic ddim ddim_cfg latent_midi_vae midi_vae]
#   PYTHON [python], TORCHRUN [torchrun], CPU_WORKERS [8]
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
GPUS="${GPUS:-8}"
DATA="$(realpath -m "${CFMUSIC_DATA_ROOT:-$ROOT/data}")"
OUT="$(realpath -m "${CFMUSIC_OUTPUT_ROOT:-$ROOT}")"
CLAMP3="$(realpath -m "${CLAMP3_REPOSITORY:-$ROOT/external/clamp3}")"
METHODS=(${METHODS:-otcfm_hsic otcfm_no_hsic ddim ddim_cfg latent_midi_vae midi_vae})
PY=("${PYTHON:-python}")
DDP=("${TORCHRUN:-torchrun}" --standalone "--nproc_per_node=$GPUS")
CPU_WORKERS="${CPU_WORKERS:-8}"
LOGS="$OUT/logs"
export CLAMP3_REPOSITORY="$CLAMP3" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
DRY_RUN=false
if (( $# )); then [[ $# == 1 && "$1" == --dry-run ]] || { echo "usage: $0 [--dry-run]" >&2; exit 2; }; DRY_RUN=true; fi
(( 128 % GPUS == 0 && 64 % GPUS == 0 )) || { echo "GPUS must divide 64 (global codec batches)" >&2; exit 2; }
[[ "$GPUS" == 8 ]] || echo "Warning: the published transport recipe uses 8 GPUs (global batch 768)." >&2
"$DRY_RUN" || mkdir -p "$LOGS"

stage() {
    local name="$1" expected="$2"
    shift 2
    printf '\n[%s]\n%s\n' "$name" "$*"
    if "$DRY_RUN"; then return; fi
    if [[ -e "$LOGS/$name.done" ]]; then
        echo "Skip finished stage $name"
        return
    fi
    "$@" 2>&1 | tee "$LOGS/$name.log"
    (( PIPESTATUS[0] == 0 )) || { echo "Stage failed: $name (log: $LOGS/$name.log)" >&2; return 1; }
    [[ -e "$expected" ]] || { echo "Missing output of $name: $expected" >&2; return 1; }
    touch "$LOGS/$name.done"
}

label_workers() {
    local rank pids=()
    for ((rank = 0; rank < GPUS; rank++)); do
        CUDA_VISIBLE_DEVICES="$rank" HIP_VISIBLE_DEVICES="$rank" "${PY[@]}" scripts/prepare_xmidi_16bar.py \
            label-worker --data-root "$DATA" --clamp3 "$CLAMP3" --rank "$rank" --workers 4 --batch-size 64 \
            > "$LOGS/label_worker_$rank.log" 2>&1 &
        pids+=("$!")
    done
    for rank in "${!pids[@]}"; do
        wait "${pids[$rank]}" || { echo "Label worker $rank failed: $LOGS/label_worker_$rank.log" >&2; return 1; }
    done
}

PATHS=("paths.project_root=$ROOT" "paths.data_root=$DATA" "paths.checkpoints_dir=$OUT/checkpoints"
    "paths.artifacts_dir=$OUT/artifacts" "paths.reports_dir=$OUT/reports" "paths.runs_dir=$OUT/runs")
P16=("${PATHS[@]}" "paths.processed_dir=$DATA/processed_16bar" "paths.latent_dir=$DATA/latents_16bar")
CODEC8="$OUT/checkpoints/xmidi_codec_8bar/codec/xmidi/last.pt"
CODEC="$OUT/checkpoints/xmidi_codec_16bar/codec/xmidi/last.pt"

# 1. Data and codec -------------------------------------------------------------------------
stage download "$DATA/raw/xmidi/download_manifest.json" "${PY[@]}" -m cfmusic.commands.download \
    'datasets=[xmidi]' "${PATHS[@]}" "license.accept=${ACCEPT_XMIDI_LICENSE:-false}" \
    "license.acknowledge_unknown=${ACCEPT_XMIDI_LICENSE:-false}"
stage prepare "$DATA/processed/xmidi/manifest.parquet" "${PY[@]}" -m cfmusic.commands.prepare \
    data=xmidi "${PATHS[@]}" "preprocessing.workers=$CPU_WORKERS"
stage codec_8bar "$CODEC8" "${DDP[@]}" -m cfmusic.commands.train_codec experiment=xmidi_codec_8bar \
    "${PATHS[@]}" "codec.training.batch_size=$((128 / GPUS))" resume=true
stage segments "$DATA/processed_16bar/xmidi/complete.json" "${PY[@]}" scripts/prepare_xmidi_16bar.py \
    segments --data-root "$DATA" --workers "$CPU_WORKERS"
stage codec_16bar "$CODEC" "${DDP[@]}" -m cfmusic.commands.train_codec experiment=xmidi_codec_16bar \
    "${P16[@]}" "codec.training.initialize_from=$CODEC8" "codec.training.batch_size=$((64 / GPUS))" resume=true
stage cache "$DATA/latents_16bar/xmidi/cache_metadata.json" "${DDP[@]}" -m cfmusic.commands.cache_latents \
    experiment=xmidi_codec_16bar "${P16[@]}" "codec_checkpoint=$CODEC"

# 2. Six-genre CLaMP3 labels and the q30 training filter ------------------------------------
stage label_prepare "$DATA/labels/xmidi_16bar/protocol.json" "${PY[@]}" scripts/prepare_xmidi_16bar.py \
    label-prepare --data-root "$DATA" --clamp3 "$CLAMP3" --gpus "$GPUS"
stage label_workers "$DATA/labels/xmidi_16bar/status-0.npy" label_workers
stage label_publish "$DATA/labels/xmidi_16bar/q30/complete.json" "${PY[@]}" scripts/prepare_xmidi_16bar.py \
    label-publish --data-root "$DATA"

# 3. Models, paired test, evaluation --------------------------------------------------------
EXPERIMENTS=()
for method in "${METHODS[@]}"; do
    recipe="xmidi_$method"
    trainer=cfmusic.commands.train_transport
    generator=cfmusic.commands.generate_counterfactuals
    checkpoint="$OUT/checkpoints/$method/transport/last.pt"
    expected="$checkpoint"
    INFER=()
    case "$method" in
        otcfm_hsic|otcfm_no_hsic) INFER=(transport.prediction_guidance_scale=4.0) ;;  # target CFG=4
        latent_midi_vae|midi_vae)
            trainer=cfmusic.commands.train_baseline
            checkpoint="$OUT/checkpoints/$method/transport/best.pt"
            expected="$OUT/checkpoints/$method/transport/training_complete.json" ;;
    esac
    [[ "$method" == midi_vae ]] && generator=cfmusic.commands.generate_midi_vae
    stage "train_$method" "$expected" "${DDP[@]}" -m "$trainer" "experiment=$recipe" \
        "experiment.name=$method" "${P16[@]}" resume=true
    TEST=("experiment=$recipe" "experiment.name=${method}_test" "${P16[@]}" "${INFER[@]}")
    WEIGHTS=("transport_checkpoint=$checkpoint")
    ART="$OUT/artifacts/${method}_test/xmidi"
    stage "generate_$method" "$ART/generation_manifest.json" "${DDP[@]}" -m "$generator" "${TEST[@]}" \
        "codec_checkpoint=$CODEC" "${WEIGHTS[@]}"
    stage "evaluate_$method" "$ART/evaluation/aggregate.json" "${PY[@]}" -m cfmusic.commands.evaluate "${TEST[@]}"
    stage "temporal_$method" "$ART/evaluation/temporal_results.csv" "${PY[@]}" -m cfmusic.commands.evaluate_temporal \
        "${TEST[@]}"
    stage "leakage_$method" "$ART/evaluation/noise_leakage.json" "${DDP[@]}" \
        -m cfmusic.commands.evaluate_factual_leakage "${TEST[@]}" "${WEIGHTS[@]}" leakage.split=test
    EXPERIMENTS+=("${method}_test")
done
LIST=$(IFS=,; printf '%s' "${EXPERIMENTS[*]}")
stage report "$OUT/reports/main_results.md" "${PY[@]}" -m cfmusic.commands.build_report \
    experiment=xmidi_otcfm_hsic "${P16[@]}" "report.experiments=[$LIST]"
"$DRY_RUN" || echo "Done: $OUT/reports/main_results.md"

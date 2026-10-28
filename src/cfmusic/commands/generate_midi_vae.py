"""Generate direct MIDI-VAE artifacts under the common held-out evaluation contract."""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf

from cfmusic.baselines.factory import baseline_model
from cfmusic.baselines.midi_vae import MIDIVAE, ROLL_SCHEMA, MIDIRollDataset, rolls_to_midi
from cfmusic.baselines.models import swap_style
from cfmusic.commands.generate_counterfactuals import (
    _safe_component,
    _selected_source_manifest,
    proportional_target_assignments,
    sampling_weights,
    select_source_indices,
)
from cfmusic.config import CONFIG_DIR, dataset_asset_path, prepare_config
from cfmusic.data.midi_io import load_midi
from cfmusic.distributed import cleanup_distributed, distributed_barrier, initialize_distributed
from cfmusic.latent.dataset import LatentDataset
from cfmusic.memory import autocast_context
from cfmusic.progress import track
from cfmusic.tokenization.factory import tokenizer_from_config


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    context = initialize_distributed()
    try:
        paths = prepare_config(cfg)
        split = str(cfg.counterfactual.split)
        if split not in {"validation", "test"}:
            raise ValueError("Direct baseline generation requires an explicit held-out split")
        dataset = LatentDataset(
            paths["latent_dir"] / str(cfg.data.name),
            index_path=Path(str(cfg.data.latent_index)),
            split=split,
        )
        card = json.loads(dataset_asset_path(cfg, paths, "dataset_card.json").read_text())
        names = card["style_vocabulary"]
        weights = sampling_weights(cfg.counterfactual, names)
        require_all = bool(cfg.counterfactual.require_all_transitions)
        if (
            str(cfg.counterfactual.source_sampling_policy) != "proportional"
            or str(cfg.counterfactual.target_policy) != "proportional"
        ):
            raise ValueError(
                "MIDI-VAE common protocol uses proportional source and target sampling"
            )
        if set(dataset.frame.style_id) != set(range(len(names))):
            raise ValueError("Selected split lacks required styles")
        chosen = select_source_indices(
            dataset.frame,
            strata=["style_id"],
            max_per_stratum=int(cfg.counterfactual.max_sources_per_style),
            max_total=int(cfg.counterfactual.max_total_sources),
            unique_sources=True,
            seed=int(cfg.seed),
            policy="proportional",
            minimum_per_stratum=len(names) - 1 if require_all else 0,
            population_weights=weights,
        )
        targets = proportional_target_assignments(
            dataset.frame,
            chosen,
            stratum="style_id",
            unique_sources=True,
            seed=int(cfg.seed),
            require_all_transitions=require_all,
            population_weights=weights,
        )
        local = chosen[context.rank :: context.world_size]
        selected = dataset.frame.iloc[local].copy()
        manifest = _selected_source_manifest(
            dataset_asset_path(cfg, paths, "manifest.parquet"), selected
        )
        rows = selected[["segment_id", "sample_id", "style_id"]].merge(
            manifest[["segment_id", "source_midi_path", "start_bar", "num_bars"]],
            on="segment_id",
            how="left",
            validate="one_to_one",
            sort=False,
        )
        if len(rows) != len(local) or rows.source_midi_path.isna().any():
            raise ValueError("Direct MIDI-VAE evaluation segments differ from common source plan")
        rolls = MIDIRollDataset(rows, voices=int(cfg.transport.model.voices))
        checkpoint_path = Path(str(cfg.transport_checkpoint)).resolve()
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
        if payload["config"]["model"] != OmegaConf.to_container(cfg.transport.model, resolve=True):
            raise ValueError(
                "MIDI-VAE checkpoint architecture differs from requested configuration"
            )
        model = baseline_model(cfg.transport).to(context.device)
        if not isinstance(model, MIDIVAE):
            raise ValueError("Direct generation requires MIDI-VAE")
        model.load_state_dict(payload["model"])
        model.eval()
        tokenizer = tokenizer_from_config(
            cfg.tokenizer, max_sequence_length=int(cfg.codec.max_sequence_length)
        )
        root = paths["artifacts_dir"] / str(cfg.experiment.name) / str(cfg.data.name)
        completed = []
        batch_size = int(cfg.counterfactual.get("decode_batch_size", 8))
        for start in track(
            range(0, len(rolls), batch_size), description="MIDI-VAE generation", unit="batch"
        ):
            end = min(len(rolls), start + batch_size)
            items = [rolls[index] for index in range(start, end)]
            batch = {
                key: torch.stack([item[key] for item in items]).to(context.device)
                for key in items[0]
            }
            target = torch.tensor(
                [targets[local[i]] for i in range(start, end)], device=context.device
            )
            with (
                torch.inference_mode(),
                autocast_context(context.device, str(cfg.transport.training.precision)),
            ):
                mean, _ = model.encode(batch)
                reconstruction = model.decode(mean)
                generated = model.decode(
                    swap_style(mean, batch["style_id"], target, model.num_styles)
                )
            for j, index in enumerate(range(start, end)):
                row = rows.iloc[index]
                source_id, target_id = int(row.style_id), int(target[j])
                directory = (
                    root
                    / _safe_component(str(row.sample_id))
                    / _safe_component(str(row.segment_id))
                    / names[target_id]
                )
                directory.mkdir(parents=True, exist_ok=True)
                metadata_path = directory / "counterfactual_metadata.json"
                completed.append(str(metadata_path.relative_to(root)))
                if (
                    bool(cfg.counterfactual.skip_existing)
                    and metadata_path.exists()
                    and all(
                        (directory / p).exists()
                        for p in ("source.mid", "counterfactual.mid", "same_style_reconstruction.mid")
                    )
                ):
                    continue
                source = load_midi(Path(str(row.source_midi_path)))
                tokenized = tokenizer.encode(
                    source, start_bar=int(row.start_bar), num_bars=int(row.num_bars)
                )
                source_midi = tokenizer.decode(tokenized)
                source_midi.dump(str(directory / "source.mid"))
                tempo = (
                    float(source_midi.tempo_changes[0].tempo)
                    if source_midi.tempo_changes
                    else 120.0
                )
                for prediction, filename in (
                    (generated, "counterfactual.mid"),
                    (reconstruction, "same_style_reconstruction.mid"),
                ):
                    midi = rolls_to_midi(
                        prediction["pitch_logits"][j].argmax(-1),
                        prediction["velocity"][j],
                        prediction["program_logits"][j].argmax(-1),
                        tempo=tempo,
                    )
                    midi.dump(str(directory / filename))
                metadata = {
                    "dataset": str(cfg.data.name),
                    "sample_id": str(row.sample_id),
                    "segment_id": str(row.segment_id),
                    "source_style_id": source_id,
                    "target_style_id": target_id,
                    "source_style": names[source_id],
                    "target_style": names[target_id],
                    "representation_kind": model.representation_kind,
                    "baseline": "MIDI-VAE",
                    "baseline_representation": ROLL_SCHEMA,
                    "split": split,
                    "shared_codec_used_for_generation": False,
                    "inverse_nfe": 1,
                    "forward_nfe": 2,
                }
                metadata_path.write_text(json.dumps(metadata, indent=2))
        gathered: list[object] = [completed]
        if context.world_size > 1:
            gathered = [None] * context.world_size
            dist.all_gather_object(gathered, completed)
        if context.is_main:
            records = sorted(
                path for group in gathered if isinstance(group, list) for path in group
            )
            if len(records) != len(chosen):
                raise ValueError("Incomplete MIDI-VAE distributed generation")
            pairs: dict[str, int] = {}
            for index, target_id in targets.items():
                key = f"{names[int(dataset.frame.iloc[index].style_id)]}_to_{names[target_id]}"
                pairs[key] = pairs.get(key, 0) + 1
            root.mkdir(parents=True, exist_ok=True)
            (root / "generation_manifest.json").write_text(
                json.dumps(
                    {
                        "version": 3,
                        "split": split,
                        "selected_sources": len(chosen),
                        "planned_transitions": len(chosen),
                        "metadata_files": records,
                        "transition_pair_counts": pairs,
                        "all_directed_pairs_required": require_all,
                        "source_sampling_policy": "proportional",
                        "target_policy": "proportional",
                        "sampling_population": "unique_sources",
                    },
                    indent=2,
                )
            )
        distributed_barrier(context)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()

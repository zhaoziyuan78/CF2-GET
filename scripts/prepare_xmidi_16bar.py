"""Build the 16-bar XMIDI windows and their six-genre CLaMP3 label overlay.

Phases (run in this order; each one is resumable and refuses changed inputs):

  segments       16-bar/hop-8 windows of every song in the 8-bar `cfmusic.commands.prepare`
                 manifest, with the content-quality filter below. Existing song splits are kept.
  label-prepare  After `cfmusic.commands.cache_latents`: CLaMP3 genre prototypes and one job
                 partition per GPU over every cached window.
  label-worker   One process per GPU: CLaMP3 C2 global features of each window (BEAT round
                 trip -> MTF -> first/middle/last 512-patch views) scored against the prototypes.
  label-publish  argmax genre labels, the per-genre q30 training-margin filter, and the latent
                 label overlays (`<labels>/index.parquet` full, `<labels>/q30/index.parquet`).

Default layout under --data-root (matching scripts/run_xmidi_16bar.sh and the configs):
  processed/xmidi          8-bar manifest from `cfmusic.commands.prepare`
  processed_16bar/xmidi    16-bar manifest (segments)
  latents_16bar/xmidi      latent cache of the 16-bar chunk VAE
  labels/xmidi_16bar       label work directory and full overlay; q30/ is the training overlay
"""

from __future__ import annotations

import argparse
import io
import json
import multiprocessing
import os
import time
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from itertools import islice
from pathlib import Path
from types import MethodType

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

GENRES = ["classical", "jazz", "rock", "pop", "country", "latin"]
QUALITY = dict(
    num_bars=16,
    hop_bars=8,
    minimum_note_onsets=64,
    minimum_active_bars=12,
    minimum_active_beat_fraction=0.6,
    maximum_silent_beats=8,
    max_tokens=5120,
)
ROOT = Path(__file__).resolve().parents[1]
_TOKENIZER = None
_PATCHILIZER = None
_MIDI_CACHE: OrderedDict = OrderedDict()


class Layout:
    def __init__(self, data_root: Path) -> None:
        self.parent = data_root / "processed/xmidi"
        self.processed = data_root / "processed_16bar/xmidi"
        self.cache = data_root / "latents_16bar/xmidi"
        self.work = data_root / "labels/xmidi_16bar"
        self.filtered = self.work / "q30"
        self.reports = self.work / "reports"


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".pending")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def parquet(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".pending")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


def file_identity(path: Path) -> dict:
    info = path.stat()
    return {"path": str(path.resolve()), "bytes": info.st_size, "mtime_ns": info.st_mtime_ns}


# --------------------------------------------------------------------------- segments


def silence_run(active):
    padded = np.r_[True, active, True]
    edges = np.flatnonzero(padded)
    return int(np.diff(edges).max(initial=1) - 1)


def quality_windows(midi, segments):
    """Count representable note onsets and sounding beat coverage (sustain included)."""
    size = max((s.start_bar + s.num_bars) * 4 for s in segments) if segments else 0
    active = np.zeros(size, dtype=bool)
    onsets = np.zeros(size, dtype=np.int64)
    for inst in midi.instruments:
        for note in inst.notes:
            if not 21 <= note.pitch <= 108:
                continue
            left = int(note.start // midi.ticks_per_beat)
            right = left + 1 if inst.is_drum else int(np.ceil(note.end / midi.ticks_per_beat))
            if 0 <= left < size:
                onsets[left] += 1
            active[max(0, left) : min(size, right)] = True
    prefix = np.r_[0, onsets.cumsum()]
    result = []
    for s in segments:
        left, right = s.start_bar * 4, (s.start_bar + s.num_bars) * 4
        view = active[left:right]
        notes = int(prefix[right] - prefix[left])
        bars = int(view.reshape(-1, 4).any(axis=1).sum())
        fraction = float(view.mean())
        gap = silence_run(view)
        result.append(
            dict(
                start_bar=s.start_bar,
                num_bars=s.num_bars,
                segment_index=s.index,
                segment_note_onsets=notes,
                active_bars=bars,
                active_beat_fraction=fraction,
                maximum_silent_beats=gap,
                too_few_notes=notes < QUALITY["minimum_note_onsets"],
                too_few_active_bars=bars < QUALITY["minimum_active_bars"],
                too_sparse=fraction < QUALITY["minimum_active_beat_fraction"],
                long_silence=gap > QUALITY["maximum_silent_beats"],
            )
        )
    return result


def prepare_song(item):
    from cfmusic.data.midi_io import load_midi
    from cfmusic.data.segmentation import enumerate_segments
    from cfmusic.tokenization.beat import BeatTokenizer, BeatTokenizerConfig

    sid, path, expected = item
    midi = load_midi(Path(path))
    if sum(len(i.notes) for i in midi.instruments) != expected:
        raise ValueError(f"Raw MIDI changed: {path}")
    segments = enumerate_segments(midi, num_bars=16, hop_bars=8, drop_incomplete_final_segment=True)
    rows = quality_windows(midi, segments)
    tokenizer = BeatTokenizer(BeatTokenizerConfig(max_sequence_length=5120))
    lengths = tokenizer.encoded_segment_lengths(midi, [(s.start_bar, s.num_bars) for s in segments])
    for row, length in zip(rows, lengths, strict=True):
        row.update(
            sample_id=sid, raw_token_count=length, token_count=length, overlength=length > 5120
        )
        row["retained"] = not any(
            row[k]
            for k in (
                "too_few_notes",
                "too_few_active_bars",
                "too_sparse",
                "long_silence",
                "overlength",
            )
        )
    return rows


def segments(layout: Layout, workers: int) -> None:
    destination = layout.processed / "complete.json"
    if destination.exists():
        print(destination.read_text(), flush=True)
        return
    source = layout.parent / "manifest.parquet"
    stat = source.stat()
    spec = dict(
        source=str(source.resolve()),
        source_bytes=stat.st_size,
        source_mtime_ns=stat.st_mtime_ns,
        quality=QUALITY,
        split_policy="preserve existing sample_id/group_id splits",
    )
    receipt = layout.processed / "preparation_protocol.json"
    if receipt.exists() and json.loads(receipt.read_text()) != spec:
        raise ValueError("Preparation inputs or thresholds changed")
    dump(receipt, spec)
    original = pd.read_parquet(source)
    if original.groupby("group_id").split.nunique().max() != 1:
        raise ValueError("Original split groups overlap")
    songs = original.drop_duplicates("sample_id").reset_index(drop=True)
    del original
    items = list(
        songs[["sample_id", "source_midi_path", "num_notes"]].itertuples(index=False, name=None)
    )
    parts = layout.processed / "preparation_parts"
    parts.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        for offset in tqdm(range(0, len(items), 1024), desc="Prepare 16-bar songs"):
            target = parts / f"part-{offset:06d}.parquet"
            if target.exists():
                continue
            groups = pool.map(prepare_song, items[offset : offset + 1024], chunksize=8)
            rows = [r for group in groups for r in group]
            parquet(target, pd.DataFrame(rows))
    audits = pd.concat(
        [pd.read_parquet(p) for p in sorted(parts.glob("part-*.parquet"))], ignore_index=True
    )
    metadata = songs.drop(columns=[c for c in audits.columns if c != "sample_id" and c in songs])
    metadata = metadata.drop(columns=["segment_id"], errors="ignore")
    combined = audits.merge(metadata, on="sample_id", validate="many_to_one")
    frame = combined.loc[combined.retained].copy()
    frame["segment_id"] = frame.sample_id + ":bar" + frame.start_bar.astype(str) + ":n16"
    frame["valid"] = True
    assert frame.segment_id.is_unique and frame.groupby("group_id").split.nunique().max() == 1
    parquet(layout.processed / "manifest.parquet", frame)
    card = json.loads((layout.parent / "dataset_card.json").read_text())
    card.update(
        segments=len(frame),
        splits=frame.split.value_counts().to_dict(),
        segmentation=QUALITY,
        quality_filter=QUALITY,
        parent_manifest=str(source.resolve()),
    )
    dump(layout.processed / "dataset_card.json", card)
    cols = [
        "too_few_notes",
        "too_few_active_bars",
        "too_sparse",
        "long_silence",
        "overlength",
        "retained",
    ]
    reports = layout.processed / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    combined.groupby(["split", "style_label"])[cols].agg(["sum", "count"]).to_csv(
        reports / "quality_filter_by_split_genre.csv"
    )
    summary = dict(
        candidate_windows=len(combined),
        retained_windows=len(frame),
        retained_fraction=len(frame) / len(combined),
        splits=frame.split.value_counts().to_dict(),
        counts={c: int(combined[c].sum()) for c in cols},
        quality=QUALITY,
        songs_before=len(songs),
        songs_retained=int(frame.sample_id.nunique()),
        note="Reason counts overlap; rules apply equally to all splits. q30 later applies to train only.",
    )
    dump(reports / "quality_filter_summary.json", summary)
    dump(destination, summary)
    print(json.dumps(summary, indent=2), flush=True)


# --------------------------------------------------------------------------- labels


def label_prepare(layout: Layout, repository: Path, gpus: int, name: str) -> None:
    from cfmusic.evaluation.clamp3 import embedding_protocol
    from cfmusic.evaluation.clamp3_labels import prompt_embeddings, singleton_taxonomy

    work = layout.work
    if (work / "protocol.json").exists():
        protocol = json.loads((work / "protocol.json").read_text())
        if any(file_identity(Path(v["path"])) != v for v in protocol["inputs"].values()):
            raise ValueError("Label input changed")
        if protocol["num_gpus"] != gpus:
            raise ValueError("Label partitions were prepared for a different GPU count")
        return
    embedding_model = embedding_protocol(repository)
    taxonomy = singleton_taxonomy(embedding_model, name)
    dump(work / "taxonomy.json", taxonomy)
    prompts = work / "candidate_prompts.npy"
    if not prompts.exists():
        np.save(prompts, prompt_embeddings(repository))
    card = json.loads((layout.processed / "dataset_card.json").read_text())
    card.update(
        style_vocabulary=GENRES,
        genre_vocabulary=GENRES,
        label_source=name,
        genre_membership={g: [g] for g in GENRES},
        taxonomy=str(work / "taxonomy.json"),
    )
    dump(work / "source_card.json", card)
    index = pd.read_parquet(layout.cache / "index.parquet")
    index.insert(0, "row_id", np.arange(len(index), dtype=np.int64))
    details = pd.read_parquet(
        layout.processed / "manifest.parquet",
        columns=[
            "segment_id",
            "sample_id",
            "split",
            "source_midi_path",
            "start_bar",
            "num_bars",
            "group_id",
        ],
    )
    jobs = index[["row_id", "sample_id", "segment_id", "split", "style_id"]].merge(
        details, on=["sample_id", "segment_id", "split"], validate="one_to_one"
    )
    assert len(jobs) == len(index)
    ranks = pd.factorize(jobs.sample_id, sort=True)[0] % gpus
    for rank in range(gpus):
        parquet(
            work / f"jobs-{rank}.parquet",
            jobs.loc[ranks == rank].sort_values(["sample_id", "start_bar"]),
        )
    paths = dict(
        index=layout.cache / "index.parquet",
        manifest=layout.processed / "manifest.parquet",
        card=work / "source_card.json",
        taxonomy=work / "taxonomy.json",
        prompts=prompts,
    )
    dump(
        work / "protocol.json",
        dict(
            schema=name,
            inputs={k: file_identity(p) for k, p in paths.items()},
            num_gpus=gpus,
            vocabulary=GENRES,
            rows=len(jobs),
            base_rows=len(index),
            songs=int(jobs.sample_id.nunique()),
            split_counts=jobs.split.value_counts().to_dict(),
            embedding_model=embedding_model,
            max_sequence_length=5120,
            midi_view_policy="three",
            midi_representation="BEAT token roundtrip, matching source.mid",
            label_assignment_hash=name,
            selection="all content-filtered 16-bar cached segments",
        ),
    )


def _initialize(repository: str, max_sequence_length: int) -> None:
    from cfmusic.commands.clamp3_batched_worker import _load_official_module
    from cfmusic.tokenization.beat import BeatTokenizer, BeatTokenizerConfig

    global _TOKENIZER, _PATCHILIZER
    _TOKENIZER = BeatTokenizer(BeatTokenizerConfig(max_sequence_length=max_sequence_length))
    _PATCHILIZER = _load_official_module(Path(repository)).M3Patchilizer()


def _prepare_segment(item: tuple) -> tuple:
    import miditoolkit
    import mido

    local_id, path, start, bars = item
    try:
        midi = _MIDI_CACHE.get(path)
        if midi is None:
            midi = miditoolkit.MidiFile(path)
            _MIDI_CACHE[path] = midi
            if len(_MIDI_CACHE) > 4:
                _MIDI_CACHE.popitem(last=False)
        else:
            _MIDI_CACHE.move_to_end(path)
        tokens = _TOKENIZER.encode_untruncated(midi, start_bar=int(start), num_bars=int(bars))
        if len(tokens) > _TOKENIZER.config.max_sequence_length:
            raise ValueError(f"Cached window has {len(tokens)} tokens, exceeding codec limit")
        stream = io.BytesIO()
        # Default ticks match generate_counterfactuals' source.mid, not raw-song ticks.
        _TOKENIZER.decode(tokens).dump(file=stream)
        stream.seek(0)
        decoded = mido.MidiFile(file=stream)
        ignored = {
            "text", "copyright", "track_name", "instrument_name", "lyrics", "marker",
            "cue_marker", "device_name", "sequencer_specific",
        }
        lines = [f"ticks_per_beat {decoded.ticks_per_beat}"]
        for message in decoded.merged_track:
            if (message.is_meta and message.type in ignored) or message.type == "sysex":
                continue
            encoded = " ".join(str(value) for value in message.dict().values())
            lines.append(encoded.encode("unicode_escape").decode("utf-8"))
        patches = np.asarray(
            _PATCHILIZER.encode("\n".join(lines), add_special_patches=True), dtype=np.int64
        )
        starts = sorted({0, max(0, (len(patches) - 512) // 2), max(0, len(patches) - 512)})
        windows = [patches[start:start + 512] for start in starts]
        return local_id, windows, None
    except Exception as error:
        return local_id, None, f"{type(error).__name__}: {error}"


def label_worker(layout: Layout, repository: Path, rank: int, workers: int, batch_size: int) -> None:
    import torch

    from cfmusic.commands.clamp3_batched_worker import _load_official_model
    from cfmusic.evaluation.clamp3 import embedding_protocol
    from cfmusic.evaluation.clamp3_labels import (
        family_scores,
        gpu_patch_forward,
        normalize,
        open_array,
        write_json,
    )

    work = layout.work
    protocol = json.loads((work / "protocol.json").read_text())
    if protocol["embedding_model"] != embedding_protocol(repository):
        raise ValueError("CLaMP3 changed after preparation")
    if not 0 <= rank < protocol["num_gpus"]:
        raise ValueError("Worker rank outside the prepared partitions")
    jobs = pd.read_parquet(work / f"jobs-{rank}.parquet")
    if jobs.empty:
        raise ValueError("Empty worker partition")
    scores = open_array(work / f"scores-{rank}.npy", (len(jobs), len(protocol["vocabulary"])), np.float32)
    status = open_array(work / f"status-{rank}.npy", (len(jobs),), np.uint8)
    pending = jobs.loc[status != 1].copy()
    if pending.empty:
        print(f"Rank {rank}: all {len(jobs):,} segments complete", flush=True)
        return
    taxonomy = json.loads(Path(protocol["inputs"]["taxonomy"]["path"]).read_text())
    prototypes = normalize(np.load(protocol["inputs"]["prompts"]["path"]).mean(axis=1))
    model, patchilizer, _ = _load_official_model(repository, "cuda")
    sample = torch.randint(0, 128, (2, 512, 64), device=model.device)
    masks = torch.ones((2, 512), device=model.device)
    with torch.inference_mode():
        expected = model.get_symbolic_features(sample, masks, get_global=True)
        model.symbolic_model.forward = MethodType(gpu_patch_forward, model.symbolic_model)
        actual = model.get_symbolic_features(sample, masks, get_global=True)
    if not torch.allclose(expected, actual, atol=1e-5, rtol=1e-5):
        raise ValueError("Device-local patch implementation changed official output")
    del sample, masks, expected, actual
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    processed = 0
    records = list(pending[["source_midi_path", "start_bar", "num_bars"]].itertuples(name=None))
    progress = tqdm(total=len(jobs), initial=int((status == 1).sum()),
                    desc=f"Label segments rank {rank}", mininterval=10)
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
        initializer=_initialize, initargs=(str(repository), int(protocol["max_sequence_length"])),
    ) as executor:
        for offset in range(0, len(records), 512):
            # Bounded submissions: do not queue millions of large patch arrays.
            iterator = executor.map(_prepare_segment, records[offset:offset + 512], chunksize=8)
            while group := list(islice(iterator, batch_size)):
                windows, owners, lengths, local_ids = [], [], [], []
                for local_id, patches, error in group:
                    if error is not None:
                        write_json(work / f"failure-{rank}.json", {
                            "local_id": local_id, "segment_id": jobs.iloc[local_id].segment_id,
                            "error": error,
                        })
                        raise RuntimeError(f"Cannot label segment {local_id}: {error}")
                    local_ids.append(local_id)
                    for patch in patches:
                        windows.append(patch)
                        owners.append(local_id)
                        lengths.append(len(patch))
                pooled = {local_id: np.zeros(768, dtype=np.float64) for local_id in local_ids}
                for start in range(0, len(windows), batch_size):
                    subset = windows[start:start + batch_size]
                    inputs = torch.full((len(subset), 512, 64), patchilizer.pad_token_id,
                                        dtype=torch.long, device=model.device)
                    masks = torch.zeros((len(subset), 512), dtype=torch.long, device=model.device)
                    for i, patch in enumerate(subset):
                        inputs[i, :len(patch)] = torch.from_numpy(patch).to(model.device)
                        masks[i, :len(patch)] = 1
                    with torch.inference_mode():
                        output = model.get_symbolic_features(inputs, masks, get_global=True)
                    output = output.float().cpu().numpy()
                    if not np.isfinite(output).all():
                        raise ValueError("Non-finite CLaMP3 features")
                    for i, feature in enumerate(output, start=start):
                        pooled[owners[i]] += feature.astype(np.float64) * lengths[i]
                features = np.stack([pooled[i] for i in local_ids])
                if (np.linalg.norm(features, axis=1) <= 0).any():
                    raise ValueError("Zero CLaMP3 embedding")
                scores[local_ids] = family_scores(features, prototypes, taxonomy)
                scores.flush()
                status[local_ids] = 1
                status.flush()
                processed += len(group)
                progress.update(len(group))
            elapsed = time.monotonic() - started
            write_json(work / f"progress-{rank}.json", {
                "rank": rank, "complete": int((status == 1).sum()), "total": len(jobs),
                "seconds": elapsed, "segments_per_second": processed / max(elapsed, 1),
                "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
            })
    progress.close()


def filter_train(index):
    """Keep train windows whose top-1 minus top-2 margin reaches their genre's 30th percentile."""
    train = index.split.eq("train")
    if not np.isfinite(index.genre_margin).all() or (index.genre_margin < 0).any():
        raise ValueError("Invalid relabel margins")
    thresholds = index.loc[train].groupby("style_id").genre_margin.quantile(.3)
    keep = ~train | index.genre_margin.ge(index.style_id.map(thresholds))
    filtered = index.loc[keep].reset_index(drop=True)
    pd.testing.assert_frame_equal(index.loc[~train].reset_index(drop=True),
                                  filtered.loc[filtered.split.ne("train")].reset_index(drop=True))
    counts = []
    for style_id, genre in enumerate(GENRES):
        before = index.loc[train & index.style_id.eq(style_id)]
        after = filtered.loc[filtered.split.eq("train") & filtered.style_id.eq(style_id)]
        if before.empty or after.empty:
            raise ValueError(f"No supported train samples for {genre}")
        counts.append(dict(style_id=style_id, genre=genre, threshold=float(thresholds[style_id]),
                           original_segments=len(before), retained_segments=len(after),
                           retained_songs=int(after.sample_id.nunique()),
                           original_mean_margin=float(before.genre_margin.mean()),
                           retained_mean_margin=float(after.genre_margin.mean())))
    return filtered, pd.DataFrame(counts)


def label_publish(layout: Layout, name: str) -> None:
    work, filtered_dir = layout.work, layout.filtered
    if (filtered_dir / "complete.json").exists():
        return
    protocol = json.loads((work / "protocol.json").read_text())
    index = pd.read_parquet(layout.cache / "index.parquet")
    scores = np.empty((len(index), 6), np.float32)
    seen = np.zeros(len(index), bool)
    for rank in range(protocol["num_gpus"]):
        ids = pd.read_parquet(work / f"jobs-{rank}.parquet").row_id.to_numpy()
        status = np.load(work / f"status-{rank}.npy")
        local = np.load(work / f"scores-{rank}.npy")
        if not (status == 1).all() or seen[ids].any() or local.shape != (len(ids), 6):
            raise ValueError("Incomplete label partition")
        scores[ids], seen[ids] = local, True
    assert seen.all() and np.isfinite(scores).all()
    predicted = scores.argmax(1)
    top = np.sort(scores, axis=1)
    index["style_id"] = index["genre_id"] = predicted
    index["genre_margin"], index["genre_similarity"] = top[:, -1] - top[:, -2], top[:, -1]
    index["label_source"] = name
    filtered, counts = filter_train(index)
    manifest = pd.read_parquet(layout.processed / "manifest.parquet")
    manifest = manifest.loc[manifest.segment_id.isin(index.segment_id)].copy()
    lookup = pd.Series(predicted, index=index.segment_id)
    manifest["style_id"] = manifest["genre_id"] = manifest.segment_id.map(lookup).astype(int)
    manifest["style_label"] = manifest["genre_label"] = manifest.style_id.map(
        dict(enumerate(GENRES))
    )
    manifest["label_source"] = name
    parquet(work / "manifest.parquet", manifest)
    # Test sources are later sampled in proportion to each train song's modal window label.
    song_counts = (
        index.loc[index.split.eq("train")]
        .groupby(["sample_id", "style_id"])
        .size()
        .rename("n")
        .reset_index()
    )
    songs = song_counts.sort_values(
        ["sample_id", "n", "style_id"], ascending=[True, False, True]
    ).drop_duplicates("sample_id")
    songs["genre_label"] = songs.style_id.map(dict(enumerate(GENRES)))
    parquet(work / "source_labels.parquet", songs[["sample_id", "genre_label"]])
    base = json.loads((layout.cache / "cache_metadata.json").read_text())
    card = json.loads((work / "source_card.json").read_text())
    for directory, current, suffix in [(work, index, "full"), (filtered_dir, filtered, "train-q30")]:
        directory.mkdir(parents=True, exist_ok=True)
        if directory == filtered_dir and not (directory / "manifest.parquet").exists():
            (directory / "manifest.parquet").symlink_to(work / "manifest.parquet")
        parquet(directory / "index.parquet", current)
        dump(
            directory / "dataset_card.json",
            dict(
                card,
                segments=len(current),
                splits=current.split.value_counts().to_dict(),
                label_assignment_hash=f"{name}:{suffix}",
                source_labels=str(work / "source_labels.parquet"),
            ),
        )
        dump(
            directory / "index.metadata.json",
            dict(
                schema_version="cfmusic.latent-label-overlay.v1",
                base_dataset_manifest_hash=base["dataset_manifest_hash"],
                label_assignment_hash=f"{name}:{suffix}",
                label_source=name,
                rows=len(current),
                unique_samples=int(current.sample_id.nunique()),
                genre_vocabulary=GENRES,
                dataset_card=str((directory / "dataset_card.json").resolve()),
                manifest=str((directory / "manifest.parquet").resolve()),
                relabel_config=str((work / "taxonomy.json").resolve()),
            ),
        )
    layout.reports.mkdir(parents=True, exist_ok=True)
    counts.to_csv(layout.reports / "training_filter_by_genre.csv", index=False)
    summary = dict(
        all_segments=len(index),
        train_before=int(index.split.eq("train").sum()),
        train_after=int(filtered.split.eq("train").sum()),
        heldout_unchanged=True,
        outputs=[
            file_identity(filtered_dir / f)
            for f in ["index.parquet", "index.metadata.json", "dataset_card.json"]
        ],
    )
    dump(filtered_dir / "complete.json", summary)
    dump(layout.reports / "relabel_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phase", choices=["segments", "label-prepare", "label-worker", "label-publish"])
    parser.add_argument("--data-root", type=Path,
                        default=Path(os.environ.get("CFMUSIC_DATA_ROOT", ROOT / "data")))
    parser.add_argument("--clamp3", type=Path,
                        default=Path(os.environ.get("CLAMP3_REPOSITORY", ROOT / "external/clamp3")))
    parser.add_argument("--name", default="xmidi_16bar_q30", help="label source / overlay identity")
    parser.add_argument("--workers", type=int, default=8, help="CPU processes (per GPU for labels)")
    parser.add_argument("--gpus", type=int, default=8, help="label partitions (label-prepare)")
    parser.add_argument("--rank", type=int, default=0, help="label partition (label-worker)")
    parser.add_argument("--batch-size", type=int, default=64, help="CLaMP3 views per batch")
    args = parser.parse_args()
    layout = Layout(args.data_root.resolve())
    repository = args.clamp3.resolve()
    if args.phase == "segments":
        segments(layout, args.workers)
    elif args.phase == "label-prepare":
        label_prepare(layout, repository, args.gpus, args.name)
    elif args.phase == "label-worker":
        label_worker(layout, repository, args.rank, args.workers, args.batch_size)
    else:
        label_publish(layout, args.name)


if __name__ == "__main__":
    main()

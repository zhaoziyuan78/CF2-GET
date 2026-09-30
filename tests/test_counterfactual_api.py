import json
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from cfmusic.commands.generate_counterfactuals import (
    _artifact_matches_generation,
    _safe_component,
    _selected_source_manifest,
    concatenate_conditions,
    largest_remainder_quotas,
    proportional_target_assignments,
    select_source_indices,
)
from cfmusic.conditioning.schema import ConditionBatch
from cfmusic.transport.conditional_flow import ConditionalFlow


def test_artifact_names_work_on_shared_filesystems(tmp_path: Path) -> None:
    names = ["clip:bar0:n8", "clip_bar0_n8", "clip%3Abar0%3An8", "a/b", "a\\b", ".."]
    encoded = [_safe_component(name) for name in names]
    assert len(set(encoded)) == len(names)
    for name in encoded:
        assert not any(char in name for char in '/\\:<>"|?*\r\n')
        (tmp_path / name).mkdir()


def test_idle_generation_rank_does_not_read_manifest(tmp_path: Path) -> None:
    selected = pd.DataFrame(columns=["sample_id", "segment_id"])
    manifest = _selected_source_manifest(tmp_path / "unread.parquet", selected)
    assert manifest.empty
    assert "source_midi_path" in manifest.columns


def test_generation_manifest_selects_sample_or_segment(tmp_path: Path) -> None:
    path = tmp_path / "manifest.parquet"
    pd.DataFrame(
        {
            "sample_id": ["a", "a", "b", "c"],
            "segment_id": ["a0", "a1", "b0", "c0"],
            "source_midi_path": ["a.mid", "a.mid", "b.mid", "c.mid"],
            "token_count": [10] * 4,
            "start_bar": [0, 4, 0, 0],
            "num_bars": [8] * 4,
        }
    ).to_parquet(path)
    selected = pd.DataFrame({"sample_id": ["a"], "segment_id": ["b0"]})
    manifest = _selected_source_manifest(path, selected)
    assert manifest["segment_id"].tolist() == ["a0", "a1", "b0"]


class ConstantField(nn.Module):
    def forward(
        self, state: torch.Tensor, time: torch.Tensor, condition: ConditionBatch
    ) -> torch.Tensor:
        return condition.style_id.to(state)[:, None, None].expand_as(state) + 1


def test_shared_noise_counterfactual_api() -> None:
    transport = ConditionalFlow(ConstantField(), solver_method="heun")
    latent = torch.randn(3, 2, 4)
    zeros = torch.zeros(3, dtype=torch.long)
    source = ConditionBatch(zeros, zeros, zeros)
    target = ConditionBatch(zeros, zeros, torch.ones(3, dtype=torch.long))
    output = transport.counterfactual(latent, source, target, num_steps=4)
    assert torch.allclose(output.reconstructed_source_latent, latent, atol=1e-6)
    assert torch.allclose(output.counterfactual_latent, latent + 1, atol=1e-6)
    assert output.inverse_nfe == 8 and output.forward_nfe == 16


def test_source_selection_is_balanced_unique_and_deterministic() -> None:
    frame = pd.DataFrame(
        {
            "sample_id": ["a", "a", "b", "c", "d", "e", "f"],
            "style_id": [0, 0, 0, 0, 1, 1, 1],
        }
    )
    arguments = {
        "strata": ["style_id"],
        "max_per_stratum": 2,
        "max_total": 4,
        "unique_sources": True,
        "seed": 7,
    }

    selected = select_source_indices(frame, **arguments)

    assert selected == select_source_indices(frame, **arguments)
    selected_frame = frame.iloc[selected]
    assert selected_frame["sample_id"].nunique() == 4
    assert selected_frame["style_id"].value_counts().to_dict() == {0: 2, 1: 2}


def test_source_selection_can_follow_empirical_proportions_exactly() -> None:
    frame = pd.DataFrame(
        {
            "sample_id": [f"sample-{index}" for index in range(100)],
            "style_id": [0] * 40 + [1] * 30 + [2] * 20 + [3] * 10,
        }
    )

    selected = select_source_indices(
        frame,
        strata=["style_id"],
        max_per_stratum=100,
        max_total=30,
        unique_sources=True,
        seed=11,
        policy="proportional",
    )

    assert frame.iloc[selected]["style_id"].value_counts().to_dict() == {
        0: 12,
        1: 9,
        2: 6,
        3: 3,
    }


def test_proportional_targets_exclude_source_and_use_exact_quotas() -> None:
    frame = pd.DataFrame(
        {
            "sample_id": [f"sample-{index}" for index in range(10)],
            "style_id": [0] * 4 + [1] * 3 + [2] * 2 + [3],
        }
    )
    selected = list(range(4))

    assignments = proportional_target_assignments(
        frame,
        selected,
        stratum="style_id",
        unique_sources=True,
        seed=5,
    )

    assert set(assignments) == set(selected)
    assert all(frame.iloc[index]["style_id"] != target for index, target in assignments.items())
    assert pd.Series(assignments).value_counts().to_dict() == {1: 2, 2: 1, 3: 1}


def test_largest_remainder_quota_respects_capacity() -> None:
    assert largest_remainder_quotas(
        {"large": 9, "small": 1}, total=5, capacity={"large": 3, "small": 2}
    ) == {"large": 3, "small": 2}


def test_condition_concatenation_preserves_factorial_fields() -> None:
    first = ConditionBatch(
        torch.tensor([0]),
        torch.tensor([1]),
        torch.tensor([2]),
        torch.tensor([3]),
        torch.tensor([4]),
    )
    second = ConditionBatch(
        torch.tensor([5]),
        torch.tensor([6]),
        torch.tensor([7]),
        torch.tensor([8]),
        torch.tensor([9]),
    )

    combined = concatenate_conditions([first, second])

    assert combined.batch_size == 2
    assert combined.style_id.tolist() == [2, 7]
    assert combined.genre_id is not None and combined.genre_id.tolist() == [3, 8]
    assert combined.emotion_id is not None and combined.emotion_id.tolist() == [4, 9]


def test_existing_artifact_must_match_current_generation_identity(tmp_path: Path) -> None:
    metadata_path = tmp_path / "counterfactual_metadata.json"
    identity = {
        "codec_checkpoint_hash": "codec-new",
        "transport_checkpoint_hash": "transport-new",
        "generation_config_hash": "config-new",
    }
    metadata_path.write_text(json.dumps({**identity, "sample_id": "example"}), encoding="utf-8")

    assert _artifact_matches_generation(metadata_path, identity)
    assert not _artifact_matches_generation(
        metadata_path, {**identity, "transport_checkpoint_hash": "transport-other"}
    )

"""Weight-only codec initialization with an explicitly expanded position table."""

from collections.abc import Mapping

import torch
from torch import nn


def initialize_longer_codec(model: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    """Keep old positions bitwise; new positions retain the seeded model initialization."""
    expected = model.state_dict()
    if set(expected) != set(state):
        raise ValueError("Warm start requires the same codec architecture and state keys")
    output = {}
    for name, target in expected.items():
        old = state[name]
        if old.shape == target.shape:
            output[name] = old
        elif (
            name == "position_embedding"
            and old.ndim == target.ndim == 2
            and old.shape[1] == target.shape[1]
            and old.shape[0] < target.shape[0]
        ):
            value = target.clone()
            value[: len(old)] = old.to(value)
            output[name] = value
        else:
            raise ValueError(
                f"Unsupported warm-start shape change: {name}: {old.shape} -> {target.shape}"
            )
    model.load_state_dict(output, strict=True)

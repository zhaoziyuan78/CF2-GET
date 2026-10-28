"""Decoder-only overlays: never replace encoder or shared embedding parameters."""

from collections.abc import Mapping

import torch
from torch import nn

DECODER_PREFIXES = ("decoder.", "latent_projection.", "output.")


def decoder_parameter(name: str) -> bool:
    return name.startswith(DECODER_PREFIXES)


def freeze_encoder(model: nn.Module) -> list[str]:
    """Shared token/position embeddings are part of the encoder and stay frozen."""
    trainable = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(decoder_parameter(name))
        if parameter.requires_grad:
            trainable.append(name)
    return trainable


def decoder_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if decoder_parameter(name)
    }


def load_decoder_overlay(model: nn.Module, checkpoint: Mapping) -> None:
    state = checkpoint.get("decoder")
    expected = {name for name in model.state_dict() if decoder_parameter(name)}
    if not isinstance(state, Mapping) or set(state) != expected:
        raise ValueError("Decoder overlay must contain all and only decoder-specific weights")
    for name, value in state.items():
        if value.shape != model.state_dict()[name].shape or not torch.isfinite(value).all():
            raise ValueError(f"Invalid decoder overlay tensor: {name}")
    model.load_state_dict(state, strict=False)


def decode_completion(
    tokens: list[int], vocabulary, *, requested_bars: int | None, token_budget: int,
    chunked: bool = False,
) -> dict:
    """Inspect raw tokens before MIDI serialization loses trailing rests."""
    eos = vocabulary.eos_id in tokens
    effective = tokens[: tokens.index(vocabulary.eos_id) + 1] if eos else tokens
    bars = effective.count(vocabulary.id("BAR"))
    beats = effective.count(vocabulary.id("BEAT")) + effective.count(vocabulary.id("REST"))
    full = requested_bars is None or (bars >= requested_bars and beats >= 4 * requested_bars)
    return dict(
        eos=eos,
        token_count=len(effective),
        token_budget=token_budget,
        # A chunked decoder omits joint EOS when any local budget is exhausted;
        # short sibling chunks can leave the joint sequence below its full cap.
        budget_exhausted=not eos and (chunked or len(effective) >= token_budget),
        bars=bars,
        timeline_beats=beats,
        requested_bars=requested_bars,
        complete=bool(eos and full),
    )

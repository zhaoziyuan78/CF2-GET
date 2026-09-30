"""Parameter-free cross-style residual preservation in normalized codec latents."""

from __future__ import annotations

import torch
from torch import Tensor


def centered_content_loss(
    generated: Tensor, source: Tensor, source_ids: Tensor, target_ids: Tensor, means: Tensor
) -> Tensor:
    """Permit class-mean displacement while anchoring within-class source detail.

    Means are estimated offline from training data only, never from held-out or
    generated MIDI. This is a latent geometric prior, not identified causal content.
    """
    if source.shape != generated.shape or source.ndim != 3:
        raise ValueError("Expected matching [batch, tokens, channels] latents")
    if means.ndim != 3 or means.shape[1:] != source.shape[1:] or means.shape[0] < 2:
        raise ValueError("Class means do not match the latent shape")
    if means.requires_grad or not torch.isfinite(means).all():
        raise ValueError("Class means must be fixed and finite")
    for ids in (source_ids, target_ids):
        if ids.ndim != 1 or ids.shape[0] != source.shape[0] or ids.dtype != torch.long:
            raise ValueError("Expected one integer class id per sample")
        if ((ids < 0) | (ids >= means.shape[0])).any():
            raise ValueError("Class id outside fixed vocabulary")
    reference = source.detach().float() - means.index_select(0, source_ids).float()
    residual = generated.float() - means.index_select(0, target_ids).float()
    return (residual - reference).square().mean()

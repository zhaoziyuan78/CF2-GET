"""Deterministic DDIM update equations and timestep grids."""

from __future__ import annotations

import torch
from torch import Tensor


def ddim_timesteps(
    train_timesteps: int, num_steps: int, *, descending: bool, max_timestep: int | None = None
) -> Tensor:
    top = train_timesteps - 1 if max_timestep is None else max_timestep
    if not 0 <= top < train_timesteps:
        raise ValueError("max_timestep must lie in [0, train_timesteps)")
    if not 1 <= num_steps <= top + 1:
        raise ValueError("num_steps must lie in [1, max_timestep + 1]")
    grid = torch.linspace(0, top, num_steps).round().long().unique()
    return grid.flip(0) if descending else grid


def predict_clean(state: Tensor, epsilon: Tensor, alpha: Tensor | float) -> Tensor:
    alpha_tensor = torch.as_tensor(alpha, device=state.device, dtype=state.dtype)
    return (state - (1 - alpha_tensor).sqrt() * epsilon) / alpha_tensor.sqrt().clamp_min(1e-8)


def deterministic_ddim_update(
    state: Tensor,
    epsilon: Tensor,
    alpha_current: Tensor | float,
    alpha_next: Tensor | float,
) -> Tensor:
    clean = predict_clean(state, epsilon, alpha_current)
    next_alpha = torch.as_tensor(alpha_next, device=state.device, dtype=state.dtype)
    return next_alpha.sqrt() * clean + (1 - next_alpha).sqrt() * epsilon

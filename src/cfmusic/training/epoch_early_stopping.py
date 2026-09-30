"""Serializable epoch-boundary validation plateau detection; no objective changes."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass
class EpochEarlyStopping:
    patience: int = 5
    min_epochs: int = 10
    min_steps: int = 3000
    relative_delta: float = 0.001
    absolute_delta: float = 0.0001
    best: float | None = None
    bad_epochs: int = 0
    last_epoch: int = 0
    stopped: bool = False

    def __post_init__(self) -> None:
        if self.patience < 1 or self.min_epochs < 1 or self.min_steps < 0:
            raise ValueError("Invalid early-stopping patience/minimum budget")
        if any(not math.isfinite(x) or x < 0 for x in (self.relative_delta, self.absolute_delta)):
            raise ValueError("Early-stopping tolerances must be finite and nonnegative")
        if self.best is not None and not math.isfinite(self.best):
            raise ValueError("Invalid saved plateau reference")
        if self.bad_epochs < 0 or self.last_epoch < 0:
            raise ValueError("Invalid saved early-stopping cursor")

    def observe(self, loss: float, *, completed_epochs: int, global_step: int) -> bool:
        """Call exactly once per complete epoch, never on a mid-epoch checkpoint."""
        if not math.isfinite(loss):
            raise FloatingPointError("Non-finite validation loss cannot signal convergence")
        if completed_epochs <= self.last_epoch or global_step < 0:
            raise ValueError("Duplicate/backwards early-stopping observation")
        if self.stopped:
            raise ValueError("Cannot advance a completed early-stopping run")
        self.last_epoch = completed_epochs
        if completed_epochs < self.min_epochs or global_step < self.min_steps:
            # Do not spend patience on initialization or the LR warmup.
            self.best, self.bad_epochs = None, 0
            return False
        threshold = max(self.absolute_delta, abs(self.best or 0.0) * self.relative_delta)
        if self.best is None or self.best - loss > threshold:
            self.best, self.bad_epochs = loss, 0
        else:
            self.bad_epochs += 1
        self.stopped = self.bad_epochs >= self.patience
        return self.stopped

    def state_dict(self) -> dict:
        return asdict(self)

    def load_state_dict(self, state: dict) -> None:
        restored = type(self)(**state)
        fields = ("patience", "min_epochs", "min_steps", "relative_delta", "absolute_delta")
        if any(getattr(restored, key) != getattr(self, key) for key in fields):
            raise ValueError("Early-stopping recipe changed on resume")
        self.__dict__.update(restored.__dict__)

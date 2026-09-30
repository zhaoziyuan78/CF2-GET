import math

import pytest

from cfmusic.training.epoch_early_stopping import EpochEarlyStopping


def test_requires_full_minimum_budget_and_patience():
    stop = EpochEarlyStopping(patience=2, min_epochs=3, min_steps=50)
    for epoch in range(1, 6):
        assert not stop.observe(1.0, completed_epochs=epoch, global_step=epoch * 10)
    assert stop.bad_epochs == 0
    assert not stop.observe(1.0, completed_epochs=6, global_step=60)
    assert stop.observe(1.0, completed_epochs=7, global_step=70)


def test_significant_improvement_resets_patience_but_small_drift_accumulates():
    stop = EpochEarlyStopping(
        patience=2, min_epochs=1, min_steps=0, relative_delta=0.01, absolute_delta=0
    )
    assert not stop.observe(1, completed_epochs=1, global_step=1)
    assert not stop.observe(0.995, completed_epochs=2, global_step=2)
    assert not stop.observe(0.98, completed_epochs=3, global_step=3)
    assert stop.bad_epochs == 0
    assert not stop.observe(0.98, completed_epochs=4, global_step=4)
    assert stop.observe(0.981, completed_epochs=5, global_step=5)


def test_full_state_resume_preserves_plateau_history():
    stop = EpochEarlyStopping(patience=2, min_epochs=1, min_steps=0)
    stop.observe(1, completed_epochs=1, global_step=10)
    stop.observe(1, completed_epochs=2, global_step=20)
    resumed = EpochEarlyStopping(patience=2, min_epochs=1, min_steps=0)
    resumed.load_state_dict(stop.state_dict())
    assert resumed.observe(1, completed_epochs=3, global_step=30)
    assert resumed.stopped
    with pytest.raises(ValueError, match="recipe changed"):
        EpochEarlyStopping().load_state_dict(stop.state_dict())


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_loss_is_failure_not_early_stop(value):
    with pytest.raises(FloatingPointError):
        EpochEarlyStopping().observe(value, completed_epochs=1, global_step=10)


def test_duplicate_epoch_rejected():
    stop = EpochEarlyStopping()
    stop.observe(1, completed_epochs=1, global_step=10)
    with pytest.raises(ValueError, match="Duplicate"):
        stop.observe(1, completed_epochs=1, global_step=10)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(patience=0),
        dict(min_epochs=0),
        dict(min_steps=-1),
        dict(relative_delta=math.nan),
        dict(absolute_delta=-1),
    ],
)
def test_bad_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        EpochEarlyStopping(**kwargs)

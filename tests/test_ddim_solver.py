import torch
from torch import nn

from cfmusic.solvers.ddim import ddim_timesteps, deterministic_ddim_update
from cfmusic.solvers.schedules import cosine_alpha_cumprod
from cfmusic.transport.conditional_ddim import ConditionalDDIM


def test_schedule_and_timestep_indexing() -> None:
    alpha = cosine_alpha_cumprod(100)
    assert torch.all(alpha[1:] <= alpha[:-1])
    descending = ddim_timesteps(100, 10, descending=True)
    assert descending[0] == 99 and descending[-1] == 0
    assert torch.all(descending[1:] < descending[:-1])


def test_ddim_update_is_deterministic() -> None:
    state, epsilon = torch.randn(2, 3, 4), torch.randn(2, 3, 4)
    first = deterministic_ddim_update(state, epsilon, 0.4, 0.6)
    second = deterministic_ddim_update(state, epsilon, 0.4, 0.6)
    assert torch.equal(first, second)


def test_timestep_grid_respects_max_timestep() -> None:
    grid = ddim_timesteps(1000, 32, descending=True, max_timestep=955)
    assert grid[0] == 955 and grid[-1] == 0
    assert len(grid) == 32


class BiasedGaussianDenoiser(nn.Module):
    """Exact posterior epsilon for x0 ~ N(0, I), plus a small systematic error."""

    def __init__(self, alpha_cumprod: torch.Tensor, bias: float) -> None:
        super().__init__()
        self.alpha_cumprod = alpha_cumprod
        self.bias = bias

    def forward(self, state: torch.Tensor, time: torch.Tensor, condition: object) -> torch.Tensor:
        timestep = (time * (len(self.alpha_cumprod) - 1)).round().long()
        alpha = self.alpha_cumprod[timestep][:, None, None]
        return (1 - alpha).sqrt() * state + self.bias


def test_ddim_sampling_excludes_singular_cosine_endpoint() -> None:
    alpha = cosine_alpha_cumprod(1000)
    exact = BiasedGaussianDenoiser(alpha, bias=0.0)
    biased = BiasedGaussianDenoiser(alpha, bias=0.02)
    noise = torch.randn(64, 8, 16, generator=torch.Generator().manual_seed(0))
    latent = torch.randn(64, 8, 16, generator=torch.Generator().manual_seed(1))

    def run(floor: float | None) -> tuple[float, float]:
        options = {} if floor is None else {"min_alpha_cumprod": floor}
        model = ConditionalDDIM(biased, **options)  # type: ignore[arg-type]
        reference = ConditionalDDIM(exact, **options)  # type: ignore[arg-type]
        sample = model.predict(noise, None, num_steps=32)  # type: ignore[arg-type]
        target = reference.predict(noise, None, num_steps=32)  # type: ignore[arg-type]
        abducted = model.abduct(latent, None, num_steps=32)  # type: ignore[arg-type]
        reconstructed = model.predict(abducted, None, num_steps=32)  # type: ignore[arg-type]
        return (
            float((sample - target).square().mean().sqrt()),
            float((reconstructed - latent).square().mean()),
        )

    default = ConditionalDDIM(exact)  # type: ignore[arg-type]
    assert alpha[default.max_timestep] >= default.min_alpha_cumprod > alpha[default.max_timestep + 1]
    # Including the clamped endpoint (sigma = 1000) amplifies a 2% epsilon error ~50x.
    singular_error, singular_roundtrip = run(1e-7)
    fixed_error, fixed_roundtrip = run(None)
    assert singular_error > 0.5 and singular_roundtrip > 0.5
    assert fixed_error < 0.15 and fixed_roundtrip < 0.02

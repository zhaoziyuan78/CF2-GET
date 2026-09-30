"""Latent-MIDI-VAE: MIDI-VAE style-coordinate intervention on frozen BEAT latents.

MIDI-VAE's first K posterior coordinates carry a softmax style label. Transfer
swaps source/target coordinates (Brunner et al., 2018, section 3.3). The latent
variant uses the same intervention on an additional VAE above frozen BEAT codes.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from cfmusic.conditioning.schema import ConditionBatch
from cfmusic.transport.counterfactual import CounterfactualOutput


def swap_style(code: Tensor, source: Tensor, target: Tensor, num_styles: int) -> Tensor:
    if num_styles > code.shape[-1] or (source < 0).any() or (target < 0).any():
        raise ValueError("Invalid style coordinates")
    if (source >= num_styles).any() or (target >= num_styles).any():
        raise ValueError("Style outside active vocabulary")
    result = code.clone()
    rows = torch.arange(len(code), device=code.device)
    result[rows, source] = code[rows, target]
    result[rows, target] = code[rows, source]
    return result


def gaussian_kl(mean: Tensor, logvar: Tensor) -> Tensor:
    return 0.5 * (mean.square() + logvar.exp() - 1 - logvar).mean()


class LatentMIDIVAE(nn.Module):
    representation_kind = "style_content_posterior_not_gaussian_noise"
    uses_source_condition = True

    def __init__(
        self,
        *,
        latent_tokens: int,
        latent_dim: int,
        hidden_dim: int,
        code_dim: int,
        num_styles: int,
        layers: int = 2,
        beta: float = 0.001,
        style_weight: float = 0.1,
    ) -> None:
        super().__init__()
        if code_dim <= num_styles:
            raise ValueError("MIDI-VAE requires content dimensions beyond the style coordinates")
        self.num_styles, self.beta, self.style_weight = num_styles, beta, style_weight
        self.input = nn.Linear(latent_dim, hidden_dim)
        self.encoder = nn.GRU(hidden_dim, hidden_dim, layers, batch_first=True)
        self.posterior = nn.Linear(hidden_dim, code_dim * 2)
        self.decoder_input = nn.Linear(code_dim, hidden_dim)
        self.slots = nn.Parameter(torch.randn(1, latent_tokens, hidden_dim) * 0.02)
        self.decoder = nn.GRU(hidden_dim, hidden_dim, layers, batch_first=True)
        self.output = nn.Linear(hidden_dim, latent_dim)

    def encode(self, value: Tensor) -> tuple[Tensor, Tensor]:
        _, hidden = self.encoder(torch.tanh(self.input(value)))
        mean, logvar = self.posterior(hidden[-1]).chunk(2, -1)
        return mean, logvar.clamp(-12, 8)

    def decode(self, code: Tensor) -> Tensor:
        inputs = torch.tanh(self.decoder_input(code))[:, None] + self.slots
        hidden, _ = self.decoder(inputs)
        return self.output(hidden)

    def forward(self, value: Tensor, labels: Tensor) -> dict[str, Tensor]:
        mean, logvar = self.encode(value)
        code = mean + torch.randn_like(mean) * (0.5 * logvar).exp() if self.training else mean
        reconstruction = F.mse_loss(self.decode(code), value)
        kl = gaussian_kl(mean, logvar)
        style = F.cross_entropy(mean[:, : self.num_styles], labels)
        return {
            "loss": reconstruction + self.beta * kl + self.style_weight * style,
            "reconstruction_loss": reconstruction,
            "kl_loss": kl,
            "style_loss": style,
        }

    def abduct(
        self,
        latent: Tensor,
        condition: ConditionBatch,
        *,
        num_steps: int = 1,
        track_grad: bool = False,
    ) -> Tensor:
        del condition, num_steps, track_grad
        return self.encode(latent)[0][:, None]

    def predict(
        self,
        noise: Tensor,
        condition: ConditionBatch,
        *,
        num_steps: int = 1,
        source_condition: ConditionBatch | None = None,
        track_grad: bool = False,
    ) -> Tensor:
        del num_steps, track_grad
        if source_condition is None:
            raise ValueError(
                "MIDI-VAE transfer requires observed source style, not an inferred label"
            )
        code = swap_style(
            noise[:, 0], source_condition.style_id, condition.style_id, self.num_styles
        )
        return self.decode(code)

    def counterfactual(
        self, latent: Tensor, source: ConditionBatch, target: ConditionBatch, *, num_steps: int = 1
    ) -> CounterfactualOutput:
        noise = self.abduct(latent, source)
        return CounterfactualOutput(
            latent,
            noise,
            self.decode(noise[:, 0]),
            self.predict(noise, target, source_condition=source),
            source,
            target,
            1,
            2,
        )

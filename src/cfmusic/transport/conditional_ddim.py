"""Conditional deterministic DDIM generation, vanilla inversion, and FPI."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from cfmusic.conditioning.schema import ConditionBatch, apply_condition_dropout
from cfmusic.models.latent_vector_field import ConditionalVectorField
from cfmusic.progress import track
from cfmusic.solvers.ddim import ddim_timesteps, deterministic_ddim_update
from cfmusic.solvers.ddim_fpi import fixed_point_inversion_step
from cfmusic.solvers.schedules import cosine_alpha_cumprod
from cfmusic.transport.counterfactual import CounterfactualOutput

# Epsilon prediction recovers x0 = (x - sqrt(1 - a) eps) / sqrt(a), so an eps error is
# amplified by sigma = sqrt((1 - a) / a).  The cosine schedule reaches a = 0 at t = T - 1
# (clamped to 1e-6, sigma = 1000), where a 2% eps error becomes a ~20-std x0 error and the
# denoised latents blow up.  Sampling and inversion therefore stop at the last timestep
# with a >= this floor: Stable Diffusion's terminal alpha_cumprod (4.66e-3, sigma ~ 14.6).
# Training still covers every timestep, so the floor needs no retraining.
DEFAULT_MIN_ALPHA_CUMPROD = 4.6e-3


class ConditionalDDIM(nn.Module):
    alpha_cumprod: Tensor

    def __init__(
        self,
        denoiser: ConditionalVectorField,
        *,
        train_timesteps: int = 1000,
        inversion_method: str = "vanilla",
        fpi_iterations: int = 3,
        fpi_tolerance: float = 1e-5,
        fpi_stop_on_convergence: bool = True,
        min_alpha_cumprod: float = DEFAULT_MIN_ALPHA_CUMPROD,
        classifier_free_guidance: bool = False,
        condition_dropout: float = 0.0,
        abduction_guidance_scale: float = 1.0,
        reconstruction_guidance_scale: float = 1.0,
        prediction_guidance_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if inversion_method not in {"vanilla", "fixed_point"}:
            raise ValueError(f"Unknown DDIM inversion method: {inversion_method}")
        if not 0.0 < min_alpha_cumprod < 1.0:
            raise ValueError("min_alpha_cumprod must lie in (0, 1)")
        if not 0.0 <= condition_dropout < 1.0:
            raise ValueError("condition_dropout must be in [0, 1)")
        if classifier_free_guidance and condition_dropout <= 0:
            raise ValueError("Classifier-free guidance training requires positive condition_dropout")
        scales = (abduction_guidance_scale, reconstruction_guidance_scale, prediction_guidance_scale)
        if any(scale < 0 for scale in scales):
            raise ValueError("Guidance scales must be non-negative")
        # Same contract as ConditionalFlow: without CFG training there is no null
        # condition, so every stage runs unguided whatever scales are configured.
        self.classifier_free_guidance = classifier_free_guidance
        self.condition_dropout = condition_dropout if classifier_free_guidance else 0.0
        self.abduction_guidance_scale = abduction_guidance_scale
        self.reconstruction_guidance_scale = reconstruction_guidance_scale
        self.prediction_guidance_scale = prediction_guidance_scale
        self.denoiser = denoiser
        self.train_timesteps = train_timesteps
        self.inversion_method = inversion_method
        self.fpi_iterations = fpi_iterations
        self.fpi_tolerance = fpi_tolerance
        self.fpi_stop_on_convergence = fpi_stop_on_convergence
        self.register_buffer("alpha_cumprod", cosine_alpha_cumprod(train_timesteps))
        admissible = torch.nonzero(self.alpha_cumprod >= min_alpha_cumprod).flatten()
        if admissible.numel() == 0:
            raise ValueError("min_alpha_cumprod excludes every diffusion timestep")
        self.min_alpha_cumprod = min_alpha_cumprod
        self.max_timestep = int(admissible.max())
        self.last_nfe = 0
        self.last_fixed_point_residuals: list[float] = []

    def _predict(self, state: Tensor, timestep: int, condition: ConditionBatch) -> Tensor:
        normalized = state.new_full((state.shape[0],), timestep / max(1, self.train_timesteps - 1))
        return self.denoiser(state, normalized, condition)

    def _guided_predict(
        self, state: Tensor, timestep: int, condition: ConditionBatch, guidance_scale: float
    ) -> tuple[Tensor, int]:
        """Return the classifier-free guided epsilon and its network-evaluation count."""

        if not self.classifier_free_guidance or guidance_scale == 1.0:
            return self._predict(state, timestep, condition), 1
        # One batched call: the first copy is masked to the null condition.
        count = condition.batch_size
        branches = condition.index_select(
            torch.arange(count, device=state.device).repeat(2)
        ).with_condition_mask(torch.cat([state.new_zeros(count), state.new_ones(count)]))
        unconditional, conditional = self._predict(
            torch.cat([state, state]), timestep, branches
        ).chunk(2)
        return unconditional + guidance_scale * (conditional - unconditional), 2

    def training_loss(
        self,
        latent: Tensor,
        condition: ConditionBatch,
        *,
        negative_condition: ConditionBatch | None = None,
        condition_contrast_weight: float = 0.0,
        condition_contrast_margin: float = 0.0,
        condition_contrast_samples: int | None = None,
        sample_weight: Tensor | None = None,
    ) -> dict[str, Tensor]:
        timesteps = torch.randint(0, self.train_timesteps, (latent.shape[0],), device=latent.device)
        alpha = self.alpha_cumprod[timesteps].to(latent)[:, None, None]
        epsilon = torch.randn_like(latent)
        noisy = alpha.sqrt() * latent + (1 - alpha).sqrt() * epsilon
        normalized = timesteps.to(latent.dtype) / max(1, self.train_timesteps - 1)
        model_condition, dropout_fraction = apply_condition_dropout(condition, self.condition_dropout)
        prediction = self.denoiser(noisy, normalized, model_condition)
        per_sample_error = (
            functional.mse_loss(prediction, epsilon, reduction="none").flatten(1).mean(1)
        )
        if sample_weight is None:
            ddim = per_sample_error.mean()
        else:
            weights = sample_weight.to(per_sample_error)
            if weights.ndim != 1 or weights.shape[0] != latent.shape[0]:
                raise ValueError("sample_weight must contain one value per latent")
            ddim = (per_sample_error * weights).sum() / weights.sum().clamp_min(1e-8)

        contrast = latent.new_zeros(())
        condition_gap = latent.new_zeros(())
        condition_accuracy = latent.new_zeros(())
        condition_correct_error = latent.new_zeros(())
        condition_wrong_error = latent.new_zeros(())
        if negative_condition is not None and condition_contrast_weight > 0:
            count = latent.shape[0]
            if condition_contrast_samples is not None:
                count = min(count, max(1, condition_contrast_samples))
            # High diffusion timesteps are the DDIM analogue of the Gaussian
            # CFM endpoint and therefore receive the condition-discrimination
            # budget.  The standard epsilon objective still samples all times.
            indices = torch.topk(normalized, count, largest=True, sorted=False).indices
            negative_prediction = self.denoiser(
                noisy.index_select(0, indices),
                normalized.index_select(0, indices),
                negative_condition.index_select(indices),
            )
            negative_error = (
                functional.mse_loss(
                    negative_prediction, epsilon.index_select(0, indices), reduction="none"
                )
                .flatten(1)
                .mean(1)
            )
            if self.condition_dropout > 0 or condition.condition_mask is not None:
                # The primary prediction may have seen a dropped (null) condition.
                correct_prediction = self.denoiser(
                    noisy.index_select(0, indices),
                    normalized.index_select(0, indices),
                    condition.index_select(indices),
                )
                correct_error = (
                    functional.mse_loss(
                        correct_prediction, epsilon.index_select(0, indices), reduction="none"
                    )
                    .flatten(1)
                    .mean(1)
                )
            else:
                correct_error = per_sample_error.index_select(0, indices)
            gaps = negative_error - correct_error
            contrast_values = functional.relu(condition_contrast_margin - gaps)
            if sample_weight is None:
                contrast = contrast_values.mean()
            else:
                contrast_weights = sample_weight.index_select(0, indices).to(contrast_values)
                contrast = (
                    contrast_values * contrast_weights
                ).sum() / contrast_weights.sum().clamp_min(1e-8)
            condition_gap = gaps.mean()
            condition_accuracy = (gaps > 0).to(latent.dtype).mean()
            condition_correct_error = correct_error.mean()
            condition_wrong_error = negative_error.mean()
        loss = ddim + condition_contrast_weight * contrast
        return {
            "loss": loss,
            "ddim_loss": ddim,
            "condition_contrast_loss": contrast,
            "condition_gap": condition_gap,
            "condition_accuracy": condition_accuracy,
            "condition_correct_error": condition_correct_error,
            "condition_wrong_error": condition_wrong_error,
            "condition_dropout_fraction": dropout_fraction,
        }

    def predict(
        self,
        noise: Tensor,
        condition: ConditionBatch,
        *,
        num_steps: int,
        track_grad: bool = False,
        guidance_scale: float | None = None,
    ) -> Tensor:
        scale = self.prediction_guidance_scale if guidance_scale is None else guidance_scale
        context = torch.enable_grad() if track_grad else torch.no_grad()
        state = noise
        timesteps = ddim_timesteps(
            self.train_timesteps, num_steps, descending=True, max_timestep=self.max_timestep
        ).tolist()
        inference_steps = (
            timesteps
            if track_grad
            else track(
                timesteps,
                description="DDIM denoise",
                total=len(timesteps),
                unit="step",
                leave=False,
                position=2,
            )
        )
        nfe = 0
        with context:
            for index, timestep in enumerate(inference_steps):
                next_timestep = timesteps[index + 1] if index + 1 < len(timesteps) else None
                alpha_next: Tensor | float = (
                    self.alpha_cumprod[next_timestep] if next_timestep is not None else 1.0
                )
                epsilon, evaluations = self._guided_predict(state, timestep, condition, scale)
                state = deterministic_ddim_update(
                    state, epsilon, self.alpha_cumprod[timestep], alpha_next
                )
                nfe += evaluations
        self.last_nfe = nfe
        return state

    def reconstruct(
        self, noise: Tensor, condition: ConditionBatch, *, num_steps: int, track_grad: bool = False
    ) -> Tensor:
        return self.predict(
            noise,
            condition,
            num_steps=num_steps,
            track_grad=track_grad,
            guidance_scale=self.reconstruction_guidance_scale,
        )

    def abduct(
        self,
        latent: Tensor,
        condition: ConditionBatch,
        *,
        num_steps: int,
        track_grad: bool = False,
        guidance_scale: float | None = None,
    ) -> Tensor:
        scale = self.abduction_guidance_scale if guidance_scale is None else guidance_scale
        context = torch.enable_grad() if track_grad else torch.no_grad()
        state = latent
        timesteps = ddim_timesteps(
            self.train_timesteps, num_steps, descending=False, max_timestep=self.max_timestep
        ).tolist()
        inversion_steps = (
            timesteps
            if track_grad
            else track(
                timesteps,
                description="DDIM invert",
                total=len(timesteps),
                unit="step",
                leave=False,
                position=2,
            )
        )
        current_timestep: int | None = None
        nfe = 0
        self.last_fixed_point_residuals = []
        with context:
            for target_timestep in inversion_steps:
                alpha_current: Tensor | float = (
                    1.0 if current_timestep is None else self.alpha_cumprod[current_timestep]
                )
                prediction_timestep = 0 if current_timestep is None else current_timestep
                epsilon, evaluations = self._guided_predict(
                    state, prediction_timestep, condition, scale
                )
                initial = deterministic_ddim_update(
                    state, epsilon, alpha_current, self.alpha_cumprod[target_timestep]
                )
                nfe += evaluations
                if self.inversion_method == "fixed_point":

                    def predict_upper(candidate: Tensor, step: int = target_timestep) -> Tensor:
                        return self._guided_predict(candidate, step, condition, scale)[0]

                    result = fixed_point_inversion_step(
                        state,
                        initial,
                        predict_upper,
                        alpha_lower=alpha_current,
                        alpha_upper=self.alpha_cumprod[target_timestep],
                        iterations=self.fpi_iterations,
                        tolerance=self.fpi_tolerance,
                        stop_on_convergence=self.fpi_stop_on_convergence,
                    )
                    state = result.state
                    nfe += result.nfe * evaluations
                    self.last_fixed_point_residuals.extend(result.residuals)
                else:
                    state = initial
                current_timestep = target_timestep
        self.last_nfe = nfe
        return state

    def counterfactual(
        self,
        latent: Tensor,
        source_condition: ConditionBatch,
        target_condition: ConditionBatch,
        *,
        num_steps: int,
    ) -> CounterfactualOutput:
        noise = self.abduct(latent, source_condition, num_steps=num_steps)
        inverse_nfe = self.last_nfe
        reconstructed = self.reconstruct(noise, source_condition, num_steps=num_steps)
        reconstruction_nfe = self.last_nfe
        counterfactual = self.predict(noise, target_condition, num_steps=num_steps)
        return CounterfactualOutput(
            latent,
            noise,
            reconstructed,
            counterfactual,
            source_condition,
            target_condition,
            inverse_nfe,
            reconstruction_nfe + self.last_nfe,
        )

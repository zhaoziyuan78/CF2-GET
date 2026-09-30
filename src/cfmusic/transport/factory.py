"""Build CFM, DDIM, or Latent-MIDI-VAE transports from composed Hydra config."""

from __future__ import annotations

from collections.abc import Mapping

from omegaconf import DictConfig

from cfmusic.baselines.models import LatentMIDIVAE
from cfmusic.conditioning.embeddings import AdditiveConditionEmbedding
from cfmusic.models.latent_denoiser import ConditionalLatentDenoiser
from cfmusic.models.latent_vector_field import ConditionalVectorField
from cfmusic.transport.conditional_ddim import DEFAULT_MIN_ALPHA_CUMPROD, ConditionalDDIM
from cfmusic.transport.conditional_flow import ConditionalFlow


def validate_guidance_checkpoint(
    checkpoint: Mapping[str, object],
    cfg: DictConfig,
    *,
    exact_training_match: bool,
) -> None:
    """Prevent CFG inference/resume with a checkpoint that never learned a null condition."""

    requested = bool(cfg.get("classifier_free_guidance", False))
    saved_config = checkpoint.get("config")
    saved = (
        bool(saved_config.get("classifier_free_guidance", False))
        if isinstance(saved_config, Mapping)
        else False
    )
    saved_dropout = (
        float(saved_config.get("condition_dropout", 0.0))
        if isinstance(saved_config, Mapping)
        else 0.0
    )
    requested_dropout = float(cfg.get("condition_dropout", 0.0))
    if requested and (not saved or saved_dropout <= 0):
        raise ValueError(
            "Classifier-free guidance requires a checkpoint trained with positive condition "
            "dropout; disable CFG for this checkpoint or train the unified CFG recipe"
        )
    if exact_training_match and (
        requested != saved or (requested and abs(requested_dropout - saved_dropout) > 1e-12)
    ):
        raise ValueError(
            "Cannot resume with different classifier-free-guidance or condition-dropout "
            "training settings; inference guidance scales may be changed safely"
        )


def _embedding(cfg: DictConfig) -> AdditiveConditionEmbedding:
    return AdditiveConditionEmbedding(
        num_datasets=int(cfg.num_datasets),
        num_tasks=int(cfg.num_tasks),
        num_styles=int(cfg.num_styles),
        num_genres=int(cfg.num_genres),
        num_emotions=int(cfg.num_emotions),
        embedding_dim=int(cfg.embedding_dim),
    )


def create_transport(cfg: DictConfig) -> ConditionalFlow | ConditionalDDIM | LatentMIDIVAE:
    model_cfg = cfg.model
    if str(cfg.type) == "latent_midi_vae":
        return LatentMIDIVAE(
            latent_tokens=int(model_cfg.latent_tokens),
            latent_dim=int(model_cfg.latent_dim),
            hidden_dim=int(model_cfg.hidden_dim),
            layers=int(model_cfg.layers),
            num_styles=int(cfg.conditioning.num_styles),
            code_dim=int(model_cfg.code_dim),
            beta=float(cfg.loss.beta),
            style_weight=float(cfg.loss.style_weight),
        )
    embedding = _embedding(cfg.conditioning)
    model_type = ConditionalLatentDenoiser if str(cfg.type) == "ddim" else ConditionalVectorField
    backbone = model_type(
        latent_dim=int(model_cfg.latent_dim),
        hidden_dim=int(model_cfg.hidden_dim),
        layers=int(model_cfg.layers),
        heads=int(model_cfg.heads),
        mlp_ratio=int(model_cfg.mlp_ratio),
        dropout=float(model_cfg.dropout),
        condition_embedding=embedding,
        zero_init_output=bool(model_cfg.zero_init_output),
        gradient_checkpointing=bool(model_cfg.get("gradient_checkpointing", False)),
        latent_tokens=model_cfg.get("latent_tokens"),
        slot_identity=bool(model_cfg.get("slot_identity", False)),
    )
    legacy_guidance_scale = float(cfg.get("guidance_scale", 1.0))
    if str(cfg.type) == "ddim":
        if float(cfg.get("source_repulsion_scale", 0.0)) > 0 or cfg.get(
            "prediction_guidance_interval"
        ) is not None:
            raise ValueError("DDIM supports plain classifier-free guidance only")
        return ConditionalDDIM(
            backbone,
            train_timesteps=int(cfg.diffusion.train_timesteps),
            inversion_method=str(cfg.ddim_inversion.method),
            fpi_iterations=int(cfg.ddim_inversion.iterations),
            fpi_tolerance=float(cfg.ddim_inversion.tolerance),
            fpi_stop_on_convergence=bool(cfg.ddim_inversion.stop_on_convergence),
            min_alpha_cumprod=float(
                cfg.diffusion.get("min_alpha_cumprod", DEFAULT_MIN_ALPHA_CUMPROD)
            ),
            classifier_free_guidance=bool(cfg.get("classifier_free_guidance", False)),
            condition_dropout=float(cfg.get("condition_dropout", 0.0)),
            abduction_guidance_scale=float(cfg.get("abduction_guidance_scale", 1.0)),
            reconstruction_guidance_scale=float(cfg.get("reconstruction_guidance_scale", 1.0)),
            prediction_guidance_scale=float(
                cfg.get("prediction_guidance_scale", legacy_guidance_scale)
            ),
        )
    ot_cfg = cfg.flow.get("ot")
    return ConditionalFlow(
        backbone,
        solver_method=str(cfg.solver.method),
        time_sampling=str(cfg.flow.time_sampling),
        ot_solver=str(ot_cfg.solver) if str(cfg.flow.path) == "ot" and ot_cfg else None,
        ot_projection_dim=int(ot_cfg.cost_projection_dim) if ot_cfg else 128,
        ot_regularization=float(ot_cfg.regularization) if ot_cfg else 0.05,
        classifier_free_guidance=bool(cfg.get("classifier_free_guidance", False)),
        condition_dropout=float(cfg.get("condition_dropout", 0.0)),
        guidance_scale=legacy_guidance_scale,
        abduction_guidance_scale=float(cfg.get("abduction_guidance_scale", 1.0)),
        reconstruction_guidance_scale=float(cfg.get("reconstruction_guidance_scale", 1.0)),
        prediction_guidance_scale=float(
            cfg.get("prediction_guidance_scale", legacy_guidance_scale)
        ),
        source_repulsion_scale=float(cfg.get("source_repulsion_scale", 0.0)),
        prediction_guidance_interval=(
            tuple(float(value) for value in cfg.prediction_guidance_interval)
            if cfg.get("prediction_guidance_interval") is not None else None
        ),
    )

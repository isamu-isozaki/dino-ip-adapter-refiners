import torch
from torch import Tensor
from torch.nn import functional as F
from dino_ip_adapter_refiners.diffusion_utils import (
    scale_loss,
    sample_noise,
    sample_timesteps,
    add_noise_to_latents,
    TimestepSampler,
    LossScaler,
)
from dino_ip_adapter_refiners.mixin import IPAdapterMixin, EvaluationMixin, AMPMixin, DataMixin
from dino_ip_adapter_refiners.data import BatchOnlyImage, Batch
from dino_ip_adapter_refiners.config import SaveAdapterConfig
from refiners.training_utils import CallbackConfig, ModelConfig

from refiners.foundationals.latent_diffusion import SD1UNet

from typing import TypeVar, Generic
import os
from refiners.training_utils.callback import Callback
from refiners.fluxion.utils import save_to_safetensors
from refiners.training_utils.trainer import register_callback

BatchT = TypeVar("BatchT", bound="BatchOnlyImage | Batch")
class ComputeGradNormCallback(Callback["BaseTrainer"]):
    """Callback to compute gradient norm"""

    def on_backward_end(self, trainer: "BaseTrainer") -> None:
        if trainer.clock.is_evaluation_step:
            for name, param in trainer.ip_adapter.named_parameters():
                if param.grad is not None:
                    grads = param.grad.detach().data
                    grad_norm = (grads.norm(p=2) / grads.numel()).item()
                    trainer.wandb_log(data={"grad_norm/" + name: grad_norm})
        return super().on_backward_end(trainer)


class ComputeParamNormCallback(Callback["BaseTrainer"]):
    """Callback to compute gradient norm"""

    def on_backward_end(self, trainer: "BaseTrainer") -> None:
        if trainer.clock.is_evaluation_step:
            for name, param in trainer.ip_adapter.named_parameters():
                if param.grad is not None:
                    data = param.data.detach()
                    data_norm = (data.norm(p=2) / data.numel()).item()
                    trainer.wandb_log(data={"param_norm/" + name: data_norm})
        return super().on_backward_end(trainer)


class SaveAdapterCallback(Callback["BaseTrainer"]):
    """Callback to save the adapter when a checkpoint is saved."""
    def __init__(self) -> None:
        super().__init__()

    def on_backward_end(self, trainer: "BaseTrainer") -> None:
        if trainer.clock.iteration % trainer.config.save_adapter.checkpoint_steps == 0:
            os.makedirs(trainer.config.save_adapter.save_folder, exist_ok=True)
            cross_attention_adapters = trainer.ip_adapter.sub_adapters
            image_proj = trainer.ip_adapter.image_proj

            tensors: dict[str, Tensor] = {}
            tensors |= {f"image_proj.{key}": value for key, value in image_proj.state_dict().items()}
            for i, cross_attention_adapter in enumerate(cross_attention_adapters):
                tensors |= {f"ip_adapter.{i:03d}.{key}": value for key, value in cross_attention_adapter.state_dict().items()}
            save_to_safetensors(
                path= f"{trainer.config.save_adapter.save_folder}/step{trainer.clock.iteration}.safetensors",
                tensors=tensors,
            )

class BaseTrainer(
    Generic[BatchT],
    IPAdapterMixin[BatchT],
    EvaluationMixin,
    AMPMixin[BatchT],
    DataMixin[BatchT]
):
    @register_callback()
    def compute_grad_norms(self, config: CallbackConfig) -> ComputeGradNormCallback:
        return ComputeGradNormCallback()
    @register_callback()
    def compute_param_norms(self, config: CallbackConfig) -> ComputeParamNormCallback:
        return ComputeParamNormCallback()
    @register_callback()
    def save_adapter(self, config: SaveAdapterConfig) -> SaveAdapterCallback:
        return SaveAdapterCallback()
    @register_model()
    def unet(self, config: ModelConfig) -> SD1UNet:
        unet = SD1UNet(in_channels=4, device=self.device, dtype=self.dtype)
        unet.load_from_safetensors(self.config.extra_training.unet_checkpoint)
        return unet
class Trainer(
    BaseTrainer[Batch]
):
    def compute_loss(self, batch: Batch) -> torch.Tensor:
        batch = batch.to(device=self.device, dtype=self.dtype)

        timesteps = sample_timesteps(
            len(batch), sampler=TimestepSampler.UNIFORM, device=self.device
        )
        self.unet.set_timestep(timesteps)

        noise = sample_noise(batch.latent.shape, device=self.device, dtype=self.dtype)
        noisy_latent = add_noise_to_latents(
            latents=batch.latent, noise=noise, solver=self.solver, timesteps=timesteps
        )

        image_embedding = self.ip_adapter.get_image_embedding(
            batch.dino_embedding, drop_rate=0.1
        )
        self.ip_adapter.set_image_context(image_embedding)

        predicted_noise = self.unet(noisy_latent)
        loss = F.mse_loss(input=predicted_noise, target=noise, reduction="none")
        rescaled_loss = scale_loss(
            loss,
            timesteps=timesteps,
            scaler=LossScaler.LEGACY,
            solver=self.solver,
        )

        return rescaled_loss.mean()

class TrainerOnlyImage(
    BaseTrainer[BatchOnlyImage]
):
    def compute_loss(self, batch: BatchOnlyImage) -> torch.Tensor:
        batch = batch.to(device=self.device, dtype=self.dtype)

        timesteps = sample_timesteps(
            len(batch), sampler=TimestepSampler.UNIFORM, device=self.device
        )
        self.unet.set_timestep(timesteps)

        noise = sample_noise(batch.latent.shape, device=self.device, dtype=self.dtype)
        noisy_latent = add_noise_to_latents(
            latents=batch.latent, noise=noise, solver=self.solver, timesteps=timesteps
        )

        image_embedding = self.ip_adapter.get_image_embedding(
            batch.dino_embedding, drop_rate=0.1
        )
        self.ip_adapter.set_image_context(image_embedding)

        predicted_noise = self.unet(noisy_latent)
        loss = F.mse_loss(input=predicted_noise, target=noise, reduction="none")
        rescaled_loss = scale_loss(
            loss,
            timesteps=timesteps,
            scaler=LossScaler.LEGACY,
            solver=self.solver,
        )

        return rescaled_loss.mean()

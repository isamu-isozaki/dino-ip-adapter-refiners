from refiners.foundationals.latent_diffusion.solvers import DDPM
import torch
from torch.nn import functional as F
from refiners.training_utils import Trainer as AbstractTrainer
from dino_ip_adapter_refiners.config import Config
from dino_ip_adapter_refiners.diffusion_utils import (
    scale_loss,
    sample_noise,
    sample_timesteps,
    add_noise_to_latents,
    TimestepSampler,
    LossScaler,
)
from dino_ip_adapter_refiners.mixin.sd1 import SD1Mixin
from dino_ip_adapter_refiners.mixin.evaluation import EvaluationMixin
from dino_ip_adapter_refiners.mixin.ip_adapter import IPAdapterMixin
from dino_ip_adapter_refiners.mixin.webdataset_data import WebdatasetDataMixin, Batch


class Trainer(
    WebdatasetDataMixin,
    SD1Mixin,
    EvaluationMixin,
    IPAdapterMixin,
    AbstractTrainer[Config, Batch],
):
    @property
    def solver(self) -> DDPM:
        return DDPM(1000, device=self.device)

    def compute_loss(self, batch: Batch) -> torch.Tensor:
        batch_size = len(batch)

        timesteps = sample_timesteps(
            batch_size, sampler=TimestepSampler.UNIFORM, device=self.device
        )
        self.unet.set_context("timesteps", timesteps)

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

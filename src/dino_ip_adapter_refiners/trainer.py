from functools import cached_property
from refiners.foundationals.latent_diffusion import SD1UNet
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
from dino_ip_adapter_refiners.mixin.evaluation import EvaluationMixin
from dino_ip_adapter_refiners.mixin.ip_adapter import IPAdapterMixin
from dino_ip_adapter_refiners.data import DatasetAdapter, BatchOnlyImage, Batch

class Trainer(
    EvaluationMixin,
    IPAdapterMixin,
    AbstractTrainer[Config, Batch],
):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        assert config.dataset.only_image
        # TODO: Change this to Mixin once I figure out __init__ for multi inheritance
        self.dataset_adapter = DatasetAdapter(config.dataset)
    def get_item(self, index: int) -> Batch:
        item =  self.dataset_adapter.get_item(index)
        assert isinstance(item, Batch)
        return item
    def collate_fn(self, batch: list[Batch]) -> Batch:
        output = self.dataset_adapter.collate_fn(batch)
        assert isinstance(output, Batch)
        return output
    @cached_property
    def unet(self) -> SD1UNet:
        return SD1UNet(in_channels=4, device=self.device, dtype=self.dtype)

    @property
    def solver(self) -> DDPM:
        return DDPM(1000, device=self.device)

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
    EvaluationMixin,
    IPAdapterMixin,
    AbstractTrainer[Config, BatchOnlyImage],
):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        assert config.dataset.only_image
        # TODO: Change this to Mixin once I figure out __init__ for multi inheritance
        self.dataset_adapter = DatasetAdapter(config.dataset)
    def get_item(self, index: int) -> BatchOnlyImage:
        item =  self.dataset_adapter.get_item(index)
        assert isinstance(item, BatchOnlyImage)
        return item
    def collate_fn(self, batch: list[BatchOnlyImage]) -> BatchOnlyImage:
        output = self.dataset_adapter.collate_fn(batch)
        assert isinstance(output, BatchOnlyImage)
        return output
    @cached_property
    def unet(self) -> SD1UNet:
        return SD1UNet(in_channels=4, device=self.device, dtype=self.dtype)

    @property
    def solver(self) -> DDPM:
        return DDPM(1000, device=self.device)

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

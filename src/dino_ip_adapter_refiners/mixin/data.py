from dino_ip_adapter_refiners.mixin.base import BaseMixin, BatchT
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
from dino_ip_adapter_refiners.utils import register_model

from torch.utils.data import DataLoader
from typing import Any, TypeVar, Generic
from torch import Tensor, float16, nn
from torch.cuda.amp import GradScaler, autocast
from refiners.training_utils.trainer import backward
from refiners.fluxion.utils import no_grad
from refiners.training_utils.common import (
    scoped_seed,
)
class DataMixin(
    Generic[BatchT],
    BaseMixin[BatchT]
):
    def __init__(self, config: Config):
        self.dataset_adapter = DatasetAdapter(config.dataset)
        super().__init__(config)
    @register_model()
    def unet(self) -> SD1UNet:
        unet = SD1UNet(in_channels=4, device=self.device, dtype=self.dtype)
        unet.load_from_safetensors(self.config.extra_training.unet_checkpoint)
        return unet
    @property
    def solver(self) -> DDPM:
        return DDPM(1000, device=self.device)
    def get_item(self, index: int) -> BatchT:
        return self.dataset_adapter.get_item(index)
    def collate_fn(self, batch: list[BatchT]) -> BatchT:
        return self.dataset_adapter.collate_fn(batch)
    @cached_property
    def dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.dataset, batch_size=self.config.training.batch_size, num_workers=self.config.dataset.dataset_workers, shuffle=True, collate_fn=self.collate_fn
        )
    @property
    def dataset_length(self) -> int:
        return self.dataset_adapter.dataset_length
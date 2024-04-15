from refiners.training_utils import Trainer as AbstractTrainer
from dino_ip_adapter_refiners.config import Config
from dino_ip_adapter_refiners.data import BatchOnlyImage, Batch
from typing import TypeVar, Generic
from refiners.training_utils.wandb import WandbMixin
from refiners.foundationals.latent_diffusion.solvers import DDPM

BatchT = TypeVar("BatchT", bound="BatchOnlyImage | Batch")

class BaseMixin(
    Generic[BatchT],
    AbstractTrainer[Config, BatchT],
    WandbMixin
):
    @property
    def solver(self) -> DDPM:
        return DDPM(1000, device=self.device)
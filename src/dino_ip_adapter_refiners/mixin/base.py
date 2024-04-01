from refiners.training_utils import Trainer as AbstractTrainer
from dino_ip_adapter_refiners.config import Config
from dino_ip_adapter_refiners.data import BatchOnlyImage, Batch
from typing import TypeVar, Generic
from refiners.training_utils.wandb import WandbMixin

BatchT = TypeVar("BatchT", bound="BatchOnlyImage | Batch")

class BaseMixin(
    Generic[BatchT],
    AbstractTrainer[Config, BatchT],
    WandbMixin
):
    pass
from dino_ip_adapter_refiners.mixin.base import BaseMixin, BatchT
from functools import cached_property
from refiners.foundationals.latent_diffusion.solvers import DDPM
from dino_ip_adapter_refiners.config import Config

from dino_ip_adapter_refiners.data import DatasetAdapter

from torch.utils.data import DataLoader
from typing import Any, Generic
class DataMixin(
    Generic[BatchT],
    BaseMixin[BatchT]
):
    def __init__(self, config: Config):
        self.dataset_adapter = DatasetAdapter(config.dataset)
        super().__init__(config)
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
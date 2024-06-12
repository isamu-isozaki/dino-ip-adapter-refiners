from dino_ip_adapter_refiners.mixin.base import BaseMixin, BatchT
from functools import cached_property
from dino_ip_adapter_refiners.config import Config
from dino_ip_adapter_refiners.data import DataLoaderAdapter

from torch.utils.data import DataLoader
from typing import Any, Generic
class DataMixin(
    Generic[BatchT],
    BaseMixin[BatchT]
):
    def __init__(self, config: Config):
        self.dataloader_adapter = DataLoaderAdapter(config.dataset, config.training.batch_size, config.ip_adapter.pop)
        super().__init__(config)
    def get_item(self, index: int) -> BatchT:
        return self.dataloader_adapter.get_item(index)
    def collate_fn(self, batch: list[BatchT]) -> BatchT:
        return self.dataloader_adapter.collate_fn(batch)
    @property
    def dataset_length(self) -> int:
        return self.dataloader_adapter.dataset_length
    @cached_property
    def dataloader(self) -> DataLoader[Any]:
        return self.dataloader_adapter.dataloader
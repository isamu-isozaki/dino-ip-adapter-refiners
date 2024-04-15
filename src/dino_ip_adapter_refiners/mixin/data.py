from dino_ip_adapter_refiners.mixin.base import BaseMixin, BatchT
from dino_ip_adapter_refiners.config import Config
from dino_ip_adapter_refiners.data import DataLoaderAdapter
from typing import Generic
class DataMixin(
    Generic[BatchT],
    BaseMixin[BatchT]
):
    def __init__(self, config: Config):
        self.dataloader = DataLoaderAdapter(config.dataset, config.training.batch_size).dataloader
        super().__init__(config)
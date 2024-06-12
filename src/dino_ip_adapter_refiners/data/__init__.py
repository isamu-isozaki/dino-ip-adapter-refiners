from dino_ip_adapter_refiners.config import DatasetConfig
from dino_ip_adapter_refiners.data.mosaic import MosaicAdapter
from dino_ip_adapter_refiners.data.webdataset import WebdatasetAdapter
from dino_ip_adapter_refiners.data.utils import BatchOnlyImage, Batch
from functools import cached_property
from torch.utils.data import DataLoader
from typing import Any


class DataLoaderAdapter:
    def __init__(self, config: DatasetConfig, batch_size: int = 1, pop: bool = False):
        self.config = config
        if config.is_mosaic:
            self.dataset = MosaicAdapter(config, batch_size=batch_size, pop=pop)
        elif config.is_webdataset:
            self.dataset = WebdatasetAdapter(config, batch_size=batch_size, pop=pop)
        else:
            raise Exception("is_mosaic or is_webdataset must be true for the dataset")
        self.dataset_length = self.dataset.dataset_length
        self.dataloader = self.dataset.dataloader
    def collate_fn(self, batch: list[BatchOnlyImage] | list[Batch]) -> BatchOnlyImage | Batch:
        return self.dataset.collate_fn(batch=batch)
    def get_item(self, index: int) -> BatchOnlyImage | Batch:
        if self.config.is_webdataset:
            raise NotImplementedError("Getting individual items with webdataset is not currently supported")
        assert isinstance(self.dataset, MosaicAdapter)
        return self.dataset.get_item(index)
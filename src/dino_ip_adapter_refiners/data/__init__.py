from dino_ip_adapter_refiners.config import DatasetConfig
from dino_ip_adapter_refiners.data.mosaic import MosaicAdapter
from dino_ip_adapter_refiners.data.webdataset import WebdatasetAdapter
from dino_ip_adapter_refiners.data.utils import BatchOnlyImage, Batch
from functools import cached_property
from torch.utils.data import DataLoader
from typing import Any


class DataLoaderAdapter:
    def __init__(self, config: DatasetConfig, batch_size: int = 1):
        if config.is_mosaic:
            self.dataset = MosaicAdapter(config, batch_size=batch_size)
        elif config.is_webdataset:
            self.dataset = WebdatasetAdapter(config, batch_size=batch_size)
        else:
            raise Exception("is_mosaic or is_webdataset must be true for the dataset")
        self.dataset_length = self.dataset.dataset_length
    @cached_property
    def dataloader(self) -> DataLoader[Any]:
        return self.dataset.dataloader
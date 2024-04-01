from dino_ip_adapter_refiners.config import DatasetConfig
from dino_ip_adapter_refiners.data.mosaic import MosaicAdapter
from dino_ip_adapter_refiners.data.webdataset import WebdatasetAdapter
from dino_ip_adapter_refiners.data.utils import BatchOnlyImage, Batch

class DatasetAdapter:
    def __init__(self, config: DatasetConfig):
        if config.is_mosaic:
            self.dataset = MosaicAdapter(config.train_shards_path_or_url, config.cache_dir, shuffle=config.shuffle, cache_limit=config.cache_limit, only_image=config.only_image)
        elif config.is_webdataset:
            self.dataset = WebdatasetAdapter(config.train_shards_path_or_url, config.dataset_length, only_image=config.only_image)
        else:
            raise Exception("is_mosaic or is_webdataset must be true for the dataset")
    def get_item(self, index: int) -> BatchOnlyImage | Batch:
        return self.dataset.get_item(index)
    def collate_fn(self, batch: list[BatchOnlyImage] | list[Batch]) -> BatchOnlyImage | Batch:
        return self.dataset.collate_fn(batch)
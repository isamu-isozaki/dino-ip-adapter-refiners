from functools import cached_property
import torch
from streaming import StreamingDataset  # type: ignore

from dino_ip_adapter_refiners.data.utils import BatchOnlyImage, Batch, BaseDataAdapter
from dino_ip_adapter_refiners.config import DatasetConfig
from torch.utils.data import DataLoader
from typing import Any
from torch import cat, tensor
class MosaicAdapter(BaseDataAdapter):
    def __init__(self, config: DatasetConfig, batch_size: int =1, pop: bool = False):
        super().__init__(only_image=config.only_image)
        self.config = config
        self.train_shards_path_or_url = config.train_shards_path_or_url
        self.cache_dir = config.cache_dir
        self.shuffle = config.shuffle
        self.cache_limit = config.cache_limit
        self.only_image = config.only_image
        self.batch_size = batch_size
        self.predownload = config.predownload
        self.download_retry = config.download_retry
        self.download_timeout = config.download_timeout
        self.dataset_length = len(self.dataset)
        self.pop = pop
    @cached_property
    def dataset(self) -> StreamingDataset:
        return StreamingDataset(remote=self.train_shards_path_or_url, local=self.cache_dir, shuffle=self.shuffle, cache_limit=self.cache_limit, predownload=self.predownload, batch_size=self.batch_size, download_retry=self.download_retry, download_timeout=self.download_timeout, validate_hash=None, keep_zip=False)
    def collate_fn_from_numpy(self, batch: list[dict]) -> BatchOnlyImage | Batch:
        latents = cat(tensors=[tensor(item["sd15_lda.pth"][None]) for item in batch])
        dino_embeddings = cat([tensor(item["dinov2_vitl14_reg4_pretrain_popped.pth" if self.pop else "dinov2_vitl14_reg4_pretrain.pth"][None]) for item in batch])
        if self.only_image:
            return BatchOnlyImage(latent=latents, dino_embedding=dino_embeddings)
        else:
            text_embeddings = cat(tensors=[tensor(item["clipl.pth"][None]) for item in batch])
            return Batch(latent=latents, dino_embedding=dino_embeddings, text_embedding=text_embeddings)
    def get_item(self, index: int) -> BatchOnlyImage | Batch:
        item = self.dataset[index]
        dino_embedding = torch.tensor(item["dinov2_vitl14_reg4_pretrain_popped.pth" if self.pop else "dinov2_vitl14_reg4_pretrain.pth"])
        latent = torch.tensor(item["sd15_lda.pth"])
        if self.only_image:
            return BatchOnlyImage(
                latent=latent.unsqueeze(0),
                dino_embedding=dino_embedding.unsqueeze(0),
            )
        else:
            assert "clipl.pth" in item
            text_embedding = torch.tensor(item["clipl.pth"])
            return Batch(
                latent=latent.unsqueeze(0),
                dino_embedding=dino_embedding.unsqueeze(0),
                text_embedding=text_embedding.unsqueeze(0)
            )
    @cached_property
    def dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.dataset, batch_size=self.batch_size, num_workers=self.config.dataset_workers, collate_fn=self.collate_fn_from_numpy, persistent_workers=True
        )
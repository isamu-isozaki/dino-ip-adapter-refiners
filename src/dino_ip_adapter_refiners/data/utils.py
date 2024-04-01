from dataclasses import dataclass
from functools import cached_property
import io
import torch
from webdataset import WebDataset  # type: ignore

from dino_ip_adapter_refiners.batch import BaseBatch

class BatchOnlyImage(BaseBatch):
    """Structure of the data in the IPDataset."""

    latent: torch.Tensor
    dino_embedding: torch.Tensor

class Batch(BaseBatch):
    """Structure of the data in the IPDataset."""

    latent: torch.Tensor
    dino_embedding: torch.Tensor
    text_embedding: torch.Tensor

class BaseDataAdapter:
    def __init__(self, only_image: bool = False):
        self.only_image = only_image
    def collate_fn(self, batch: list[BatchOnlyImage] | list[Batch]) -> BatchOnlyImage | Batch:
        if self.only_image:
            assert isinstance(batch[0], BatchOnlyImage)
            return BatchOnlyImage.collate(batch)
        assert isinstance(batch[0], Batch)
        return Batch.collate(batch)
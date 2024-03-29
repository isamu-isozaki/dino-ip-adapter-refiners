from dino_ip_adapter_refiners.batch import BaseBatch
import torch


class Batch(BaseBatch):
    """Structure of the data in the IPDataset."""

    latent: torch.Tensor
    dino_embedding: torch.Tensor


class WebdatasetDataMixin:
    def get_item(self, index: int) -> Batch:
        raise NotImplementedError()

    def collate_fn(self, batch: list[Batch]) -> Batch:
        return Batch.collate(batch)

    @property
    def dataset_length(self) -> int:
        return 10

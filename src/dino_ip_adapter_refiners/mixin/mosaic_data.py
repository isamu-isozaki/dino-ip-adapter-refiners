from dataclasses import dataclass
from functools import cached_property
import io
import torch
from streaming import StreamingDataset  # type: ignore

from dino_ip_adapter_refiners.batch import BaseBatch
from streaming import StreamingDataset
text_embedding="CLIPL.pth".lower()
latent="sd15_lda.pth",
image_embedding="dinov2_vitl14_reg4_pretrain.pth",
class Batch(BaseBatch):
    """Structure of the data in the IPDataset."""

    latent: torch.Tensor
    dino_embedding: torch.Tensor
    text_embedding: torch.Tensor



class MosaicDataMixin:
    @cached_property
    def dataset(self) -> StreamingDataset:
        # Remote path where full dataset is persistently stored
        remote = 's3://my-bucket/path-to-dataset'

        # Local working dir where dataset is cached during operation
        local = '/tmp/path-to-dataset'

        # Create streaming dataset
        dataset = StreamingDataset(local=local, remote=remote, shuffle=True)

        return StreamingDataset("/home/trom/00000.tar")

    def get_next_item(self) -> Item:
        obj = next(iter(self.webdataset))  # type: ignore
        return Item(
            dino_embedding=obj["dinov2_vitl14_reg4_pretrain.pth"],  # type: ignore
            latent=obj["sd15_lda.pth"],  # type: ignore
            json=obj["json"],  # type: ignore
        )

    def get_item(self, index: int) -> Batch:
        item = self.get_next_item()
        latent = torch.load(io.BytesIO(item.latent))  # type: ignore
        dino_embedding = torch.load(io.BytesIO(item.dino_embedding))  # type: ignore
        return Batch(
            latent=latent.unsqueeze(0),
            dino_embedding=dino_embedding.unsqueeze(0),
        )

    def collate_fn(self, batch: list[Batch]) -> Batch:
        return Batch.collate(batch)

    @property
    def dataset_length(self) -> int:
        return 10_000

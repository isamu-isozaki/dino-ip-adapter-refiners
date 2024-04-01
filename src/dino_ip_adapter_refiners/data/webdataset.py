from dataclasses import dataclass
from functools import cached_property
import io
import torch
from webdataset import WebDataset  # type: ignore

from dino_ip_adapter_refiners.data.utils import BatchOnlyImage, Batch, BaseDataAdapter

@dataclass
class Item:
    dino_embedding: bytes
    latent: bytes
    json: bytes
    text_embedding: bytes | None = None


class WebdatasetAdapter(BaseDataAdapter):
    def __init__(self, train_shards_path_or_url: str, dataset_length: int, only_image: bool = False):
        super().__init__(only_image=only_image)
        self.train_shards_path_or_url = train_shards_path_or_url
        self.dataset_length = dataset_length
        self.only_image = only_image
    @cached_property
    def dataset(self) -> WebDataset:
        return WebDataset(self.train_shards_path_or_url)

    def get_next_item(self) -> Item:
        obj = next(iter(self.dataset))  # type: ignore
        return Item(
            dino_embedding=obj["dinov2_vitl14_reg4_pretrain.pth"],  # type: ignore
            latent=obj["sd15_lda.pth"],  # type: ignore
            json=obj["json"],  # type: ignore
            text_embedding=None if self.only_image else obj["clipl.pth"] # type: ignore
        )

    def get_item(self, index: int) -> BatchOnlyImage | Batch:
        item = self.get_next_item()
        latent = torch.load(io.BytesIO(item.latent))  # type: ignore
        dino_embedding = torch.load(io.BytesIO(item.dino_embedding))  # type: ignore
        if self.only_image:
            return BatchOnlyImage(
                latent=latent.unsqueeze(0),
                dino_embedding=dino_embedding.unsqueeze(0),
            )
        else:
            assert item.text_embedding is not None
            text_embedding = torch.load(io.BytesIO(item.text_embedding)) # type: ignore
            return Batch(
                latent=latent.unsqueeze(0),
                dino_embedding=dino_embedding.unsqueeze(0),
                text_embedding=text_embedding.unsqueeze(0)
            )
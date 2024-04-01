from functools import cached_property
import torch
from streaming import StreamingDataset  # type: ignore

from dino_ip_adapter_refiners.data.utils import BatchOnlyImage, Batch, BaseDataAdapter

class MosaicAdapter(BaseDataAdapter):
    def __init__(self, train_shards_path_or_url: str, cache_dir: str, shuffle: bool = True, cache_limit: str = '100gb', only_image: bool = False):
        super().__init__(only_image=only_image)
        self.train_shards_path_or_url = train_shards_path_or_url
        self.cache_dir = cache_dir
        self.shuffle = shuffle
        self.cache_limit = cache_limit
        self.only_image = only_image
        self.dataset_length = len(self.dataset)
    @cached_property
    def dataset(self) -> StreamingDataset:
        return StreamingDataset(remote=self.train_shards_path_or_url, local=self.cache_dir, shuffle=self.shuffle, cache_limit=self.cache_limit)
    def get_item(self, index: int) -> BatchOnlyImage | Batch:
        item = self.dataset[index]
        dino_embedding = torch.tensor(item["dinov2_vitl14_reg4_pretrain.pth"])
        latent = torch.tensor(item["sd15_lda.pth"])
        if self.only_image:
            return BatchOnlyImage(
                latent=latent.unsqueeze(0),
                dino_embedding=dino_embedding.unsqueeze(0),
            )
        else:
            assert item.text_embedding is not None
            text_embedding = torch.tensor(item["clipl.pth"])
            return Batch(
                latent=latent.unsqueeze(0),
                dino_embedding=dino_embedding.unsqueeze(0),
                text_embedding=text_embedding.unsqueeze(0)
            )
from dataclasses import dataclass
from functools import cached_property
import io
import torch
from webdataset import WebDataset  # type: ignore
from dino_ip_adapter_refiners.config import DatasetConfig
from dino_ip_adapter_refiners.data.utils import BatchOnlyImage, Batch, BaseDataAdapter
from torch.utils.data import DataLoader
from typing import Any
import webdataset as wds
from torch import cat
from webdataset.tariterators import (
    base_plus_ext,
    tar_file_expander,
    url_opener,
    valid_sample,
)
import math

def filter_keys(key_set):
    def _f(dictionary):
        return {k: v for k, v in dictionary.items() if k in key_set}

    return _f
def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=wds.warn_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples

@dataclass
class Item:
    dino_embedding: bytes
    latent: bytes
    json: bytes
    text_embedding: bytes | None = None


class WebdatasetAdapter(BaseDataAdapter):
    def __init__(self, config: DatasetConfig, batch_size: int = 1):
        super().__init__(only_image=config.only_image)
        self.config = config
        self.batch_size = batch_size
        self.train_shards_path_or_url = config.train_shards_path_or_url
        self.dataset_length = config.dataset_length
        self.only_image = config.only_image
    def collate_fn_from_dict(self, batch: list[dict]) -> BatchOnlyImage | Batch:
        latents = cat(tensors=[item["latent"][None] for item in batch])
        dino_embeddings = cat([item["dino_embedding"][None] for item in batch])
        if self.only_image:
            return BatchOnlyImage(latent=latents, dino_embedding=dino_embeddings)
        else:
            text_embeddings = cat(tensors=[item["text_embedding"][None] for item in batch])
            return Batch(latent=latents, dino_embedding=dino_embeddings, text_embedding=text_embeddings)
    @cached_property
    def dataloader(self) -> DataLoader[Any]:
        all_keys = ["text_embedding", "latent", "dino_embedding"]
        if self.only_image:
            all_keys.pop(0)
        processing_pipeline = [
            wds.decode(wds.handle_extension("pth", wds.autodecode.torch_loads), handler=wds.ignore_and_continue),
            wds.rename(
                text_embedding="clipl.pth",
                latent="sd15_lda.pth",
                dino_embedding="dinov2_vitl14_reg4_pretrain_popped.pth",
                handler=wds.warn_and_continue,
            ),
            wds.map(filter_keys(set(all_keys))),
        ]
        pipeline = [
            wds.ResampledShards(self.train_shards_path_or_url),
            tarfile_to_samples_nothrow,
            wds.shuffle(self.config.shuffle_buffer_size),
            *processing_pipeline,
            wds.batched(self.batch_size, partial=False, collation_fn=self.collate_fn_from_dict),
        ]
        global_batch_size = self.batch_size
        num_workers = self.config.dataset_workers
        num_train_examples = self.config.dataset_length
        num_worker_batches = math.ceil(num_train_examples / (global_batch_size * num_workers))  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        # each worker is iterating over this
        data_pipeline = wds.DataPipeline(*pipeline).with_epoch(num_worker_batches)
        dataloader = wds.WebLoader(
            data_pipeline,
            batch_size=None,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
        )
        dataloader.num_batches = num_batches
        dataloader.num_samples = num_samples
        return dataloader
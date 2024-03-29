from typing import Any
from torch import nn
import torch
from refiners.foundationals.latent_diffusion.cross_attention import CrossAttentionBlock
from refiners.foundationals.latent_diffusion.image_prompt import PerceiverResampler
from refiners.fluxion.adapters import Adapter
from refiners.foundationals.latent_diffusion import SD1UNet
from refiners.training_utils import register_model

from refiners.fluxion import layers as fl

from dino_ip_adapter_refiners.config import IPAdapterConfig


class CrossAttentionAdapter(fl.Chain, Adapter[CrossAttentionBlock]):
    def __init__(
        self,
        target: CrossAttentionBlock,
    ) -> None:
        with self.setup_adapter(target):
            cross_attention = target.layer(1, fl.Residual)
            layer_norm = cross_attention.layer(0, fl.LayerNorm)
            attention = cross_attention.layer(2, fl.Attention).structural_copy()

            distribute = attention.layer(0, fl.Distribute)
            query_matrix = distribute.layer((0), fl.Linear)
            self._key_matrix = [
                fl.Linear(
                    in_features=1024,
                    out_features=query_matrix.out_features,
                    device=target.device,
                    dtype=target.dtype,
                )
            ]
            self._value_matrix = [
                fl.Linear(
                    in_features=1024,
                    out_features=query_matrix.out_features,
                    device=target.device,
                    dtype=target.dtype,
                )
            ]

            new_distribute = fl.Distribute(
                query_matrix,
                self.key_matrix,
                self.value_matrix,
            )

            attention.replace(distribute, new_distribute)

            super().__init__(
                layer_norm,
                fl.Parallel(
                    fl.Identity(),
                    fl.UseContext(context="ip_adapter", key="image_embedding"),
                    fl.UseContext(context="ip_adapter", key="image_embedding"),
                ),
                attention,
            )

    @property
    def key_matrix(self) -> fl.Linear:
        return self._key_matrix[0]

    @property
    def value_matrix(self) -> fl.Linear:
        return self._value_matrix[0]

    def enable_gradients(self, enable: bool) -> None:
        self.key_matrix.weight.requires_grad_(enable)
        self.value_matrix.weight.requires_grad_(enable)


class DinoIPAdapter(Adapter[SD1UNet], fl.Chain):
    def __init__(self, target: SD1UNet) -> None:
        with self.setup_adapter(target):
            super().__init__(target)

        self._image_proj = [
            PerceiverResampler(
                latents_dim=1024,
                num_attention_layers=8,
                num_attention_heads=24,
                head_dim=64,
                num_tokens=128,
                input_dim=1024,
                output_dim=1024,
                device=target.device,
                dtype=target.dtype,
            )
        ]
        self._unconditional_image_embedding = [
            nn.Parameter(
                torch.randn(1, 128, 1024, device=target.device, dtype=target.dtype)
            )
        ]

        self.sub_adapters = [
            CrossAttentionAdapter(cross_attn)
            for cross_attn in target.layers(CrossAttentionBlock)
        ]

    @property
    def image_proj(self) -> PerceiverResampler:
        return self._image_proj[0]

    @property
    def unconditional_image_embedding(self) -> torch.Tensor:
        return self._unconditional_image_embedding[0]

    def inject(self, parent: fl.Chain | None = None):
        for sub_adapter in self.sub_adapters:
            sub_adapter.inject(self)

        return self

    def enable_gradients(self, enable: bool) -> None:
        self.image_proj.requires_grad_(enable)
        for sub_adapter in self.sub_adapters:
            sub_adapter.enable_gradients(enable)

    def get_image_embedding(
        self, dino_embedding: torch.Tensor, /, drop_rate: float = 0.1
    ) -> torch.Tensor:
        if torch.rand(1) > drop_rate:
            return self.image_proj(dino_embedding)

        return self.unconditional_image_embedding

    def set_image_context(self, image_embedding: torch.Tensor, /) -> None:
        self.set_context("ip_adapter", {"image_embedding": image_embedding})


class IPAdapterMixin:
    @register_model()
    def ip_adapter(self: Any, config: IPAdapterConfig) -> DinoIPAdapter:
        ip_adapter = DinoIPAdapter(
            target=self.unet,
        ).inject()
        ip_adapter.enable_gradients(True)
        return ip_adapter


if __name__ == "__main__":
    import torch

    unet = SD1UNet(4)
    adapter = DinoIPAdapter(unet).inject()

    timestep = torch.randn(1, 1)
    unet.set_timestep(timestep)

    dino_embedding = torch.randn(1, 1370, 1024)
    image_embedding = adapter.image_proj(dino_embedding)
    unet.set_context("ip_adapter", {"image_embedding": image_embedding})

    x = torch.randn(1, 4, 32, 32)
    y = adapter(x)

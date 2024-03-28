from typing import Any
from refiners.foundationals.latent_diffusion.image_prompt import PerceiverResampler
from torch import device as Device, dtype as DType
from refiners.fluxion.adapters import Adapter
from refiners.foundationals.latent_diffusion import SD1UNet
from refiners.training_utils import register_model

from refiners.fluxion import layers as fl

from dino_ip_adapter_refiners.config import IPAdapterConfig


class ImageCrossAttention(fl.Chain):
    def __init__(
        self,
        query_matrix: fl.Linear,
        /,
        dim: int = 1024,
        num_heads: int = 64,
        scale: float = 1.0,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        self.dim = dim
        self._multiply = [fl.Multiply(scale)]
        super().__init__(
            fl.Distribute(
                query_matrix,
                fl.Chain(
                    fl.UseContext(context="ip_adapter", key="clip_image_embedding"),
                    fl.Linear(
                        in_features=dim,
                        out_features=dim,
                        device=device,
                        dtype=dtype,
                    ),
                ),
                fl.Chain(
                    fl.UseContext(context="ip_adapter", key="clip_image_embedding"),
                    fl.Linear(
                        in_features=dim,
                        out_features=dim,
                        device=device,
                        dtype=dtype,
                    ),
                ),
            ),
            fl.ScaledDotProductAttention(num_heads=num_heads),
            self.multiply,
        )

    @property
    def multiply(self) -> fl.Multiply:
        return self._multiply[0]

    @property
    def scale(self) -> float:
        return self.multiply.scale

    @scale.setter
    def scale(self, value: float) -> None:
        self.multiply.scale = value

    @property
    def key_projection(self) -> fl.Linear:
        return self.layer(("Distribute", 1, "Linear"), fl.Linear)

    @property
    def value_projection(self) -> fl.Linear:
        return self.layer(("Distribute", 2, "Linear"), fl.Linear)

    def enable_gradients(self, requires_grad: bool = True, /) -> None:
        self.key_projection.requires_grad_(requires_grad=requires_grad)
        self.value_projection.requires_grad_(requires_grad=requires_grad)


class CrossAttentionAdapter(fl.Chain, Adapter[fl.Attention]):
    def __init__(
        self,
        target: fl.Attention,
        scale: float = 1.0,
    ) -> None:
        with self.setup_adapter(target):
            query_matrix = target.layer(("Distribute", 0, "Linear"), fl.Linear)

            super().__init__(
                ImageCrossAttention(
                    query_matrix,
                    scale=scale,
                ),
            )

    @property
    def image_cross_attention(self) -> ImageCrossAttention:
        return self.layer(0, ImageCrossAttention)

    @property
    def scale(self) -> float:
        return self.image_cross_attention.scale

    @scale.setter
    def scale(self, value: float) -> None:
        self.image_cross_attention.scale = value

    def enable_gradients(self, requires_grad: bool = True, /) -> None:
        self.image_cross_attention.enable_gradients(requires_grad)


class DinoIPAdapter(Adapter[SD1UNet], fl.Chain):
    def __init__(self, target: SD1UNet, scale: float = 1.0) -> None:
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

        self.sub_adapters = [
            CrossAttentionAdapter(cross_attn, scale=scale)
            for cross_attn in filter(
                lambda attn: type(attn) != fl.SelfAttention, target.layers(fl.Attention)
            )
        ]

    @property
    def image_proj(self) -> PerceiverResampler:
        return self._image_proj[0]

    def enable_gradients(self, requires_grad: bool = True, /) -> None:
        self.image_proj.requires_grad_(requires_grad=requires_grad)
        for sub_adapter in self.sub_adapters:
            sub_adapter.enable_gradients(requires_grad)


class IPAdapterMixin:
    @register_model()
    def ip_adapter(self: Any, config: IPAdapterConfig) -> DinoIPAdapter:
        ip_adapter = DinoIPAdapter(
            target=self.unet,
        )
        ip_adapter.enable_gradients(True)
        ip_adapter.inject()
        return ip_adapter

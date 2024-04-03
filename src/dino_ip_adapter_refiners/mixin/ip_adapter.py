from typing import Any, Generic
from torch import nn
import torch
from refiners.foundationals.latent_diffusion.cross_attention import CrossAttentionBlock
from refiners.foundationals.latent_diffusion.image_prompt import PerceiverResampler
from refiners.fluxion.adapters import Adapter
from refiners.foundationals.latent_diffusion import SD1UNet
from dino_ip_adapter_refiners.utils import register_model
from refiners.fluxion import layers as fl
from dino_ip_adapter_refiners.config import IPAdapterConfig
from dino_ip_adapter_refiners.mixin.base import BaseMixin, BatchT
from torch import Tensor

from torch.nn import Module, Linear, Embedding, LayerNorm
from torch.nn.init import trunc_normal_
from torch import float32
from refiners.training_utils import ModelConfig
from refiners.foundationals.latent_diffusion import SD1UNet
from refiners.fluxion.layers.attentions import ScaledDotProductAttention
from jaxtyping import Float

# Adapted from https://github.com/huggingface/open-muse
def _init_learnable_weights(module: Module, initializer_range: float):
    """
    Initialize the weights according to the original implementation.
    https://github.com/google-research/maskgit/blob/main/maskgit/nets/maskgit_transformer.py#L37
    """

    # TODO: make this configurable
    if isinstance(module, Linear):
        if module.weight.requires_grad:
            if initializer_range == 0:
                module.weight.data.zero_()
            else:
                trunc_normal_(module.weight, std=initializer_range)
        if module.bias is not None and module.bias.requires_grad:
            module.bias.data.zero_()
    elif isinstance(module, Embedding):
        if module.weight.requires_grad:
            if initializer_range == 0:
                module.weight.data.zero_()
            else:
                trunc_normal_(module.weight, std=initializer_range)
    elif isinstance(module, (LayerNorm)):
        if hasattr(module, "weight") and module.weight.requires_grad:
            module.weight.data.fill_(1.0)
        if hasattr(module, "bias") and module.bias is not None and module.bias.requires_grad:
            module.bias.data.zero_()


def expand_dim(x: Float[Tensor, "batch embed_dim"], sequence_length: int = -1) -> Float[Tensor, "batch seq_len embed_dim"]:
    if sequence_length == -1:
        return x
    return x[:, None, :].repeat([1, sequence_length, 1])

class ImageCrossAttention(fl.Chain):
    def __init__(
        self,
        text_cross_attention: fl.Attention,
        scale: float = 1.0,
        use_timestep_embedding: bool = False,
        sequence_length: int = -1,
    ) -> None:
        self._scale = scale
        self.sequence_length = sequence_length
        key_contexts: list[fl.Chain] = [
            fl.Chain(
                fl.UseContext(context="ip_adapter", key="image_embedding"),
                fl.Linear(
                    in_features=text_cross_attention.key_embedding_dim,
                    out_features=text_cross_attention.inner_dim,
                    bias=text_cross_attention.use_bias,
                    device=text_cross_attention.device,
                    dtype=text_cross_attention.dtype,
                ),
            ),
        ]
        query_contexts: list[fl.Chain] = [
            fl.Chain(
                fl.UseContext(context="ip_adapter", key="image_embedding"),
                fl.Linear(
                    in_features=text_cross_attention.value_embedding_dim,
                    out_features=text_cross_attention.inner_dim,
                    bias=text_cross_attention.use_bias,
                    device=text_cross_attention.device,
                    dtype=text_cross_attention.dtype,
                ),
            ),
        ]
        if use_timestep_embedding:
            key_contexts.append(
                fl.Chain(
                    fl.UseContext(context="range_adapter", key="timestep_embedding"),
                    fl.Linear(
                        in_features=1280,
                        out_features=text_cross_attention.inner_dim,
                        bias=text_cross_attention.use_bias,
                        device=text_cross_attention.device,
                        dtype=text_cross_attention.dtype,
                    ),
                    fl.Lambda(lambda x: expand_dim(x, sequence_length=sequence_length)),
                )
            )
            query_contexts.append(
                fl.Chain(
                    fl.UseContext(context="range_adapter", key="timestep_embedding"),
                    fl.Linear(
                        in_features=1280,
                        out_features=text_cross_attention.inner_dim,
                        bias=text_cross_attention.use_bias,
                        device=text_cross_attention.device,
                        dtype=text_cross_attention.dtype,
                    ),
                    fl.Lambda(lambda x: expand_dim(x, sequence_length=sequence_length))
                )
            )


        super().__init__(
            fl.Distribute(
                fl.Identity(),
                fl.Sum(
                    *key_contexts
                ),
                fl.Sum(
                    *query_contexts
                ),
            ),
            ScaledDotProductAttention(
                num_heads=text_cross_attention.num_heads, is_causal=text_cross_attention.is_causal
            ),
            fl.Multiply(self.scale),
        )
    @property
    def scale(self) -> float:
        return self._scale

    @scale.setter
    def scale(self, value: float) -> None:
        self._scale = value
        self.ensure_find(fl.Multiply).scale = value

class WeightedSum(fl.Chain):
    _TAG = "WeightedSum"
    def __init__(self, chain_1: fl.Module, chain_2: fl.Module) -> None:
        super().__init__(chain_1, chain_2)

    def forward(self, *args: Tensor) -> Tensor:
        ref = self[0](*args)
        output =  ref + self[1](*args)
        return (output / output.norm()) * ref.norm()

class CrossAttentionAdapter(fl.Chain, Adapter[fl.Attention]):
    def __init__(
        self,
        target: fl.Attention,
        scale: float = 1.0,
        use_timestep_embedding: bool = False,
        sequence_length: int = -1,
        weighted_sum: bool = False
    ) -> None:
        self._scale = scale
        sum_method = fl.Sum if not weighted_sum else WeightedSum
        with self.setup_adapter(target):
            clone = target.structural_copy()
            scaled_dot_product = clone.ensure_find(ScaledDotProductAttention)
            image_cross_attention = ImageCrossAttention(
                text_cross_attention=clone,
                scale=self.scale,
                use_timestep_embedding=use_timestep_embedding,
                sequence_length=sequence_length
            )
            clone.replace(
                old_module=scaled_dot_product,
                new_module=sum_method(
                    scaled_dot_product,
                    image_cross_attention,
                ),
            )
            super().__init__(
                clone,
            )

    @property
    def image_cross_attention(self) -> ImageCrossAttention:
        return self.ensure_find(ImageCrossAttention)

    @property
    def scale(self) -> float:
        return self._scale

    @scale.setter
    def scale(self, value: float) -> None:
        self._scale = value
        self.image_cross_attention.scale = value
    def enable_gradients(self, enable: bool) -> None:
        self.image_cross_attention.requires_grad_(enable)

class CrossAttentionAdapterOnlyImage(fl.Chain, Adapter[CrossAttentionBlock]):
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
    def __init__(self, target: SD1UNet, image_proj: PerceiverResampler | None = None, unconditional_image_embedding: Tensor | None = None,  use_timestep_embedding: bool = False, use_unconditional_image_embedding: bool = True, only_image: bool = False, weighted_sum: bool = True, weights: dict[str, Tensor] | None = None) -> None:
        with self.setup_adapter(target):
            super().__init__(target)

        self._image_proj = [
            image_proj if image_proj is not None else PerceiverResampler(
                latents_dim=768,
                num_attention_layers=4,
                num_attention_heads=12,
                head_dim=64,
                num_tokens=16,
                input_dim=1024,
                output_dim=768,
                device=target.device,
                dtype=target.dtype,
            )
        ]
        self.use_unconditional_image_embedding = use_unconditional_image_embedding
        if use_unconditional_image_embedding:
            self.unconditional_image_embedding = unconditional_image_embedding if unconditional_image_embedding is not None else nn.Parameter(
                torch.randn(1, 16, 768)
            )
        if only_image:
            self.sub_adapters = [
                CrossAttentionAdapterOnlyImage(cross_attn)
                for cross_attn in target.layers(CrossAttentionBlock)
            ]
        else:
            self.sub_adapters = [
                 CrossAttentionAdapter(cross_attn, use_timestep_embedding=use_timestep_embedding, weighted_sum=weighted_sum)
                 for cross_attn in filter(lambda attn: type(attn) != fl.SelfAttention, target.layers(fl.Attention))
            ]
        if weights is not None:
            image_proj_state_dict: dict[str, Tensor] = {
                k.removeprefix("image_proj."): v for k, v in weights.items() if k.startswith("image_proj.")
            }

            self.image_proj.load_state_dict(image_proj_state_dict, strict=True)


            for i, cross_attn in enumerate(self.sub_adapters):
                cross_attention_weights: dict[str, Tensor] = {}
                for k, v in weights.items():
                    prefix = f"ip_adapter.{i:03d}."
                    if not k.startswith(prefix):
                        continue
                    cross_attention_weights[k[len(prefix):]] = v
                cross_attn.load_state_dict(cross_attention_weights, strict=False)
            if use_unconditional_image_embedding:
                self.unconditional_image_embedding = weights["unconditional_image_embedding"]

    @property
    def image_proj(self) -> PerceiverResampler:
        return self._image_proj[0]

    def inject(self, parent: fl.Chain | None = None):
        for sub_adapter in self.sub_adapters:
            sub_adapter.inject(self)

        return self

    def initialize_weights(self, initializer_range: float = 0.02):
        for sub_adapter in self.sub_adapters:
            sub_adapter.to(dtype=float32)
            for module in sub_adapter.modules():
                _init_learnable_weights(module, initializer_range)
    def enable_gradients(self, enable: bool) -> None:
        self.image_proj.requires_grad_(enable)
        for sub_adapter in self.sub_adapters:
            sub_adapter.enable_gradients(enable)
        self.unconditional_image_embedding.requires_grad_(enable)

    def get_image_embedding(
        self, dino_embedding: torch.Tensor
    ) -> torch.Tensor:
        return self.image_proj(dino_embedding)

    def set_image_context(self, image_embedding: torch.Tensor, /) -> None:
        self.set_context("ip_adapter", {"image_embedding": image_embedding})


class IPAdapterMixin(
    Generic[BatchT],
    BaseMixin[BatchT]
):
    @register_model()
    def unet(self, config: ModelConfig) -> SD1UNet:
        unet = SD1UNet(in_channels=4, device=self.device, dtype=self.dtype)
        unet.load_from_safetensors(self.config.extra_training.unet_checkpoint)
        return unet
    @register_model()
    def image_proj(self, config: ModelConfig) -> PerceiverResampler:
        image_proj = PerceiverResampler(
            latents_dim=768,
            num_attention_layers=4,
            num_attention_heads=12,
            head_dim=64,
            num_tokens=16,
            input_dim=1024,
            output_dim=768,
            device=self.device,
            dtype=self.dtype,
        )
        image_proj.to(dtype=float32)
        image_proj.requires_grad_(True)
        for module in image_proj.modules():
            _init_learnable_weights(module, self.config.ip_adapter.initializer_range)
        return image_proj

    @register_model()
    def ip_adapter(self, config: IPAdapterConfig) -> DinoIPAdapter:
        ip_adapter = DinoIPAdapter(
            target=self.unet,
            image_proj=self.image_proj,
        ).inject()
        ip_adapter.enable_gradients(True)
        ip_adapter.initialize_weights(config.initializer_range)
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

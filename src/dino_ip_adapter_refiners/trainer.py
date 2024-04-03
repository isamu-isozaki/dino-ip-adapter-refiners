import torch
from torch import Tensor, zeros, float32, randn_like
from torch.nn import functional as F
from dino_ip_adapter_refiners.diffusion_utils import (
    scale_loss,
    sample_noise,
    sample_timesteps,
    add_noise_to_latents,
    TimestepSampler,
    LossScaler,
)
import numpy as np
from functools import cached_property
from dino_ip_adapter_refiners.mixin import IPAdapterMixin, AMPMixin, DataMixin
from dino_ip_adapter_refiners.data import BatchOnlyImage, Batch
from dino_ip_adapter_refiners.config import SaveAdapterConfig
from dino_ip_adapter_refiners.utils import pil2tensor
from refiners.training_utils import CallbackConfig

from typing import TypeVar, Generic
import os
from refiners.training_utils.callback import Callback
from refiners.fluxion.utils import save_to_safetensors
from refiners.training_utils.trainer import register_callback
from dino_ip_adapter_refiners.utils import register_model
from refiners.fluxion import layers as fl
import torchvision.transforms.functional as TF
from refiners.fluxion.utils import normalize
from torchvision.transforms import InterpolationMode

import random
from refiners.foundationals.dinov2 import (
    DINOv2_small,
    DINOv2_small_reg,
    DINOv2_base,
    DINOv2_base_reg,
    DINOv2_large,
    DINOv2_large_reg,
    ViT,
)
from refiners.foundationals.clip.text_encoder import CLIPTextEncoderL
from refiners.foundationals.latent_diffusion.solvers.dpm import DPMSolver
from refiners.foundationals.latent_diffusion.stable_diffusion_1.model import SD1Autoencoder, StableDiffusion_1
from refiners.training_utils.wandb import WandbLoggable
import PIL
from PIL import Image
from torch import (
    Tensor,
    cat,
    randn,
)
from loguru import logger

BatchT = TypeVar("BatchT", bound="BatchOnlyImage | Batch")
class ComputeGradNormCallback(Callback["BaseTrainer"]):
    """Callback to compute gradient norm"""

    def on_backward_end(self, trainer: "BaseTrainer") -> None:
        if trainer.clock.is_evaluation_step:
            for name, param in trainer.ip_adapter.named_parameters():
                if param.grad is not None:
                    grads = param.grad.detach().data
                    grad_norm = (grads.norm(p=2) / grads.numel()).item()
                    trainer.wandb_log(data={"grad_norm/" + name: grad_norm})
            for name, param in trainer.image_proj.named_parameters():
                if param.grad is not None:
                    grads = param.grad.detach().data
                    grad_norm = (grads.norm(p=2) / grads.numel()).item()
                    trainer.wandb_log(data={"grad_norm/" + name: grad_norm})
            for name, param in trainer.uncond_image_embedding.named_parameters():
                if param.grad is not None:
                    grads = param.grad.detach().data
                    grad_norm = (grads.norm(p=2) / grads.numel()).item()
                    trainer.wandb_log(data={"grad_norm/" + name: grad_norm})
        return super().on_backward_end(trainer)


class ComputeParamNormCallback(Callback["BaseTrainer"]):
    """Callback to compute gradient norm"""

    def on_backward_end(self, trainer: "BaseTrainer") -> None:
        if trainer.clock.is_evaluation_step:
            for name, param in trainer.ip_adapter.named_parameters():
                if param.grad is not None:
                    data = param.data.detach()
                    data_norm = (data.norm(p=2) / data.numel()).item()
                    trainer.wandb_log(data={"param_norm/" + name: data_norm})
            for name, param in trainer.image_proj.named_parameters():
                if param.grad is not None:
                    data = param.data.detach()
                    data_norm = (data.norm(p=2) / data.numel()).item()
                    trainer.wandb_log(data={"param_norm/" + name: data_norm})
            for name, param in trainer.uncond_image_embedding.named_parameters():
                if param.grad is not None:
                    data = param.data.detach()
                    data_norm = (data.norm(p=2) / data.numel()).item()
                    trainer.wandb_log(data={"param_norm/" + name: data_norm})
        return super().on_backward_end(trainer)


class SaveAdapterCallback(Callback["BaseTrainer"]):
    """Callback to save the adapter when a checkpoint is saved."""
    def __init__(self) -> None:
        super().__init__()

    def on_backward_end(self, trainer: "BaseTrainer") -> None:
        if trainer.clock.iteration % trainer.config.save_adapter.checkpoint_steps == 0:
            os.makedirs(trainer.config.save_adapter.save_folder, exist_ok=True)
            cross_attention_adapters = trainer.ip_adapter.sub_adapters
            image_proj = trainer.ip_adapter.image_proj
            uncond_image_embedding = None
            if trainer.ip_adapter.use_uncond_image_embedding:
                uncond_image_embedding = trainer.ip_adapter.unconditional_image_embedding

            tensors: dict[str, Tensor] = {}
            tensors |= {f"image_proj.{key}": value for key, value in image_proj.state_dict().items()}
            for i, cross_attention_adapter in enumerate(cross_attention_adapters):
                tensors |= {f"ip_adapter.{i:03d}.{key}": value for key, value in cross_attention_adapter.state_dict().items()}
            if trainer.ip_adapter.use_uncond_image_embedding:
                assert isinstance(uncond_image_embedding, Tensor)
                tensors |= {f"uncond_image_embedding": uncond_image_embedding}
            save_to_safetensors(
                path= f"{trainer.config.save_adapter.save_folder}/step{trainer.clock.iteration}.safetensors",
                tensors=tensors,
            )

class BaseTrainer(
    Generic[BatchT],
    IPAdapterMixin[BatchT],
    AMPMixin[BatchT],
    DataMixin[BatchT]
):
    @register_callback()
    def compute_grad_norms(self, config: CallbackConfig) -> ComputeGradNormCallback:
        return ComputeGradNormCallback()
    @register_callback()
    def compute_param_norms(self, config: CallbackConfig) -> ComputeParamNormCallback:
        return ComputeParamNormCallback()
    @register_callback()
    def save_adapter(self, config: SaveAdapterConfig) -> SaveAdapterCallback:
        return SaveAdapterCallback()
    def compute_evaluation(self) -> None:
        text_encoder = CLIPTextEncoderL(self.device, self.dtype)
        text_encoder.load_from_safetensors(self.config.extra_training.text_encoder_checkpoint)
        lda = SD1Autoencoder(self.device, self.dtype)
        lda.load_from_safetensors(self.config.extra_training.lda_checkpoint)
        image_encoder = DINOv2_large_reg(self.device, self.dtype)
        image_encoder.load_from_safetensors(self.config.extra_training.image_encoder_checkpoint)
        image_encoder.pop()
        image_encoder.layer((-1), fl.Chain).pop()

        # initialize an SD1.5 pipeline using the trainer's models
        pipeline_dtype = None if self.config.extra_training.automatic_mixed_precision else self.dtype
        sd = StableDiffusion_1(
            unet=self.unet,
            lda=lda,
            solver=DPMSolver(num_inference_steps=self.config.test_ldm.num_inference_steps),
            device=self.device,
            dtype=pipeline_dtype
        )
        self.ip_adapter.scale = self.config.ip_adapter.inference_scale
        # retreive data from config
        prompts = self.config.test_ldm.prompts
        validation_image_paths = self.config.test_ldm.validation_image_paths
        assert len(prompts) == len(validation_image_paths)
        num_images_per_prompt = self.config.test_ldm.num_images_per_prompt
        if self.config.test_ldm.use_short_prompts:
            prompts = [prompt.split(sep=",")[0] for prompt in prompts]
        cond_images = [Image.open(validation_image_path) for validation_image_path in validation_image_paths]

        # for each prompt generate `num_images_per_prompt` images
        # TODO: remove this for loop, batch things up
        images: dict[str, WandbLoggable] = {}
        for i in range(len(cond_images)):
            images[f"condition images_{i}"] = cond_images[i]
        for prompt, cond_image in zip(prompts, cond_images):
            canvas_image = Image.new(mode="RGB", size=(512, 512 * num_images_per_prompt))
            conditional_embedding = text_encoder(prompt)
            negative_embedding = text_encoder("")
            clip_text_embedding = cat(tensors=(negative_embedding, conditional_embedding), dim=0)

            cond_resolution = self.config.ip_adapter.resolution
            cond_image = pil2tensor(cond_image)
            cond_image = TF.resize(
                cond_image,
                size=cond_resolution,
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )

            cond_image = TF.center_crop(cond_image, cond_resolution)
            cond_image = normalize(
                cond_image,
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            )[None]
            image_embedding = image_encoder(cond_image)
            image_embedding = self.image_proj(image_embedding)

            # TODO: pool text according to end of text id for pooled text embeds if given option
            for i in range(num_images_per_prompt):
                logger.info(f"Generating image {i+1}/{num_images_per_prompt} for prompt: {prompt}")
                x = randn(1, 4, 64, 64, device=self.device, dtype=self.dtype)
                self.ip_adapter.set_image_context(image_embedding)
                for step in sd.steps:
                    x = sd(
                        x=x,
                        step=step,
                        clip_text_embedding=clip_text_embedding,
                        condition_scale=self.config.test_ldm.condition_scale
                    )
                canvas_image.paste(sd.lda.decode_latents(x=x), box=(0, 512 * i))
            images[prompt] = canvas_image
        # log images to wandb
        self.wandb_log(data=images)
        self.ip_adapter.scale = self.config.ip_adapter.scale
        del text_encoder
        del lda
        del image_encoder
    @cached_property
    def empty_text_embedding(self) -> Tensor:
        text_encoder = CLIPTextEncoderL(self.device, self.dtype)
        text_encoder.load_from_safetensors(self.config.extra_training.text_encoder_checkpoint)
        output = text_encoder("").float().cpu()
        del text_encoder
        return output
    @cached_property
    def black_image_embedding(self) -> Tensor:
        if self.config.ip_adapter.use_uncond_image_embedding:
            return self.ip_adapter.unconditional_image_embedding
        cond_resolution = self.config.ip_adapter.resolution
        image_encoder = DINOv2_large_reg(self.device, self.dtype)
        image_encoder.load_from_safetensors(self.config.extra_training.image_encoder_checkpoint)
        image_encoder.pop()
        image_encoder.layer((-1), fl.Chain).pop()
        output = image_encoder(zeros((1, 3, cond_resolution, cond_resolution)).to(self.device, dtype=self.dtype)).float().cpu()
        del image_encoder
        return output
    def drop_latents(self, image_embedding: Tensor, text_embedding: Tensor) -> tuple[Tensor, Tensor]:
        dataset_config = self.config.dataset
        rand_num = random.random()
        if rand_num < dataset_config.image_drop_rate:
            image_embedding = self.black_image_embedding
        elif rand_num < (dataset_config.image_drop_rate + dataset_config.text_drop_rate):
            text_embedding = self.empty_text_embedding
        elif rand_num < (
            dataset_config.image_drop_rate + dataset_config.text_drop_rate + dataset_config.text_and_image_drop_rate
        ):
            text_embedding = self.empty_text_embedding
            image_embedding = self.black_image_embedding
        return image_embedding, text_embedding
    def drop_image_latents(self, image_embedding: Tensor) -> Tensor:
        dataset_config = self.config.dataset
        rand_num = random.random()
        if rand_num < dataset_config.image_drop_rate + dataset_config.text_and_image_drop_rate:
            image_embedding = self.black_image_embedding
        return image_embedding

def compute_loss(self: BaseTrainer[BatchT], batch: BatchT, only_image: bool = False):
    batch = batch.to(device=self.device, dtype=self.dtype)
    latents = batch.latent
    image_embeddings = batch.dino_embedding
    text_embeddings = None
    if not only_image:
        text_embeddings = batch.text_embedding
    batch_size = latents.shape[0]
    image_embeddings = self.image_proj(image_embeddings)
    for i in range(batch_size):
        if only_image:
            image_embeddings[i] = self.drop_image_latents(image_embeddings[i])
        else:
            assert isinstance(text_embeddings, Tensor)
            image_embeddings[i], text_embeddings[i] = self.drop_latents(image_embeddings[i], text_embeddings[i])
    self.ip_adapter.set_image_context(image_embeddings)
    if not only_image:
        self.unet.set_clip_text_embedding(clip_text_embedding=text_embeddings)

    timesteps = sample_timesteps(
        len(batch), sampler=TimestepSampler.UNIFORM, device=self.device
    )
    self.unet.set_timestep(timesteps)

    noise = sample_noise(latents.shape, device=self.device, dtype=self.dtype)
    input_perturbation = self.config.extra_training.input_perturbation
    if input_perturbation > 0:
        new_noise = noise + input_perturbation * randn_like(noise)
        noisy_latents = add_noise_to_latents(latents=latents, noise=new_noise, solver=self.solver, timesteps=timesteps)
    else:
        noisy_latents = add_noise_to_latents(latents=latents, noise=noise, solver=self.solver, timesteps=timesteps)


    predicted_noise = self.unet(noisy_latents)
    loss = F.mse_loss(input=predicted_noise, target=noise, reduction="none")
    rescaled_loss = scale_loss(
        loss,
        timesteps=timesteps,
        scaler=self.config.extra_training.loss_scaler,
        solver=self.solver,
    )

    return rescaled_loss.mean()
class Trainer(
    BaseTrainer[Batch]
):
    def compute_loss(self, batch: Batch) -> torch.Tensor:
        return compute_loss(self, batch, only_image=False)

class TrainerOnlyImage(
    BaseTrainer[BatchOnlyImage]
):
    def compute_loss(self, batch: BatchOnlyImage) -> torch.Tensor:
        return compute_loss(self, batch, only_image=True)


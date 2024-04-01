from functools import cached_property
from refiners.foundationals.latent_diffusion import SD1UNet
from refiners.foundationals.latent_diffusion.solvers import DDPM
import torch
from torch.nn import functional as F
from refiners.training_utils import Trainer as AbstractTrainer
from dino_ip_adapter_refiners.config import Config
from dino_ip_adapter_refiners.diffusion_utils import (
    scale_loss,
    sample_noise,
    sample_timesteps,
    add_noise_to_latents,
    TimestepSampler,
    LossScaler,
)
from dino_ip_adapter_refiners.mixin.evaluation import EvaluationMixin
from dino_ip_adapter_refiners.mixin.ip_adapter import IPAdapterMixin
from dino_ip_adapter_refiners.data import DatasetAdapter, BatchOnlyImage, Batch
from torch.utils.data import DataLoader
from typing import Any, TypeVar, Generic, Callable
from torch import Tensor, device as Device, dtype as DType, float16, float32, nn
from torch.cuda.amp import GradScaler, autocast
from refiners.training_utils.trainer import ModelConfigT, ModuleT, ModelItem, backward
from refiners.training_utils import BaseConfig
from refiners.fluxion import layers as fl
from functools import cached_property, wraps
from refiners.fluxion.utils import no_grad
from refiners.training_utils.common import (
    scoped_seed,
)

BatchT = TypeVar("BatchT", bound="BatchOnlyImage | Batch")

def register_model():
    def decorator(func: Callable[[Any, ModelConfigT], ModuleT]) -> ModuleT:
        @wraps(func)
        def wrapper(self: AbstractTrainer[Config, BatchT], config: ModelConfigT) -> fl.Module:
            name = func.__name__
            model = func(self, config)
            model = model.to(self.device, dtype=self.dtype)
            if config.requires_grad is not None:
                model.requires_grad_(requires_grad=config.requires_grad)
            learnable_parameters = [param for param in model.parameters() if param.requires_grad]
            if self.config.extra_training.automatic_mixed_precision:
                # For all parameters we train in automatic mixed precision we want them to be in float32.
                for learnable_parameter in learnable_parameters:
                    learnable_parameter.to(dtype=float32)
            self.models[name] = ModelItem(
                name=name, config=config, model=model, learnable_parameters=learnable_parameters
            )
            setattr(self, name, self.models[name].model)
            return model

        return wrapper  # type: ignore

    return decorator

# from https://github.com/finegrain-ai/refiners/pull/290
class AMPTrainer(
    Generic[BatchT],
    EvaluationMixin,
    IPAdapterMixin,
    AbstractTrainer[Config, BatchT],
):
    @cached_property
    def scaler(self) -> GradScaler | None:
        if self.dtype != float16 or not self.config.extra_training.automatic_mixed_precision:
            return None
        return GradScaler()
    def backward_step(self, scaled_loss: Tensor) -> None:
        if self.scaler is None:
            backward(tensors=scaled_loss)
            return
        self.scaler.scale(scaled_loss).backward()  # type: ignore
    def optimizer_step(self) -> None:
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
        max_norm = self.config.training.gradient_clipping_max_norm or float("inf")
        self.grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.learnable_parameters, max_norm=max_norm).item()
        if self.scaler is None:
            self.optimizer.step()
            return
        self.scaler.step(self.optimizer)  # type: ignore
        self.scaler.update()
    def backward(self) -> None:
        """Backward pass on the loss."""
        self._call_callbacks(event_name="on_backward_begin")
        scaled_loss = self.loss / self.clock.num_step_per_iteration
        self.backward_step(scaled_loss)
        self._call_callbacks(event_name="on_backward_end")
        if self.clock.is_optimizer_step:
            self._call_callbacks(event_name="on_optimizer_step_begin")
            self.optimizer_step()
            self.optimizer.zero_grad()
            self._call_callbacks(event_name="on_optimizer_step_end")
        if self.clock.is_lr_scheduler_step:
            self._call_callbacks(event_name="on_lr_scheduler_step_begin")
            self.lr_scheduler.step()
            self._call_callbacks(event_name="on_lr_scheduler_step_end")
        if self.clock.is_evaluation_step:
            self.evaluate()
    def step(self, batch: BatchT) -> None:
        """Perform a single training step."""
        self._call_callbacks(event_name="on_compute_loss_begin")
        with autocast(dtype=self.dtype, enabled=self.config.training.automatic_mixed_precision):
            loss = self.compute_loss(batch=batch)
        self.loss = loss
        self._call_callbacks(event_name="on_compute_loss_end")
        self.backward()

    @no_grad()
    @scoped_seed(seed=AbstractTrainer.get_evaluation_seed)
    def evaluate(self) -> None:
        """Evaluate the model."""
        self.set_models_to_mode(mode="eval")
        self._call_callbacks(event_name="on_evaluate_begin")
        with autocast(dtype=self.dtype, enabled=self.config.training.automatic_mixed_precision):
            self.compute_evaluation()
        self._call_callbacks(event_name="on_evaluate_end")
        self.set_models_to_mode(mode="train")

class BaseTrainer(AMPTrainer[BatchT]):
    def __init__(self, config: Config):
        super().__init__(config)
        self.dataset_adapter = DatasetAdapter(config.dataset)
    def get_item(self, index: int) -> BatchT:
        return self.dataset_adapter.get_item(index)
    def collate_fn(self, batch: list[BatchT]) -> BatchT:
        return self.dataset_adapter.collate_fn(batch)
    @cached_property
    def dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.dataset, batch_size=self.config.training.batch_size, num_workers=self.config.dataset.dataset_workers, shuffle=True, collate_fn=self.collate_fn
        )
class Trainer(
    BaseTrainer[Batch]
):

    @cached_property
    def unet(self) -> SD1UNet:
        return SD1UNet(in_channels=4, device=self.device, dtype=self.dtype)

    @property
    def solver(self) -> DDPM:
        return DDPM(1000, device=self.device)

    def compute_loss(self, batch: Batch) -> torch.Tensor:
        batch = batch.to(device=self.device, dtype=self.dtype)

        timesteps = sample_timesteps(
            len(batch), sampler=TimestepSampler.UNIFORM, device=self.device
        )
        self.unet.set_timestep(timesteps)

        noise = sample_noise(batch.latent.shape, device=self.device, dtype=self.dtype)
        noisy_latent = add_noise_to_latents(
            latents=batch.latent, noise=noise, solver=self.solver, timesteps=timesteps
        )

        image_embedding = self.ip_adapter.get_image_embedding(
            batch.dino_embedding, drop_rate=0.1
        )
        self.ip_adapter.set_image_context(image_embedding)

        predicted_noise = self.unet(noisy_latent)
        loss = F.mse_loss(input=predicted_noise, target=noise, reduction="none")
        rescaled_loss = scale_loss(
            loss,
            timesteps=timesteps,
            scaler=LossScaler.LEGACY,
            solver=self.solver,
        )

        return rescaled_loss.mean()

class TrainerOnlyImage(
    BaseTrainer[BatchOnlyImage]
):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        # TODO: Change this to Mixin once I figure out __init__ for multi inheritance
        self.dataset_adapter = DatasetAdapter(config.dataset)
    def get_item(self, index: int) -> BatchOnlyImage:
        item =  self.dataset_adapter.get_item(index)
        assert isinstance(item, BatchOnlyImage)
        return item
    def collate_fn(self, batch: list[BatchOnlyImage]) -> BatchOnlyImage:
        output = self.dataset_adapter.collate_fn(batch)
        assert isinstance(output, BatchOnlyImage)
        return output
    @cached_property
    def unet(self) -> SD1UNet:
        return SD1UNet(in_channels=4, device=self.device, dtype=self.dtype)

    @property
    def solver(self) -> DDPM:
        return DDPM(1000, device=self.device)

    def compute_loss(self, batch: BatchOnlyImage) -> torch.Tensor:
        batch = batch.to(device=self.device, dtype=self.dtype)

        timesteps = sample_timesteps(
            len(batch), sampler=TimestepSampler.UNIFORM, device=self.device
        )
        self.unet.set_timestep(timesteps)

        noise = sample_noise(batch.latent.shape, device=self.device, dtype=self.dtype)
        noisy_latent = add_noise_to_latents(
            latents=batch.latent, noise=noise, solver=self.solver, timesteps=timesteps
        )

        image_embedding = self.ip_adapter.get_image_embedding(
            batch.dino_embedding, drop_rate=0.1
        )
        self.ip_adapter.set_image_context(image_embedding)

        predicted_noise = self.unet(noisy_latent)
        loss = F.mse_loss(input=predicted_noise, target=noise, reduction="none")
        rescaled_loss = scale_loss(
            loss,
            timesteps=timesteps,
            scaler=LossScaler.LEGACY,
            solver=self.solver,
        )

        return rescaled_loss.mean()

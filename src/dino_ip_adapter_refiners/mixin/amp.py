from dino_ip_adapter_refiners.mixin.base import BaseMixin, BatchT
from functools import cached_property
from typing import Generic
from torch import Tensor, float16, nn
from torch.cuda.amp import GradScaler, autocast
from refiners.training_utils.trainer import backward
from refiners.fluxion.utils import no_grad
from refiners.training_utils.common import (
    scoped_seed,
)
# from https://github.com/finegrain-ai/refiners/pull/290
class AMPMixin(
    Generic[BatchT],
    BaseMixin[BatchT]
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
        for item in self.models.values():
            for name, param in item.model.named_parameters():
                if param.grad is not None:
                    print(name)
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
        with autocast(dtype=self.dtype, enabled=self.config.extra_training.automatic_mixed_precision):
            loss = self.compute_loss(batch=batch)
        self.loss = loss
        self._call_callbacks(event_name="on_compute_loss_end")
        self.backward()

    @no_grad()
    @scoped_seed(seed=BaseMixin.get_evaluation_seed)
    def evaluate(self) -> None:
        """Evaluate the model."""
        self.set_models_to_mode(mode="eval")
        self._call_callbacks(event_name="on_evaluate_begin")
        with autocast(dtype=self.dtype, enabled=self.config.extra_training.automatic_mixed_precision):
            self.compute_evaluation()
        self._call_callbacks(event_name="on_evaluate_end")
        self.set_models_to_mode(mode="train")
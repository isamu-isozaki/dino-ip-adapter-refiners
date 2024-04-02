
from refiners.training_utils import Trainer as AbstractTrainer
from dino_ip_adapter_refiners.config import Config

from typing import Any, Callable
from torch import float32
from refiners.training_utils.trainer import ModelConfigT, ModuleT, ModelItem
from refiners.fluxion import layers as fl
from functools import wraps
def register_model():
    def decorator(func: Callable[[Any, ModelConfigT], ModuleT]) -> ModuleT:
        @wraps(func)
        def wrapper(self: AbstractTrainer[Config, Any], config: ModelConfigT) -> fl.Module:
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
            print(name)
            setattr(self, name, self.models[name].model)
            return model

        return wrapper  # type: ignore

    return decorator
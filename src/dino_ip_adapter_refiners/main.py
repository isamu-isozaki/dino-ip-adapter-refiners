from refiners.training_utils import (
    LRSchedulerConfig,
    LRSchedulerType,
    OptimizerConfig,
    Optimizers,
    TrainingConfig,
    WandbConfig,
)
from refiners.training_utils.common import TimeUnit, TimeValue
from dino_ip_adapter_refiners.config import Config, IPAdapterConfig
from dino_ip_adapter_refiners.trainer import Trainer
import sys

if __name__ == "__main__":
    config = Config(
        training=TrainingConfig(
            device="cuda",
            dtype="bfloat16",
            duration=TimeValue(number=10, unit=TimeUnit.EPOCH),
            batch_size=46,
            gradient_clipping_max_norm=2.0,
        ),
        optimizer=OptimizerConfig(
            optimizer=Optimizers.AdamW8bit,
            learning_rate=1e-4,
            betas=(0.9, 0.999),
            weight_decay=1e-2,
        ),
        lr_scheduler=LRSchedulerConfig(
            type=LRSchedulerType.CONSTANT_LR,
            warmup=TimeValue(number=200, unit=TimeUnit.STEP),
        ),
        ip_adapter=IPAdapterConfig(),
        wandb=WandbConfig(
            mode="online",
            project="debug-dino-ip-adapter-refiners",
            entity="ben-selas",
        )
    )
    config_path = sys.argv[1]
    config = AdapterLatentDiffusionConfig.load_from_toml(toml_path=config_path)
    trainer = Trainer(config)

    trainer.train()

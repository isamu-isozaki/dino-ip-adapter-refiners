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
from dino_ip_adapter_refiners.trainer import TrainerOnlyImage
import sys
from dotenv import dotenv_values


if __name__ == "__main__":
    config_path = sys.argv[1]
    config = dotenv_values(".env")

    TRAINING_PATH = config.get("TRAINING_PATH")
    config = Config.load_from_toml(toml_path=config_path)
    config.train_shards_path_or_url = TRAINING_PATH
    trainer = TrainerOnlyImage(config)

    trainer.train()

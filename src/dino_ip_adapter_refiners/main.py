
from dino_ip_adapter_refiners.config import Config
from dino_ip_adapter_refiners.trainer import TrainerOnlyImage
import sys
from dotenv import dotenv_values


if __name__ == "__main__":
    config_path = sys.argv[1]
    config = dotenv_values(".env")

    TRAINING_PATH = config.get("TRAINING_PATH")
    assert isinstance(TRAINING_PATH, str)
    config = Config.load_from_toml(toml_path=config_path)
    config.dataset.train_shards_path_or_url = TRAINING_PATH
    trainer = TrainerOnlyImage(config)

    trainer.train()

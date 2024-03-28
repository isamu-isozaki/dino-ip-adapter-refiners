from dino_ip_adapter_refiners.config import Config
from dino_ip_adapter_refiners.trainer import Trainer

if __name__ == "__main__":
    config = Config()
    trainer = Trainer(config)

    trainer.train()

from refiners.training_utils import BaseConfig, ModelConfig, WandbConfig


class IPAdapterConfig(ModelConfig):
    pass


class Config(BaseConfig):
    ip_adapter: IPAdapterConfig
    wandb: WandbConfig

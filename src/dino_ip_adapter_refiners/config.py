from refiners.training_utils import BaseConfig, ModelConfig


class IPAdapterConfig(ModelConfig):
    pass


class Config(BaseConfig):
    ip_adapter: IPAdapterConfig

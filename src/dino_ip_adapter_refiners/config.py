from refiners.training_utils import BaseConfig, ModelConfig, WandbConfig, CallbackConfig, TrainingConfig
from pydantic import BaseModel

class TestDiffusionConfig(BaseModel):
    seed: int = 0
    num_inference_steps: int = 30
    use_short_prompts: bool = False
    prompts: list[str] = []
    num_images_per_prompt: int = 1
    condition_scale: float = 7.5


class IPAdapterConfig(ModelConfig):
    """Configuration for the IP adapter."""

    image_encoder_type: str
    resolution: int = 518
    scale: float = 1.0
    inference_scale: float = 0.75
    use_timestep_embedding: bool = False
    fine_grained: bool = False
    initialize_model: bool = True
    initializer_range: float = 0.02
    use_rescaler: bool = False
    weighted_sum: bool = False
    timestep_bias_strategy: str = "none"
    timestep_bias_portion: float = 0.5
    timestep_bias_begin: int = 0
    timestep_bias_end: int = 1000
    timestep_bias_multiplier: float = 1.0
    use_unconditional_image_embedding: bool = True

class IPTrainingConfig(TrainingConfig):
    automatic_mixed_precision: bool = (
        True  # Enables automatic mixed precision which allows float32 gradients while working with lower precision. This only has effect when dtype is not float32
    )
    unet_checkpoint: str = "checkpoints/unet.safetensors"
    text_encoder_checkpoint: str = "checkpoints/CLIPTextEncoderL.safetensors"
    lda_checkpoint: str = "checkpoints/CLIPTextEncoderL.safetensors"
    image_encoder_checkpoint: str = "checkpoints/dinov2_vitl14_reg4_pretrain.safetensors"
    ip_adapter_checkpoint: str | None = None
    input_pertubation: float = 0.0
    loss_scaler: str = "legacy"

class SaveAdapterConfig(CallbackConfig):
    checkpoint_steps: int = 2000
    save_folder: str | None = None

class DatasetConfig(BaseModel):
    """Configuration for the dataset."""

    horizontal_flip_probability: float = 0.5
    image_drop_rate: float = 0.05
    text_drop_rate: float = 0.05
    text_and_image_drop_rate: float = 0.05
    train_shards_path_or_url: str = ""
    cache_dir: str
    shuffle: bool = True
    cache_limit: str = "100gb"
    is_mosaic: bool = False
    is_webdataset: bool = False
    only_image: bool = False
    dataset_length: int = 567597
    dataset_workers: int = 4
    predownload: int = 1000
    download_retry: int = 2
    download_timeout: float = 120

class TestIPDiffusionConfig(TestDiffusionConfig):
    """Configuration to test the diffusion model, during the `evaluation` loop of the trainer."""

    validation_image_paths: list[str]

class Config(BaseConfig):
    """Finetunning configuration.

    Contains the configs of the dataset, the latent diffusion model and the adapter.
    """

    dataset: DatasetConfig
    extra_training: IPTrainingConfig
    test_ldm: TestIPDiffusionConfig
    compute_grad_norms: CallbackConfig
    compute_param_norms: CallbackConfig
    save_adapter: SaveAdapterConfig
    wandb: WandbConfig
    unet: ModelConfig
    # image proj has to be after image encoder or it fails
    image_proj: ModelConfig
    # adapter needs to be initialized later for this to work
    ip_adapter: IPAdapterConfig

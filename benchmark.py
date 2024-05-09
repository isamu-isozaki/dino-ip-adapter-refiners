# type: ignore
# test image score
import os
import json
from refiners.foundationals.clip.image_encoder import (
    CLIPImageEncoderH,
    ViTEmbeddings,
    TransformerLayer
)
from torch import Tensor, device as Device, dtype as DType

from torch import zeros
from refiners.fluxion import layers as fl

from refiners.foundationals.latent_diffusion.stable_diffusion_xl.text_encoder import (
    CLIPTextEncoderL,
)
import PIL
import PIL.Image
import torch
from refiners.fluxion.utils import image_to_tensor, normalize
from tqdm.auto import tqdm
from dino_ip_adapter_refiners.mixin.ip_adapter import DinoIPAdapter
from refiners.foundationals.latent_diffusion.stable_diffusion_1.model import (
    SD1Autoencoder,
    SD1UNet,
    StableDiffusion_1,
)
from refiners.foundationals.dinov2 import (
    DINOv2_large_reg,
)
from refiners.foundationals.latent_diffusion.cross_attention import (
    CrossAttentionBlock2d,
)
from refiners.fluxion.utils import load_from_safetensors
from refiners.foundationals.latent_diffusion.solvers.dpm import DPMSolver
from torch import Tensor, cat, randn
from refiners.fluxion.utils import manual_seed
import gc
import numpy as np
import argparse
import pandas as pd
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from typing import cast, TypeVar

from jaxtyping import Float

from refiners.fluxion.adapters.adapter import Adapter
from refiners.fluxion.context import Contexts
from refiners.foundationals.clip.tokenizer import CLIPTokenizer

T = TypeVar("T", bound="CLIPTextEncoderG | CLIPTextEncoderL")

class TextEncoderWithPoolingGeneral(fl.Chain, Adapter[T]):
    def __init__(
        self,
        target: T,
        projection: fl.Linear | None = None,
        pool_features: int = 1280,
        slice_target: bool = True
    ) -> None:
        with self.setup_adapter(target=target):
            tokenizer = target.ensure_find(CLIPTokenizer)
            super().__init__(
                tokenizer,
                fl.SetContext(
                    context="text_encoder_pooling", key="end_of_text_index", callback=self.set_end_of_text_index
                ),
                target[1:-2] if slice_target else target[1:],
                fl.Parallel(
                    fl.Identity(),
                    fl.Chain(
                        target[-2:] if slice_target else fl.Identity(),
                        projection
                        or fl.Linear(
                            in_features=pool_features,
                            out_features=pool_features,
                            bias=False,
                            device=target.device,
                            dtype=target.dtype,
                        ),
                        fl.Lambda(func=self.pool),
                    ),
                ),
            )

    def init_context(self) -> Contexts:
        return {"text_encoder_pooling": {"end_of_text_index": []}}

    def __call__(self, text: str) -> tuple[Float[Tensor, "1 77 embed_dim"], Float[Tensor, "1 embed_dim"]]:
        return super().__call__(text)

    @property
    def tokenizer(self) -> CLIPTokenizer:
        return self.ensure_find(CLIPTokenizer)

    def set_end_of_text_index(self, end_of_text_index: list[int], tokens: Tensor) -> None:
        position = (tokens == self.tokenizer.end_of_text_token_id).nonzero(as_tuple=True)[1][0].item()
        end_of_text_index.append(cast(int, position))

    def pool(self, x: Float[Tensor, "1 77 embed_dim"]) -> Float[Tensor, "1 embed_dim"]:
        end_of_text_index = self.use_context(context_name="text_encoder_pooling").get("end_of_text_index", [])
        assert len(end_of_text_index) == 1, "End of text index not found."
        return x[:, end_of_text_index[0], :]


class TextEncoderWithPoolingL(TextEncoderWithPoolingGeneral[CLIPTextEncoderL]):
    def __init__(
        self,
        target: CLIPTextEncoderL,
        projection: fl.Linear | None = None,
    ) -> None:
        super().__init__(target, projection, pool_features=768, slice_target=False)
class CLIPImageEncoder(fl.Chain):
    """Contrastive Language-Image Pretraining (CLIP) image encoder.

    See [[arXiv:2103.00020] Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
    for more details.
    """

    def __init__(
        self,
        image_size: int = 224,
        embedding_dim: int = 768,
        output_dim: int = 512,
        patch_size: int = 32,
        num_layers: int = 12,
        num_attention_heads: int = 12,
        feedforward_dim: int = 3072,
        layer_norm_eps: float = 1e-5,
        use_quick_gelu: bool = False,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        """Initialize a CLIP image encoder.

        Args:
            image_size: The size of the input image.
            embedding_dim: The dimension of the embedding.
            output_dim: The dimension of the output.
            patch_size: The size of the patches.
            num_layers: The number of layers.
            num_attention_heads: The number of attention heads.
            feedforward_dim: The dimension of the feedforward layer.
            layer_norm_eps: The epsilon value for normalization.
            device: The PyTorch device to use.
            dtype: The PyTorch data type to use.
        """
        self.image_size = image_size
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.feedforward_dim = feedforward_dim
        cls_token_pooling: Callable[[Tensor], Tensor] = lambda x: x[:, 0, :]
        super().__init__(
            ViTEmbeddings(
                image_size=image_size, embedding_dim=embedding_dim, patch_size=patch_size, device=device, dtype=dtype
            ),
            fl.LayerNorm(normalized_shape=embedding_dim, eps=layer_norm_eps, device=device, dtype=dtype),
            fl.Chain(
                TransformerLayer(
                    embedding_dim=embedding_dim,
                    feedforward_dim=feedforward_dim,
                    num_attention_heads=num_attention_heads,
                    layer_norm_eps=layer_norm_eps,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ),
            fl.Lambda(func=cls_token_pooling),
            fl.LayerNorm(normalized_shape=embedding_dim, eps=layer_norm_eps, device=device, dtype=dtype),
            fl.Linear(in_features=embedding_dim, out_features=output_dim, bias=False, device=device, dtype=dtype),
        )
        if use_quick_gelu:
            for gelu, parent in self.walk(predicate=lambda m, _: isinstance(m, fl.GeLU)):
                parent.replace(
                    old_module=gelu,
                    new_module=fl.GeLU(approximation=fl.GeLUApproximation.SIGMOID),
                )

class CLIPImageEncoderL(CLIPImageEncoder):
    def __init__(self, device: Device | str | None = None, dtype: DType | None = None) -> None:
        super().__init__(
            embedding_dim = 1024,
            output_dim = 768,
            patch_size = 14,
            num_layers = 24,
            num_attention_heads = 16,
            feedforward_dim = 4096,
            use_quick_gelu=True,
            device=device,
            dtype=dtype,
        )


def clip_transform(
    image: PIL.Image.Image,
    device: torch.device,
    dtype: torch.dtype,
    size: tuple[int, int] = (224, 224),
    mean: list[float] | None = None,
    std: list[float] | None = None,
) -> torch.Tensor:
    # Default mean and std are parameters from https://github.com/openai/CLIP
    return normalize(
        image_to_tensor(image.resize(size), device=device, dtype=dtype),
        mean=[0.48145466, 0.4578275, 0.40821073] if mean is None else mean,
        std=[0.26862954, 0.26130258, 0.27577711] if std is None else std,
    )


def calculate_clip_score(args):
    os.makedirs(args.output_path, exist_ok=True)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    scales = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    custom_concept_base_path = "/home/isamu/custom_concept_101"
    db_name = "benchmark_dataset"
    prompts_and_config = (
        "/home/isamu/custom_concept_101/custom-diffusion/customconcept101"
    )
    single_concept_json = "dataset.json"
    text_encoder_path = (
        "/home/isamu/refiners/tests/weights/CLIPLWithProjection.safetensors"
    )
    image_encoder_path = (
        "/home/isamu/refiners/tests/weights/clip_image_l_vision.safetensors"
    )

    num_prompts = args.num_prompts
    num_images_per_prompt = args.num_images_per_prompt
    generation_path = args.generation_path
    csv_path = f"{args.output_path}/num_prompts_{num_prompts}_num_images_per_prompt_{num_images_per_prompt}.csv"
    output = {
        "name": [],
        "text alignment": [],
        "image alignment": [],
        "adjusted text alignment": [],
        "sum of alignment": [],
    }
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        output = df.to_dict()
    else:
        df = pd.DataFrame.from_dict(output)
    print(output)
    with open(os.path.join(prompts_and_config, single_concept_json)) as f:
        data = json.load(f)
    text_encoder_l_with_projection = CLIPTextEncoderL(device=device, dtype=dtype)
    clip_text_encoder = TextEncoderWithPoolingL(
        target=text_encoder_l_with_projection
    ).inject()
    text_encoder = clip_text_encoder.load_from_safetensors(text_encoder_path).to(
        device, dtype=dtype
    )
    clip_image_encoder = (
        CLIPImageEncoderL()
        .load_from_safetensors(image_encoder_path)
        .to(device, dtype=dtype)
    )
    base_data = {}

    for elem in data:
        class_prompt = elem["class_prompt"]

        prompt_file = os.path.join(prompts_and_config, elem["prompt_filename"])
        benchmark_dataset = os.path.join(custom_concept_base_path, db_name)
        images_folder = elem["instance_data_dir"].replace(
            "./benchmark_dataset", benchmark_dataset
        )
        validation_image_paths = []
        for image in os.listdir(images_folder):
            image_path = os.path.join(images_folder, image)
            validation_image_paths.append(image_path)
            break
        prompts = []
        with open(prompt_file) as f:
            for i, prompt in enumerate(f.readlines()):
                if i >= num_prompts:
                    break
                prompts.append(prompt.replace("{}", class_prompt))
        cond_image = PIL.Image.open(validation_image_paths[0])
        if cond_image.mode != "RGB":
            cond_image = cond_image.convert("RGB")
        base_data[class_prompt] = {}
        base_data[class_prompt]["image"] = cond_image
        base_data[class_prompt]["prompts"] = prompts
    clip_text = {}
    clip_image = {}
    w = 2.5
    with torch.no_grad():
        for scale in tqdm(scales):
            name = f"{args.checkpoint_path}_{scale}"
            if name in output["name"]:
                continue

            clip_text[scale] = []
            clip_image[scale] = []

            scale_dir = os.path.join(generation_path, str(scale))
            for elem in data:
                class_prompt = elem["class_prompt"]
                cond_image = base_data[class_prompt]["image"]
                preprocessed_cond_image = clip_transform(cond_image, device, dtype)
                cond_embed = clip_image_encoder(preprocessed_cond_image)
                cond_embed = cond_embed / cond_embed.norm(p=2, dim=-1, keepdim=True)
                prompts = base_data[class_prompt]["prompts"]
                for idx, prompt in enumerate(prompts):
                    prompt_embeds = text_encoder(prompt)[1]
                    prompt_embeds = prompt_embeds / prompt_embeds.norm(
                        p=2, dim=-1, keepdim=True
                    )
                    for i in range(num_images_per_prompt):
                        generated_image = PIL.Image.open(
                            os.path.join(scale_dir, f"{class_prompt}_{idx}_{i}.png")
                        )
                        preprocessed_generated_image = clip_transform(
                            generated_image, device, dtype
                        )
                        generated_embeds = clip_image_encoder(
                            preprocessed_generated_image
                        )
                        generated_embeds = generated_embeds / generated_embeds.norm(
                            p=2, dim=-1, keepdim=True
                        )
                        clip_text_alignment = torch.sum(
                            prompt_embeds * generated_embeds, dim=1
                        )
                        clip_text_alignment = torch.clip(clip_text_alignment * w, 0)
                        clip_text_alignment = (
                            clip_text_alignment.float().cpu().detach().numpy()
                        )
                        clip_text[scale].append(clip_text_alignment)
                        clip_image_alignment = torch.sum(
                            cond_embed * generated_embeds, dim=1
                        )
                        clip_image_alignment = torch.clip(clip_image_alignment, 0)
                        clip_image_alignment = (
                            clip_image_alignment.float().cpu().detach().numpy()
                        )
                        clip_image[scale].append(clip_image_alignment)
    print(f"For {args.checkpoint_path}")
    for scale in scales:
        name = f"{args.checkpoint_path}_{scale}"
        if name in output["name"]:
            continue
        print(
            f"At scale: {scale}. Text alignment is {np.mean(clip_text[scale])/w} and Image alignment is {np.mean(clip_image[scale])}"
        )
        output["name"].append(name)
        output["text alignment"].append(np.mean(clip_text[scale]) / w)
        output["image alignment"].append(np.mean(clip_image[scale]))
        output["adjusted text alignment"].append(np.mean(clip_text[scale]))
        output["sum of alignment"].append(
            np.mean(clip_text[scale]) + np.mean(clip_image[scale])
        )
    df = pd.DataFrame.from_dict(output)
    df = df.sort_values(by="sum of alignment", ascending=False)
    df.to_csv(csv_path, index=False, header=True)


def generation_and_clip_score_calc(args):
    device = torch.device("cuda")
    dtype = torch.bfloat16
    scales = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    custom_concept_base_path = "/home/isamu/custom_concept_101"
    db_name = "benchmark_dataset"
    prompts_and_config = (
        "/home/isamu/custom_concept_101/custom-diffusion/customconcept101"
    )
    single_concept_json = "dataset.json"
    clip_image_encoder_h_path = (
        "/home/isamu/refiners/tests/weights/CLIPImageEncoderH.safetensors"
    )
    cond_resolution = 518

    num_prompts = args.num_prompts
    num_images_per_prompt = args.num_images_per_prompt
    condition_scale = args.condition_scale
    generation_path = args.generation_path
    checkpoint_path = args.checkpoint_path
    use_timestep_embedding = args.use_timestep_embedding

    os.makedirs(generation_path, exist_ok=True)

    with open(os.path.join(prompts_and_config, single_concept_json)) as f:
        data = json.load(f)
    unet = (
        SD1UNet(in_channels=4)
        .load_from_safetensors("/home/isamu/refiners/tests/weights/unet.safetensors")
        .to(device, dtype=dtype)
        .eval()
    )
    if args.clip_image_encoder:
        cond_resolution = 224
        image_encoder = (
            CLIPImageEncoderH()
            .load_from_safetensors(clip_image_encoder_h_path)
            .to(device, dtype=dtype)
            .eval()
        )
    else:
        image_encoder = (
            DINOv2_large_reg()
            .load_from_safetensors(
                "/home/isamu/refiners/tests/weights/dinov2_vitl14_reg4_pretrain.safetensors"
            )
            .to(device, dtype=dtype)
            .eval()
        )
        image_encoder.pop()
        image_encoder.layer((-1), fl.Chain).pop()
    lda = (
        SD1Autoencoder()
        .load_from_safetensors("/home/isamu/refiners/tests/weights/lda.safetensors")
        .to(device, dtype=dtype)
        .eval()
    )
    text_encoder = (
        CLIPTextEncoderL()
        .load_from_safetensors(
            "/home/isamu/refiners/tests/weights/CLIPTextEncoderL.safetensors"
        )
        .to(device, dtype=dtype)
        .eval()
    )

    adapter = (
        DinoIPAdapter(
            target=unet,
            weights=load_from_safetensors(checkpoint_path),
            finegrained=True,
            scale=1,
            use_unconditional_image_embedding = False,
            use_timestep_embedding=use_timestep_embedding,
        )
        .inject()
        .to(device, dtype=dtype)
        .eval()
    )

    manual_seed(9752)
    sd = StableDiffusion_1(
        unet=unet,
        lda=lda,
        solver=DPMSolver(num_inference_steps=30),
        device=device,
        dtype=dtype,
    )
    with torch.no_grad():
        gc.collect()
        torch.cuda.empty_cache()
        negative_image_embedding = adapter.image_proj(image_encoder(zeros((1, 3, cond_resolution, cond_resolution)).to(device, dtype=dtype)))

        for scale in tqdm(scales):
            scale_dir = os.path.join(generation_path, str(scale))
            os.makedirs(scale_dir, exist_ok=True)
            adapter.scale = scale

            for elem in data:
                class_prompt = elem["class_prompt"]

                prompt_file = os.path.join(prompts_and_config, elem["prompt_filename"])
                benchmark_dataset = os.path.join(custom_concept_base_path, db_name)
                images_folder = elem["instance_data_dir"].replace(
                    "./benchmark_dataset", benchmark_dataset
                )
                validation_image_paths = []
                for image in os.listdir(images_folder):
                    image_path = os.path.join(images_folder, image)
                    validation_image_paths.append(image_path)
                    break
                prompts = []
                with open(prompt_file) as f:
                    for i, prompt in enumerate(f.readlines()):
                        if i >= num_prompts:
                            break
                        prompts.append(prompt.replace("{}", class_prompt))

                cond_image = PIL.Image.open(validation_image_paths[0])
                if cond_image.mode != "RGB":
                    cond_image = cond_image.convert("RGB")
                cond_image = image_to_tensor(cond_image).to(device, dtype=dtype)
                cond_image = TF.resize(
                    cond_image,
                    size=cond_resolution,
                    interpolation=InterpolationMode.BILINEAR,
                    antialias=True,
                )
                cond_image = TF.center_crop(cond_image, cond_resolution)
                cond_image = normalize(
                    cond_image,
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                )
                image_embedding = image_encoder(cond_image)
                image_embedding = adapter.image_proj(image_embedding)
                image_embedding = cat(tensors=(negative_image_embedding, image_embedding), dim=0)

                # for each prompt generate `num_images_per_prompt` images
                # TODO: remove this for loop, batch things up
                for idx, prompt in enumerate(prompts):
                    conditional_embedding = text_encoder(prompt)
                    negative_embedding = text_encoder("")

                    assert isinstance(text_encoder, CLIPTextEncoderL)
                    assert isinstance(negative_embedding, Tensor)
                    assert isinstance(conditional_embedding, Tensor)
                    clip_text_embedding = cat(
                        tensors=(negative_embedding, conditional_embedding), dim=0
                    )

                    # TODO: pool text according to end of text id for pooled text embeds if given option
                    for i in range(num_images_per_prompt):
                        file_path = os.path.join(
                            scale_dir, f"{class_prompt}_{idx}_{i}.png"
                        )
                        x = randn(1, 4, 64, 64, device=device, dtype=dtype)
                        adapter.set_image_context(image_embedding)
                        for step in sd.steps:
                            x = sd(
                                x=x,
                                step=step,
                                clip_text_embedding=clip_text_embedding,
                                condition_scale=condition_scale,
                            )
                        output_image = sd.lda.decode_latents(x=x)
                        output_image.save(file_path)

    adapter.eject()
    del unet
    del image_encoder
    del lda
    del text_encoder
    del adapter
    calculate_clip_score(args)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Converts a CLIPTextEncoder from the library transformers from the HuggingFace Hub to refiners."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="/home/isamu/checkpoints_rescaler_fixed/step400000.safetensors",
        help=("Path to checkpoint"),
    )
    parser.add_argument(
        "--generation_path",
        type=str,
        default="/home/isamu/generation",
        help=("Path for image generation"),
    )
    parser.add_argument(
        "--use_timestep_embedding",
        action="store_true",
        help=("Use timestep embed"),
    )
    parser.add_argument(
        "--num_prompts",
        type=int,
        default=5,
        help=("Number of prompts"),
    )
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=1,
        help=("Number of images to generate per prompt"),
    )
    parser.add_argument(
        "--condition_scale",
        type=float,
        default=7.5,
        help=("Condition scale"),
    )
    parser.add_argument(
        "--clip_image_encoder",
        action="store_true",
        help=("use clip image encoder"),
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="eval",
        help=("CSV path for eval script"),
    )
    parser.add_argument(
        "--no_generation",
        action="store_true",
        help=("use clip image encoder"),
    )
    args = parser.parse_args()
    if args.no_generation:
        calculate_clip_score(args)
    else:
        generation_and_clip_score_calc(args)


if __name__ == "__main__":
    main()

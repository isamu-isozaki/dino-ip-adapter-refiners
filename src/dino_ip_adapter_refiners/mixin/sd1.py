from functools import cached_property

from refiners.foundationals.latent_diffusion import StableDiffusion_1, SD1UNet


class SD1Mixin:
    @cached_property
    def sd(self) -> StableDiffusion_1:
        return StableDiffusion_1()

    @property
    def unet(self) -> SD1UNet:
        return self.sd.unet

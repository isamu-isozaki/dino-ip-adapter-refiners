from enum import Enum
from refiners.foundationals.latent_diffusion import Solver
import torch


class LossScaler(str, Enum):
    """Loss scalers"""

    NONE = "none"
    LEGACY = "legacy"
    RESCALER = "rescaler"
    MIN_SNR_1 = "min_snr_1"
    MIN_SNR_5 = "min_snr_5"


def approximate_loss_legacy(timesteps: torch.Tensor, /) -> torch.Tensor:
    """Legacy loss approximation fitted on stable-diffusion-1-5"""
    steps = 999 - timesteps
    a = 3.1198626909458634e-08
    exponent = 2.3683577564059
    b = -0.3560275587290773
    c = -13.269541143845919
    C = 0.36245161978354973

    return a * steps**exponent + b * torch.exp(-c / (steps - 1001)) + C


def min_snr(
    timesteps: torch.Tensor, solver: Solver, gamma: float = 1.0
) -> torch.Tensor:
    """Compute the Min-SNR weight for each timestep. This supposes we're predicting the noise."""
    signal_to_noise_ratios = solver.signal_to_noise_ratios[timesteps]
    signal_to_noise_ratios = signal_to_noise_ratios.exp() ** 2
    return torch.minimum(gamma / signal_to_noise_ratios, torch.ones_like(timesteps))


def scale_loss(
    loss: torch.Tensor,
    /,
    timesteps: torch.Tensor,
    scaler: LossScaler = LossScaler.NONE,
    curve: torch.Tensor | None = None,
    solver: Solver | None = None,
) -> torch.Tensor:
    """Scale loss"""
    match scaler:
        case LossScaler.NONE:
            return loss
        case LossScaler.LEGACY:
            return loss / approximate_loss_legacy(timesteps).reshape(-1, 1, 1, 1)
        case LossScaler.RESCALER:
            assert curve is not None, "losses must be provided when using RESCALER"
            return loss / curve[timesteps].reshape(-1, 1, 1, 1)
        case LossScaler.MIN_SNR_1:
            assert solver is not None, "solver must be provided when using MIN_SNR"
            return loss * min_snr(
                timesteps=timesteps, solver=solver, gamma=1.0
            ).reshape(-1, 1, 1, 1)
        case LossScaler.MIN_SNR_5:
            assert solver is not None, "solver must be provided when using MIN_SNR"
            return loss * min_snr(
                timesteps=timesteps, solver=solver, gamma=5.0
            ).reshape(-1, 1, 1, 1)


class TimestepSampler(str, Enum):
    """Timestep sampler"""

    UNIFORM = "uniform"
    CUBIC = "cubic"
    BETA = "reverse_beta"


def sample_timesteps(
    batch_size: int,
    /,
    sampler: TimestepSampler = TimestepSampler.UNIFORM,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Sample timesteps"""
    match sampler:
        case TimestepSampler.UNIFORM:
            return torch.randint(
                low=0, high=1000, size=(batch_size,), dtype=torch.int64, device=device
            )
        case TimestepSampler.CUBIC:
            # https://github.com/TencentARC/T2I-Adapter/blob/main/train_sketch.py#L675
            uniform_samples = torch.rand(
                size=(batch_size,), dtype=torch.float32, device=device
            )
            timesteps = (1 - uniform_samples**3) * 1000
            return timesteps.long()
        case TimestepSampler.BETA:
            beta_samples = (
                torch.distributions.Beta(2, 5)
                .sample(torch.Size((batch_size,)))
                .to(device)
            )
            return ((1 - beta_samples) * 1000).round().to(torch.int64)


def sample_noise(
    size: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
    offset_noise: float = 0.1,
) -> torch.Tensor:
    """Sample noise from a normal distribution.
    If `offset_noise` is more than 0, the noise will be offset by a small amount. It allows the model to generate
    images with a wider range of contrast https://www.crosslabs.org/blog/diffusion-with-offset-noise.
    """
    noise = torch.randn(*size, device=device, dtype=dtype)
    return noise + offset_noise * torch.randn(
        *size[:2], 1, 1, device=device, dtype=dtype
    )


def add_noise_to_latents(
    latents: torch.Tensor,
    noise: torch.Tensor,
    solver: Solver,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    """Add noise to latents."""
    steps = 999 - timesteps
    return torch.cat(
        [
            solver.add_noise(
                latents[i : i + 1],
                noise[i : i + 1],
                int(steps[i].item()),
            )
            for i in range(latents.shape[0])
        ],
        dim=0,
    )

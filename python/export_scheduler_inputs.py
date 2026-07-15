"""Export deterministic LCM scheduler inputs for board-side generation."""

from pathlib import Path
import argparse

import numpy as np
import torch
from diffusers import LCMScheduler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="D:/models/stable-diffusion-v1-5")
    parser.add_argument("--output_dir", default="testdata/prompt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, choices=[4, 6, 8], default=4)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scheduler = LCMScheduler.from_pretrained(args.model_dir, subfolder="scheduler")
    scheduler.set_timesteps(args.steps)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    latents = torch.randn((1, 4, 64, 64), generator=generator, dtype=torch.float32)
    latents.numpy().tofile(output_dir / "initial_latents.bin")
    scheduler.timesteps.numpy().astype(np.float32).tofile(output_dir / "timesteps.bin")
    coeffs = []
    for index, timestep in enumerate(scheduler.timesteps):
        prev_timestep = scheduler.timesteps[index + 1] if index + 1 < args.steps else torch.tensor(-1)
        alpha_t = scheduler.alphas_cumprod[timestep]
        alpha_prev = scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else scheduler.final_alpha_cumprod
        beta_t = 1 - alpha_t
        beta_prev = 1 - alpha_prev
        scaled_timestep = timestep * scheduler.config.timestep_scaling
        sigma_data = 0.5
        coeffs.append([
            alpha_t.sqrt().item(), beta_t.sqrt().item(), alpha_prev.sqrt().item(),
            beta_prev.sqrt().item(),
            (sigma_data**2 / (scaled_timestep**2 + sigma_data**2)).item(),
            (scaled_timestep / (scaled_timestep**2 + sigma_data**2) ** 0.5).item(),
        ])
        if index + 1 < args.steps:
            noise = torch.randn(latents.shape, generator=generator, dtype=torch.float32)
            noise.numpy().tofile(output_dir / f"scheduler_noise_{index}.bin")
    np.asarray(coeffs, dtype=np.float32).tofile(output_dir / "scheduler_coeffs.bin")
    print(f"Saved seed={args.seed}, steps={args.steps} scheduler inputs to {output_dir}")


if __name__ == "__main__":
    main()

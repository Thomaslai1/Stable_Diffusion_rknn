"""Create deterministic inputs for PC and RK3588 comparison."""

from pathlib import Path

import numpy as np


def main():
    output_dir = Path(__file__).resolve().parent.parent / "testdata"
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1234)

    vae_latents = rng.standard_normal((1, 4, 64, 64), dtype=np.float32)
    np.save(output_dir / "vae_latents.npy", vae_latents)
    vae_latents.tofile(output_dir / "vae_latents.bin")

    unet_sample = rng.standard_normal((1, 4, 64, 64), dtype=np.float32)
    unet_timestep = np.array([999.0], dtype=np.float32)
    unet_hidden_states = rng.standard_normal((1, 77, 768), dtype=np.float32)
    np.save(output_dir / "unet_sample.npy", unet_sample)
    np.save(output_dir / "unet_timestep.npy", unet_timestep)
    np.save(output_dir / "unet_encoder_hidden_states.npy", unet_hidden_states)
    unet_sample.tofile(output_dir / "unet_sample.bin")
    unet_timestep.tofile(output_dir / "unet_timestep.bin")
    unet_hidden_states.tofile(output_dir / "unet_encoder_hidden_states.bin")
    print(f"Saved test inputs to {output_dir.resolve()}")


if __name__ == "__main__":
    main()

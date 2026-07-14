"""Export deterministic fixed-prompt inputs and references for Android."""

from pathlib import Path
import argparse

import numpy as np
import torch
from diffusers import DiffusionPipeline, LCMScheduler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="D:/models/stable-diffusion-v1-5")
    parser.add_argument("--lcm_lora_dir", default="D:/models/lcm-lora-sdv1-5")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).resolve().parent.parent / "fixed_prompt"
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt = "a photo of a cat"
    seed = 42
    steps = 4

    pipe = DiffusionPipeline.from_pretrained(
        args.model_dir,
        torch_dtype=torch.float32,
        safety_checker=None,
    )
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(args.lcm_lora_dir)
    pipe.fuse_lora()
    pipe = pipe.to("cpu")
    pipe.scheduler.set_timesteps(steps)

    prompt_embeds, _ = pipe.encode_prompt(
        prompt=prompt,
        device="cpu",
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    latents = pipe.prepare_latents(
        batch_size=1,
        num_channels_latents=4,
        height=512,
        width=512,
        dtype=torch.float32,
        device="cpu",
        generator=generator,
    )

    np.save(output_dir / "prompt_embeds.npy", prompt_embeds.detach().numpy())
    np.save(output_dir / "initial_latents.npy", latents.detach().numpy())
    np.save(output_dir / "timesteps.npy", pipe.scheduler.timesteps.numpy())
    prompt_embeds.detach().numpy().tofile(output_dir / "prompt_embeds.bin")
    latents.detach().numpy().tofile(output_dir / "initial_latents.bin")
    pipe.scheduler.timesteps.numpy().astype(np.float32).tofile(output_dir / "timesteps.bin")

    scheduler_coeffs = []

    for step_index, timestep in enumerate(pipe.scheduler.timesteps):
        model_input = pipe.scheduler.scale_model_input(latents, timestep)
        noise_pred = pipe.unet(
            model_input,
            timestep,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
        )[0]
        np.save(output_dir / f"unet_input_{step_index}.npy", model_input.detach().numpy())
        np.save(output_dir / f"unet_output_{step_index}.npy", noise_pred.detach().numpy())

        prev_timestep = (
            pipe.scheduler.timesteps[step_index + 1]
            if step_index + 1 < len(pipe.scheduler.timesteps)
            else torch.tensor(-1)
        )
        alpha_t = pipe.scheduler.alphas_cumprod[timestep]
        alpha_prev = (
            pipe.scheduler.alphas_cumprod[prev_timestep]
            if prev_timestep >= 0
            else pipe.scheduler.final_alpha_cumprod
        )
        beta_t = 1 - alpha_t
        beta_prev = 1 - alpha_prev
        scaled_timestep = timestep * pipe.scheduler.config.timestep_scaling
        sigma_data = 0.5
        c_skip = sigma_data**2 / (scaled_timestep**2 + sigma_data**2)
        c_out = scaled_timestep / (scaled_timestep**2 + sigma_data**2) ** 0.5
        scheduler_coeffs.append(
            [
                alpha_t.sqrt().item(),
                beta_t.sqrt().item(),
                alpha_prev.sqrt().item(),
                beta_prev.sqrt().item(),
                c_skip.item(),
                c_out.item(),
            ]
        )
        predicted_original = (latents - beta_t.sqrt() * noise_pred) / alpha_t.sqrt()
        denoised = c_out * predicted_original + c_skip * latents
        if step_index + 1 < len(pipe.scheduler.timesteps):
            noise = torch.randn(latents.shape, generator=generator, dtype=latents.dtype)
            np.save(output_dir / f"scheduler_noise_{step_index}.npy", noise.numpy())
            noise.numpy().tofile(output_dir / f"scheduler_noise_{step_index}.bin")
            latents = alpha_prev.sqrt() * denoised + beta_prev.sqrt() * noise
        else:
            latents = denoised
        np.save(output_dir / f"latent_{step_index + 1}.npy", latents.detach().numpy())

    np.save(output_dir / "scheduler_coeffs.npy", np.asarray(scheduler_coeffs, dtype=np.float32))
    np.asarray(scheduler_coeffs, dtype=np.float32).tofile(output_dir / "scheduler_coeffs.bin")

    image = pipe.vae.decode(
        latents / pipe.vae.config.scaling_factor,
        return_dict=False,
    )[0]
    image = pipe.image_processor.postprocess(image.detach(), output_type="pil")[0]
    image.save(output_dir / "reference.png")
    print(f"Saved fixed-prompt artifacts to {output_dir}")


if __name__ == "__main__":
    main()

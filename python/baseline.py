"""Generate a reproducible Stable Diffusion 1.5 + LCM baseline image."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from diffusers import DiffusionPipeline, LCMScheduler


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="a photo of a cat")
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument("--model_id", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--lcm_lora_id", default="latent-consistency/lcm-lora-sdv1-5")
    parser.add_argument("--steps", type=int, default=4, choices=(4, 8))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--output", default="../results/baseline.png")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.height != 512 or args.width != 512:
        raise ValueError("The first RKNN baseline only supports 512x512.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    pipe = DiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.float32,
        safety_checker=None,
    )
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(args.lcm_lora_id)
    pipe.fuse_lora()
    pipe = pipe.to("cpu")

    image = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.steps,
        guidance_scale=1.0,
        height=args.height,
        width=args.width,
        generator=generator,
    ).images[0]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    print(f"Saved baseline image to {output.resolve()}")


if __name__ == "__main__":
    main()

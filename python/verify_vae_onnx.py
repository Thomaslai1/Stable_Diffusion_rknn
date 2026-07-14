"""Compare the PyTorch VAE decoder output with ONNX Runtime."""

import argparse

import numpy as np
import onnxruntime as ort
import torch
from diffusers import AutoencoderKL


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae_dir", required=True)
    parser.add_argument("--onnx", default="../model/vae_decoder.onnx")
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    latents = torch.randn(1, 4, 64, 64, dtype=torch.float32)
    vae = AutoencoderKL.from_pretrained(args.vae_dir, torch_dtype=torch.float32)
    vae.eval()

    with torch.no_grad():
        torch_output = vae.decode(
            latents / vae.config.scaling_factor,
            return_dict=False,
        )[0].numpy()

    session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    onnx_output = session.run(["image"], {"latents": latents.numpy()})[0]

    absolute_error = np.abs(torch_output - onnx_output)
    print(f"max_abs_error: {absolute_error.max():.8f}")
    print(f"mean_abs_error: {absolute_error.mean():.8f}")
    print(f"torch_range: [{torch_output.min():.6f}, {torch_output.max():.6f}]")
    print(f"onnx_range:  [{onnx_output.min():.6f}, {onnx_output.max():.6f}]")


if __name__ == "__main__":
    main()

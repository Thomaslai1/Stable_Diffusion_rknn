"""Export the Stable Diffusion VAE decoder to a fixed-shape ONNX model."""

import argparse
from pathlib import Path

import onnx
import torch
from diffusers import AutoencoderKL


class VAEDecoderWrapper(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latents):
        latents = latents / self.vae.config.scaling_factor
        return self.vae.decode(latents, return_dict=False)[0]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae_dir", required=True)
    parser.add_argument("--output", default="../model/vae_decoder.onnx")
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def main():
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    vae = AutoencoderKL.from_pretrained(args.vae_dir, torch_dtype=torch.float32)
    vae.eval()
    decoder = VAEDecoderWrapper(vae).eval()
    latents = torch.randn(1, 4, 64, 64, dtype=torch.float32)

    with torch.no_grad():
        torch.onnx.export(
            decoder,
            (latents,),
            str(output),
            input_names=["latents"],
            output_names=["image"],
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=False,
        )

    onnx.checker.check_model(str(output))
    model = onnx.load(str(output), load_external_data=False)
    print(f"Saved ONNX model to {output.resolve()}")
    for value in model.graph.input:
        print(f"input: {value.name}")
    for value in model.graph.output:
        print(f"output: {value.name}")


if __name__ == "__main__":
    main()

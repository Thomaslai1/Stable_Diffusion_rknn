"""Export the fused Stable Diffusion UNet to a fixed-shape ONNX model."""

import argparse
from pathlib import Path

import onnx
import torch
from diffusers import DiffusionPipeline


class UNetWrapper(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, sample, timestep, encoder_hidden_states):
        return self.unet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
        )[0]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--lcm_lora_id", required=True)
    parser.add_argument("--output", default="../model/unet.onnx")
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def main():
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    pipe = DiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.float32,
        safety_checker=None,
    )
    pipe.load_lora_weights(args.lcm_lora_id)
    pipe.fuse_lora()

    unet = UNetWrapper(pipe.unet).eval()
    sample = torch.randn(1, 4, 64, 64, dtype=torch.float32)
    timestep = torch.tensor([999.0], dtype=torch.float32)
    encoder_hidden_states = torch.randn(1, 77, 768, dtype=torch.float32)

    with torch.no_grad():
        torch.onnx.export(
            unet,
            (sample, timestep, encoder_hidden_states),
            str(output),
            input_names=["sample", "timestep", "encoder_hidden_states"],
            output_names=["out_sample"],
            opset_version=args.opset,
            do_constant_folding=True,
            external_data=True,
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

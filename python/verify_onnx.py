"""Compare the fused PyTorch UNet output with ONNX Runtime."""

import argparse

import numpy as np
import onnxruntime as ort
import torch
from diffusers import DiffusionPipeline


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--lcm_lora_id", required=True)
    parser.add_argument("--onnx", default="../model/unet.onnx")
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    sample = torch.randn(1, 4, 64, 64, dtype=torch.float32)
    timestep = torch.tensor([999.0], dtype=torch.float32)
    encoder_hidden_states = torch.randn(1, 77, 768, dtype=torch.float32)

    pipe = DiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.float32,
        safety_checker=None,
    )
    pipe.load_lora_weights(args.lcm_lora_id)
    pipe.fuse_lora()
    pipe.unet.eval()

    with torch.no_grad():
        torch_output = pipe.unet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
        )[0].numpy()

    session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    onnx_output = session.run(
        ["out_sample"],
        {
            "sample": sample.numpy(),
            "timestep": timestep.numpy(),
            "encoder_hidden_states": encoder_hidden_states.numpy(),
        },
    )[0]

    absolute_error = np.abs(torch_output - onnx_output)
    print(f"max_abs_error: {absolute_error.max():.8f}")
    print(f"mean_abs_error: {absolute_error.mean():.8f}")
    print(f"torch_range: [{torch_output.min():.6f}, {torch_output.max():.6f}]")
    print(f"onnx_range:  [{onnx_output.min():.6f}, {onnx_output.max():.6f}]")


if __name__ == "__main__":
    main()

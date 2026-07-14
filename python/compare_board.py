"""Compare a board-produced output with the ONNX Runtime reference."""

import argparse

import numpy as np
import onnxruntime as ort


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["unet", "vae"])
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--board_output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.kind == "unet":
        names = ["sample", "timestep", "encoder_hidden_states"]
        files = [
            "unet_sample.npy",
            "unet_timestep.npy",
            "unet_encoder_hidden_states.npy",
        ]
        output_name = "out_sample"
    else:
        names = ["latents"]
        files = ["vae_latents.npy"]
        output_name = "image"

    inputs = {
        name: np.load(f"{args.input_dir}/{file}")
        for name, file in zip(names, files)
    }
    reference = ort.InferenceSession(
        args.onnx,
        providers=["CPUExecutionProvider"],
    ).run([output_name], inputs)[0]
    board_output = np.load(args.board_output)
    absolute_error = np.abs(reference - board_output)
    print(f"max_abs_error: {absolute_error.max():.8f}")
    print(f"mean_abs_error: {absolute_error.mean():.8f}")
    print(f"onnx_range: [{reference.min():.6f}, {reference.max():.6f}]")
    print(f"board_range: [{board_output.min():.6f}, {board_output.max():.6f}]")


if __name__ == "__main__":
    main()

"""Run one RKNN model on a Rockchip board with RKNNLite."""

import argparse
from pathlib import Path

import numpy as np
from rknnlite.api import RKNNLite


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["unet", "vae"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)

    if args.kind == "unet":
        inputs = [
            np.load(input_dir / "unet_sample.npy"),
            np.load(input_dir / "unet_timestep.npy"),
            np.load(input_dir / "unet_encoder_hidden_states.npy"),
        ]
    else:
        inputs = [np.load(input_dir / "vae_latents.npy")]

    rknn = RKNNLite()
    ret = rknn.load_rknn(args.model)
    if ret != 0:
        raise RuntimeError(f"load_rknn failed: {ret}")
    ret = rknn.init_runtime()
    if ret != 0:
        raise RuntimeError(f"init_runtime failed: {ret}")

    outputs = rknn.inference(inputs=inputs)
    np.save(args.output, outputs[0])
    print(f"Saved output to {Path(args.output).resolve()}")
    print(f"output shape: {outputs[0].shape}")

    rknn.release()


if __name__ == "__main__":
    main()

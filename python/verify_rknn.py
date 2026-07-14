"""Compare an FP RKNN model with its ONNX Runtime reference output."""

import argparse

import numpy as np
import onnxruntime as ort
from rknn.api import RKNN


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["unet", "vae"])
    parser.add_argument("--rknn", required=True)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--simulate", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    if args.kind == "unet":
        inputs = [
            rng.standard_normal((1, 4, 64, 64), dtype=np.float32),
            np.array([999.0], dtype=np.float32),
            rng.standard_normal((1, 77, 768), dtype=np.float32),
        ]
        input_names = ["sample", "timestep", "encoder_hidden_states"]
        output_name = "out_sample"
    else:
        inputs = [rng.standard_normal((1, 4, 64, 64), dtype=np.float32)]
        input_names = ["latents"]
        output_name = "image"

    onnx_session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    reference = onnx_session.run(
        [output_name],
        dict(zip(input_names, inputs)),
    )[0]

    rknn = RKNN(verbose=False)
    try:
        if args.simulate:
            ret = rknn.config(target_platform="rk3588", optimization_level=3)
            if ret != 0:
                raise RuntimeError(f"RKNN config failed: {ret}")
            ret = rknn.load_onnx(model=args.onnx)
            if ret != 0:
                raise RuntimeError(f"RKNN load_onnx failed: {ret}")
            ret = rknn.build(do_quantization=False)
            if ret != 0:
                raise RuntimeError(f"RKNN build failed: {ret}")
        else:
            ret = rknn.load_rknn(args.rknn)
            if ret != 0:
                raise RuntimeError(f"RKNN load_rknn failed: {ret}")
        ret = rknn.init_runtime()
        if ret != 0:
            raise RuntimeError(f"RKNN init_runtime failed: {ret}")
        result = rknn.inference(inputs=inputs)[0]
    finally:
        rknn.release()

    absolute_error = np.abs(reference - result)
    print(f"max_abs_error: {absolute_error.max():.8f}")
    print(f"mean_abs_error: {absolute_error.mean():.8f}")
    print(f"onnx_range: [{reference.min():.6f}, {reference.max():.6f}]")
    print(f"rknn_range: [{result.min():.6f}, {result.max():.6f}]")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Convert a fixed-shape Stable Diffusion ONNX submodel to RKNN."""

import argparse
from pathlib import Path

from rknn.api import RKNN


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("onnx_model", help="Fixed-shape UNet or VAE ONNX model")
    parser.add_argument("target_platform", choices=["rk3588"])
    parser.add_argument("dtype", nargs="?", choices=["i8", "fp"], default="fp")
    parser.add_argument("output_rknn_path", nargs="?")
    parser.add_argument("--dataset", help="Calibration dataset for INT8 conversion")
    return parser.parse_args()


def main():
    args = parse_args()
    onnx_path = Path(args.onnx_model).resolve()
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    output_path = (
        Path(args.output_rknn_path).resolve()
        if args.output_rknn_path
        else onnx_path.with_suffix(".rknn")
    )
    if args.dtype == "i8" and not args.dataset:
        raise ValueError("--dataset is required for INT8 conversion")

    rknn = RKNN(verbose=False)
    try:
        print("--> Config model")
        ret = rknn.config(
            target_platform=args.target_platform,
            optimization_level=3,
        )
        if ret != 0:
            raise RuntimeError(f"RKNN config failed: {ret}")

        print("--> Load ONNX model")
        ret = rknn.load_onnx(model=str(onnx_path))
        if ret != 0:
            raise RuntimeError(f"RKNN load_onnx failed: {ret}")

        print(f"--> Build RKNN model ({args.dtype})")
        build_kwargs = {"do_quantization": args.dtype == "i8"}
        if args.dtype == "i8":
            build_kwargs["dataset"] = str(Path(args.dataset).resolve())
        ret = rknn.build(**build_kwargs)
        if ret != 0:
            raise RuntimeError(f"RKNN build failed: {ret}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        print("--> Export RKNN model")
        ret = rknn.export_rknn(str(output_path))
        if ret != 0:
            raise RuntimeError(f"RKNN export_rknn failed: {ret}")
        print(f"Done: {output_path}")
    finally:
        rknn.release()


if __name__ == "__main__":
    main()

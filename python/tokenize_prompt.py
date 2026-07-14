"""Tokenize a Stable Diffusion prompt and save the 1x77 input_ids tensor."""

from pathlib import Path
import argparse

import numpy as np
from transformers import CLIPTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt")
    parser.add_argument("--tokenizer_dir", default="D:/models/stable-diffusion-v1-5/tokenizer")
    parser.add_argument("--output_dir", default="testdata/prompt")
    args = parser.parse_args()
    tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_dir)
    input_ids = tokenizer(
        args.prompt,
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="np",
    )["input_ids"].astype(np.int32)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "input_ids.npy", input_ids)
    input_ids.tofile(output_dir / "input_ids.bin")
    print(f"Saved {input_ids.shape} to {output_dir}")


if __name__ == "__main__":
    main()

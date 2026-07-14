"""Export the CLIP text encoder used by Stable Diffusion to ONNX."""

from pathlib import Path
import argparse

import torch
from transformers import CLIPTextModel


class TextEncoderWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        return self.model(input_ids=input_ids.to(torch.long), return_dict=False)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="D:/models/stable-diffusion-v1-5/text_encoder")
    parser.add_argument("--output", default="model/text_encoder.onnx")
    args = parser.parse_args()

    model = CLIPTextModel.from_pretrained(args.model_dir, torch_dtype=torch.float32)
    model.eval()
    wrapper = TextEncoderWrapper(model)
    input_ids = torch.zeros((1, 77), dtype=torch.int32)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (input_ids,),
            output,
            input_names=["input_ids"],
            output_names=["prompt_embeds"],
            opset_version=17,
            dynamo=False,
        )
    print(f"Saved {output}")


if __name__ == "__main__":
    main()

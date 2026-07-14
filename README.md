# stable_diffusion

## Table of contents

- [1. Description](#1-description)
- [2. Current Support Platform](#2-current-support-platform)
- [3. Pretrained Model](#3-pretrained-model)
- [4. Convert to RKNN](#4-convert-to-rknn)
- [5. Python Verification](#5-python-verification)
- [6. Android Demo](#6-android-demo)
- [7. Linux Demo](#7-linux-demo)
- [8. Expected Results](#8-expected-results)
- [9. Board Validation](#9-board-validation)
- [10. Reports](#10-reports)
- [11. Limitations](#11-limitations)

## 1. Description

This example deploys Stable Diffusion 1.5 with LCM-LoRA to RK3588 using RKNN-Toolkit2.

The deployment is split into CPU and NPU components:

| Component | Runtime |
| --- | --- |
| Tokenizer and text encoder | CPU |
| UNet denoiser | RK3588 NPU |
| VAE decoder | RK3588 NPU or CPU fallback during validation |
| Scheduler, CFG and latent update | CPU |

The first version uses a fixed 512x512 input, batch size 1 and 4 or 8 inference steps.

## 2. Current Support Platform

| Platform | Status | Notes |
| --- | --- | --- |
| RK3588 Linux aarch64 | Planned | Requires RKNN-Toolkit2 and RKNPU2 runtime |
| RK3588 Android arm64-v8a | Planned | Requires compatible Android NDK and runtime libraries |

## 3. Pretrained Model

The first baseline uses Stable Diffusion 1.5 with the official LCM-LoRA weights:

- Base model: `runwayml/stable-diffusion-v1-5`
- LCM-LoRA: `latent-consistency/lcm-lora-sdv1-5`

Install the baseline dependencies from `requirements-baseline.txt`, then run:

```shell
cd python
python baseline.py \
  --prompt "a photo of a cat" \
  --seed 42 \
  --steps 4 \
  --output ../results/baseline.png
```

The first run downloads the model from Hugging Face. The generated image is the reference output for later ONNX and RKNN comparisons.

## 4. Convert to RKNN

Conversion instructions will be added after the ONNX input/output contract is fixed.

## 5. Python Verification

Python verification instructions will be added together with the reference implementation.

## 6. Android Demo

Android build, deployment and run instructions will be added after the Linux path is validated.

## 7. Linux Demo

Linux build, deployment and run instructions will be added after the RKNN Python verification passes.

## 8. Expected Results

Performance and output examples will be recorded after board validation.

## 9. Board Validation

Board validation results will be recorded after the RK3588 runtime is available.

## 10. Reports

Conversion, accuracy and performance reports will be added during validation.

## 11. Limitations

The initial example does not support dynamic resolution, batch inference, ControlNet, Img2Img, Inpainting or dynamic LoRA switching.
# 任意 Prompt 阶段

Text Encoder 的输入是 tokenizer 生成的 `1×77` 个 token id，输出是 UNet 使用的 `1×77×768` prompt embedding。

先在电脑上生成测试输入：

```bash
python python/tokenize_prompt.py "a photo of a dog" --output_dir testdata/prompt
```

再将 Text Encoder 转换为 RKNN：

```bash
python python/export_text_encoder_onnx.py --output model/text_encoder.onnx
python python/convert.py model/text_encoder.onnx rk3588 fp model/text_encoder_fp.rknn
```

Android Demo 的输入是 `input_ids.bin`，输出是 `prompt_embeds.bin`。该 embedding 可以直接替换固定 prompt Demo 使用的 `prompt_embeds.bin`。

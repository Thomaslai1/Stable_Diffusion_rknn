# stable_diffusion

Stable Diffusion 1.5 + LCM-LoRA on RK3588 with RKNN. The current Android demo supports arbitrary text prompts and fixed 512×512 image generation in four steps.

## Table of contents

- [1. Description](#1-description)
- [2. Current Support Platform](#2-current-support-platform)
- [3. Directory Structure](#3-directory-structure)
- [4. Pretrained Model](#4-pretrained-model)
- [5. Environment](#5-environment)
- [6. Export and Convert](#6-export-and-convert)
- [7. Android Deployment](#7-android-deployment)
- [8. One-command Image Generation](#8-one-command-image-generation)
- [9. Validation Results](#9-validation-results)
- [10. Reports](#10-reports)
- [11. Limitations](#11-limitations)

## 1. Description

The pipeline is split between the CPU and RK3588 NPU:

| Component | Runtime |
| --- | --- |
| Tokenizer | Windows Python during the current demo stage |
| Text Encoder | RK3588 NPU through RKNN |
| UNet denoiser | RK3588 NPU through RKNN |
| VAE decoder | RK3588 NPU through RKNN |
| LCM scheduler and latent update | Android C++ CPU |

The default configuration uses batch size 1, 512×512 resolution, FP RKNN models and four LCM steps.

## 2. Current Support Platform

| Platform | Status |
| --- | --- |
| RK3588 Android arm64-v8a | Tested |
| RK3588 Linux aarch64 | Model conversion tested; demo packaging pending |

## 3. Directory Structure

```text
cpp/       Android C++ demos and RKNN API header
python/    baseline, export, conversion and tokenizer scripts
scripts/   one-command generation scripts
model/     local ONNX/RKNN artifacts, ignored by Git
testdata/  local test inputs and outputs, ignored by Git
results/   generated images, ignored by Git
```

## 4. Pretrained Model

The example uses:

- Base model: Stable Diffusion 1.5
- LoRA: `latent-consistency/lcm-lora-sdv1-5`
- Resolution: 512×512
- Sampling: LCM, four steps

Keep the Diffusers model in a local directory such as:

```text
D:/models/stable-diffusion-v1-5
D:/models/lcm-lora-sdv1-5
```

Large model files are intentionally excluded from Git.

## 5. Environment

Windows is used for baseline, tokenizer and image conversion. WSL is used for RKNN Toolkit2 conversion.

The existing local environments are:

```text
Windows: D:/HuaweiMoveData/Users/laiy5/Desktop/stable_diffusion_rknn/.venv
WSL:     /home/laiy5/rknn312
```

Activate the WSL conversion environment with:

```bash
source /home/laiy5/rknn312/bin/activate
```

## 6. Export and Convert

Export the Text Encoder to ONNX and prepare a prompt input:

```powershell
python python/export_text_encoder_onnx.py --output model/text_encoder.onnx
python python/tokenize_prompt.py "a photo of a dog" --output_dir testdata/prompt
```

Convert the Text Encoder to RKNN in WSL:

```bash
python python/convert.py model/text_encoder.onnx rk3588 fp model/text_encoder_fp.rknn
```

UNet and VAE conversion uses the same `python/convert.py` entry point. The generated RKNN files remain in the ignored `model/` directory.

## 7. Android Deployment

The board is Android arm64-v8a and must have the RKNN runtime available. The current scripts use Windows ADB:

```text
C:/Users/laiy5/AppData/Local/Android/Sdk/platform-tools/adb.exe
```

The board-side directory expected by the one-command script is:

```text
/data/local/tmp/stable_diffusion/
```

It must contain:

```text
fixed_prompt_rknn_demo
unet_fp.rknn
vae_decoder_fp.rknn
fixed_prompt/
```

The Text Encoder demo also needs `text_encoder_rknn_demo`, `text_encoder_fp.rknn`, `librknnrt.so` and `libc++_shared.so`. The script pushes these files automatically from the local workspace.

## 8. One-command Image Generation

From PowerShell:

```powershell
cd D:\HuaweiMoveData\Users\laiy5\Desktop\Stable_Diffusion_rknn_repo
powershell -ExecutionPolicy Bypass -File .\scripts\generate.ps1 "a photo of a dog"
```

The script passes the current prompt embedding explicitly to the C++ demo, so an older prompt embedding cannot be reused accidentally. It performs:

```text
prompt → tokenizer → Text Encoder → UNet → scheduler → VAE → PNG
```

The default output is:

```text
results/generated.png
```

Specify another output path with:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\generate.ps1 `
  "a golden robot in a library" `
  -Output results\robot.png
```

## 9. Validation Results

| Stage | Result |
| --- | --- |
| PyTorch → ONNX UNet | max error about `7.4e-6` |
| PyTorch → ONNX VAE | max error about `7.3e-5` |
| VAE board validation | completed |
| UNet board validation | completed |
| Text Encoder board validation | mean error about `0.00846` after INT32 input fix |
| Fixed prompt board generation | completed |
| Arbitrary prompt board generation | completed |

## 10. Reports

- [PITFALLS_REPORT.md](PITFALLS_REPORT.md): deployment issues and solutions.

## 11. Limitations

The current demo is limited to 512×512, batch size 1, four LCM steps and FP RKNN models. Android performs tokenizer preprocessing on the host through the PowerShell wrapper; a fully native Android tokenizer is a future improvement. INT8 quantization, dynamic resolution, Img2Img, Inpainting, ControlNet and dynamic LoRA switching are not included.

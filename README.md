[简体中文](README_CN.md) | [English](README.md)

# Stable Diffusion RKNN Demo

## Description

This example deploys Stable Diffusion 1.5 with LCM-LoRA to an RK3588 Android board based on the RKNN-Toolkit2 toolchain. It includes the process of exporting the Text Encoder, UNet and VAE models to ONNX, converting them to RKNN models, and running the complete text-to-image pipeline with the Android C++ API.

- Support `RK3588` Android `arm64-v8a`.
- Support arbitrary text prompts through the CLIP tokenizer and Text Encoder.
- Support 512×512 image generation with four LCM steps.
- Use RKNN FP16 models for Text Encoder, UNet and VAE inference.

The current pipeline is split into CPU and NPU components:

| Component | Runtime |
| --- | --- |
| Tokenizer | Host-side Python |
| Text Encoder | RK3588 NPU through RKNN |
| UNet | RK3588 NPU through RKNN |
| VAE decoder | RK3588 NPU through RKNN |
| LCM scheduler | Android C++ CPU |

## Dependency library installation

This example relies on `RKNN-Toolkit2` for model conversion and the Android NDK for compiling the C++ demo. Please refer to the [RKNN-Toolkit2 Quick Start](https://github.com/airockchip/rknn-toolkit2/tree/master/doc) for the official installation instructions.

- `RKNN-Toolkit2 >= 2.3.2` is required for model conversion.
- Android NDK `r19` is recommended for compiling the Android demo.
- The board must provide a compatible `librknnrt.so`.
- The host environment requires Python, PyTorch, Diffusers, Transformers, ONNX and Pillow.

The tested local environments are:

```text
Windows Python: <project-root>/.venv
WSL RKNN:       <wsl-home>/rknn312
Android NDK:    <android-sdk>/ndk/19.2.5345600
```

## Model support

| Category | Name | Dtype | Model files | Support platform |
| --- | --- | --- | --- | --- |
| Text-to-Image | [Stable Diffusion v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5) | FP16 | Text Encoder, UNet, VAE | RK3588 Android |
| Text-to-Image | [LCM-LoRA](https://huggingface.co/latent-consistency/lcm-lora-sdv1-5) | FP16 | Fused into UNet during export | RK3588 Android |

The model files are not included in this repository because of their size. Keep the Diffusers model and LCM-LoRA in local directories such as:

```text
<models-root>/stable-diffusion-v1-5
<models-root>/lcm-lora-sdv1-5
```

## Directory structure

```text
cpp/       Android C++ demos and RKNN API header
python/    baseline, export, conversion and tokenizer scripts
scripts/   one-command prompt generation script
model/     generated ONNX/RKNN files, ignored by Git
testdata/  local inputs and validation outputs, ignored by Git
results/   generated images, ignored by Git
```

## Convert model

Activate the WSL RKNN environment:

```sh
source <wsl-home>/rknn312/bin/activate
cd <project-root>
```

Export the Text Encoder and convert it to RKNN:

```sh
python python/export_text_encoder_onnx.py --output model/text_encoder.onnx
python python/convert.py model/text_encoder.onnx rk3588 fp model/text_encoder_fp.rknn
```

The UNet and VAE use the same conversion entry point:

```sh
python python/convert.py model/unet.onnx rk3588 fp model/unet_fp.rknn
python python/convert.py model/vae_decoder.onnx rk3588 fp model/vae_decoder_fp.rknn
```

## Compile Demo

For Android development boards, set the Android NDK path and compile the C++ demos with the Android clang toolchain:

```powershell
$clang = "<android-sdk>\ndk\19.2.5345600\toolchains\llvm\prebuilt\windows-x86_64\bin\aarch64-linux-android21-clang++.cmd"
New-Item -ItemType Directory -Force build | Out-Null
```

The following demos are included:

```text
text_encoder_rknn_demo
fixed_prompt_rknn_demo
unet_rknn_demo
vae_rknn_demo
```

The generated executables and runtime libraries are local deployment artifacts and are not pushed to Git.

## Run Demo

The board-side directory is:

```text
/data/local/tmp/stable_diffusion/
```

It must contain the UNet, VAE and scheduler artifacts used by `fixed_prompt_rknn_demo`. The one-command script automatically pushes the Text Encoder model and runtime libraries, then runs the complete pipeline.

```powershell
cd <project-root>
powershell -ExecutionPolicy Bypass -File .\scripts\generate.ps1 "a photo of a dog"
```

Specify the output file with:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\generate.ps1 `
  "a golden robot in a library" `
  -Output results\robot.png
```

The output image is saved to `results/generated.png` by default.

The seed and LCM step count can be changed without editing code:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\generate.ps1 `
  "a photo of a dog" -Seed 123 -Steps 6 -Output results\dog_6steps.png
```

Supported step counts are `4`, `6` and `8`.

## Model performance benchmark

The current implementation is a functional baseline. Text Encoder timing has been measured; model files are uploaded only on the first run and reused on subsequent runs.

| Component | Configuration | Status |
| --- | --- | --- |
| Text Encoder | RKNN FP16, 1×77 input | Board validated |
| UNet | RKNN FP16, 4 denoising steps | Board validated |
| VAE decoder | RKNN FP16, 512×512 output | Board validated |
| End-to-end generation | 512×512, four LCM steps | Board validated |

The measured Text Encoder board inference time is about 46–78 ms in the current test. Inference time does not yet include a formal multi-core and INT8 comparison.

Scheduler input generation was checked with seed `42`:

| Steps | `timesteps.bin` | `scheduler_coeffs.bin` | Noise files | Status |
| ---: | ---: | ---: | ---: | --- |
| 4 | 16 B | 96 B | 3 | Passed |
| 6 | 24 B | 144 B | 5 | Passed |
| 8 | 32 B | 192 B | 7 | Passed |

The initial latent file is 65,536 B for all configurations (`1×4×64×64` FLOAT32). Full board-side image quality and timing comparisons for 4, 6 and 8 steps are still pending.

## Validation results

| Item | Result |
| --- | --- |
| PyTorch to ONNX UNet | maximum error about `7.4e-6` |
| PyTorch to ONNX VAE | maximum error about `7.3e-5` |
| VAE RKNN board inference | Passed |
| UNet RKNN board inference | Passed |
| Text Encoder board inference | Passed; mean error about `0.00846` after INT32 input fix |
| Arbitrary prompt Text Encoder output | Passed; different prompts produce different embeddings |
| Arbitrary prompt end-to-end generation | Single-prompt path passed; two-prompt regression test pending |

## Reports

- [PITFALLS_REPORT.md](PITFALLS_REPORT.md): deployment issues, causes and solutions.

## Release Notes

| Version | Description |
| --- | --- |
| 0.3.0 | Add arbitrary prompt generation and one-command Android pipeline. |
| 0.2.0 | Add Android UNet and fixed-prompt image generation. |
| 0.1.0 | Add PyTorch baseline, ONNX export and VAE/UNet RKNN validation. |

## Environment dependencies

The models were converted with RKNN-Toolkit2 2.3.2 and tested with the Android RKNN runtime supplied for RK3588. Using a different Toolkit2/runtime version may change model loading, performance or output accuracy.

| Component | Tested version |
| --- | --- |
| RKNN-Toolkit2 | 2.3.2 |
| Android NDK | 19.2.5345600 |
| Target ABI | arm64-v8a |
| Android | 14 |

## RKNPU Resource

- [RKNN-Toolkit2](https://github.com/airockchip/rknn-toolkit2)
- [RKNN Model Zoo](https://github.com/airockchip/rknn_model_zoo)
- [Stable Diffusion v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
- [LCM-LoRA for Stable Diffusion v1.5](https://huggingface.co/latent-consistency/lcm-lora-sdv1-5)

## License

Please follow the licenses of Stable Diffusion v1.5, LCM-LoRA, RKNN-Toolkit2 and all dependent components.

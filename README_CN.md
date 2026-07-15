[简体中文](README_CN.md) | [English](README.md)

# Stable Diffusion RKNN 示例

## Description

本示例基于 RKNN-Toolkit2，将 Stable Diffusion 1.5 和 LCM-LoRA 部署到 RK3588 Android 开发板。示例包含 Text Encoder、UNet、VAE 的 ONNX 导出、RKNN 转换，以及使用 Android C++ API 执行完整文生图流程。

- 支持 `RK3588` Android `arm64-v8a` 平台。
- 支持通过 CLIP Tokenizer 和 Text Encoder 输入任意文本 prompt。
- 支持 512×512 分辨率、4 步 LCM 生图。
- Text Encoder、UNet、VAE 当前使用 RKNN FP16 模型推理。

当前流程分为 CPU 和 NPU 两部分：

| 组件 | 运行位置 |
| --- | --- |
| Tokenizer | 主机 Python |
| Text Encoder | RK3588 NPU，通过 RKNN 推理 |
| UNet | RK3588 NPU，通过 RKNN 推理 |
| VAE 解码器 | RK3588 NPU，通过 RKNN 推理 |
| LCM Scheduler | Android C++ CPU |

## Dependency library installation

本示例使用 `RKNN-Toolkit2` 完成模型转换，使用 Android NDK 编译 C++ Demo。依赖安装请参考 [RKNN-Toolkit2 Quick Start](https://github.com/airockchip/rknn-toolkit2/tree/master/doc)。

- 模型转换需要 `RKNN-Toolkit2 >= 2.3.2`。
- Android Demo 推荐使用 Android NDK `r19` 编译。
- RK3588 板端需要兼容的 `librknnrt.so`。
- 主机需要安装 Python、PyTorch、Diffusers、Transformers、ONNX 和 Pillow。

本项目验证使用的本地环境如下：

```text
Windows Python: <project-root>/.venv
WSL RKNN:       <wsl-home>/rknn312
Android NDK:    <android-sdk>/ndk/19.2.5345600
```

## Model support

| 类别 | 模型 | 类型 | 模型文件 | 支持平台 |
| --- | --- | --- | --- | --- |
| 文生图 | [Stable Diffusion v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5) | FP16 | Text Encoder、UNet、VAE | RK3588 Android |
| 文生图 | [LCM-LoRA](https://huggingface.co/latent-consistency/lcm-lora-sdv1-5) | FP16 | 导出 UNet 时融合 | RK3588 Android |

由于模型文件较大，本仓库不包含模型权重。请将 Diffusers 基础模型和 LCM-LoRA 放在本地目录，例如：

```text
<models-root>/stable-diffusion-v1-5
<models-root>/lcm-lora-sdv1-5
```

## Directory structure

```text
cpp/       Android C++ Demo 和 RKNN API 头文件
python/    基线、导出、转换和 Tokenizer 脚本
scripts/   一键 prompt 生图脚本
model/     生成的 ONNX/RKNN 文件，Git 忽略
testdata/  测试输入和验证输出，Git 忽略
results/   生成图片，Git 忽略
```

## Convert model

激活 WSL 中的 RKNN 环境：

```sh
source <wsl-home>/rknn312/bin/activate
cd <project-root>
```

导出 Text Encoder 并转换为 RKNN：

```sh
python python/export_text_encoder_onnx.py --output model/text_encoder.onnx
python python/convert.py model/text_encoder.onnx rk3588 fp model/text_encoder_fp.rknn
```

UNet 和 VAE 使用相同的转换入口：

```sh
python python/convert.py model/unet.onnx rk3588 fp model/unet_fp.rknn
python python/convert.py model/vae_decoder.onnx rk3588 fp model/vae_decoder_fp.rknn
```

## Compile Demo

编译 Android Demo 时，请根据本机环境设置 Android NDK 路径，并使用 Android clang 工具链：

```powershell
$clang = "<android-sdk>\ndk\19.2.5345600\toolchains\llvm\prebuilt\windows-x86_64\bin\aarch64-linux-android21-clang++.cmd"
New-Item -ItemType Directory -Force build | Out-Null
```

本示例包含以下 Demo：

```text
text_encoder_rknn_demo
fixed_prompt_rknn_demo
unet_rknn_demo
vae_rknn_demo
```

生成的可执行文件和运行库只用于本地部署，不提交到 Git。

## Run Demo

板端工作目录为：

```text
/data/local/tmp/stable_diffusion/
```

该目录需要包含 UNet、VAE 以及 `fixed_prompt_rknn_demo` 所需的 Scheduler 测试文件。一键脚本会自动上传 Text Encoder 模型和运行库，并执行完整流程。

```powershell
cd <project-root>
powershell -ExecutionPolicy Bypass -File .\scripts\generate.ps1 "a photo of a dog"
```

指定输出文件：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\generate.ps1 `
  "a golden robot in a library" `
  -Output results\robot.png
```

默认输出文件为 `results/generated.png`。

## Model performance benchmark

当前实现首先保证功能正确。Text Encoder 的板端耗时已经测量；由于当前脚本每次运行都会上传模型文件，完整端到端耗时仍在补充测试。

| 组件 | 配置 | 状态 |
| --- | --- | --- |
| Text Encoder | RKNN FP16，输入 1×77 | 已完成板端验证 |
| UNet | RKNN FP16，4 步去噪 | 已完成板端验证 |
| VAE 解码器 | RKNN FP16，输出 512×512 | 已完成板端验证 |
| 端到端生图 | 512×512，4 步 LCM | 已完成板端验证 |

当前 Text Encoder 板端实测耗时约为 46–78 ms。该数据还没有包含多核和 INT8 的正式对比；当前脚本的 ADB 上传耗时也不代表实际部署后的应用耗时。

## Validation results

| 项目 | 结果 |
| --- | --- |
| PyTorch 到 ONNX 的 UNet | 最大误差约 `7.4e-6` |
| PyTorch 到 ONNX 的 VAE | 最大误差约 `7.3e-5` |
| VAE RKNN 板端推理 | 通过 |
| UNet RKNN 板端推理 | 通过 |
| Text Encoder 板端推理 | 通过；修复 INT32 输入后平均误差约 `0.00846` |
| 任意 prompt Text Encoder 输出 | 通过；不同 prompt 会生成不同 embedding |
| 任意 prompt 端到端生图 | 单 prompt 流程通过；双 prompt 回归测试待完成 |

## Reports

- [PITFALLS_REPORT.md](PITFALLS_REPORT.md)：部署过程中的问题、原因和解决方法。

## Release Notes

| 版本 | 说明 |
| --- | --- |
| 0.3.0 | 增加任意 prompt 生图和一键 Android 生图流程。 |
| 0.2.0 | 增加 Android UNet、固定 prompt 和板端生图。 |
| 0.1.0 | 增加 PyTorch 基线、ONNX 导出以及 VAE/UNet RKNN 验证。 |

## Environment dependencies

模型使用 RKNN-Toolkit2 2.3.2 转换，并使用 RK3588 Android RKNN Runtime 验证。若更换 Toolkit2 或 Runtime 版本，模型加载、性能和精度可能发生变化。

| 组件 | 验证版本 |
| --- | --- |
| RKNN-Toolkit2 | 2.3.2 |
| Android NDK | 19.2.5345600 |
| 目标 ABI | arm64-v8a |
| Android | 14 |

## RKNPU Resource

- [RKNN-Toolkit2](https://github.com/airockchip/rknn-toolkit2)
- [RKNN Model Zoo](https://github.com/airockchip/rknn_model_zoo)
- [Stable Diffusion v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
- [LCM-LoRA for Stable Diffusion v1.5](https://huggingface.co/latent-consistency/lcm-lora-sdv1-5)

## License

请遵循 Stable Diffusion v1.5、LCM-LoRA、RKNN-Toolkit2 以及所有依赖组件各自的许可证。

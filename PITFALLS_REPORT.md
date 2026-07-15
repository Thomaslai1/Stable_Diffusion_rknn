Exit code: 0
Wall time: 2.7 seconds
Output:
# RK3588 Stable Diffusion 部署踩坑报告

## 1. 项目目标

将 Stable Diffusion 1.5 + LCM LoRA 部署到 RK3588：

- UNet、VAE 使用 RKNN，在板端 NPU 推理。
- Text Encoder 暂时保留在 CPU 侧。
- 使用 512×512、LCM 4 步推理。
- 先完成固定 prompt 生图，再扩展到任意 prompt。

## 2. 模型文件相关问题

### 2.1 不同组件有同名文件

Text Encoder、UNet、VAE 都可能出现 `config.json` 或权重文件。不能只看文件名判断组件，必须结合所在目录和权重 key 判断：

- 出现 `text_model.*`：通常是 Text Encoder。
- 出现 `conv_in.*`：通常是 UNet。
- 出现 `decoder.*`：通常是 VAE。

### 2.2 下载了不需要的大文件

同一个模型可能同时提供 `.bin`、`.safetensors`、FP16、EMA、剪枝等多个版本。当前推理只需要与 Diffusers 目录结构匹配的一套权重，其他大文件不需要全部下载，也不能提交到 Git。

### 2.3 文件名和目录必须符合 Diffusers 约定

脚本依赖固定目录，例如 `unet/`、`vae/`、`text_encoder/`、`tokenizer/`。文件名不一致、目录多一层或带空格，都可能导致加载失败。

## 3. Python 和 Diffusers 问题

### 3.1 依赖没有装在当前虚拟环境

运行脚本时必须先激活正确的环境。Windows 项目使用 `.venv`，RKNN 转换使用 WSL 中的 `rknn312`。两个环境用途不同，不能混用。

### 3.2 缺少 `peft`

加载 LCM LoRA 需要 `peft`。缺少它时，基础模型可以加载，但 LoRA 无法融合。

### 3.3 `encode_prompt` 返回值发生变化

不同 Diffusers 版本的 `encode_prompt` 返回值结构可能不同。不能盲目解包，应根据当前版本确认返回值，并取出真正的 prompt embedding。

### 3.4 Tensor 不能直接转 NumPy

带梯度的 Tensor 不能直接调用 `.numpy()`，需要先 `detach()`。导出阶段是推理流程，应使用 `torch.no_grad()`。

### 3.5 脚本中的相对路径容易出错

从不同目录启动脚本时，`../testdata` 等相对路径可能指向错误位置。后来统一以脚本所在目录的上级目录作为项目根目录。

## 4. ONNX 导出问题

### 4.1 UNet 文件过大

UNet 超过 2 GiB，不能只保存成一个普通 ONNX 文件，需要启用 external data。这样会生成一个 ONNX 主文件和一组外部数据文件，必须整体保留，不能只复制主文件。

### 4.2 ONNX 检查方式不正确

带 external data 的模型必须在外部数据文件仍在原目录时检查。移动或只复制 `.onnx` 主文件，会出现模型损坏或找不到权重的问题。

### 4.3 不能只看导出成功

ONNX 导出没有报错，不代表结果正确。必须用同一组输入分别跑 PyTorch 和 ONNX Runtime，再比较输出误差。

本项目结果：

- UNet：最大误差约 `7.4e-6`。
- VAE：最大误差约 `7.3e-5`。

## 5. RKNN 转换问题

### 5.1 Windows 环境不适合直接做 RKNN 转换

RKNN Toolkit2 已安装在 WSL 的 `/home/laiy5/rknn312`。转换前需要：

```bash
source /home/laiy5/rknn312/bin/activate
```

### 5.2 WSL 内存不足

构建 VAE 或 UNet RKNN 时进程曾被系统直接 `Killed`。原因是转换阶段需要较多内存。后来增加了 WSL 内存和 swap，FP 转换才成功。

### 5.3 NumPy/SciPy 警告

RKNN Toolkit2 输出了 NumPy 与 SciPy 版本范围不完全匹配的警告。该警告没有阻止本次构建，但后续应固定一套经过验证的依赖版本。

### 5.4 PC 端不能直接模拟已导出的 RKNN

RKNN PC 模拟器通常要求在同一进程中完成 ONNX 加载和构建，不能简单地把已经保存的 RKNN 文件当作普通 ONNX 模型加载。因此最终验证采用真实 RK3588 板端运行。

## 6. RK3588 板端问题

### 6.1 ONNX 输入格式和 RKNN 实际输入格式不同

VAE 和 UNet 的 RKNN 输入实际是：

- `FLOAT16`。
- `NHWC`。

而 PyTorch/ONNX 常用的是 `FLOAT32`、`NCHW`。如果不转换，程序会在 `rknn_inputs_set` 附近崩溃。

解决方式是：

1. `FLOAT32` 转 `FLOAT16`。
2. 采样输入从 NCHW 转 NHWC。
3. timestep 和文本 embedding 按 RKNN 查询到的属性设置输入。
4. 输出再按实际格式转换回 CPU 侧使用的格式。

### 6.2 Android 系统没有 Python

板端执行：

```bash
adb shell "python3 --version"
```

显示 Python 不存在。Android 镜像没有预装 Python，也没有 `rknnlite`，因此板端不能直接运行 Python 推理脚本，改用 NDK C++ 调用 RKNN Runtime。

### 6.3 WSL 的 adb 看不到 Windows 设备

WSL 中的 Linux `adb` 与 Windows adb 服务不是同一个环境。最终使用 Windows SDK 的 adb：

```bash
/mnt/c/Users/laiy5/AppData/Local/Android/Sdk/platform-tools/adb.exe
```

### 6.4 需要同时准备运行库

板端除了推送 `.rknn` 和可执行文件，还需要：

- `librknnrt.so`。
- NDK 的 `libc++_shared.so`。
- 正确设置 `LD_LIBRARY_PATH`。

缺少 C++ 运行库或动态库路径错误时，程序可能无法启动。

### 6.5 RKNN Runtime 版本要匹配

转换端使用的 RKNN Toolkit2 版本、板端 `librknnrt.so` 版本应尽量匹配。版本差异可能导致模型加载失败或运行结果异常。

### 6.6 `rknn_input` 初始化容易写错

C++ 结构体使用聚合初始化时，字段顺序一旦写错，可能导致类型、格式或内存大小错误。更稳妥的方式是逐字段赋值，并以 `rknn_query` 查询到的属性为准。

## 7. 完整生图流程问题

### 7.1 只有 UNet 和 VAE 还不能生图

完整流程还需要：

1. Tokenizer 将 prompt 转成 token。
2. Text Encoder 将 token 转成文本 embedding。
3. Scheduler 更新 latent。
4. UNet 预测噪声。
5. VAE 将 latent 解码为图片。

### 7.2 LCM Scheduler 不能省略

LCM 不是简单地重复调用 UNet。每一步还要使用 scheduler 的系数和随机噪声。缺少这些参数时，即使 UNet 和 VAE 都能运行，也无法得到正确图片。

### 7.3 Android 图片保存依赖问题

为了减少 Android 端依赖，固定 prompt Demo 先输出 PPM 图片，再在电脑端转换为 PNG。后续如果需要 App 集成，再接入 Android Bitmap 或 PNG 编码流程。

## 8. 当前完成情况

| 阶段 | 状态 |
|---|---|
| PyTorch 基线 | 已完成 |
| ONNX 导出与误差验证 | 已完成 |
| VAE FP RKNN 转换 | 已完成 |
| VAE RK3588 推理 | 已完成 |
| VAE 板端误差验证 | 已完成 |
| UNet RKNN 板端验证 | 已完成 |
| 固定 prompt 板端生图 | 已完成 |
| 任意 prompt 生图 | 待完成 |
| INT8 量化与性能优化 | 待完成 |

## 9. 后续建议

建议按以下顺序继续：

1. 先提交当前阶段的小 commit，并确认不包含大模型文件。
2. 增加 Tokenizer 和 Text Encoder 的 CPU 推理流程。
3. 将任意 prompt 生成的 embedding 传给板端 UNet。
4. 再进行 INT8 量化、速度和内存测试。
5. 最后整理 README、目录结构和 PR 说明。

## 10. 最新测试记录

### 10.1 Scheduler 参数文件

使用 seed `42` 分别生成 4、6、8 步输入，文件尺寸符合预期：

| Steps | timesteps.bin | scheduler_coeffs.bin | noise 文件数 | 结果 |
| --- | ---: | ---: | ---: | --- |
| 4 | 16 B | 96 B | 3 | 通过 |
| 6 | 24 B | 144 B | 5 | 通过 |
| 8 | 32 B | 192 B | 7 | 通过 |

三种配置的 `initial_latents.bin` 都是 65,536 B，对应 `1×4×64×64` 的 FLOAT32 latent。

### 10.2 代码和脚本检查

- Python 脚本编译检查：通过。
- PowerShell 生图脚本语法检查：通过。
- README 脱敏检查：通过，未包含个人用户名、盘符或本机目录。
- Android C++ fixed prompt Demo 编译：通过。

### 10.3 当前未完成的测试

4、6、8 步的 Scheduler 输入已经生成并通过尺寸检查，且已完成板端 4、6、8 步推理。测试中发现旧版可执行文件的耗时标签固定写成 `UNet 4 steps`，但实际循环步数正确；随后已改为动态打印实际步数并重新编译。图片主观质量还需要进一步人工对比，不能只根据运行成功判断质量提升。

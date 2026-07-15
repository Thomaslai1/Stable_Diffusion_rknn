param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Prompt,
    [string]$Output = "results\generated.png",
    [string]$Adb = "C:\Users\laiy5\AppData\Local\Android\Sdk\platform-tools\adb.exe",
    [string]$BoardDir = "/data/local/tmp/stable_diffusion",
    [string]$TextBoardDir = "/data/local/tmp/stable_diffusion_text"
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
$Standalone = "D:\HuaweiMoveData\Users\laiy5\Desktop\stable_diffusion_rknn"
$Python = Join-Path $Standalone ".venv\Scripts\python.exe"
$Tokenize = Join-Path $Repo "python\tokenize_prompt.py"
$InputDir = Join-Path $Repo "testdata\prompt"
$OutputPath = Join-Path $Repo $Output
$TempOutput = Join-Path $Repo "testdata\prompt\board_image.ppm"

if (-not (Test-Path $Adb)) { throw "找不到 adb: $Adb" }
if (-not (Test-Path $Python)) { throw "找不到 Python 虚拟环境: $Python" }
if (-not (Test-Path (Join-Path $Repo "model\text_encoder_fp.rknn"))) { throw "找不到 model\text_encoder_fp.rknn" }
if (-not (Test-Path (Join-Path $Repo "build\text_encoder_rknn_demo"))) { throw "找不到 build\text_encoder_rknn_demo，请先编译 Android Demo" }
if (-not (Test-Path (Join-Path $Repo "build\fixed_prompt_rknn_demo"))) { throw "找不到 build\fixed_prompt_rknn_demo，请先编译 Android Demo" }

New-Item -ItemType Directory -Force $InputDir | Out-Null
New-Item -ItemType Directory -Force (Split-Path $OutputPath) | Out-Null

function Push-IfMissing([string]$LocalPath, [string]$RemotePath) {
    & $Adb shell "test -f $RemotePath"
    if ($LASTEXITCODE -ne 0) {
        & $Adb push $LocalPath $RemotePath
    }
}

& $Python $Tokenize $Prompt --output_dir $InputDir
& $Adb shell "mkdir -p $TextBoardDir"
Push-IfMissing (Join-Path $Repo "model\text_encoder_fp.rknn") "$TextBoardDir/text_encoder_fp.rknn"
& $Adb push (Join-Path $InputDir "input_ids.bin") "$TextBoardDir/input_ids.bin"
& $Adb push (Join-Path $Repo "build\text_encoder_rknn_demo") "$TextBoardDir/text_encoder_rknn_demo"
Push-IfMissing (Join-Path $Standalone "cpp\third_party\librknnrt.so") "$TextBoardDir/librknnrt.so"
Push-IfMissing (Join-Path $Standalone "cpp\third_party\libc++_shared.so") "$TextBoardDir/libc++_shared.so"
& $Adb shell "chmod 755 $TextBoardDir/text_encoder_rknn_demo"
& $Adb shell "LD_LIBRARY_PATH=$TextBoardDir $TextBoardDir/text_encoder_rknn_demo $TextBoardDir/text_encoder_fp.rknn $TextBoardDir/input_ids.bin $TextBoardDir/prompt_embeds.bin"
& $Adb push (Join-Path $Repo "build\fixed_prompt_rknn_demo") "$BoardDir/fixed_prompt_rknn_demo"
& $Adb shell "chmod 755 $BoardDir/fixed_prompt_rknn_demo"
& $Adb shell "LD_LIBRARY_PATH=$BoardDir $BoardDir/fixed_prompt_rknn_demo $BoardDir/unet_fp.rknn $BoardDir/vae_decoder_fp.rknn $BoardDir/fixed_prompt $TextBoardDir/prompt_embeds.bin"
& $Adb pull "$BoardDir/fixed_prompt/board_image.ppm" $TempOutput

& $Python -c "from PIL import Image; Image.open(r'$TempOutput').save(r'$OutputPath')"
Write-Output "图片已生成: $OutputPath"

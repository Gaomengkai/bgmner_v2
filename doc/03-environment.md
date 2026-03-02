# 03. 环境与依赖

## 1. Python 基础依赖

工程包名：`bgmner-bert`（`pyproject.toml`）。

安装方式：

```powershell
mamba activate bgmner
python -m pip install -e .
```

主要依赖：

- `transformers`
- `accelerate`
- `seqeval`
- `onnx` / `onnxruntime` / `onnxscript`
- `fastapi` / `uvicorn`

## 2. Windows + AMD（ROCm Torch）

脚本：

- `scripts/install_torch_rocm_win_amd.ps1`

执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install_torch_rocm_win_amd.ps1
```

特性：

- 仅支持 Windows
- 要求 Python `3.12`
- 检查 AMD/Radeon 显卡
- 安装指定 ROCm 7.2 + torch/torchaudio/torchvision 轮子

可用 `-Force` 先清理旧 torch：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install_torch_rocm_win_amd.ps1 -Force
```

## 3. ONNX Runtime + DirectML 正规来源

不要依赖 `Office` 或 `System32` 自带 DLL，请使用 NuGet 官方包。

- `Microsoft.AI.DirectML`
- `Microsoft.ML.OnnxRuntime`
- `Microsoft.ML.OnnxRuntime.DirectML`

参考：

- `https://www.nuget.org/packages/Microsoft.AI.DirectML/`
- `https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime/`
- `https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.DirectML/`

下载示例：

```powershell
$dmlVer = "1.15.4"
$ortVer = "1.24.0"
$outDir = ".\third_party\nuget"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

Invoke-WebRequest "https://www.nuget.org/api/v2/package/Microsoft.AI.DirectML/$dmlVer" -OutFile "$outDir\Microsoft.AI.DirectML.$dmlVer.nupkg"
Invoke-WebRequest "https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime/$ortVer" -OutFile "$outDir\Microsoft.ML.OnnxRuntime.$ortVer.nupkg"
Invoke-WebRequest "https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime.DirectML/$ortVer" -OutFile "$outDir\Microsoft.ML.OnnxRuntime.DirectML.$ortVer.nupkg"
```

建议保持 `OnnxRuntime` 与 `OnnxRuntime.DirectML` 同版本。

## 4. 环境变量

Rust ONNX 动态加载库路径可通过：

```powershell
$env:ORT_DYLIB_PATH="E:\conda\envs\bgmner\Lib\site-packages\onnxruntime\capi\onnxruntime.dll"
```

说明：

- 空字符串会被视为未设置并忽略
- 文件不存在会报错

## 5. 验证命令

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

## 6. 常见环境误区

- 在错误环境安装依赖：先用 `python -c "import sys; print(sys.executable)"` 确认路径。
- 已经 `mamba activate bgmner` 后，不需要再把每条命令包一层 `mamba run -n bgmner`。


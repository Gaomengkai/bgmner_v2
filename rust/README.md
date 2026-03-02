# bgmner-rs（Rust ONNX 推理子工程）

这个目录是独立 Rust 子工程，只做 ONNX 推理能力：

- Web API（`/health` + `/predict`）
- 批量文件推理（JSONL / 纯文本）

接口协议与 Python `bgmner-api --backend onnx` 对齐。

## 1. 编译

```powershell
cd .\rust
cargo check
cargo test
```

## 2. ONNX Runtime 动态库

本工程使用 `ort(load-dynamic)`，需要可用的 ONNX Runtime 动态库。

- 推荐：先激活你的 `bgmner` 环境再运行（会自动优先探测 `CONDA_PREFIX`）。
- 若仍找不到，请手动设置：

```powershell
$env:ORT_DYLIB_PATH="E:\conda\envs\bgmner\Lib\site-packages\onnxruntime\capi\onnxruntime.dll"
```

### 2.1 官方获取来源（NuGet）

不要依赖 `Office` 或 `System32` 里的 `DirectML.dll`，请从 NuGet 获取可再分发版本：

- `Microsoft.AI.DirectML`（提供 `DirectML.dll`）
- `Microsoft.ML.OnnxRuntime`（CPU ORT）
- `Microsoft.ML.OnnxRuntime.DirectML`（DirectML ORT）

NuGet 页面：

- `https://www.nuget.org/packages/Microsoft.AI.DirectML/`
- `https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime/`
- `https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.DirectML/`

下载示例（PowerShell）：

```powershell
$dmlVer = "1.15.4"
$ortVer = "1.24.0"
$outDir = ".\\third_party\\nuget"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

Invoke-WebRequest "https://www.nuget.org/api/v2/package/Microsoft.AI.DirectML/$dmlVer" -OutFile "$outDir\\Microsoft.AI.DirectML.$dmlVer.nupkg"
Invoke-WebRequest "https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime/$ortVer" -OutFile "$outDir\\Microsoft.ML.OnnxRuntime.$ortVer.nupkg"
Invoke-WebRequest "https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime.DirectML/$ortVer" -OutFile "$outDir\\Microsoft.ML.OnnxRuntime.DirectML.$ortVer.nupkg"
```

运行 `bgmner-rs.exe` 时，建议 `exe` 同目录至少包含：

- `onnxruntime.dll`
- `onnxruntime_providers_shared.dll`
- `DirectML.dll`

并保持 `Microsoft.ML.OnnxRuntime` 与 `Microsoft.ML.OnnxRuntime.DirectML` 同版本。

### 2.2 macOS（Apple Silicon）CoreML 版 ORT（GitHub Release）

macOS 建议直接使用 ONNX Runtime 官方发布包（含 `libonnxruntime.dylib`）。

示例（`v1.24.2`）：

```bash
cd /path/to/bgmner_v2
mkdir -p third_party/ort_1.24.2

curl -fL -o /tmp/onnxruntime-osx-arm64-1.24.2.tgz \
  https://github.com/microsoft/onnxruntime/releases/download/v1.24.2/onnxruntime-osx-arm64-1.24.2.tgz

tar -xzf /tmp/onnxruntime-osx-arm64-1.24.2.tgz \
  -C third_party/ort_1.24.2 \
  --strip-components=1

# 可选：若被 Gatekeeper 标记隔离
xattr -dr com.apple.quarantine third_party/ort_1.24.2

export ORT_DYLIB_PATH="$PWD/third_party/ort_1.24.2/lib/libonnxruntime.dylib"
```

可用 `coreml,cpu` 作为 provider 回退链：

```bash
cd rust
./target/release/bgmner-rs batch \
  --model-dir ../runs/bgm_ner_20ep_mmBERT_small/best_model \
  --onnx-path ../runs/bgm_ner_20ep_mmBERT_small/onnx/model.int8.dynamic.onnx \
  --provider coreml,cpu \
  --text "测试标题"
```

## 3. 启动 Web API

```powershell
cargo run -- serve `
  --model-dir ..\runs\bgm_ner_20ep_xlmr\best_model `
  --onnx-path ..\runs\bgm_ner_20ep_xlmr\onnx\model.onnx `
  --provider dml,cpu `
  --dml-device-id 1 `
  --host 127.0.0.1 `
  --port 8000 `
  --batch-size 32 `
  --max-length 256
```

### 3.1 健康检查

```powershell
curl http://127.0.0.1:8000/health
```

### 3.2 单条预测

```powershell
curl.exe -X POST "http://127.0.0.1:8000/predict" `
  -H "Content-Type: application/json" `
  -d "{\"text\":\"[桜都字幕组] 迷宫饭 [15][1080p]\"}"
```

### 3.3 多条 + items（含 id）

```powershell
curl.exe -X POST "http://127.0.0.1:8000/predict" `
  -H "Content-Type: application/json" `
  -d "{\"texts\":[\"标题A\",\"标题B\"],\"items\":[{\"id\":\"x-1\",\"text\":\"标题C\"}],\"batch_size\":16,\"max_length\":256}"
```

## 4. 批处理推理

### 4.1 直接给文本

```powershell
cargo run -- batch `
  --model-dir ..\runs\bgm_ner_20ep_xlmr\best_model `
  --onnx-path ..\runs\bgm_ner_20ep_xlmr\onnx\model.onnx `
  --provider dml,cpu `
  --dml-device-id 1 `
  --text "[桜都字幕组] 迷宫饭 [15][1080p]"
```

### 4.2 文件输入输出

输入支持两类行格式：

- 纯文本行
- JSON 行（含 `text` 字段；`text` 可为字符串或字符数组，`id` 会透传到输出）

```powershell
cargo run -- batch `
  --model-dir ..\runs\bgm_ner_20ep_xlmr\best_model `
  --onnx-path ..\runs\bgm_ner_20ep_xlmr\onnx\model.onnx `
  --provider dml,cpu `
  --dml-device-id 1 `
  --input-file ..\data\ner_data\dev.txt `
  --output-file .\predictions.jsonl `
  --batch-size 32 `
  --max-length 256
```

## 5. Provider 参数说明

- 支持 `--provider auto`（默认）
- 支持别名：`cpu`、`coreml`、`dml`、`cuda`、`rocm`
- 支持官方名：`CPUExecutionProvider`、`CoreMLExecutionProvider`、`DmlExecutionProvider`、`CUDAExecutionProvider`、`ROCMExecutionProvider`
- 支持链式回退（按顺序）：例如 `--provider dml,cpu`
- `--dml-device-id` 可指定 DirectML 的设备索引（默认 `0`）。多 GPU/虚拟显示设备场景下建议尝试 `1`、`2`。

### 5.1 DirectML 常见报错（0x887A0004）

如果你看到 `DmlExecutionProvider` 可用，但创建会话时报 `0x887A0004`，通常是本机 `DirectML.dll` 版本过旧或设备选择不合适。

建议：

1. 把较新的 `DirectML.dll` 放到 `bgmner-rs.exe` 同目录（本工程会优先预加载该文件）。
2. 使用 `--dml-device-id` 切换设备索引。
3. 使用 `--provider dml,cpu` 作为安全回退链。

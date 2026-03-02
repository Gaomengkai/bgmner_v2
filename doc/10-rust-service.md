# 10. Rust ONNX 服务说明

## 1. 工程位置

- 子工程目录：`rust/`
- 可执行名：`bgmner-rs`

## 2. 编译

```powershell
cd .\rust
cargo check
cargo test
cargo build
cargo build --release
```

## 3. 命令概览

```text
bgmner-rs serve ...
bgmner-rs batch ...
```

### 3.1 `serve`

启动 Web 服务（`/health`, `/predict`）。

示例：

```powershell
.\target\release\bgmner-rs.exe serve `
  --model-dir ..\runs\bgm_ner_20ep_mmBERT_small\best_model `
  --onnx-path ..\runs\bgm_ner_20ep_mmBERT_small\onnx\model.int8.dynamic.onnx `
  --provider dml,cpu `
  --dml-device-id 0 `
  --host 127.0.0.1 `
  --port 8000 `
  --batch-size 32 `
  --max-length 256
```

### 3.2 `batch`

离线批处理推理（文本或文件）。

示例：

```powershell
.\target\release\bgmner-rs.exe batch `
  --model-dir ..\runs\bgm_ner_20ep_mmBERT_small\best_model `
  --onnx-path ..\runs\bgm_ner_20ep_mmBERT_small\onnx\model.int8.dynamic.onnx `
  --provider cpu `
  --input-file ..\data\ner_data\dev.txt `
  --output-file .\predictions.jsonl `
  --batch-size 32 `
  --max-length 256
```

## 4. provider 支持

支持别名和官方名：

- `cpu` / `CPUExecutionProvider`
- `coreml` / `CoreMLExecutionProvider`
- `dml` / `DmlExecutionProvider`
- `cuda` / `CUDAExecutionProvider`
- `rocm` / `ROCMExecutionProvider`

支持回退链：`--provider dml,cpu`。

## 5. DLL 加载规则（Windows）

ONNX Runtime DLL 搜索优先级：

1. `exe` 同目录（最高优先）
2. `CONDA_PREFIX` / `VIRTUAL_ENV` 推导路径
3. `PATH` 回退

说明：

- 不使用当前工作目录（PWD）作为优先搜索路径
- `ORT_DYLIB_PATH` 空值会被忽略

### 5.1 macOS CoreML dylib（GitHub Release）

macOS（Apple Silicon）可直接使用 ONNX Runtime 官方发布包：

- `https://github.com/microsoft/onnxruntime/releases/download/v1.24.2/onnxruntime-osx-arm64-1.24.2.tgz`

示例：

```bash
cd /path/to/bgmner_v2
mkdir -p third_party/ort_1.24.2

curl -fL -o /tmp/onnxruntime-osx-arm64-1.24.2.tgz \
  https://github.com/microsoft/onnxruntime/releases/download/v1.24.2/onnxruntime-osx-arm64-1.24.2.tgz

tar -xzf /tmp/onnxruntime-osx-arm64-1.24.2.tgz \
  -C third_party/ort_1.24.2 \
  --strip-components=1

# 可选：去掉 macOS 隔离标记
xattr -dr com.apple.quarantine third_party/ort_1.24.2

export ORT_DYLIB_PATH="$PWD/third_party/ort_1.24.2/lib/libonnxruntime.dylib"
```

运行时建议使用回退链：

```bash
cd rust
./target/release/bgmner-rs batch ... --provider coreml,cpu
```

## 6. DirectML 运行时

建议把以下 DLL 放在 `bgmner-rs.exe` 同目录：

- `onnxruntime.dll`
- `onnxruntime_providers_shared.dll`
- `DirectML.dll`

`DirectML.dll` 请从 NuGet 的 `Microsoft.AI.DirectML` 获取。

## 7. 常见 DML 报错

错误码 `0x887A0004` 常见原因：

- 运行时加载到过旧 `DirectML.dll`
- 当前 `dml-device-id` 不适配

处理：

1. 更新并固定 `exe` 同目录 `DirectML.dll`
2. 切换 `--dml-device-id`
3. 使用 `--provider dml,cpu` 保底

## 8. 性能剖析开关

开启分阶段统计：

```powershell
$env:BGMNER_PROFILE_STAGES="1"
```

输出阶段：

- `encode`
- `tensor`
- `infer`
- `argmax`
- `decode`

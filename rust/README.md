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

## 3. 启动 Web API

```powershell
cargo run -- serve `
  --model-dir ..\runs\bgm_ner_20ep_xlmr\best_model `
  --onnx-path ..\runs\bgm_ner_20ep_xlmr\onnx\model.onnx `
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
  --input-file ..\data\ner_data\dev.txt `
  --output-file .\predictions.jsonl `
  --batch-size 32 `
  --max-length 256
```

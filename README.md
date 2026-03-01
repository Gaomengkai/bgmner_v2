# bgmner-bert（中文说明）

当前目录是独立 NER 工程，已整理为以下核心能力：

- 训练（支持命令行和外部 JSON 超参数）
- 推理（HF / ONNX）
- 基准测试（HF / ONNX）
- 导出 ONNX
- INT8 量化
- ONNX 评估
- Web API（支持多条输入）

---

## 1. 环境准备（Windows + AMD）

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install_torch_rocm_win_amd.ps1
python -m pip install -e .
```

---

## 2. 数据集纳入工程（复制 + 排序）

将上级目录原始数据复制进当前工程并按 `id` 排序（`train/dev`）：

```powershell
python .\scripts\sync_dataset.py `
  --src-dir ..\data\bgm\ner_data `
  --dst-dir .\data\ner_data
```

生成后数据路径：

- `data/ner_data/train.txt`
- `data/ner_data/dev.txt`
- `data/ner_data/labels.txt`

---

## 3. 底模统一策略（取消 model cache）

本工程已取消 `model cache` 概念：

- 训练时如果 `--model-name` 是 HF 模型名，会自动下载到 `backbones/<模型名>` 后再训练。
- 如果 `--model-name` 是本地目录，则直接使用。

手动下载（可选）：

```powershell
bgmner-download-backbone `
  --model-name FacebookAI/xlm-roberta-base `
  --save-dir backbones\xlm-roberta-base
```

---

## 4. 训练（支持外部 JSON 超参数）

### 4.1 命令行方式

```powershell
bgmner-train `
  --dataset-dir data\ner_data `
  --model-name backbones\xlm-roberta-base `
  --output-root runs `
  --run-name bgm_bert_base
```

### 4.2 JSON 配置方式

工程内已提供完整模板：`configs/train.default.json`。

可先复制成你自己的配置文件再修改：

```powershell
Copy-Item .\configs\train.default.json .\train_config.json
```

示例 `train_config.json`：

```json
{
  "dataset_dir": "data/ner_data",
  "output_root": "runs",
  "run_name": "bgm_bert_json_cfg",
  "model_name": "FacebookAI/xlm-roberta-base",
  "backbones_dir": "backbones",
  "max_length": 256,
  "seed": 42,
  "num_train_epochs": 5.0,
  "per_device_train_batch_size": 16,
  "per_device_eval_batch_size": 32,
  "gradient_accumulation_steps": 1,
  "learning_rate": 2e-05,
  "weight_decay": 0.01,
  "warmup_ratio": 0.1,
  "logging_steps": 50,
  "save_total_limit": 2,
  "dataloader_num_workers": 0,
  "early_stopping_patience": 2,
  "report_to": "none"
}
```

运行：

```powershell
bgmner-train --config-file .\train_config.json
```

说明：

- `--config-file` 加载后，命令行参数仍可覆盖 JSON 值。
- JSON 字段若写错名称会直接报错，避免“静默忽略”。

---

## 5. 推理

### 5.1 HF 推理

```powershell
bgmner-predict `
  --model-dir runs\bgm_bert_base\best_model `
  --text "[桜都字幕组] 摇曳露营△ 第三季 [03][1080p][简繁内封]"
```

批量文件：

```powershell
bgmner-predict `
  --model-dir runs\bgm_bert_base\best_model `
  --input-file ..\test\1000.txt `
  --output-file runs\bgm_bert_base\predictions\test1000_predictions.jsonl
```

### 5.2 ONNX 推理

```powershell
bgmner-onnx-predict `
  --onnx-path runs\bgm_bert_base\onnx\model.onnx `
  --model-dir runs\bgm_bert_base\best_model `
  --text "[LoliHouse] 迷宫饭 / Dungeon Meshi [15][1080p][繁体内嵌]"
```

---

## 6. 基准测试（HF / ONNX）

说明：

- 支持 `--text` 或 `--input-file`（可直接读取 `data/ner_data/dev.txt` 这类 JSONL）。
- 输出包含吞吐和延迟统计：`throughput_texts_per_sec`、`batch_latency_ms_p50/p95/p99`。

### 6.1 HF 基准

```powershell
bgmner-benchmark `
  --backend hf `
  --model-dir runs\bgm_bert_base\best_model `
  --input-file data\ner_data\dev.txt `
  --batch-size 32 `
  --max-length 256 `
  --warmup-runs 3 `
  --benchmark-runs 20 `
  --max-samples 300
```

### 6.2 ONNX 基准

```powershell
bgmner-benchmark `
  --backend onnx `
  --model-dir runs\bgm_bert_base\best_model `
  --onnx-path runs\bgm_bert_base\onnx\model.int8.dynamic.onnx `
  --provider CPUExecutionProvider `
  --input-file data\ner_data\dev.txt `
  --batch-size 32 `
  --max-length 256 `
  --warmup-runs 3 `
  --benchmark-runs 20 `
  --output-json runs\bgm_bert_base\metrics\benchmark_onnx.json
```

---

## 7. 导出 ONNX

```powershell
bgmner-export-onnx `
  --model-dir runs\bgm_bert_base\best_model `
  --output-path runs\bgm_bert_base\onnx\model.onnx
```

---

## 8. 量化与评估

### 7.1 INT8 量化（默认线性层）

```powershell
bgmner-quantize-int8 `
  --input-onnx runs\bgm_bert_base\onnx\model.onnx `
  --output-onnx runs\bgm_bert_base\onnx\model.int8.dynamic.onnx `
  --op-types "MatMul,Gemm"
```

### 7.2 含 Embedding 量化（可选）

```powershell
bgmner-quantize-int8 `
  --input-onnx runs\bgm_bert_base\onnx\model.onnx `
  --output-onnx runs\bgm_bert_base\onnx\model.int8.dynamic.with_gather.onnx `
  --op-types "MatMul,Gemm,Gather,EmbedLayerNormalization" `
  --weight-type qint8 `
  --per-channel
```

### 7.3 ONNX 评估

```powershell
bgmner-eval-onnx `
  --onnx-path runs\bgm_bert_base\onnx\model.int8.dynamic.onnx `
  --model-dir runs\bgm_bert_base\best_model `
  --dataset-file data\ner_data\dev.txt `
  --provider CPUExecutionProvider
```

---

## 9. Web API（支持多条输入）

启动 HF 后端：

```powershell
bgmner-api `
  --backend hf `
  --model-dir runs\bgm_bert_base\best_model `
  --host 0.0.0.0 `
  --port 8000
```

启动 ONNX 后端：

```powershell
bgmner-api `
  --backend onnx `
  --model-dir runs\bgm_bert_base\best_model `
  --onnx-path runs\bgm_bert_base\onnx\model.int8.dynamic.onnx `
  --provider CPUExecutionProvider `
  --host 0.0.0.0 `
  --port 8000
```

### 8.1 请求格式（支持三种输入）

- `text`：单条字符串
- `texts`：字符串数组
- `items`：对象数组（支持附带 `id`）

示例：

```json
{
  "texts": ["标题A", "标题B"],
  "items": [{"id": "x-1", "text": "标题C"}],
  "batch_size": 32,
  "max_length": 256
}
```

健康检查：

```powershell
curl http://127.0.0.1:8000/health
```

预测：

```powershell
curl -X POST http://127.0.0.1:8000/predict `
  -H "Content-Type: application/json" `
  -d "{\"text\":\"[桜都字幕组] 迷宫饭 [15][1080p]\"}"
```

---

## 10. 测试

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

---

## 11. 目录结构

```text
./
  data/ner_data/
    train.txt
    dev.txt
    labels.txt
  backbones/
  runs/
  scripts/
    install_torch_rocm_win_amd.ps1
    sync_dataset.py
  configs/
    train.default.json
  src/bgmner_bert/
    api.py
    download_backbone.py
    train.py
    predict.py
    benchmark.py
    export_onnx.py
    onnx_predict.py
    quantize_int8.py
    eval_onnx.py
    data.py
    metrics.py
    bio_decode.py
    inference_utils.py
    config.py
  tests/
    test_api.py
    test_bio_decode.py
    test_data_alignment.py
    test_benchmark.py
    test_quantize_int8.py
    test_train_config_json.py
    test_onnx_parity.py
  pyproject.toml
  README.md
```

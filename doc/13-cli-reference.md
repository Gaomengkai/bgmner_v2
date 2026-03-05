# 13. CLI 参数参考

本页按当前源码参数定义整理（`src/bgmner_bert/*.py`, `rust/src/main.rs`, `scripts/*.py|ps1`）。

## 1. `bgmner-download-backbone`

用途：下载底模与 tokenizer 到本地目录。

参数：

- `--model-name`：HF 模型名或本地路径。默认 `FacebookAI/xlm-roberta-base`
- `--save-dir`：目标目录。默认 `backbones/<sanitized_model_name>`

## 2. `bgmner-train`

用途：训练 NER 模型。

参数：

- `--config-file`：外部 JSON 配置文件
- `--dataset-dir`：默认 `data/ner_data`
- `--output-root`：默认 `runs`
- `--run-name`：默认自动生成 `run_YYYYMMDD_HHMMSS`
- `--model-name`：默认 `FacebookAI/xlm-roberta-base`
- `--backbones-dir`：默认 `backbones`
- `--max-length`：默认 `256`
- `--seed`：默认 `42`
- `--num-train-epochs`：默认 `5.0`
- `--per-device-train-batch-size`：默认 `16`
- `--per-device-eval-batch-size`：默认 `32`
- `--gradient-accumulation-steps`：默认 `1`
- `--learning-rate`：默认 `2e-5`
- `--weight-decay`：默认 `0.01`
- `--warmup-ratio`：默认 `0.1`
- `--logging-steps`：默认 `50`
- `--save-total-limit`：默认 `2`
- `--dataloader-num-workers`：默认 `0`
- `--early-stopping-patience`：默认 `2`
- `--report-to`：默认 `none`

## 3. `bgmner-predict`

用途：HF 推理。

参数：

- `--model-dir`：必填，`best_model` 路径
- `--text`：单条文本
- `--input-file`：输入文件（每行一条）
- `--output-file`：输出 JSONL
- `--batch-size`：默认 `32`
- `--max-length`：默认 `256`
- `--device`：`auto|cpu|cuda|mps`，默认 `auto`（优先级 `cuda -> mps -> cpu`）

## 4. `bgmner-onnx-predict`

用途：ONNX 推理。

参数：

- `--onnx-path`：必填，ONNX 文件
- `--model-dir`：必填，`best_model` 路径（用于 tokenizer/config）
- `--text`
- `--input-file`
- `--output-file`
- `--batch-size`：默认 `32`
- `--max-length`：默认 `256`
- `--provider`：默认 `auto`（macOS 默认 `cpu`）

provider 支持：

- 别名：`cpu|coreml|dml|cuda|rocm`
- 也支持官方 Provider 名称
- 支持链式回退，如 `dml,cpu`

## 5. `bgmner-export-onnx`

用途：从 HF 导出 ONNX。

参数：

- `--model-dir`：必填
- `--output-path`：默认自动推导到 `../onnx/model.onnx`
- `--opset`：默认 `17`
- `--max-length`：默认 `32`
- `--optimize / --no-optimize`：默认 `False`
- `--optimize-level`：`basic|extended|all`，默认 `all`
- `--optimize-output-path`：默认 `<output>.opt.onnx`

## 6. `bgmner-quantize-int8`

用途：ONNX 动态 INT8 量化。

参数：

- `--input-onnx`：必填
- `--output-onnx`：默认 `<input>.int8.dynamic.onnx`
- `--weight-type`：`qint8|quint8`，默认 `qint8`
- `--per-channel / --no-per-channel`：默认 `True`
- `--reduce-range / --no-reduce-range`：默认 `False`
- `--op-types`：逗号分隔，默认 `Gather,EmbedLayerNormalization`
- `--preprocess / --no-preprocess`：默认 `True`
- `--preprocess-skip-optimization / --no-preprocess-skip-optimization`：默认 `False`
- `--preprocess-skip-onnx-shape / --no-preprocess-skip-onnx-shape`：默认 `False`
- `--preprocess-skip-symbolic-shape / --no-preprocess-skip-symbolic-shape`：默认 `True`
- `--meta-output`：默认 `<output_dir>/quantize_meta.json`

## 7. `bgmner-eval-onnx`

用途：在数据集上评估 ONNX。

参数：

- `--onnx-path`：必填
- `--model-dir`：必填
- `--dataset-file`：默认 `data/ner_data/dev.txt`
- `--provider`：默认 `auto`（macOS 默认 `cpu`）
- `--batch-size`：默认 `32`
- `--max-length`：默认 `256`
- `--max-samples`：默认 `0`（不截断）
- `--output-json`
- `--output-report`

## 8. `bgmner-benchmark`

用途：离线 HF/ONNX 推理基准。

参数：

- `--backend`：`hf|onnx`，默认 `hf`
- `--model-dir`：必填
- `--onnx-path`：当 `backend=onnx` 必填
- `--provider`：默认 `auto`
- `--device`：`auto|cpu|cuda|mps`，默认 `auto`（优先级 `cuda -> mps -> cpu`）
- `--text`
- `--input-file`
- `--max-samples`：默认 `0`
- `--batch-size`：默认 `32`
- `--max-length`：默认 `256`
- `--warmup-runs`：默认 `3`
- `--benchmark-runs`：默认 `20`
- `--output-json`

## 9. `bgmner-api`

用途：启动 Python Web API。

参数：

- `--backend`：`hf|onnx`，默认 `hf`
- `--model-dir`：必填
- `--onnx-path`：当 `backend=onnx` 必填
- `--device`：`auto|cpu|cuda|mps`，默认 `auto`（优先级 `cuda -> mps -> cpu`）
- `--provider`：默认 `auto`
- `--batch-size`：默认 `32`
- `--max-length`：默认 `256`
- `--host`：默认 `0.0.0.0`
- `--port`：默认 `8000`
- `--log-level`：默认 `info`

## 10. `scripts/benchmark_api_batch.py`

用途：对 `/predict` 兼容 API 做 HTTP 基准。

参数：

- `--url`：必填，完整 URL
- `--text`：可重复
- `--input-file`
- `--max-samples`：默认 `0`
- `--batch-size`：默认 `32`
- `--max-length`：默认 `256`
- `--payload-mode`：`texts|items`，默认 `texts`
- `--warmup-runs`：默认 `3`
- `--benchmark-runs`：默认 `20`
- `--timeout-sec`：默认 `30.0`
- `--header`：可重复，格式 `Key:Value`
- `--continue-on-error`
- `--output-json`

## 11. `scripts/sync_dataset.py`

用途：复制并排序数据集。

参数：

- `--src-dir`：默认 `../data/bgm/ner_data`
- `--dst-dir`：默认 `data/ner_data`

## 12. `scripts/install_torch_rocm_win_amd.ps1`

用途：Windows AMD 安装 ROCm Torch。

参数：

- `-Force`：先卸载现有 torch/rocm 包

## 13. `bgmner-rs`（Rust）

### 13.1 `bgmner-rs serve`

参数：

- `--model-dir`：必填
- `--onnx-path`：必填
- `--host`：默认 `0.0.0.0`
- `--port`：默认 `8000`
- `--batch-size`：默认 `32`
- `--max-length`：默认 `256`
- `--provider`：默认 `auto`
- `--dml-device-id`：默认 `0`
- `--log-level`：默认 `info`

### 13.2 `bgmner-rs batch`

参数：

- `--model-dir`：必填
- `--onnx-path`：必填
- `--text`：可重复
- `--input-file`
- `--output-file`
- `--batch-size`：默认 `32`
- `--max-length`：默认 `256`
- `--provider`：默认 `auto`
- `--dml-device-id`：默认 `0`
- `--log-level`：默认 `info`

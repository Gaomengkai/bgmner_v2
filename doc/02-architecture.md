# 02. 系统架构

## 1. 目标

本工程是一个面向字幕标题场景的多语种 NER 系统，支持：

- 基于 `AutoModelForTokenClassification` 的训练
- HF 推理
- ONNX 推理
- ONNX 动态量化与评估
- Python Web API
- Rust ONNX Web API / Batch

## 2. 核心模块（Python）

源码目录：`src/bgmner_bert/`

- `train.py`: 训练主流程
- `config.py`: 训练参数（支持 JSON 配置）
- `data.py`: 数据加载、BIO 映射、token 对齐
- `metrics.py`: seqeval 指标计算
- `predict.py`: HF 推理
- `onnx_predict.py`: ONNX 推理
- `export_onnx.py`: 导出与可选图优化
- `quantize_int8.py`: 动态 INT8 量化
- `eval_onnx.py`: ONNX 离线评估
- `api.py`: FastAPI 服务
- `benchmark.py`: HF/ONNX 本地推理基准
- `onnx_runtime.py`: provider 别名与会话构建

## 3. 模型结构

训练模型来自 `transformers` 的 `AutoModelForTokenClassification`，通常为：

1. Backbone 编码器（例如 XLM-R、MiniLM）
2. Token classification head（线性层）

预测结果在 token 级别得到标签后，再映射回字级标签并解码 BIO 实体区间。

## 4. 数据流

1. 数据集（JSONL，字级 `text` + `labels`）读入
2. 构建 BIO 标签映射（`O/B-*/I-*`）
3. fast tokenizer 编码，基于 `word_ids()` 对齐标签
4. 训练/评估输出 token logits
5. `argmax` 得到标签 ID
6. 按 word 首 token 回填字级标签
7. BIO 解码实体

## 5. 产物结构

训练 `run_name=xxx` 的典型目录：

- `runs/xxx/best_model`: HF 可加载模型目录（推理必需）
- `runs/xxx/onnx`: ONNX 文件与导出元数据
- `runs/xxx/metrics`: 指标与分类报告
- `runs/xxx/predictions`: 开发集预测明细
- `runs/xxx/meta`: 训练参数、标签映射、trainer state

## 6. Rust 子系统

源码目录：`rust/src/`

- `main.rs`: `serve`/`batch` CLI
- `engine.rs`: tokenizer + ONNX session + 推理核心
- `provider.rs`: provider 解析（别名/官方名）
- `api.rs`: Axum API，兼容 Python API 协议
- `batch.rs`: JSONL/文本批量输入输出

Rust 路径仅做 ONNX 推理，不参与训练。


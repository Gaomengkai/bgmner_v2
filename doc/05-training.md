# 05. 训练与超参数

## 1. 训练入口

命令：

```powershell
bgmner-train --help
```

实现入口：

- `src/bgmner_bert/train.py`
- `src/bgmner_bert/config.py`

## 2. 底模来源策略（统一到 `backbones/`）

`--model-name` 支持两类值：

1. 本地目录：直接使用
2. HF 模型名：自动下载到 `--backbones-dir`（默认 `backbones/`）后使用

这意味着工程已取消分散的 model cache 概念。

## 3. 超参数配置方式

支持两种方式：

1. 直接 CLI 传参
2. `--config-file <json>` 外部配置

优先级：

- CLI 参数覆盖 JSON
- JSON 覆盖 dataclass 默认值

如果 JSON 中出现未知字段，会直接报错（不会静默忽略）。

## 4. 默认超参数（`TrainConfig`）

- `dataset_dir`: `data/ner_data`
- `output_root`: `runs`
- `run_name`: 自动时间戳
- `model_name`: `FacebookAI/xlm-roberta-base`
- `backbones_dir`: `backbones`
- `max_length`: `256`
- `seed`: `42`
- `num_train_epochs`: `5.0`
- `per_device_train_batch_size`: `16`
- `per_device_eval_batch_size`: `32`
- `gradient_accumulation_steps`: `1`
- `learning_rate`: `2e-5`
- `weight_decay`: `0.01`
- `warmup_ratio`: `0.1`
- `logging_steps`: `50`
- `save_total_limit`: `2`
- `dataloader_num_workers`: `0`
- `early_stopping_patience`: `2`
- `report_to`: `none`

## 5. 配置文件示例

文件：`configs/train.default.json` 可直接作为模板。

例如 20 epoch：

```json
{
  "dataset_dir": "data/ner_data",
  "output_root": "runs",
  "run_name": "bgm_ner_20ep_xlmr",
  "model_name": "backbones/xlm-roberta-base",
  "backbones_dir": "backbones",
  "max_length": 256,
  "num_train_epochs": 20,
  "per_device_train_batch_size": 16,
  "per_device_eval_batch_size": 32,
  "learning_rate": 2e-5,
  "weight_decay": 0.01,
  "warmup_ratio": 0.1,
  "early_stopping_patience": 0,
  "report_to": "none"
}
```

执行：

```powershell
bgmner-train --config-file .\configs\train.20ep.json
```

## 6. 常用训练命令

### 6.1 XLM-R

```powershell
bgmner-train `
  --dataset-dir data\ner_data `
  --model-name backbones\xlm-roberta-base `
  --output-root runs `
  --run-name bgm_ner_20ep_xlmr `
  --num-train-epochs 20
```

### 6.2 MiniLM

```powershell
bgmner-train `
  --dataset-dir data\ner_data `
  --model-name backbones\microsoft_Multilingual-MiniLM-L12-H384 `
  --output-root runs `
  --run-name bgm_ner_20ep_mmBERT_small `
  --num-train-epochs 20
```

## 7. 训练产物

`runs/<run_name>/` 下的关键文件：

- `best_model/`: 推理与导出的主入口
- `metrics/eval_metrics.json`: 验证集指标
- `metrics/train_metrics.json`: 训练过程指标
- `metrics/classification_report.txt`: seqeval 报告
- `predictions/dev_predictions.jsonl`: 开发集预测结果
- `meta/train_args.json`: 实际训练参数
- `meta/label_mappings.json`: 标签映射
- `meta/trainer_state.json`: Trainer 状态

## 8. 中断行为（Ctrl+C）

当前实现中 `best_model` 的最终保存发生在训练完成后。  
如果中途 `Ctrl+C`，可能只留下 `trainer_output` 的中间 checkpoint，而没有完整 `best_model` 目录。

建议：

- 重要训练不要中断
- 若必须中断，优先检查 `trainer_output` 中是否已有可恢复 checkpoint


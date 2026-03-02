# 06. 推理指南（HF / ONNX）

## 1. HF 推理

命令：

```powershell
bgmner-predict `
  --model-dir runs\bgm_ner_20ep_xlmr\best_model `
  --text "[桜都字幕组] 迷宫饭 [15][1080p]"
```

批量文件：

```powershell
bgmner-predict `
  --model-dir runs\bgm_ner_20ep_xlmr\best_model `
  --input-file data\ner_data\dev.txt `
  --batch-size 32 `
  --max-length 256 `
  --output-file runs\bgm_ner_20ep_xlmr\predictions\dev_hf.jsonl
```

`--device` 可选：`auto` / `cpu` / `cuda`。

## 2. ONNX 推理

命令：

```powershell
bgmner-onnx-predict `
  --onnx-path runs\bgm_ner_20ep_xlmr\onnx\model.int8.dynamic.onnx `
  --model-dir runs\bgm_ner_20ep_xlmr\best_model `
  --provider cpu `
  --text "[桜都字幕组] 迷宫饭 [15][1080p]"
```

批量文件：

```powershell
bgmner-onnx-predict `
  --onnx-path runs\bgm_ner_20ep_xlmr\onnx\model.int8.dynamic.onnx `
  --model-dir runs\bgm_ner_20ep_xlmr\best_model `
  --provider cpu `
  --input-file data\ner_data\dev.txt `
  --batch-size 32 `
  --max-length 256 `
  --output-file runs\bgm_ner_20ep_xlmr\predictions\dev_onnx.jsonl
```

## 3. ONNX 推理时 `model-dir` 的最低要求

ONNX 路径只提供图与权重，`model-dir` 仍用于：

- tokenizer
- `config.json` 中的 `id2label/label2id`

因此即使 `model-dir` 没有 `model.safetensors`，ONNX 推理仍可运行；但至少需要 tokenizer 和 config 相关文件。

## 4. 输入格式

HF/ONNX CLI 的 `--input-file` 默认逐行读取。  
对于基准工具和 Rust batch，额外支持 JSONL 行（带 `text` 字段）。

## 5. 输出字段

每条推理结果包含：

- `text`: 原始输入
- `truncated_text`: 按 `max_length` 截断后的文本
- `entities`: 实体区间映射
- `pred_labels`: 字级预测标签序列

## 6. provider 说明（ONNX）

支持别名：

- `cpu`
- `coreml`
- `dml`
- `cuda`
- `rocm`

也支持官方名称（如 `CPUExecutionProvider`）。  
可链式设置回退，例如：`--provider dml,cpu`。


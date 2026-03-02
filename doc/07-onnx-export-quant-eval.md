# 07. ONNX 导出、量化、评估

## 1. 导出 ONNX

命令：

```powershell
bgmner-export-onnx `
  --model-dir runs\bgm_ner_20ep_xlmr\best_model `
  --output-path runs\bgm_ner_20ep_xlmr\onnx\model.onnx `
  --opset 17 `
  --max-length 32
```

可选优化：

```powershell
bgmner-export-onnx `
  --model-dir runs\bgm_ner_20ep_xlmr\best_model `
  --output-path runs\bgm_ner_20ep_xlmr\onnx\model.onnx `
  --optimize `
  --optimize-level all
```

说明：

- `--optimize-level`: `basic` / `extended` / `all`
- 优化后文件默认为 `<output>.opt.onnx`

## 2. INT8 动态量化

推荐（保真优先）：

```powershell
bgmner-quantize-int8 `
  --input-onnx runs\bgm_ner_20ep_xlmr\onnx\model.onnx `
  --output-onnx runs\bgm_ner_20ep_xlmr\onnx\model.int8.dynamic.onnx `
  --op-types "Gather,EmbedLayerNormalization"
```

说明：

- 默认开启 ONNX 预处理（`quant_pre_process`）
- 默认 `weight-type=qint8`
- 默认 `per-channel=true`

预处理相关开关：

- `--preprocess / --no-preprocess`
- `--preprocess-skip-optimization`
- `--preprocess-skip-onnx-shape`
- `--preprocess-skip-symbolic-shape`

## 3. 激进量化（需谨慎）

```powershell
bgmner-quantize-int8 `
  --input-onnx runs\bgm_ner_20ep_xlmr\onnx\model.onnx `
  --output-onnx runs\bgm_ner_20ep_xlmr\onnx\model.int8.aggressive.onnx `
  --op-types "MatMul,Gemm" `
  --weight-type qint8 `
  --per-channel
```

实践结论：

- `MatMul/Gemm` 量化可能带来明显 F1 劣化
- `Gather,EmbedLayerNormalization` 更容易兼顾效果与体积

## 4. ONNX 评估

```powershell
bgmner-eval-onnx `
  --onnx-path runs\bgm_ner_20ep_xlmr\onnx\model.int8.dynamic.onnx `
  --model-dir runs\bgm_ner_20ep_xlmr\best_model `
  --dataset-file data\ner_data\dev.txt `
  --provider cpu `
  --batch-size 32 `
  --max-length 256 `
  --output-json runs\bgm_ner_20ep_xlmr\metrics\eval_onnx.json `
  --output-report runs\bgm_ner_20ep_xlmr\metrics\eval_onnx_report.txt
```

指标包括：

- `precision`
- `recall`
- `f1`
- `accuracy`
- `runtime_sec`
- `samples_per_sec`

## 5. 推荐验证流程

1. 先评估 HF `best_model` 作为基线
2. 导出 ONNX（可选优化）
3. 评估 FP32 ONNX
4. 量化 INT8
5. 再评估 INT8 ONNX，比较 F1 与吞吐


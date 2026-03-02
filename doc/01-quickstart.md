# 01. 快速开始

本页给出从零到可用的一条最短路径（Windows + `bgmner` 环境）。

## 1. 进入工程

```powershell
cd D:\Code\1\py\bgmner\BERT-BILSTM-CRF-main\experimental
```

## 2. 激活环境并安装工程

```powershell
mamba activate bgmner
python -m pip install -e .
```

说明：

- 如果当前终端已经在 `bgmner` 环境，可直接运行命令，不必再套 `mamba run -n bgmner ...`。

## 3. 同步数据集到工程内

```powershell
python .\scripts\sync_dataset.py `
  --src-dir ..\data\bgm\ner_data `
  --dst-dir .\data\ner_data
```

## 4. 下载底模到 `backbones/`

```powershell
bgmner-download-backbone `
  --model-name FacebookAI/xlm-roberta-base `
  --save-dir backbones\xlm-roberta-base
```

## 5. 训练（20 epoch 示例）

```powershell
bgmner-train `
  --dataset-dir data\ner_data `
  --model-name backbones\xlm-roberta-base `
  --output-root runs `
  --run-name bgm_ner_20ep_xlmr `
  --num-train-epochs 20
```

训练完成后，核心产物位于：

- `runs\bgm_ner_20ep_xlmr\best_model`

## 6. ONNX 导出与量化

```powershell
bgmner-export-onnx `
  --model-dir runs\bgm_ner_20ep_xlmr\best_model `
  --output-path runs\bgm_ner_20ep_xlmr\onnx\model.onnx `
  --optimize --optimize-level all
```

```powershell
bgmner-quantize-int8 `
  --input-onnx runs\bgm_ner_20ep_xlmr\onnx\model.onnx `
  --output-onnx runs\bgm_ner_20ep_xlmr\onnx\model.int8.dynamic.onnx `
  --op-types "Gather,EmbedLayerNormalization"
```

## 7. 评估 ONNX

```powershell
bgmner-eval-onnx `
  --onnx-path runs\bgm_ner_20ep_xlmr\onnx\model.int8.dynamic.onnx `
  --model-dir runs\bgm_ner_20ep_xlmr\best_model `
  --dataset-file data\ner_data\dev.txt `
  --provider cpu
```

## 8. 启动 API（ONNX）

```powershell
bgmner-api `
  --backend onnx `
  --model-dir runs\bgm_ner_20ep_xlmr\best_model `
  --onnx-path runs\bgm_ner_20ep_xlmr\onnx\model.int8.dynamic.onnx `
  --provider cpu `
  --host 127.0.0.1 `
  --port 8000
```

测试：

```powershell
$payload = @{ text = "[桜都字幕组] 迷宫饭 [15][1080p]" } | ConvertTo-Json -Compress
curl.exe -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d $payload
```

---

更多细节请继续阅读：

- [03-环境与依赖](./03-environment.md)
- [05-训练与超参数](./05-training.md)
- [08-Web API](./08-web-api.md)
- [12-故障排查](./12-troubleshooting.md)


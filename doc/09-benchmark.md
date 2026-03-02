# 09. Benchmark 指南

## 1. 离线推理基准（本地函数调用）

命令：`bgmner-benchmark`

### 1.1 HF

```powershell
bgmner-benchmark `
  --backend hf `
  --model-dir runs\bgm_ner_20ep_xlmr\best_model `
  --input-file data\ner_data\dev.txt `
  --batch-size 32 `
  --max-length 256 `
  --warmup-runs 3 `
  --benchmark-runs 20 `
  --output-json runs\bench_hf.json
```

### 1.2 ONNX

```powershell
bgmner-benchmark `
  --backend onnx `
  --model-dir runs\bgm_ner_20ep_xlmr\best_model `
  --onnx-path runs\bgm_ner_20ep_xlmr\onnx\model.int8.dynamic.onnx `
  --provider cpu `
  --input-file data\ner_data\dev.txt `
  --batch-size 32 `
  --max-length 256 `
  --warmup-runs 3 `
  --benchmark-runs 20 `
  --output-json runs\bench_onnx.json
```

## 2. API 基准（HTTP）

脚本：`scripts/benchmark_api_batch.py`

```powershell
mamba run -n bgmner python scripts\benchmark_api_batch.py `
  --url http://127.0.0.1:8000/predict `
  --input-file data\ner_data\dev.txt `
  --batch-size 32 `
  --warmup-runs 3 `
  --benchmark-runs 20 `
  --output-json runs\api_benchmark.json
```

支持：

- `--payload-mode texts|items`
- `--header Key:Value`（可重复）
- `--continue-on-error`

## 3. 指标解释

常用字段：

- `throughput_texts_per_sec`
- `request_latency_ms_mean/p50/p95/p99`
- `api_process_time_ms_mean/p95`（来自响应头 `X-Process-Time-Ms`）
- `request_success_rate`
- `text_success_rate`

## 4. 对比公平性清单

做 Python vs Rust 对比时，务必统一：

1. 相同模型文件（同一个 ONNX）
2. 相同 `batch-size`、`max-length`
3. 相同 provider（例如都用 `cpu`）
4. 串行测试（不要并发跑多个服务）
5. 都用 release 可执行（Rust）
6. 都完成 warmup

## 5. Rust CPU 性能注意点

Rust 侧已经启用 CPU arena allocator（`ep::CPU::with_arena_allocator(true)`），避免 ONNX Runtime 频繁内存分配导致吞吐大幅下降。

如需定位瓶颈，可开启阶段统计：

```powershell
$env:BGMNER_PROFILE_STAGES="1"
```

会输出 `encode/tensor/infer/argmax/decode` 各阶段耗时。


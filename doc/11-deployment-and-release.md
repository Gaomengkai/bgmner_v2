# 11. 发布与部署

## 1. 模型发布（HF / ONNX）

推荐把以下内容作为一个可复现模型包：

1. `best_model/`（tokenizer + config + label mapping）
2. `onnx/model.onnx`（或优化后的 `.opt.onnx`）
3. `onnx/model.int8.dynamic.onnx`
4. `onnx/export_meta.json`
5. `onnx/quantize_meta.json`
6. `metrics/eval_metrics.json`

说明：

- ONNX 推理需要 `model-dir` 读取 tokenizer/config，不能只给 `.onnx` 文件。

## 2. 是否可单独发布 ONNX 到 Hugging Face

可以。建议结构：

- `model.onnx` / `model.int8.dynamic.onnx`
- `tokenizer.json`、`tokenizer_config.json`
- `config.json`（包含 `id2label/label2id`）
- README（provider、量化策略、评测指标）

## 3. Rust 可执行发布（Windows）

最小运行时文件（同目录）：

- `bgmner-rs.exe`
- `onnxruntime.dll`
- `onnxruntime_providers_shared.dll`
- `DirectML.dll`（如使用 DML）

可额外附带：

- `README.md`
- `start_cpu.bat`
- `start_dml.bat`

## 4. 版本对齐建议

1. `Microsoft.ML.OnnxRuntime` 与 `Microsoft.ML.OnnxRuntime.DirectML` 同版本
2. `DirectML.dll` 来源固定（NuGet），不要依赖系统/Office 路径
3. 导出 ONNX 后记录 `export_meta.json`
4. 量化后记录 `quantize_meta.json`

## 5. 独立仓库发布建议

如果把 `experimental/` 单独作为仓库发布，建议纳入：

- `doc/` 全套文档
- `configs/` 训练配置模板
- `scripts/` 工具脚本
- `data/ner_data/`（若许可允许）
- `runs/<代表性实验>/metrics` 与配置元信息

建议排除：

- 大体积中间产物
- 本地缓存目录
- 临时 benchmark 文件


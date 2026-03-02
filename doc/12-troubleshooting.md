# 12. 故障排查

## 1. `curl` 发送 JSON 报 `json_invalid`

症状：

- `Expecting property name enclosed in double quotes`
- `curl: (3) unmatched close brace/bracket`

原因：

- PowerShell 转义导致 JSON 被破坏

建议：

```powershell
$payload = @{ text = "[桜都字幕组] 迷宫饭 [15][1080p]" } | ConvertTo-Json -Compress
curl.exe -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d $payload
```

## 2. `OrtGetApiBase` 缺失 / 动态库加载失败

症状：

- `OrtGetApiBase must be present in ONNX Runtime dylib`

原因：

- 加载到错误版本或错误文件名的 DLL

处理：

```powershell
$env:ORT_DYLIB_PATH="E:\conda\envs\bgmner\Lib\site-packages\onnxruntime\capi\onnxruntime.dll"
```

并确认 `onnxruntime.dll` 来自正确发行包。

## 3. DML 报错 `0x887A0004`

症状：

- provider 显示可用，但创建会话失败

高概率原因：

- `DirectML.dll` 太旧（例如系统目录版本）

处理：

1. 用 NuGet 的 `Microsoft.AI.DirectML` 提供新 DLL
2. 放到 `bgmner-rs.exe` 同目录（高优先级）
3. 尝试 `--dml-device-id 1/2`
4. 用 `--provider dml,cpu` 兜底

## 4. Rust CPU 比 Python CPU 慢很多

排查重点：

1. 是否用 `release` 版本
2. 是否同一 provider / 同一 ONNX / 同参数
3. 是否并发跑了多个服务
4. 是否启用 CPU arena allocator（当前代码已开启）

可打开阶段统计：

```powershell
$env:BGMNER_PROFILE_STAGES="1"
```

## 5. 量化后 F1 明显下降

原因：

- 量化 `MatMul/Gemm` 可能破坏模型精度

建议：

- 先用 `Gather,EmbedLayerNormalization`
- 固定 `--op-types` 并做量化前后评估对比

## 6. ONNX Runtime 量化预处理警告

警告：

- `Please consider to run pre-processing before quantization`

当前工具默认已预处理（`--preprocess` 默认开启）。  
如遇特定模型预处理失败，可尝试：

- `--preprocess-skip-symbolic-shape`
- `--preprocess-skip-onnx-shape`
- 或 `--no-preprocess`

## 7. `Ctrl+C` 后看似退出但端口仍占用

排查：

```powershell
Get-NetTCPConnection -LocalPort 8000 | Select-Object -ExpandProperty OwningProcess
Stop-Process -Id <PID> -Force
```

## 8. `best_model` 在中断后缺失

训练逻辑在完整结束后才保存 `best_model`。  
中断时可能只留下 `trainer_output` checkpoint。

## 9. 在错误 Conda 环境安装依赖

先确认解释器：

```powershell
python -c "import sys; print(sys.executable)"
```

若不是目标环境，切回：

```powershell
mamba activate bgmner
```

## 10. tokenizer 警告 `fix_mistral_regex`

该警告来自 transformers/tokenizers 对某些 tokenizer regex 的兼容提示。  
若业务结果正常，可先记录版本后继续；如要严格消除，可按警告指向的上游说明处理并复测精度。


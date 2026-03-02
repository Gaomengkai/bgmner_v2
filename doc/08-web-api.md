# 08. Web API（Python / Rust）

## 1. API 协议

两套服务都兼容以下接口：

- `GET /health`
- `POST /predict`

`/predict` 请求支持三种输入：

- `text`: 单条字符串
- `texts`: 字符串数组
- `items`: 对象数组（可携带 `id`）

额外可选参数：

- `batch_size`（范围 `1..1024`）
- `max_length`（范围 `8..4096`）

## 2. Python 服务启动

HF 后端：

```powershell
bgmner-api `
  --backend hf `
  --model-dir runs\bgm_ner_20ep_xlmr\best_model `
  --host 127.0.0.1 `
  --port 8000
```

ONNX 后端：

```powershell
bgmner-api `
  --backend onnx `
  --model-dir runs\bgm_ner_20ep_xlmr\best_model `
  --onnx-path runs\bgm_ner_20ep_xlmr\onnx\model.int8.dynamic.onnx `
  --provider cpu `
  --host 127.0.0.1 `
  --port 8000
```

## 3. Rust 服务启动

```powershell
.\rust\target\release\bgmner-rs.exe serve `
  --model-dir runs\bgm_ner_20ep_xlmr\best_model `
  --onnx-path runs\bgm_ner_20ep_xlmr\onnx\model.int8.dynamic.onnx `
  --provider cpu `
  --host 127.0.0.1 `
  --port 8000 `
  --batch-size 32 `
  --max-length 256
```

## 4. 请求示例

### 4.1 单条

```powershell
$payload = @{ text = "[桜都字幕组] 迷宫饭 [15][1080p]" } | ConvertTo-Json -Compress
curl.exe -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d $payload
```

### 4.2 多条（texts）

```powershell
$payload = @{
  texts = @("[标题A] [1080p]", "[标题B] [简繁]")
  batch_size = 16
  max_length = 256
} | ConvertTo-Json -Compress
curl.exe -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d $payload
```

### 4.3 items（含 id）

```powershell
$payload = @{
  items = @(
    @{ id = "a-1"; text = "[标题A] [1080p]" },
    @{ id = "a-2"; text = "[标题B] [简繁]" }
  )
  batch_size = 16
  max_length = 256
} | ConvertTo-Json -Depth 5 -Compress
curl.exe -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d $payload
```

## 5. 响应示例

```json
{
  "count": 1,
  "backend": "onnx",
  "results": [
    {
      "text": "[桜都字幕组] 迷宫饭 [15][1080p]",
      "truncated_text": "[桜都字幕组] 迷宫饭 [15][1080p]",
      "entities": {},
      "pred_labels": ["O", "..."]
    }
  ]
}
```

## 6. 处理时延

服务会在响应头附加：

- `X-Process-Time-Ms`

用于 API benchmark 统计。

## 7. Ctrl+C 与端口占用

- 正常 `Ctrl+C` 应触发优雅退出
- 如进程残留，可手动杀掉占用端口的进程后重启

示例：

```powershell
Get-NetTCPConnection -LocalPort 8000 | Select-Object -ExpandProperty OwningProcess
Stop-Process -Id <PID> -Force
```


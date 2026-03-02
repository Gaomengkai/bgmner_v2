# 04. 数据集规范

## 1. 目录结构

工程内统一使用：

- `data/ner_data/train.txt`
- `data/ner_data/dev.txt`
- `data/ner_data/labels.txt`

## 2. 文件格式

### 2.1 `train.txt` / `dev.txt`

JSONL，每行一个样本，字段：

- `id`: 样本 ID（建议整型，可排序）
- `text`: 字符数组（字粒度）
- `labels`: 与 `text` 等长的 BIO 标签数组

示例：

```json
{"id":0,"text":["[","漫","猫","字","幕","社","]"],"labels":["O","B-GR","I-GR","I-GR","I-GR","I-GR","O"]}
```

约束：

- `text` 和 `labels` 必须都是数组
- 长度必须一致
- 空行忽略

### 2.2 `labels.txt`

每行一个实体基础类别（不含 BIO 前缀），例如：

```text
GR
NB
CT
ET
JT
EP
RES
SUB
```

训练时会自动扩展为：

- `O`
- `B-<label>`
- `I-<label>`

## 3. 数据同步脚本

脚本：`scripts/sync_dataset.py`

功能：

- 从上级目录复制数据到工程内
- 按 `id` 对 `train/dev` 排序
- 统一换行与 JSON 紧凑格式

命令：

```powershell
python .\scripts\sync_dataset.py `
  --src-dir ..\data\bgm\ner_data `
  --dst-dir .\data\ner_data
```

## 4. 标签对齐逻辑

训练采用 tokenizer 的 `word_ids()` 对齐：

- 每个 word 的首个 token 继承该 word 标签
- 同一 word 的后续 sub-token 标为 `-100`（loss 忽略）
- 特殊 token（`[CLS]`/`[SEP]`/padding）标为 `-100`

## 5. 长度截断规则

`max_length` 下，字级样本先按 `max_length - 2` 截断（预留特殊 token）。

## 6. 建议

- `id` 尽量保持稳定，便于 diff 与排查
- 新增标签时同步更新 `labels.txt`
- 训练前先做一次 `scripts/sync_dataset.py`，保证排序与格式一致


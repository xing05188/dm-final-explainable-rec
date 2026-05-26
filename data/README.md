# 数据处理说明

本目录存放 Amazon Review 2018 **Electronics** 子集的原始数据与 Pipeline 产出。数据处理由 `src/preprocess.py` 一键完成（Stage 1–8），参数见项目根目录 `config.py`。

---

## 目录结构

```plain
data/
├── raw/
│   └── Electronics_5.json.gz    # 官方原始数据（JSON Lines，约 674 万条）
└── processed/                   # Pipeline 产出（勿手改）
    ├── raw_interactions.csv       # Stage 1：字段提取后的全量表
    ├── cleaned_interactions.csv   # Stage 2：K-core 后（中间缓存，体积大）
    ├── interactions.csv           # Stage 3：采样后最终交互表
    ├── train.csv / val.csv / test.csv
    ├── train_neg.csv              # 负采样快照（训练时 NCF 按 epoch 重采）
    ├── user_map.json / item_map.json
    ├── user_reviews.json / item_reviews.json
    ├── user_emb.npy / item_emb.npy
    ├── stats.json
    └── .pipeline_state            # 断点续跑状态（自动生成）
```

---

## 环境准备

```powershell
cd <项目根目录>
pip install -r requirements.txt
```

主要依赖：`pandas`、`numpy`、`sentence-transformers`（Stage 8 SBERT）。

---

## 下载原始数据

1. 打开 [Amazon Review 2018 官方页](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/)
2. 下载 **Electronics** 子集：`Electronics_5.json.gz`
3. 放到 `data/raw/Electronics_5.json.gz`

也支持未压缩的 `Electronics_5.json`，`config.py` 会自动尝试两种扩展名。

---

## 一键运行 Pipeline

**PowerShell：**

```powershell
cd <项目根目录>
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:HF_HUB_ENDPOINT = "https://hf-mirror.com"
python src/preprocess.py
```

**CMD（命令提示符）：** 不要用 `$env:...`，应写成：

```cmd
cd /d <项目根目录>
set HF_ENDPOINT=https://hf-mirror.com
set HF_HUB_ENDPOINT=https://hf-mirror.com
python src/preprocess.py
```

> 已在 `config.py` 的 `text.hf_hub_endpoint` 默认配置镜像，即使不设环境变量，Stage 8 也会尝试走 `https://hf-mirror.com`。

- **首次运行**：顺序执行 Stage 1 → 8，全量数据 Stage 1–2 可能需数十分钟至数小时（视磁盘与 CPU 而定）。
- **断点续跑**：已完成的 Stage 记录在 `data/processed/.pipeline_state`，再次执行会跳过；改逻辑后需删除对应 stage 或整个 `.pipeline_state` 再跑。
- **仅重跑文本与向量**：删除 `.pipeline_state` 中 `stage7`、`stage8` 后执行上述命令。

强制从头跑（清空断点）：

```powershell
Remove-Item "data/processed/.pipeline_state" -ErrorAction SilentlyContinue
python src/preprocess.py
```

---

## Pipeline 阶段一览

| Stage | 功能 | 主要产出 |
|-------|------|----------|
| 1 | 流式加载 JSONL、基础过滤、字段提取 | `raw_interactions.csv` + **rating 分布** |
| 2 | K-core 过滤（k=5） | `cleaned_interactions.csv` |
| 3 | ID 重编码、Top-5000 活跃用户采样、Top-3000 商品 | `interactions.csv`、`user_map.json`、`item_map.json` |
| 4 | 按时间 Leave-One-Out 划分 | `train.csv`、`val.csv`、`test.csv` |
| 5 | 负采样快照（1:4，按交互行） | `train_neg.csv` |
| 6 | 数据集统计 | `stats.json` |
| 7 | 领域噪音清洗、按时间聚合、头尾各半截断 | `user_reviews.json`、`item_reviews.json` |
| 8 | SBERT 编码（L2 归一化） | `user_emb.npy`、`item_emb.npy` |

详细设计见项目根目录 [`数据处理详细计划.md`](../数据处理详细计划.md)。

---

## 产出表说明

### `interactions.csv` / `train|val|test.csv` 列

| 列名 | 说明 |
|------|------|
| `user_id` | 重编码用户 ID（0 … n_users-1） |
| `item_id` | 重编码商品 ID |
| `rating` | 评分 1–5 |
| `review_text` | 清洗后评论正文 |
| `summary` | 清洗后摘要/标题 |
| `timestamp` | Unix 时间戳（秒） |

### `stats.json` 示例字段

`n_users`、`n_items`、`n_interactions`、`n_train`、`n_val`、`n_test`、`sparsity`、`avg_items_per_user`、`avg_users_per_item`。

### 映射文件

- `user_map.json`：`{ "原始reviewerID": 新user_id }`
- `item_map.json`：`{ "原始asin": 新item_id }`

---

## 关键配置（`config.py`）

| 配置项 | 默认值 | 含义 |
|--------|--------|------|
| `filter.k_core` | 5 | K-core 最小交互数 |
| `filter.n_users_sample` | 5000 | 采样用户数（Top 活跃用户） |
| `filter.n_items_target` | 3000 | 商品数上限 |
| `filter.sample_active_users` | True | True=最活跃 5000 用户；False=随机 |
| `negative_sampling.neg_ratio` | 4 | 负正比（按 train 行） |
| `negative_sampling.resample_per_epoch` | True | NCF 训练每 epoch 重采负样本 |
| `text.truncate_mode` | `head_tail` | 聚合文本头尾各半截断 |
| `text.max_tokens` | 128 | 聚合总长度 ≈ max_tokens×4 词 |
| `text.skip_sbert` | False | True 则跳过 Stage 8 |

---

## 交给组员的数据

| 组员 | 文件 |
|------|------|
| B / C（训练） | `train.csv`、`val.csv`、`test.csv`、`stats.json` |
| B（NCF+Review） | 另加 `user_emb.npy`、`item_emb.npy` |
| D（可解释） | `user_reviews.json`、`item_reviews.json`、embedding |

NCF / LightGCN 训练请读 **`train.csv` 正样本**；负样本由 `src/ncf.py` 在训练循环中按 `resample_per_epoch` 生成，不必依赖静态 `train_neg.csv`。

---

## 常见问题

**Q: `interactions.csv` 在编辑器里行数很多，和 `stats.json` 不一致？**  
A: 评论正文含换行符时，编辑器按行计数会偏大；以 `stats.json` 或 `pandas.read_csv` 行数为准。

**Q: Stage 8 连不上 HuggingFace？**  
A: 设置 `$env:HF_ENDPOINT="https://hf-mirror.com"` 后重跑；或临时 `text.skip_sbert: True`。

**Q: 内存不足？**  
A: Stage 1–2 已流式/分块处理；勿一次性 `read_csv` 整个 `cleaned_interactions.csv`。采样后 `interactions.csv` 较小，可正常加载。

**Q: 如何只验证 Stage 1？**  
A: 删除 `.pipeline_state` 后运行；或删除其中 `stage1` 条目。跳过缓存时仍会分块统计并打印 **rating 分布**。

---

## 相关文档

- [`数据处理详细计划.md`](../数据处理详细计划.md) — 字段、采样策略更换说明、Pipeline 设计
- [`项目路线图 & 分工 Map.md`](../项目路线图%20&%20分工%20Map.md) — 分工与依赖

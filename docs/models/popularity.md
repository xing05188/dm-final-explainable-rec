# Popularity 模型

## 原理

Popularity（热度推荐）是最简单的推荐基线：将训练集中被交互次数最多的物品推荐给所有用户，不利用任何用户个性化信息，也不利用物品特征。

## 实现

**文件**：`src/base_model/popularity.py`
**类名**：`PopularityRecommender`

### __init__

统计训练集中每个物品的出现次数，按热度降序排列，得到全局热门物品列表。

```
popular_items = arg sort(count(item_id), descending)
```

### recommend

从热门列表中依次取物品，跳过用户已在训练集中交互过的物品（`exclude`），凑齐 k 个。

若热门物品数量不足以填满 k（用户几乎买遍了所有热门品），则从未交互且未被推荐过的物品中随机补齐。

## 复杂度

| 阶段 | 时间复杂度 | 空间复杂度 |
|------|-----------|------------|
| 构建 | O(N log N)，N = train 交互数 | O(M)，M = 物品数 |
| 推荐 | O(M)，需遍历热门列表 | O(M) |

## 特点

- **不需要训练**：无损失函数，无参数更新，仅为统计
- **非个性化**：所有用户得到完全相同的推荐（排除已交互物品后略有差异）
- **基线作用**：若某复杂模型无法显著超越 Popularity，模型设计或特征存在问题
- **冷启动友好**：对行为数据少的新用户也能给出合理推荐（热门即合理）

## 推荐接口

```python
model = PopularityRecommender(train_df)
recommended = model.recommend(user_id, n_items, k, exclude=set_of_train_items)
```

与 `evaluate.py` 中的评估框架完全兼容，`exclude` 参数用于排除训练集中已交互的物品，避免"推荐已买过的物品"刷高指标。

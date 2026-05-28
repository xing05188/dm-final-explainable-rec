# UserCF（基于用户的协同过滤）
## 核心思路
核心思路时构建n×n的相似度矩阵，
对于用户u，找到topk相似的用户，for v of topk(u),如果v和itemj有交互，那么对于itemj的平方加1

## 原理

核心假设：**相似用户喜欢相似的物品**。

给定用户 u，找出与其行为最相似的 k 个邻居用户，将这些邻居购买过但 u 未买过的物品作为推荐候选，按邻居相似度加权打分。

## 计算流程

### 1. 构建用户-物品交互矩阵

```
        item_0  item_1  item_2  ...  item_2999
user_0     1       0       1    ...      0
user_1     0       1       0    ...      1
...
user_4963  1       0       0    ...      1
```

采用隐式反馈：交互过 = 1，未交互 = 0。不显式使用评分值，避免评分偏差。

### 2. 计算用户相似度

对矩阵的每一行（用户向量）计算两两余弦相似度：

```
sim(user_a, user_b) = (u_a · u_b) / (||u_a|| × ||u_b||)
```

得到 (n_users × n_users) 的相似度矩阵。UserCF 自身排除（`sim[i][i] = 0`）。

### 3. 推荐打分

对于目标用户 u：

```
score(item_j) = Σ_{v ∈ top-k neighbors(u)} sim(u, v) × I(v interacted with item_j)
```

即邻居 v 与 u 越相似，v 买过的商品得分越高。近似于"邻居们用钱包投票"。

### 4. 排除已交互物品

返回排名最高但用户 u 未交互过的 k 个物品。

### 冷启动处理

若用户无正相似邻居（相似度全 ≤ 0），退化为按全局热度推荐。

## 实现

**文件**：`src/base_model/usercf.py`
**类名**：`UserCF`

## 复杂度

| 阶段 | 时间复杂度 | 空间复杂度 |
|------|-----------|------------|
| 构建相似度矩阵 | O(n_users² × n_items) | O(n_users²) |
| 单次推荐 | O(K × n_items)，K = 邻居数 | O(n_items) |

## 特点

- **解释性强**：推荐结果可追溯到"因为和你相似的用户 A、B、C 也买了这个"
- **用户侧瓶颈**：5000 用户规模可行，用户量更大时相似度矩阵 O(n²) 会撑不住
- **冷用户不友好**：新用户或交互稀疏的用户缺乏足够邻居信号
- **与 ItemCF 对称**：一个看人，一个看物；实践中 ItemCF 在物品数 < 用户数时更高效

## 推荐接口

```python
model = UserCF(train_df, n_users, n_items, k_neighbors=50)
recommended = model.recommend(user_id, n_items, k, exclude=set_of_train_items)
```

k_neighbors 默认 50，可调整以平衡速度与覆盖。

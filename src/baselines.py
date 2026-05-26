"""Baseline recommendation models: Popularity, UserCF, ItemCF."""

import numpy as np
import pandas as pd
from collections import defaultdict


class PopularityRecommender:
    """Recommend the globally most popular items (no personalization)."""

    def __init__(self, train_df: pd.DataFrame):
        self.popular_items = (
            train_df.groupby("item_id")["user_id"]
            .count()
            .sort_values(ascending=False)
            .index.tolist()
        )

    def recommend(self, user_id, n_items, k, exclude=None):
        recs = []
        for item in self.popular_items:
            if len(recs) >= k:
                break
            if exclude and item in exclude:
                continue
            recs.append(item)
        return recs


class UserCF:
    """User-Based Collaborative Filtering with cosine similarity."""

    def __init__(self, train_df: pd.DataFrame, n_users: int, n_items: int, k_neighbors: int = 50):
        self.k = k_neighbors
        self.n_users = n_users
        self.n_items = n_items

        # Build rating vectors per user
        self.user_ratings = defaultdict(dict)
        self.user_mean = {}
        for _, row in train_df.iterrows():
            u, i, r = int(row["user_id"]), int(row["item_id"]), float(row["rating"])
            self.user_ratings[u][i] = r

        for u in self.user_ratings:
            ratings = list(self.user_ratings[u].values())
            self.user_mean[u] = np.mean(ratings)

        # Precompute user similarity (top-k neighbors)
        print("  Computing user similarities ...")
        self.user_neighbors = self._compute_similarities()

    def _compute_similarities(self):
        """Compute top-k similar users for each user using cosine similarity."""
        user_neighbors = {}
        all_users = list(self.user_ratings.keys())

        for u in all_users:
            u_items = set(self.user_ratings[u].keys())
            u_vec = np.array([self.user_ratings[u][i] for i in sorted(u_items)])
            u_norm = np.linalg.norm(u_vec)
            if u_norm == 0:
                user_neighbors[u] = []
                continue

            sims = []
            for v in all_users:
                if u == v:
                    continue
                common = u_items & set(self.user_ratings[v].keys())
                if not common:
                    continue
                u_common = np.array([self.user_ratings[u][i] for i in sorted(common)])
                v_common = np.array([self.user_ratings[v][i] for i in sorted(common)])
                sim = np.dot(u_common, v_common) / (np.linalg.norm(u_common) * np.linalg.norm(v_common) + 1e-8)
                sims.append((v, sim))

            sims.sort(key=lambda x: x[1], reverse=True)
            user_neighbors[u] = sims[:self.k]

        return user_neighbors

    def recommend(self, user_id, n_items, k, exclude=None):
        """Predict scores for all items, return top-k."""
        if user_id not in self.user_ratings:
            return self._fallback(n_items, k, exclude)

        scores = np.zeros(n_items)
        user_mean = self.user_mean.get(user_id, 3.0)

        neighbors = self.user_neighbors.get(user_id, [])
        if not neighbors:
            return self._fallback(n_items, k, exclude)

        sim_sum = defaultdict(float)
        weighted_sum = defaultdict(float)

        for v, sim in neighbors:
            v_mean = self.user_mean.get(v, 3.0)
            for item, rating in self.user_ratings[v].items():
                if item in self.user_ratings[user_id]:
                    continue
                sim_sum[item] += abs(sim)
                weighted_sum[item] += sim * (rating - v_mean)

        for item in range(n_items):
            if sim_sum[item] > 0:
                scores[item] = user_mean + weighted_sum[item] / sim_sum[item]

        # Exclude already-interacted items
        if exclude:
            for item in exclude:
                scores[item] = -999

        top_items = np.argsort(scores)[::-1][:k]
        return top_items.tolist()

    def _fallback(self, n_items, k, exclude):
        return [i for i in range(min(k, n_items))]


class ItemCF:
    """Item-Based Collaborative Filtering with cosine similarity."""

    def __init__(self, train_df: pd.DataFrame, n_users: int, n_items: int, k_neighbors: int = 50):
        self.k = k_neighbors
        self.n_items = n_items

        # Build rating vectors per item
        self.item_ratings = defaultdict(dict)
        for _, row in train_df.iterrows():
            u, i, r = int(row["user_id"]), int(row["item_id"]), float(row["rating"])
            self.item_ratings[i][u] = r

        # User history
        self.user_items = defaultdict(set)
        for _, row in train_df.iterrows():
            self.user_items[int(row["user_id"])].add(int(row["item_id"]))

        # Precompute item similarity
        print("  Computing item similarities ...")
        self.item_neighbors = self._compute_similarities()

    def _compute_similarities(self):
        """Compute top-k similar items for each item."""
        item_neighbors = {}
        all_items = list(self.item_ratings.keys())

        for i in all_items:
            i_users = set(self.item_ratings[i].keys())
            i_vec = np.array([self.item_ratings[i][u] for u in sorted(i_users)])
            i_norm = np.linalg.norm(i_vec)
            if i_norm == 0:
                item_neighbors[i] = []
                continue

            sims = []
            for j in all_items:
                if i == j:
                    continue
                common = i_users & set(self.item_ratings[j].keys())
                if not common:
                    continue
                i_common = np.array([self.item_ratings[i][u] for u in sorted(common)])
                j_common = np.array([self.item_ratings[j][u] for u in sorted(common)])
                sim = np.dot(i_common, j_common) / (np.linalg.norm(i_common) * np.linalg.norm(j_common) + 1e-8)
                sims.append((j, sim))

            sims.sort(key=lambda x: x[1], reverse=True)
            item_neighbors[i] = sims[:self.k]

        return item_neighbors

    def recommend(self, user_id, n_items, k, exclude=None):
        """Predict scores for all items based on item similarity."""
        user_history = self.user_items.get(user_id, set())
        scores = np.zeros(n_items)

        sim_sum = defaultdict(float)
        weighted_sum = defaultdict(float)

        for i in user_history:
            neighbors = self.item_neighbors.get(i, [])
            for j, sim in neighbors:
                if j in user_history:
                    continue
                sim_sum[j] += abs(sim)
                weighted_sum[j] += sim * self.item_ratings[i].get(user_id, 3.0)

        for item in range(n_items):
            if sim_sum[item] > 0:
                scores[item] = weighted_sum[item] / sim_sum[item]

        if exclude:
            for item in exclude:
                scores[item] = -999

        top_items = np.argsort(scores)[::-1][:k]
        return top_items.tolist()

"""NCF + Review Embedding Fusion Model.

Fuses behavioral embeddings (from NCF) with semantic embeddings (from SBERT).
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.utils import sample_train_negatives, ensure_binary_labels


class NCFReviewDataset(Dataset):
    def __init__(self, df, user_emb, item_emb):
        self.users = torch.LongTensor(df["user_id"].values)
        self.items = torch.LongTensor(df["item_id"].values)
        self.labels = torch.FloatTensor(df["label"].values)
        self.user_review_emb = torch.FloatTensor(user_emb)
        self.item_review_emb = torch.FloatTensor(item_emb)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        u = self.users[idx]
        i = self.items[idx]
        return (
            u, i,
            self.user_review_emb[u],
            self.item_review_emb[i],
            self.labels[idx],
        )


class NCFReview(nn.Module):
    """NCF + Review Embedding fusion model.

    Architecture:
    1. NCF path: user/item ID embeddings → MLP → behavior_vec
    2. SBERT path: user/item review embeddings → linear projection → semantic_vec
    3. Weighted fusion: final = (1-α) * behavior + α * semantic
    4. Prediction: final → MLP → score
    """

    def __init__(self, n_users, n_items, embedding_dim=64,
                 mlp_layers=(64, 32, 16), review_emb_dim=384, alpha=0.3):
        super().__init__()
        self.alpha = alpha
        self.embedding_dim = embedding_dim

        # NCF embeddings
        self.user_emb = nn.Embedding(n_users, embedding_dim)
        self.item_emb = nn.Embedding(n_items, embedding_dim)

        # Review embedding projection: 384 → embedding_dim
        self.user_review_proj = nn.Linear(review_emb_dim, embedding_dim)
        self.item_review_proj = nn.Linear(review_emb_dim, embedding_dim)

        # MLP for behavior path
        behavior_input_dim = embedding_dim * 2
        behavior_layers = []
        for dim in mlp_layers:
            behavior_layers.extend([
                nn.Linear(behavior_input_dim, dim),
                nn.ReLU(),
                nn.Dropout(0.2),
            ])
            behavior_input_dim = dim
        self.behavior_mlp = nn.Sequential(*behavior_layers)

        # Fusion MLP
        fusion_input_dim = embedding_dim  # behavior_vec + semantic_vec (both dim)
        fusion_layers = [
            nn.Linear(fusion_input_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embedding_dim // 2, 1),
            nn.Sigmoid(),
        ]
        self.fusion_mlp = nn.Sequential(*fusion_layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, user_ids, item_ids, user_review_emb, item_review_emb):
        # Behavior path
        u_beh = self.user_emb(user_ids)
        i_beh = self.item_emb(item_ids)
        behavior_input = torch.cat([u_beh, i_beh], dim=-1)
        behavior_vec = self.behavior_mlp(behavior_input)

        # Semantic path
        u_sem = self.user_review_proj(user_review_emb)
        i_sem = self.item_review_proj(item_review_emb)
        semantic_vec = u_sem + i_sem  # simple addition after projection

        # Weighted fusion
        final_vec = (1 - self.alpha) * behavior_vec + self.alpha * semantic_vec

        # Predict
        score = self.fusion_mlp(final_vec)
        return score.squeeze(-1)

    def recommend(self, user_id, n_items, k, exclude=None,
                  user_emb_tensor=None, item_emb_tensor=None, device="cpu"):
        """Generate top-k recommendations."""
        self.eval()
        with torch.no_grad():
            user_tensor = torch.LongTensor([user_id] * n_items).to(device)
            item_tensor = torch.LongTensor(list(range(n_items))).to(device)

            u_rev = user_emb_tensor[user_id].unsqueeze(0).expand(n_items, -1).to(device)
            i_rev = item_emb_tensor.to(device)

            scores = self.forward(user_tensor, item_tensor, u_rev, i_rev).cpu().numpy()

        if exclude:
            for item in exclude:
                scores[item] = -999

        top_items = np.argsort(scores)[::-1][:k]
        return top_items.tolist()


def train_ncf_review(model, train_pos_df, val_df, user_emb, item_emb, config, n_items, device="cpu"):
    """Train NCF+Review fusion model. See train_ncf for resample_per_epoch behavior."""
    neg_ratio = config["negative_sampling"]["neg_ratio"]
    resample = config["negative_sampling"].get("resample_per_epoch", True)
    base_seed = config["seed"]

    if "label" in train_pos_df.columns:
        train_pos_df = train_pos_df[train_pos_df["label"] == 1][["user_id", "item_id"]]
    else:
        train_pos_df = train_pos_df[["user_id", "item_id"]].copy()

    val_df = ensure_binary_labels(val_df)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["model"]["learning_rate"])
    criterion = nn.BCELoss()

    best_val_loss = float("inf")
    patience_counter = 0
    patience = config["model"]["early_stop_patience"]
    best_state = None

    if not resample:
        static_train = sample_train_negatives(
            train_pos_df, n_items, neg_ratio=neg_ratio, seed=base_seed
        )

    for epoch in range(config["model"]["epochs"]):
        if resample:
            train_data = sample_train_negatives(
                train_pos_df, n_items, neg_ratio=neg_ratio, seed=base_seed + epoch
            )
        else:
            train_data = static_train

        loader = DataLoader(
            NCFReviewDataset(train_data, user_emb, item_emb),
            batch_size=config["model"]["batch_size"],
            shuffle=True,
        )

        model.train()
        total_loss = 0
        n_batches = 0

        for users, items, u_rev, i_rev, labels in loader:
            users = users.to(device)
            items = items.to(device)
            u_rev = u_rev.to(device)
            i_rev = i_rev.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            preds = model(users, items, u_rev, i_rev)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    Epoch {epoch + 1}: train_loss={avg_loss:.4f}")

        if avg_loss < best_val_loss:
            best_val_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch + 1}")
                break

    if best_state:
        model.load_state_dict(best_state)
    return model

"""Neural Collaborative Filtering (NCF) model in PyTorch."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


from src.utils import sample_train_negatives, ensure_binary_labels


class InteractionDataset(Dataset):
    def __init__(self, df):
        self.users = torch.LongTensor(df["user_id"].values)
        self.items = torch.LongTensor(df["item_id"].values)
        self.labels = torch.FloatTensor(df["label"].values)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]


class NCF(nn.Module):
    """Neural Collaborative Filtering with GMF + MLP fusion."""

    def __init__(self, n_users, n_items, embedding_dim=64, mlp_layers=(64, 32, 16)):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim

        # GMF path
        self.user_emb_gmf = nn.Embedding(n_users, embedding_dim)
        self.item_emb_gmf = nn.Embedding(n_items, embedding_dim)

        # MLP path
        self.user_emb_mlp = nn.Embedding(n_users, embedding_dim)
        self.item_emb_mlp = nn.Embedding(n_items, embedding_dim)

        # MLP layers
        mlp_input_dim = embedding_dim * 2
        layers = []
        for dim in mlp_layers:
            layers.append(nn.Linear(mlp_input_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            mlp_input_dim = dim
        self.mlp = nn.Sequential(*layers)

        # Final prediction
        self.output_layer = nn.Linear(mlp_layers[-1] + embedding_dim, 1)
        self.sigmoid = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, user_ids, item_ids):
        # GMF
        u_gmf = self.user_emb_gmf(user_ids)
        i_gmf = self.item_emb_gmf(item_ids)
        gmf_out = u_gmf * i_gmf  # element-wise product

        # MLP
        u_mlp = self.user_emb_mlp(user_ids)
        i_mlp = self.item_emb_mlp(item_ids)
        mlp_input = torch.cat([u_mlp, i_mlp], dim=-1)
        mlp_out = self.mlp(mlp_input)

        # Concat and predict
        concat = torch.cat([gmf_out, mlp_out], dim=-1)
        score = self.sigmoid(self.output_layer(concat))
        return score.squeeze(-1)

    def recommend(self, user_id, n_items, k, exclude=None, device="cpu"):
        """Generate top-k recommendations for a user."""
        self.eval()
        with torch.no_grad():
            user_tensor = torch.LongTensor([user_id] * n_items).to(device)
            item_tensor = torch.LongTensor(list(range(n_items))).to(device)
            scores = self.forward(user_tensor, item_tensor).cpu().numpy()

        if exclude:
            for item in exclude:
                scores[item] = -999

        top_items = np.argsort(scores)[::-1][:k]
        return top_items.tolist()


def train_ncf(model, train_pos_df, val_df, config, n_items, device="cpu"):
    """Train NCF model with early stopping.

    train_pos_df: train.csv positives (user_id, item_id) or labeled train_neg with label=1 rows.
    When resample_per_epoch=True, negatives are redrawn each epoch (see 数据处理详细计划 §5).
    """
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
            InteractionDataset(train_data),
            batch_size=config["model"]["batch_size"],
            shuffle=True,
        )

        model.train()
        total_loss = 0
        n_batches = 0

        for users, items, labels in loader:
            users, items, labels = users.to(device), items.to(device), labels.to(device)

            optimizer.zero_grad()
            preds = model(users, items)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches

        # Validation
        model.eval()
        val_loss = 0
        val_n = 0
        val_dataset = InteractionDataset(val_df)
        val_loader = DataLoader(val_dataset, batch_size=config["model"]["batch_size"])
        with torch.no_grad():
            for users, items, labels in val_loader:
                users, items, labels = users.to(device), items.to(device), labels.to(device)
                preds = model(users, items)
                loss = criterion(preds, labels)
                val_loss += loss.item() * len(labels)
                val_n += len(labels)
        val_loss /= val_n

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    Epoch {epoch + 1}: train_loss={avg_loss:.4f}, val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
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

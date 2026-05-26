"""Matrix Factorization (SVD / FunkSVD) model for recommendation."""

import numpy as np
import pandas as pd


class MatrixFactorization:
    """SGD-based Matrix Factorization with bias terms.

    Predicts: R_hat[u, i] = mu + b_u + b_i + p_u @ q_i
    Optimizes: MSE loss with L2 regularization.
    """

    def __init__(self, n_users, n_items, n_factors=64, lr=0.001,
                 reg=0.01, n_epochs=50):
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs

        # Initialize parameters
        self.mu = 0.0
        self.b_u = np.zeros(n_users)
        self.b_i = np.zeros(n_items)
        self.P = np.random.normal(0, 0.01, (n_users, n_factors))
        self.Q = np.random.normal(0, 0.01, (n_items, n_factors))

    def fit(self, train_df, val_df=None):
        """Train with SGD."""
        self.mu = train_df["rating"].mean()

        for epoch in range(self.n_epochs):
            total_loss = 0
            n_samples = 0

            # Shuffle training data
            shuffled = train_df.sample(frac=1).reset_index(drop=True)

            for _, row in shuffled.iterrows():
                u = int(row["user_id"])
                i = int(row["item_id"])
                r = float(row["rating"])

                # Predict
                pred = self.mu + self.b_u[u] + self.b_i[i] + self.P[u] @ self.Q[i]
                err = r - pred

                # Update
                self.b_u[u] += self.lr * (err - self.reg * self.b_u[u])
                self.b_i[i] += self.lr * (err - self.reg * self.b_i[i])

                P_u_old = self.P[u].copy()
                self.P[u] += self.lr * (err * self.Q[i] - self.reg * self.P[u])
                self.Q[i] += self.lr * (err * P_u_old - self.reg * self.Q[i])

                total_loss += err ** 2
                n_samples += 1

            rmse = np.sqrt(total_loss / n_samples)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                msg = f"    Epoch {epoch + 1}: train_rmse={rmse:.4f}"
                if val_df is not None:
                    val_rmse = self._evaluate_rmse(val_df)
                    msg += f", val_rmse={val_rmse:.4f}"
                print(msg)

    def _evaluate_rmse(self, df):
        total_err = 0
        for _, row in df.iterrows():
            u = int(row["user_id"])
            i = int(row["item_id"])
            r = float(row["rating"])
            pred = self.mu + self.b_u[u] + self.b_i[i] + self.P[u] @ self.Q[i]
            total_err += (r - pred) ** 2
        return np.sqrt(total_err / len(df))

    def predict_score(self, user_id, item_id):
        return self.mu + self.b_u[user_id] + self.b_i[item_id] + self.P[user_id] @ self.Q[item_id]

    def recommend(self, user_id, n_items, k, exclude=None):
        """Generate top-k recommendations for a user."""
        scores = np.array([
            self.predict_score(user_id, i) for i in range(n_items)
        ])

        if exclude:
            for item in exclude:
                scores[item] = -999

        top_items = np.argsort(scores)[::-1][:k]
        return top_items.tolist()

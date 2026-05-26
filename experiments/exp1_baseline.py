"""Experiment 1: Baseline Comparison — Popularity, UserCF, ItemCF, NCF."""

import sys
import time
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CONFIG
from src.baselines import PopularityRecommender, UserCF, ItemCF
from src.ncf import NCF, train_ncf
from src.evaluate import evaluate_model, print_metrics


def main():
    data_dir = Path(CONFIG["data"]["output_dir"])
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "val.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    stats_file = data_dir / "stats.json"

    import json
    with open(stats_file) as f:
        stats = json.load(f)

    n_users = stats["n_users"]
    n_items = stats["n_items"]

    print(f"  Dataset: {n_users} users, {n_items} items")
    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    all_results = {}

    # ── Popularity ──
    print("\n" + "=" * 60)
    print("  Training Popularity baseline ...")
    t0 = time.time()
    pop_model = PopularityRecommender(train_df)
    metrics = evaluate_model(pop_model, test_df, train_df, n_items)
    print_metrics(metrics, "Popularity")
    all_results["Popularity"] = metrics
    print(f"  Time: {time.time() - t0:.1f}s")

    # ── UserCF ──
    print("\n" + "=" * 60)
    print("  Training UserCF ...")
    t0 = time.time()
    # Use smaller k for speed
    k_neighbors = min(50, n_users - 1)
    usercf_model = UserCF(train_df, n_users, n_items, k_neighbors=k_neighbors)
    metrics = evaluate_model(usercf_model, test_df, train_df, n_items)
    print_metrics(metrics, "UserCF")
    all_results["UserCF"] = metrics
    print(f"  Time: {time.time() - t0:.1f}s")

    # ── ItemCF ──
    print("\n" + "=" * 60)
    print("  Training ItemCF ...")
    t0 = time.time()
    k_neighbors = min(50, n_items - 1)
    itemcf_model = ItemCF(train_df, n_users, n_items, k_neighbors=k_neighbors)
    metrics = evaluate_model(itemcf_model, test_df, train_df, n_items)
    print_metrics(metrics, "ItemCF")
    all_results["ItemCF"] = metrics
    print(f"  Time: {time.time() - t0:.1f}s")

    # ── NCF ──
    print("\n" + "=" * 60)
    print("  Training NCF ...")
    t0 = time.time()
    ncf_model = NCF(
        n_users=n_users,
        n_items=n_items,
        embedding_dim=CONFIG["model"]["embedding_dim"],
        mlp_layers=CONFIG["model"]["ncf_mlp_layers"],
    ).to(device)

    # train.csv positives; negatives resampled per epoch if config says so
    ncf_model = train_ncf(ncf_model, train_df, val_df, CONFIG, n_items=n_items, device=device)
    metrics = evaluate_model(ncf_model, test_df, train_df, n_items)
    print_metrics(metrics, "NCF")
    all_results["NCF"] = metrics
    print(f"  Time: {time.time() - t0:.1f}s")

    # ── Summary table ──
    print("\n" + "=" * 60)
    print("  EXPERIMENT 1: BASELINE COMPARISON SUMMARY")
    print("=" * 60)

    k_values = sorted(set(int(k.split("@")[1]) for k in all_results["Popularity"].keys()))
    metric_names = ["Precision", "Recall", "HitRate", "MAP", "NDCG"]

    # Print as table
    for metric in metric_names:
        print(f"\n  {metric}:")
        header = f"  {'Model':<15}" + "".join(f"{'@' + str(k):<12}" for k in k_values)
        print(header)
        print(f"  {'─' * (15 + 12 * len(k_values))}")
        for model_name, res in all_results.items():
            row = f"  {model_name:<15}"
            for k in k_values:
                val = res.get(f"{metric}@{k}", 0)
                row += f"{val:<12.4f}"
            print(row)

    # Save results
    output_path = Path("outputs") / "exp1_baseline_results.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    import json
    main()

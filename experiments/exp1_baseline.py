"""Experiment 1: Baseline Comparison — Popularity, NCF."""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CONFIG
from src.base_model.popularity import PopularityRecommender
from src.base_model.usercf import UserCF
from src.ncf import NCF, train_ncf
from src.evaluate import evaluate_model, print_metrics


def main():
    parser = argparse.ArgumentParser(description="Experiment 1: Baseline Comparison")
    parser.add_argument(
        "--models", nargs="+", default=["popularity"],
        choices=["popularity", "usercf", "ncf", "all"],
        help="Which models to run (default: popularity). Use 'all' to run everything."
    )
    args = parser.parse_args()

    run_models = set(args.models)
    if "all" in run_models:
        run_models = {"popularity", "usercf", "ncf"}

    data_dir = Path(CONFIG["data"]["output_dir"])
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "val.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    with open(data_dir / "stats.json") as f:
        stats = json.load(f)

    n_users = stats["n_users"]
    n_items = stats["n_items"]

    print(f"  Dataset: {n_users} users, {n_items} items")
    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print(f"  Models to run: {sorted(run_models)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if "ncf" in run_models:
        print(f"  Device: {device}")

    all_results = {}

    # ── Popularity ──
    if "popularity" in run_models:
        print("\n" + "=" * 60)
        print("  Building Popularity baseline ...")
        t0 = time.time()
        pop_model = PopularityRecommender(train_df)
        metrics = evaluate_model(pop_model, test_df, train_df, n_items)
        print_metrics(metrics, "Popularity")
        all_results["Popularity"] = metrics
        print(f"  Time: {time.time() - t0:.1f}s")

    # ── UserCF ──
    if "usercf" in run_models:
        print("\n" + "=" * 60)
        print("  Building UserCF ...")
        t0 = time.time()
        k_neighbors = min(50, n_users - 1)
        usercf_model = UserCF(train_df, n_users, n_items, k_neighbors=k_neighbors)
        metrics = evaluate_model(usercf_model, test_df, train_df, n_items)
        print_metrics(metrics, "UserCF")
        all_results["UserCF"] = metrics
        print(f"  Time: {time.time() - t0:.1f}s")

    # ── NCF ──
    if "ncf" in run_models:
        print("\n" + "=" * 60)
        print("  Training NCF ...")
        t0 = time.time()
        ncf_model = NCF(
            n_users=n_users,
            n_items=n_items,
            embedding_dim=CONFIG["model"]["embedding_dim"],
            mlp_layers=CONFIG["model"]["ncf_mlp_layers"],
        ).to(device)

        ncf_model = train_ncf(ncf_model, train_df, val_df, CONFIG, n_items=n_items, device=device)
        metrics = evaluate_model(ncf_model, test_df, train_df, n_items)
        print_metrics(metrics, "NCF")
        all_results["NCF"] = metrics
        print(f"  Time: {time.time() - t0:.1f}s")

    if not all_results:
        print("  No models selected. Use --models popularity ncf or --models all")
        return

    # ── Summary table ──
    print("\n" + "=" * 60)
    print("  EXPERIMENT 1: BASELINE COMPARISON SUMMARY")
    print("=" * 60)

    first_result = next(iter(all_results.values()))
    k_values = sorted(set(int(k.split("@")[1]) for k in first_result.keys()))
    metric_names = ["Precision", "Recall", "HitRate", "MAP", "NDCG"]

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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path("outputs") / f"exp1_baseline_{timestamp}.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    main()

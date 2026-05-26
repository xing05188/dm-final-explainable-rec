"""Centralized configuration for the entire project."""

CONFIG = {
    "data": {
        "raw_file": "data/raw/Electronics_5.json.gz",
        "output_dir": "data/processed",
    },
    "fields": {
        "user_id": "reviewerID",
        "item_id": "asin",
        "rating": "overall",
        "review_text": "reviewText",
        "summary": "summary",
        "timestamp": "unixReviewTime",
    },
    "filter": {
        "k_core": 5,
        "n_users_sample": 5000,
        "n_items_target": 3000,
        "min_items_per_user": 3,
        "min_interactions_per_user": 5,
        "sample_active_users": True,
    },
    "negative_sampling": {
        "neg_ratio": 4,
        "resample_per_epoch": True,
    },
    "text": {
        "max_tokens": 128,
        "max_words_multiplier": 4,
        "max_words_per_review": 96,
        "truncate_mode": "head_tail",
        "review_sep": " . ",
        "sbert_model": "all-MiniLM-L6-v2",
        "batch_size": 64,
        "skip_sbert": False,
        "hf_hub_endpoint": "https://hf-mirror.com",
        "strip_prefixes": [
            "verified purchase",
            "amazon vine customer review of free product",
            "amazon vine customer review",
            "free product review",
            "received this product for free",
            "early reviewer program",
            "reviewed in the united states on",
            "originally posted on",
        ],
    },
    "model": {
        "embedding_dim": 64,
        "ncf_mlp_layers": [64, 32, 16],
        "lightgcn_layers": 3,
        "fusion_alpha": 0.3,
        "learning_rate": 0.001,
        "batch_size": 512,
        "epochs": 50,
        "early_stop_patience": 5,
    },
    "eval": {
        "top_k": [5, 10, 20],
    },
    "seed": 42,
}

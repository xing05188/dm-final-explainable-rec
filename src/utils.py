"""Utility functions: checkpoint management, logging, data validation."""

import json
import os
import re
import time
from collections import Counter
from html import unescape
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Compiled patterns for review text cleaning (domain noise, not stopwords)
_RE_HTML_TAG = re.compile(r"<[^>]+>", re.IGNORECASE)
_RE_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_RE_REPEAT_CHAR = re.compile(r"(.)\1{4,}")  # aaaaa -> a
_RE_SYMBOL_RUN = re.compile(r"[^\w\s]{5,}")
_RE_NON_ALNUM = re.compile(r"[^a-z0-9\s]")
_RE_WHITESPACE = re.compile(r"\s+")


def get_output_dir():
    from config import CONFIG
    return Path(CONFIG["data"]["output_dir"])


def checkpoint_path(stage_name: str) -> Path:
    return get_output_dir() / ".pipeline_state"


def load_checkpoint() -> set:
    p = checkpoint_path("")
    if p.exists():
        with open(p, "r") as f:
            return set(json.load(f).get("completed", []))
    return set()


def save_checkpoint(stage_name: str):
    p = checkpoint_path("")
    completed = load_checkpoint()
    completed.add(stage_name)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump({"completed": list(completed)}, f)


def should_skip(stage_name: str) -> bool:
    return stage_name in load_checkpoint()


def clean_review_text(text, strip_prefixes: Optional[list] = None) -> str:
    """Remove domain-specific noise (HTML, boilerplate prefixes, symbol runs).

    Does NOT remove general stopwords — those are handled only in KeywordExplainer.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    if strip_prefixes is None:
        from config import CONFIG
        strip_prefixes = CONFIG["text"].get("strip_prefixes", [])

    text = unescape(text)
    text = _RE_HTML_TAG.sub(" ", text)
    text = _RE_URL.sub(" ", text)
    text = text.lower()

    for prefix in strip_prefixes:
        if prefix:
            text = text.replace(prefix.lower(), " ")

    text = _RE_REPEAT_CHAR.sub(r"\1", text)
    text = _RE_SYMBOL_RUN.sub(" ", text)
    text = _RE_NON_ALNUM.sub(" ", text)
    text = _RE_WHITESPACE.sub(" ", text).strip()
    return text


def truncate_words(tokens: list, max_words: int, mode: str = "head_tail") -> str:
    """Truncate word list: head_tail keeps first/second half; head/tail only."""
    if max_words <= 0 or len(tokens) <= max_words:
        return " ".join(tokens)

    if mode == "head":
        return " ".join(tokens[:max_words])
    if mode == "tail":
        return " ".join(tokens[-max_words:])

    # head_tail (default): half from start, half from end
    half = max_words // 2
    head = tokens[:half]
    tail = tokens[-(max_words - half) :]
    return " ".join(head + tail)


def compose_review_snippet(
    review_text: str,
    summary: str,
    max_words_per_review: Optional[int] = None,
    truncate_mode: str = "head_tail",
) -> str:
    """Merge summary + body for one review; avoid duplicate summary text."""
    review = (review_text or "").strip()
    summary = (summary or "").strip()

    if not review and not summary:
        return ""

    if summary and review:
        # Summary is usually the one-line title; skip if already in body
        if summary in review or review in summary:
            text = review if len(review) >= len(summary) else summary
        else:
            text = f"{summary} {review}"
    else:
        text = review or summary

    tokens = text.split()
    if max_words_per_review and len(tokens) > max_words_per_review:
        text = truncate_words(tokens, max_words_per_review, truncate_mode)
    return text


def aggregate_review_texts(
    group: pd.DataFrame,
    max_words: int,
    max_words_per_review: Optional[int] = None,
    truncate_mode: str = "head_tail",
    review_sep: str = " . ",
) -> str:
    """Aggregate reviews for one user/item: time-ordered, deduped snippets, head_tail cap."""
    if group.empty:
        return ""

    g = group.sort_values("timestamp")
    snippets = []
    seen = set()

    for _, row in g.iterrows():
        snippet = compose_review_snippet(
            row.get("review_text", ""),
            row.get("summary", ""),
            max_words_per_review=max_words_per_review,
            truncate_mode=truncate_mode,
        )
        if not snippet:
            continue
        # Drop exact duplicate snippets (re-posted text)
        key = snippet[:200]
        if key in seen:
            continue
        seen.add(key)
        snippets.append(snippet)

    if not snippets:
        return ""

    combined = review_sep.join(snippets)
    tokens = combined.split()
    return truncate_words(tokens, max_words, truncate_mode)


def set_seed(seed: int = 42):
    np.random.seed(seed)


def sample_train_negatives(
    train_pos_df: pd.DataFrame,
    n_items: int,
    neg_ratio: int = 4,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Build train set with positives + negatives (1:neg_ratio per interaction row)."""
    if seed is not None:
        np.random.seed(seed)

    pos_df = train_pos_df[["user_id", "item_id"]].copy()
    pos_df["label"] = 1
    all_items = set(range(n_items))
    user_pos_items = pos_df.groupby("user_id")["item_id"].apply(set).to_dict()

    neg_pairs = []
    for user_id, group in pos_df.groupby("user_id"):
        neg_candidates = list(all_items - user_pos_items[user_id])
        if not neg_candidates:
            continue
        n_neg = len(group) * neg_ratio
        samples = np.random.choice(neg_candidates, size=n_neg, replace=True)
        for item_id in samples:
            neg_pairs.append((user_id, int(item_id), 0))

    neg_df = pd.DataFrame(neg_pairs, columns=["user_id", "item_id", "label"])
    out = pd.concat([pos_df, neg_df], ignore_index=True)
    return out.sample(frac=1, random_state=seed).reset_index(drop=True)


def ensure_binary_labels(df: pd.DataFrame, positive_label: float = 1.0) -> pd.DataFrame:
    """Ensure DataFrame has label column (for val positives)."""
    if "label" in df.columns:
        return df
    out = df[["user_id", "item_id"]].copy()
    out["label"] = positive_label
    return out


def print_section(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def print_df_info(df, name: str):
    print(f"\n  [{name}] shape: {df.shape}")
    print(f"  columns: {list(df.columns)}")
    if "rating" in df.columns:
        print(f"  rating distribution:\n{df['rating'].value_counts().sort_index().to_string()}")


def print_rating_distribution(
    counts: Counter,
    total: Optional[int] = None,
    name: str = "raw",
) -> None:
    """Print 1-5 star counts and percentages (streaming-friendly)."""
    if not counts:
        print(f"  [{name}] rating distribution: (empty)")
        return
    total = total or sum(counts.values())
    print(f"  [{name}] rating distribution:")
    for rating in sorted(counts.keys()):
        n = counts[rating]
        pct = 100.0 * n / total if total else 0.0
        print(f"    {rating}: {n:,} ({pct:.1f}%)")


def validate_dataframe(df, required_cols: list, name: str):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] missing columns: {missing}")
    if df.isnull().any().any():
        null_counts = df.isnull().sum()
        null_cols = null_counts[null_counts > 0]
        print(f"  WARNING [{name}] has null values:\n{null_cols}")

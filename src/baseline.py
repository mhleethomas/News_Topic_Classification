"""
baseline.py:TF-IDF (unigram + bigram) + multinomial Logistic Regression.

Reads the shared processed CSVs from data_preprocessing (text_lower, label_id).

Usage (from project root, venv activated):
    python src/baseline.py
    python src/baseline.py --smoke          # fast check on debug.csv only
    python src/baseline.py --help
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline

# Project root on path so `python src/baseline.py` works from repo root
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.data_preprocessing import ID_TO_LABEL, LABEL_MAP


# ──────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────


def processed_dir(base_dir: str) -> str:
    return os.path.join(base_dir, "data", "processed")


def outputs_dir(base_dir: str) -> str:
    return os.path.join(base_dir, "outputs")


# ──────────────────────────────────────────────
# LOAD
# ──────────────────────────────────────────────


def load_split(processed: str, name: str) -> pd.DataFrame:
    """Load train / val / test / debug CSV with columns label, label_id, text, text_lower."""
    path = os.path.join(processed, f"{name}.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing '{path}'. Run data_preprocessing or pull latest data.")
    df = pd.read_csv(path)
    needed = {"label_id", "text_lower"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    return df


# ──────────────────────────────────────────────
# MODEL
# ──────────────────────────────────────────────


def build_pipeline(
    C: float = 1.0,
    max_features: int | None = 50_000,
    min_df: int = 2,
    max_df: float = 0.95,
    ngram_max: int = 2,
) -> Pipeline:
    """TF-IDF word n-grams + logistic regression (5-class BBC)."""
    ngram_range = (1, ngram_max)
    tfidf = TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True,
        max_features=max_features,
    )
    clf = LogisticRegression(
        C=C,
        max_iter=2000,
        solver="lbfgs",
        random_state=42,
    )
    return Pipeline([("tfidf", tfidf), ("clf", clf)])


def report_split(
    name: str,
    y_true: pd.Series,
    y_pred,
    labels_sorted: list[int],
) -> None:
    target_names = [ID_TO_LABEL[i] for i in labels_sorted]
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", labels=labels_sorted)
    print(f"\n=== {name} ===")
    print(f"Accuracy     : {acc:.4f}")
    print(f"F1 (macro)   : {f1_macro:.4f}")
    print(classification_report(y_true, y_pred, labels=labels_sorted, target_names=target_names, digits=3))


def save_test_predictions(
    out_path: str,
    df_test: pd.DataFrame,
    y_pred,
) -> None:
    """Write test predictions for integration / error analysis (outputs/ is gitignored)."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out = pd.DataFrame(
        {
            "label": df_test["label"].values,
            "label_id": df_test["label_id"].values,
            "pred_label_id": y_pred,
            "pred_label": [ID_TO_LABEL[int(i)] for i in y_pred],
        }
    )
    out.to_csv(out_path, index=False)
    print(f"\n[save] Test predictions → '{out_path}'")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="TF-IDF + Logistic Regression baseline for BBC News.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Train on debug.csv only (50 rows); report val metrics. Fast pipeline check.",
    )
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Inverse regularization strength for LogisticRegression (default: 1.0).",
    )
    parser.add_argument(
        "--min-df",
        type=int,
        default=2,
        help="TfidfVectorizer min_df (use 1 for very small train sets).",
    )
    args = parser.parse_args()

    base_dir = _ROOT
    proc = processed_dir(base_dir)
    labels_sorted = sorted(LABEL_MAP.values())

    if args.smoke:
        train = load_split(proc, "debug")
        val = load_split(proc, "val")
        test = None
        # Tiny train set: allow unigrams that appear once
        min_df = 1
        print("[baseline] SMOKE MODE: training on debug.csv only (not for real scores).")
    else:
        train = load_split(proc, "train")
        val = load_split(proc, "val")
        test = load_split(proc, "test")
        min_df = args.min_df

    X_train = train["text_lower"].astype(str)
    y_train = train["label_id"].astype(int)
    X_val = val["text_lower"].astype(str)
    y_val = val["label_id"].astype(int)

    pipe = build_pipeline(C=args.C, min_df=min_df)
    print(f"[baseline] Fitting pipeline on {len(train)} train rows...")
    pipe.fit(X_train, y_train)

    y_val_pred = pipe.predict(X_val)
    report_split("Validation", y_val, y_val_pred, labels_sorted)

    if test is not None:
        X_test = test["text_lower"].astype(str)
        y_test = test["label_id"].astype(int)
        y_test_pred = pipe.predict(X_test)
        report_split("Test", y_test, y_test_pred, labels_sorted)
        pred_path = os.path.join(outputs_dir(base_dir), "baseline_test_predictions.csv")
        save_test_predictions(pred_path, test, y_test_pred)
    else:
        print("\n[baseline] Skipping test split in --smoke mode.")


if __name__ == "__main__":
    main()

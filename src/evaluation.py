"""
evaluation.py
task: Evaluation and error analysis.

Reads prediction CSVs from outputs/ (baseline) and outputs/bert/ (BERT),
computes metrics, plots confusion matrices, and saves error analysis.

Usage:
    python src/evaluation.py --model baseline
    python src/evaluation.py --model bert
    python src/evaluation.py --model all
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.data_preprocessing import ID_TO_LABEL, LABEL_MAP

LABELS_SORTED = sorted(LABEL_MAP.values())
TARGET_NAMES  = [ID_TO_LABEL[i] for i in LABELS_SORTED]

# ── paths ──────────────────────────────────────────────────────────────────

PRED_PATHS = {
    # baseline.py writes here
    "baseline": os.path.join(_ROOT, "outputs", "baseline_test_predictions.csv"),
    # bert_pipeline.py writes here
    "bert":     os.path.join(_ROOT, "outputs", "bert", "test_predictions.csv"),
}

OUTPUTS_DIR = os.path.join(_ROOT, "outputs")


# ── load ───────────────────────────────────────────────────────────────────

def load_predictions(model: str) -> pd.DataFrame:
    path = PRED_PATHS[model]
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"[{model}] Prediction file not found: '{path}'\n"
            f"  Run the {'baseline' if model == 'baseline' else 'bert_pipeline'} script first."
        )
    df = pd.read_csv(path)

    # both pipelines must have these columns
    required = {"label_id", "pred_label_id"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"[{model}] Prediction CSV missing columns: {missing}")

    df["label_id"]      = df["label_id"].astype(int)
    df["pred_label_id"] = df["pred_label_id"].astype(int)
    return df


# ── metrics ────────────────────────────────────────────────────────────────

def compute_metrics(df: pd.DataFrame, model_name: str) -> dict:
    y_true = df["label_id"]
    y_pred = df["pred_label_id"]

    acc         = accuracy_score(y_true, y_pred)
    f1_macro    = f1_score(y_true, y_pred, average="macro",    labels=LABELS_SORTED)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", labels=LABELS_SORTED)

    print(f"\n{'='*50}")
    print(f"  Model : {model_name}")
    print(f"{'='*50}")
    print(f"  Accuracy      : {acc:.4f}")
    print(f"  F1 (macro)    : {f1_macro:.4f}")
    print(f"  F1 (weighted) : {f1_weighted:.4f}")
    print()
    print(classification_report(
        y_true, y_pred,
        labels=LABELS_SORTED,
        target_names=TARGET_NAMES,
        digits=3,
    ))

    return {
        "model":       model_name,
        "accuracy":    round(acc, 4),
        "f1_macro":    round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
    }


# ── confusion matrix ───────────────────────────────────────────────────────

def plot_confusion_matrix(df: pd.DataFrame, model_name: str, out_dir: str) -> None:
    y_true = df["label_id"]
    y_pred = df["pred_label_id"]

    cm = confusion_matrix(y_true, y_pred, labels=LABELS_SORTED)

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=TARGET_NAMES,
        yticklabels=TARGET_NAMES,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"confusion_matrix_{model_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[save] Confusion matrix → '{path}'")


# ── error analysis ─────────────────────────────────────────────────────────

def error_analysis(df: pd.DataFrame, model_name: str, out_dir: str) -> None:
    wrong = df[df["label_id"] != df["pred_label_id"]].copy()
    wrong["true_label"]      = wrong["label_id"].map(ID_TO_LABEL)
    wrong["predicted_label"] = wrong["pred_label_id"].map(ID_TO_LABEL)

    total     = len(df)
    n_errors  = len(wrong)
    error_pct = n_errors / total * 100
    print(f"\n[error analysis] {model_name}: "
          f"{n_errors}/{total} misclassified ({error_pct:.1f}%)")

    # most confused category pairs
    pairs = (
        wrong.groupby(["true_label", "predicted_label"])
             .size()
             .reset_index(name="count")
             .sort_values("count", ascending=False)
    )
    print("\nMost confused category pairs:")
    print(pairs.head(5).to_string(index=False))

    # BERT adds pred_confidence — show low-confidence errors too
    if "pred_confidence" in wrong.columns:
        low_conf = wrong.sort_values("pred_confidence").head(5)
        print(f"\nLowest-confidence errors ({model_name}):")
        cols = ["true_label", "predicted_label", "pred_confidence"]
        if "text" in low_conf.columns:
            cols.append("text")
        print(low_conf[cols].to_string(index=False))

    # save full error list
    os.makedirs(out_dir, exist_ok=True)
    error_path = os.path.join(out_dir, f"errors_{model_name}.csv")
    wrong.to_csv(error_path, index=False)
    print(f"[save] All error examples → '{error_path}'")

    # save confused pairs summary
    pairs_path = os.path.join(out_dir, f"confused_pairs_{model_name}.csv")
    pairs.to_csv(pairs_path, index=False)
    print(f"[save] Confused pairs     → '{pairs_path}'")


# ── comparison ─────────────────────────────────────────────────────────────

def save_comparison(results: list[dict], out_dir: str) -> None:
    """Save side-by-side metrics table for Cindy's integration step."""
    summary = pd.DataFrame(results)
    path = os.path.join(out_dir, "metrics_comparison.csv")
    summary.to_csv(path, index=False)
    print(f"\n[save] Metrics comparison → '{path}'")
    print("\n" + summary.to_string(index=False))


# ── main ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate baseline and/or BERT predictions on BBC News test set."
    )
    parser.add_argument(
        "--model",
        choices=["baseline", "bert", "all"],
        default="all",
        help="Which model to evaluate (default: all).",
    )
    args = parser.parse_args()

    models  = ["baseline", "bert"] if args.model == "all" else [args.model]
    results = []

    for model in models:
        try:
            df = load_predictions(model)
        except FileNotFoundError as e:
            print(f"\n[skip] {e}")
            continue

        metrics = compute_metrics(df, model)
        results.append(metrics)
        plot_confusion_matrix(df, model, OUTPUTS_DIR)
        error_analysis(df, model, OUTPUTS_DIR)

    if len(results) > 1:
        save_comparison(results, OUTPUTS_DIR)
    elif len(results) == 0:
        print("\n[done] No prediction files found. Nothing to evaluate.")


if __name__ == "__main__":
    main()
"""
data_preprocessing.py
Thomas's task: Data preprocessing and dataset preparation for AG News.

AG News ships with a pre-defined train (120,000) and test (7,600) split.
We carve a validation set out of the train split (stratified).

Outputs (data/processed/):
    train.csv   — ~85 % of AG News train, stratified (~102,000 rows)
    val.csv     — ~15 % of AG News train, stratified (~18,000 rows)
    test.csv    — full AG News test split (7,600 rows)
    debug.csv   — 10 samples per class (40 total) for fast smoke-testing

Shared column schema (all four files):
    label       str   category name  e.g. "Sports"
    label_id    int   0-3, stable mapping defined in LABEL_MAP below
    text        str   cleaned text, original casing  → for BERT (Xinyan)
    text_lower  str   lowercased text                → for n-gram/LR (Ruoxuan)
"""

import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split


# ──────────────────────────────────────────────
# SHARED LABEL MAP  (import this in other scripts)
# ──────────────────────────────────────────────

LABEL_MAP: dict[str, int] = {
    "Business": 0,
    "Sci/Tech": 1,
    "Sports":   2,
    "World":    3,
}
# Reverse lookup: id → name
ID_TO_LABEL: dict[int, str] = {v: k for k, v in LABEL_MAP.items()}


# ──────────────────────────────────────────────
# 1. LOAD
# ──────────────────────────────────────────────

def load_ag_news_csv(csv_path: str) -> pd.DataFrame:
    """Load an AG News CSV file (columns: text, label)."""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    missing = {"label", "text"} - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}. Found: {list(df.columns)}")

    return df[["label", "text"]].copy()


def load_dataset(raw_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load AG News train and test splits from raw CSVs.

    Returns:
        (train_df, test_df) — both with columns [label, text]
    """
    train_path = os.path.join(raw_dir, "ag_news_train.csv")
    test_path = os.path.join(raw_dir, "ag_news_test.csv")

    missing = [p for p in (train_path, test_path) if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(
            f"AG News raw files not found: {missing}\n"
            "Run:  python src/download_data.py"
        )

    print(f"[load] Found: {train_path}")
    print(f"[load] Found: {test_path}")
    return load_ag_news_csv(train_path), load_ag_news_csv(test_path)


# ──────────────────────────────────────────────
# 2. CLEAN
# ──────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Lightweight cleaning compatible with both LR and BERT pipelines.

    - Remove non-ASCII characters
    - Collapse all whitespace (tabs, newlines) into single spaces
    - Strip leading/trailing whitespace
    Original casing is preserved (BERT is casing-sensitive).
    """
    text = text.encode("ascii", errors="ignore").decode("ascii")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Clean text, normalize labels, and add label_id + text_lower columns."""
    df = df.copy()

    df = df.dropna(subset=["label", "text"])
    df = df.drop_duplicates(subset=["text"])

    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"].str.len() > 0]

    df["label"] = df["label"].str.strip()

    # Drop any rows whose label isn't in our expected set
    unknown = ~df["label"].isin(LABEL_MAP)
    if unknown.any():
        print(f"[clean] Dropping {unknown.sum()} rows with unknown labels: "
              f"{df.loc[unknown, 'label'].unique()}")
        df = df[~unknown]

    # Standardized integer label for model training
    df["label_id"] = df["label"].map(LABEL_MAP)

    # Lowercased version for n-gram/LR baseline (Ruoxuan)
    df["text_lower"] = df["text"].str.lower()

    return df[["label", "label_id", "text", "text_lower"]].reset_index(drop=True)


# ──────────────────────────────────────────────
# 3. SPLIT
# ──────────────────────────────────────────────

def split_train_val(
    train_df: pd.DataFrame,
    val_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carve a stratified validation set out of the AG News train split.

    AG News provides its own test split, so we only split train → train/val.
    Default: 85 / 15 split of the 120,000-row train set.
    """
    train, val = train_test_split(
        train_df,
        test_size=val_size,
        random_state=random_state,
        stratify=train_df["label"],
    )
    return train.reset_index(drop=True), val.reset_index(drop=True)


# ──────────────────────────────────────────────
# 4. DEBUG SUBSET
# ──────────────────────────────────────────────

def make_debug_subset(
    train: pd.DataFrame,
    n_per_class: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Sample n_per_class examples per label from train for fast smoke-testing.

    Default: 10 × 4 classes = 40 rows.
    Sampled only from train so test/val stay untouched.
    """
    groups = [
        g.sample(min(n_per_class, len(g)), random_state=random_state)
        for _, g in train.groupby("label")
    ]
    return pd.concat(groups).reset_index(drop=True)


# ──────────────────────────────────────────────
# 5. SAVE
# ──────────────────────────────────────────────

def save_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    debug: pd.DataFrame,
    out_dir: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    train.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    val.to_csv(os.path.join(out_dir, "val.csv"), index=False)
    test.to_csv(os.path.join(out_dir, "test.csv"), index=False)
    debug.to_csv(os.path.join(out_dir, "debug.csv"), index=False)
    print(f"[save] Splits written to '{out_dir}':")
    print(f"       train={len(train)}  val={len(val)}  test={len(test)}  debug={len(debug)}")


# ──────────────────────────────────────────────
# 6. SUMMARY
# ──────────────────────────────────────────────

def print_summary(
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    total = len(train_raw) + len(test_raw)
    print("\n=== Dataset Summary (AG News) ===")
    print(f"Raw   : train={len(train_raw)}  test={len(test_raw)}  total={total}")
    print(f"Output: train={len(train)}  val={len(val)}  test={len(test)}")

    print("\nLabel map (label_id):  ", LABEL_MAP)

    print("\nTrain split distribution:")
    for label, count in train["label"].value_counts().sort_index().items():
        print(f"  {label:<12} id={LABEL_MAP[label]}  n={count:>6}  ({count/len(train)*100:.1f}%)")

    print("\nPer-split distribution:")
    for name, split in [("train", train), ("val", val), ("test", test)]:
        dist = split["label"].value_counts().sort_index()
        print(f"  [{name:5}] " + "  ".join(f"{k}: {v}" for k, v in dist.items()))

    avg_words = train["text"].apply(lambda x: len(x.split())).mean()
    print(f"\nAvg text length (train) : {avg_words:.0f} words")
    print("=================================\n")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(base_dir, "data", "raw")
    processed_dir = os.path.join(base_dir, "data", "processed", "agnews")

    train_raw, test_raw = load_dataset(raw_dir)
    print(f"[load] train={len(train_raw)}  test={len(test_raw)}  categories={train_raw['label'].nunique()}")

    train_clean = preprocess(train_raw)
    test_clean = preprocess(test_raw)
    print(f"[clean] train={len(train_clean)}  test={len(test_clean)} after cleaning.")

    train, val = split_train_val(train_clean)
    debug = make_debug_subset(train)

    save_splits(train, val, test_clean, debug, processed_dir)
    print_summary(train_raw, test_raw, train, val, test_clean)


if __name__ == "__main__":
    main()

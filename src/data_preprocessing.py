"""
data_preprocessing.py
Thomas's task: Data preprocessing and dataset preparation for BBC News.

Outputs (data/processed/):
    train.csv   — 70 % of data, stratified
    val.csv     — 15 % of data, stratified
    test.csv    — 15 % of data, stratified
    debug.csv   — 10 samples per class (50 total) for fast smoke-testing

Shared column schema (all four files):
    label       str   category name  e.g. "sport"
    label_id    int   0-4, stable mapping defined in LABEL_MAP below
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
    "business":      0,
    "entertainment": 1,
    "politics":      2,
    "sport":         3,
    "tech":          4,
}
# Reverse lookup: id → name
ID_TO_LABEL: dict[int, str] = {v: k for k, v in LABEL_MAP.items()}


# ──────────────────────────────────────────────
# 1. LOAD
# ──────────────────────────────────────────────

def load_bbc_csv(csv_path: str) -> pd.DataFrame:
    """Load BBC News from a CSV file.

    Accepts:
      - columns [category, text]  (common Kaggle/HuggingFace export)
      - columns [label, text]
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()

    rename_map = {}
    if "category" in df.columns:
        rename_map["category"] = "label"
    for alt in ["article", "content", "body"]:
        if alt in df.columns and "text" not in df.columns:
            rename_map[alt] = "text"
            break

    df = df.rename(columns=rename_map)

    missing = {"label", "text"} - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}. Found: {list(df.columns)}")

    return df[["label", "text"]].copy()


def load_bbc_folder(folder_path: str) -> pd.DataFrame:
    """Load BBC News from the raw folder layout.

    data/raw/bbc/<category>/<article_id>.txt
    """
    records = []
    for category in os.listdir(folder_path):
        cat_dir = os.path.join(folder_path, category)
        if not os.path.isdir(cat_dir):
            continue
        for fname in os.listdir(cat_dir):
            if not fname.endswith(".txt"):
                continue
            fpath = os.path.join(cat_dir, fname)
            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
            records.append({"label": category, "text": text})

    if not records:
        raise FileNotFoundError(
            f"No .txt files found under '{folder_path}'."
        )
    return pd.DataFrame(records)


def load_dataset(raw_dir: str) -> pd.DataFrame:
    """Auto-detect and load the BBC News dataset.

    Priority:
      1. data/raw/bbc-text.csv   (produced by download_data.py)
      2. data/raw/bbc/           (raw folder layout)
    """
    csv_path = os.path.join(raw_dir, "bbc-text.csv")
    bbc_folder = os.path.join(raw_dir, "bbc")

    if os.path.isfile(csv_path):
        print(f"[load] Found CSV: {csv_path}")
        return load_bbc_csv(csv_path)
    elif os.path.isdir(bbc_folder):
        print(f"[load] Found folder: {bbc_folder}")
        return load_bbc_folder(bbc_folder)
    else:
        raise FileNotFoundError(
            f"No BBC News data found in '{raw_dir}'.\n"
            "Run:  python src/download_data.py"
        )


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

    df["label"] = df["label"].str.strip().str.lower()

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

def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified 70 / 15 / 15 train / val / test split."""
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"],
    )
    relative_val = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=relative_val,
        random_state=random_state,
        stratify=train_val["label"],
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


# ──────────────────────────────────────────────
# 4. DEBUG SUBSET
# ──────────────────────────────────────────────

def make_debug_subset(
    train: pd.DataFrame,
    n_per_class: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Sample n_per_class examples per label from train for fast smoke-testing.

    Default: 10 × 5 classes = 50 rows.
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
    df: pd.DataFrame,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    print("\n=== Dataset Summary ===")
    print(f"Total : {len(df)}  |  Train : {len(train)}  |  Val : {len(val)}  |  Test : {len(test)}")

    print("\nLabel map (label_id):  ", LABEL_MAP)

    print("\nFull dataset distribution:")
    for label, count in df["label"].value_counts().sort_index().items():
        print(f"  {label:<16} id={LABEL_MAP[label]}  n={count:>4}  ({count/len(df)*100:.1f}%)")

    print("\nPer-split distribution:")
    for name, split in [("train", train), ("val", val), ("test", test)]:
        dist = split["label"].value_counts().sort_index()
        print(f"  [{name:5}] " + "  ".join(f"{k}: {v}" for k, v in dist.items()))

    avg_words = df["text"].apply(lambda x: len(x.split())).mean()
    print(f"\nAvg article length : {avg_words:.0f} words")
    print("=======================\n")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(base_dir, "data", "raw")
    processed_dir = os.path.join(base_dir, "data", "processed")

    df = load_dataset(raw_dir)
    print(f"[load] {len(df)} articles, {df['label'].nunique()} categories.")

    df = preprocess(df)
    print(f"[clean] {len(df)} articles after cleaning.")

    train, val, test = split_dataset(df)
    debug = make_debug_subset(train)

    save_splits(train, val, test, debug, processed_dir)
    print_summary(df, train, val, test)


if __name__ == "__main__":
    main()

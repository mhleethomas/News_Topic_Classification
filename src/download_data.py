"""
download_data.py
Download the AG News dataset via Hugging Face datasets library.

Usage:
    pip install datasets
    python src/download_data.py
"""

import os


# AG News label mapping (alphabetical order used by HuggingFace ClassLabel)
AGNEWS_LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]


def download_ag_news(raw_dir: str) -> None:
    os.makedirs(raw_dir, exist_ok=True)
    train_path = os.path.join(raw_dir, "ag_news_train.csv")
    test_path = os.path.join(raw_dir, "ag_news_test.csv")

    if os.path.isfile(train_path) and os.path.isfile(test_path):
        print(f"[download] Already present: '{train_path}' and '{test_path}'. Skipping.")
        return

    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "Run: pip install datasets\n"
            "Then re-run this script."
        )

    print("[download] Fetching AG News from Hugging Face (fancyzhx/ag_news)...")
    ds = load_dataset("fancyzhx/ag_news")

    for split_name, out_path in [("train", train_path), ("test", test_path)]:
        split = ds[split_name]
        df = split.to_pandas()

        label_feature = split.features["label"]
        if hasattr(label_feature, "names"):
            label_names = label_feature.names
        else:
            label_names = AGNEWS_LABEL_NAMES

        df["label"] = df["label"].apply(lambda i: label_names[int(i)])
        df = df[["text", "label"]]

        df.to_csv(out_path, index=False)
        print(f"[download] Saved {len(df)} articles → '{out_path}'")

    print(f"[download] Categories: {AGNEWS_LABEL_NAMES}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(base_dir, "data", "raw")
    download_ag_news(raw_dir)

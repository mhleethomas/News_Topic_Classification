"""
download_data.py
Download the BBC News dataset for free via Hugging Face datasets library.

Usage:
    pip install datasets
    python src/download_data.py
"""

import os


def download_bbc_news(raw_dir: str) -> None:
    os.makedirs(raw_dir, exist_ok=True)
    out_path = os.path.join(raw_dir, "bbc-text.csv")

    if os.path.isfile(out_path):
        print(f"[download] Already present: '{out_path}'. Skipping.")
        return

    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "Run: pip install datasets\n"
            "Then re-run this script."
        )

    print("[download] Fetching BBC News from Hugging Face (SetFit/bbc-news)...")
    ds = load_dataset("SetFit/bbc-news", split="train")

    df = ds.to_pandas()

    # Map integer labels to category names.
    # SetFit/bbc-news uses ClassLabel with names in alphabetical order:
    #   0=business, 1=entertainment, 2=politics, 3=sport, 4=tech
    label_feature = ds.features["label"]
    if hasattr(label_feature, "names") and not all(
        n.isdigit() for n in label_feature.names
    ):
        # Proper ClassLabel names are available
        label_names = label_feature.names
        df["label"] = df["label"].apply(lambda i: label_names[int(i)])
    else:
        # Fallback: use the known BBC News alphabetical ordering
        id_to_label = {
            0: "business",
            1: "entertainment",
            2: "politics",
            3: "sport",
            4: "tech",
        }
        df["label"] = df["label"].apply(lambda i: id_to_label[int(i)])

    # Keep only the two columns we need
    df = df[["text", "label"]]

    df.to_csv(out_path, index=False)
    print(f"[download] Saved {len(df)} articles → '{out_path}'")
    print(f"[download] Categories: {sorted(df['label'].unique())}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(base_dir, "data", "raw")
    download_bbc_news(raw_dir)

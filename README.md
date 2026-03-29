# News Topic Classification — Group 2

Comparing n-gram + Logistic Regression vs. BERT/DistilBERT on the BBC News dataset.

---

## Project Structure

```
News_Topic_Classification/
├── data/
│   ├── raw/
│   │   └── bbc-text.csv          ← committed to repo, ready to use
│   └── processed/                ← committed to repo, ready to use
│       ├── train.csv             ← 835 rows  (70%)
│       ├── val.csv               ← 179 rows  (15%)
│       ├── test.csv              ← 180 rows  (15%)
│       └── debug.csv             ←  50 rows  (10 per class, for smoke-testing)
├── src/
│   ├── download_data.py          ← Thomas: download BBC News from HuggingFace (free)
│   ├── data_preprocessing.py     ← Thomas: clean, split, save, defines LABEL_MAP
│   ├── baseline.py               ← Ruoxuan: TF-IDF n-gram + Logistic Regression
│   ├── bert_pipeline.py          ← Xinyan: BERT / DistilBERT fine-tuning
│   └── evaluation.py             ← Meiling: metrics + error analysis
├── requirements.txt
└── README.md
```

---

## Team Responsibilities

| Member  | Task |
|---------|------|
| Thomas  | Data download, preprocessing, shared data format |
| Ruoxuan | Baseline pipeline — TF-IDF n-gram + Logistic Regression |
| Xinyan  | Neural pipeline — BERT / DistilBERT fine-tuning |
| Meiling | Evaluation & error analysis |
| Cindy   | Integration, comparison, results organization |

---

## Quick Start

```bash
# Requires Python 3.9+
# Install dependencies — that's all, data is already in the repo
pip install -r requirements.txt
```

All data files are committed and ready to use in `data/processed/`.
No download or preprocessing step needed.

---

## Dataset

**Source:** BBC News, downloaded for free via [Hugging Face](https://huggingface.co/datasets/SetFit/bbc-news) (`SetFit/bbc-news`).

**Size after cleaning:** 1,194 articles across 5 categories.

| Category      | Total | Train | Val | Test |
|---------------|-------|-------|-----|------|
| business      |  196  |  137  |  29 |  30  |
| entertainment |  282  |  197  |  43 |  42  |
| politics      |  274  |  192  |  41 |  41  |
| sport         |  206  |  144  |  31 |  31  |
| tech          |  236  |  165  |  35 |  36  |
| **Total**     | **1,194** | **835** | **179** | **180** |

- Split ratio: **70 / 15 / 15** (train / val / test)
- Split is **stratified** — each split has the same class proportions
- `random_state=42` — reproducible, same split every time
- Average article length: ~395 words

---

## Shared Data Format

All four CSV files (`train`, `val`, `test`, `debug`) have the **same four columns:**

| Column | Type | Description | Used by |
|--------|------|-------------|---------|
| `label` | string | Category name, lowercase | everyone |
| `label_id` | int | Integer encoding of the label (see map below) | Ruoxuan, Xinyan |
| `text` | string | Cleaned article text, **original casing** | Xinyan (BERT) |
| `text_lower` | string | Same text, **lowercased** | Ruoxuan (n-gram) |

**Label map** (stable, alphabetical order):

| label | label_id |
|-------|----------|
| business | 0 |
| entertainment | 1 |
| politics | 2 |
| sport | 3 |
| tech | 4 |

To import the label map in your script:
```python
from src.data_preprocessing import LABEL_MAP, ID_TO_LABEL
# LABEL_MAP:   {"business": 0, "entertainment": 1, ...}
# ID_TO_LABEL: {0: "business", 1: "entertainment", ...}
```

---

## debug.csv — What It Is and How to Use It

`debug.csv` is a **tiny 50-row subset** (exactly 10 articles per class) sampled from the training set.

**Purpose:** Use it to verify your pipeline works before running on the full dataset.
- Ruoxuan: test that your TF-IDF vectorizer and LR classifier can fit and predict
- Xinyan: test that your tokenizer and model forward pass run without errors
- Meiling: test that your metrics and confusion matrix code produces output
- Saves time — full training can take minutes; debug runs in seconds

**How to load it:**
```python
import pandas as pd

debug = pd.read_csv("data/processed/debug.csv")
train = pd.read_csv("data/processed/train.csv")
val   = pd.read_csv("data/processed/val.csv")
test  = pd.read_csv("data/processed/test.csv")
```

**Example — switch between debug and full data in one line:**
```python
# During development: use debug for fast iteration
data = pd.read_csv("data/processed/debug.csv")

# When ready: swap to full training set
data = pd.read_csv("data/processed/train.csv")
```

**Important:** `debug.csv` is sampled only from `train.csv`. The `val.csv` and `test.csv`
rows are never in `debug.csv`, so your evaluation stays clean.

---

## Regenerating the Data

If you need to re-run preprocessing from scratch:

```bash
python src/download_data.py       # re-downloads raw data
python src/data_preprocessing.py  # re-generates all processed CSVs
```

The output is deterministic (`random_state=42`) — you will get the exact same splits every time.

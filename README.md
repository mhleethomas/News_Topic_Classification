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
│   ├── download_data.py          ← Ming-Hsiang: download BBC News from HuggingFace (free)
│   ├── data_preprocessing.py     ← Ming-Hsiang: clean, split, save, defines LABEL_MAP
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
| Ming-Hsiang  | Data download, preprocessing, shared data format |
| Ruoxuan | Baseline pipeline — TF-IDF n-gram + Logistic Regression |
| Xinyan  | Neural pipeline — BERT / DistilBERT fine-tuning |
| Meiling | Evaluation & error analysis |
| Tzu-Chieh   | Integration, comparison, results organization |

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
## Baseline pipeline 

**Scope:** Classical topic **classification** only —  the baseline **consumes** the shared processed CSVs and runs end-to-end through **predictions**.

| Deliverable | Where it is |
|-------------|-------------|
| Preprocessed text for n-grams | Column `text_lower` in `data/processed/*.csv` |
| Unigram + bigram **TF-IDF** + **Logistic Regression** | [`src/baseline.py`](src/baseline.py) (`TfidfVectorizer` `ngram_range=(1, 2)`, `Pipeline` + `LogisticRegression`) |
| **Baseline predictions** | After a full run: `outputs/baseline_test_predictions.csv` (true vs predicted `label` / `label_id`; `outputs/` is gitignored by default) |

**Run** (from repo root, venv activated):

```bash
python src/baseline.py              # train on train.csv → metrics on val & test → save predictions
python src/baseline.py --smoke      # train on debug.csv only → val metrics (quick sanity check)
```

**Expected ballpark** (BBC splits in repo; exact numbers depend slightly on sklearn version): validation accuracy ~0.95+, test accuracy ~0.97+.

---

## BERT pipeline

**Scope:** Fine-tune a Hugging Face **BERT** model for the same 5-way topic classification. The pipeline reads the shared processed CSVs and tokenizes the **`text`** column (original casing).

| Deliverable | Location |
|-------------|----------|
| Training / inference | [`src/bert_pipeline.py`](src/bert_pipeline.py) (`BertForSequenceClassification`, `BertTokenizerFast`) |
| **Test predictions** | `outputs/bert/test_predictions.csv` |
| Other outputs | `outputs/bert/val_predictions.csv`, `training_history.csv`, `metrics_summary.csv` |

The `outputs/` directory is gitignored by default; run the script locally to generate files.

**Run** (from repo root, venv activated; requires PyTorch + GPU/MPS recommended):

```bash
python src/bert_pipeline.py
python src/bert_pipeline.py --use-debug --epochs 1 --max-length 128   # smoke: train on debug.csv only
```

**Prediction CSV columns** (test/val): includes `label`, `label_id`, `text`, `pred_label_id`, `pred_label`, `pred_confidence`, and `row_id`. Downstream evaluation only requires `label_id` and `pred_label_id`.

---

## Evaluation

**Scope:** Load saved **test** predictions from the baseline and BERT runs, recompute metrics, plot confusion matrices, and export error-analysis tables. Implemented in [`src/evaluation.py`](src/evaluation.py).

**Prerequisites:** Generate predictions first:

1. `python src/baseline.py` → `outputs/baseline_test_predictions.csv`
2. `python src/bert_pipeline.py` → `outputs/bert/test_predictions.csv`

**Run:**

```bash
python src/evaluation.py --model all       # baseline + BERT (default)
python src/evaluation.py --model baseline
python src/evaluation.py --model bert
```

**Outputs** (under `outputs/`; `outputs/` is gitignored by default):

| File | Description |
|------|-------------|
| `metrics_comparison.csv` | Side-by-side accuracy and F1 (macro / weighted) when both models are evaluated |
| `confusion_matrix_baseline.png` | Confusion matrix for the baseline |
| `confusion_matrix_bert.png` | Confusion matrix for BERT |
| `errors_<model>.csv` | Misclassified rows |
| `confused_pairs_<model>.csv` | Counts of (true label, predicted label) for errors |

The script prints classification reports to the terminal. If a prediction file is missing, that model is skipped with a message.

---

## Regenerating the Data

If you need to re-run preprocessing from scratch:

```bash
python src/download_data.py       # re-downloads raw data
python src/data_preprocessing.py  # re-generates all processed CSVs
```

The output is deterministic (`random_state=42`) — you will get the exact same splits every time.

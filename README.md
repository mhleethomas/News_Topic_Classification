# News Topic Classification — Group 2

Comparing n-gram + Logistic Regression vs. BERT on BBC News and AG News datasets.

---

## ⚠️ Branch `feature/agnews-dataset` — Pending Changes

> **Delete this section after merging to main.**

### What changed
- Dataset switched from BBC News → **AG News** (127,600 rows, 4 classes)
- Processed data reorganised: `data/processed/bbc/` and `data/processed/agnews/`
- `LABEL_MAP` updated to 4 classes (Business / Sci/Tech / Sports / World)
- `baseline.py` and `bert_pipeline.py` default paths updated to `data/processed/bbc/`

### Action required before merge
| Who | What |
|-----|------|
| Ruoxuan | Update path in `baseline.py` to `data/processed/agnews/`, run and confirm results |
| Xinyan | Update paths + set `num_labels=4` + use `--max-length 128`, run and confirm |
| Meiling | No changes needed — verify evaluation runs after others finish |
| Tzu-Chieh | Confirm outputs look correct, approve PR |

### Merge to main when
- [ ] Baseline runs on AG News without errors
- [ ] BERT runs at least 1 epoch without errors
- [ ] At least one teammate approves the PR on GitHub

---

## Project Structure

```
News_Topic_Classification/
├── data/
│   ├── raw/
│   │   ├── bbc-text.csv            ← BBC News raw (committed)
│   │   ├── ag_news_train.csv       ← AG News raw train 120,000 rows (committed)
│   │   └── ag_news_test.csv        ← AG News raw test  7,600 rows  (committed)
│   └── processed/
│       ├── bbc/                    ← BBC News processed splits (committed)
│       │   ├── train.csv           ←   835 rows (70%)
│       │   ├── val.csv             ←   179 rows (15%)
│       │   ├── test.csv            ←   180 rows (15%)
│       │   └── debug.csv           ←    50 rows (10 per class)
│       └── agnews/                 ← AG News processed splits (committed)
│           ├── train.csv           ← 102,000 rows (85% of official train)
│           ├── val.csv             ←  18,000 rows (15% of official train)
│           ├── test.csv            ←   7,600 rows (official test, unchanged)
│           └── debug.csv           ←      40 rows (10 per class)
├── src/
│   ├── download_data.py            ← Ming-Hsiang: download AG News from HuggingFace
│   ├── data_preprocessing.py       ← Ming-Hsiang: clean, split, save, defines LABEL_MAP
│   ├── baseline.py                 ← Ruoxuan: TF-IDF n-gram + Logistic Regression
│   ├── bert_pipeline.py            ← Xinyan: BERT fine-tuning
│   └── evaluation.py               ← Meiling: metrics + error analysis
├── requirements.txt
└── README.md
```

---

## Team Responsibilities

| Member | Task |
|--------|------|
| Ming-Hsiang | Data download, preprocessing, shared data format |
| Ruoxuan | Baseline pipeline — TF-IDF n-gram + Logistic Regression |
| Xinyan | Neural pipeline — BERT fine-tuning |
| Meiling | Evaluation & error analysis |
| Tzu-Chieh | Integration, comparison, results organization |

---

## Quick Start

```bash
# Requires Python 3.9+
pip install -r requirements.txt
```

All data files are committed and ready to use — no download or preprocessing needed.

---

## Datasets

### BBC News (`data/processed/bbc/`)

**Source:** [`SetFit/bbc-news`](https://huggingface.co/datasets/SetFit/bbc-news) via Hugging Face.

**Size after cleaning:** 1,194 articles across 5 categories, ~395 words avg.

| Category | Total | Train | Val | Test |
|----------|-------|-------|-----|------|
| business | 196 | 137 | 29 | 30 |
| entertainment | 282 | 197 | 43 | 42 |
| politics | 274 | 192 | 41 | 41 |
| sport | 206 | 144 | 31 | 31 |
| tech | 236 | 165 | 35 | 36 |
| **Total** | **1,194** | **835** | **179** | **180** |

- Split ratio: **70 / 15 / 15** (train / val / test), stratified, `random_state=42`

---

### AG News (`data/processed/agnews/`)

**Source:** [`fancyzhx/ag_news`](https://huggingface.co/datasets/fancyzhx/ag_news) via Hugging Face.

**Size:** 127,600 articles across 4 categories, ~38 words avg (headline + description).

| Category | Train | Val | Test |
|----------|-------|-----|------|
| Business | 25,500 | 4,500 | 1,900 |
| Sci/Tech | 25,500 | 4,500 | 1,900 |
| Sports | 25,500 | 4,500 | 1,900 |
| World | 25,500 | 4,500 | 1,900 |
| **Total** | **102,000** | **18,000** | **7,600** |

**Split logic:**

AG News provides an official `train` (120,000) and `test` (7,600). The official test is kept as-is to allow comparison with published results. A validation set is carved from the train split only.

```
AG News raw train (120,000)
    └── split 85 / 15  (random_state=42, stratified)
            ├── train.csv  (102,000 rows)
            └── val.csv    ( 18,000 rows)

AG News raw test (7,600)
    └── used as-is → test.csv  (7,600 rows)

train.csv — 10 rows per class → debug.csv (40 rows)
```

Val and test are intentionally different sizes — val monitors training, test is for final reporting.

---

## Shared Data Format

All CSV files (`train`, `val`, `test`, `debug`) share the same four columns:

| Column | Type | Description | Used by |
|--------|------|-------------|---------|
| `label` | string | Category name | everyone |
| `label_id` | int | Integer encoding (see map below) | Ruoxuan, Xinyan |
| `text` | string | Cleaned text, original casing | Xinyan (BERT) |
| `text_lower` | string | Same text, lowercased | Ruoxuan (n-gram) |

**Label map — BBC News:**

| label | label_id |
|-------|----------|
| business | 0 |
| entertainment | 1 |
| politics | 2 |
| sport | 3 |
| tech | 4 |

**Label map — AG News:**

| label | label_id |
|-------|----------|
| Business | 0 |
| Sci/Tech | 1 |
| Sports | 2 |
| World | 3 |

To import in your script:
```python
from src.data_preprocessing import LABEL_MAP, ID_TO_LABEL
```

---

## debug.csv — What It Is and How to Use It

A tiny subset (10 articles per class) sampled from `train.csv` only — val and test rows are never included.

**AG News debug:** 40 rows (4 classes × 10)
**BBC News debug:** 50 rows (5 classes × 10)

Use it to verify your pipeline runs before committing to a full training run.

```python
import pandas as pd

# AG News
train = pd.read_csv("data/processed/agnews/train.csv")
val   = pd.read_csv("data/processed/agnews/val.csv")
test  = pd.read_csv("data/processed/agnews/test.csv")
debug = pd.read_csv("data/processed/agnews/debug.csv")

# BBC News
train = pd.read_csv("data/processed/bbc/train.csv")
```

---

## Baseline Pipeline

TF-IDF (unigram + bigram) + Logistic Regression. Reads `text_lower` and `label_id` from the processed CSVs.

```bash
python src/baseline.py              # full run → val + test metrics + predictions
python src/baseline.py --smoke      # train on debug.csv only (quick check)
```

Outputs: `outputs/baseline_test_predictions.csv`

Default paths point to `data/processed/bbc/`. To run on AG News:
```bash
# baseline.py reads from processed_dir — update the path in the script or pass splits manually
```

---

## BERT Pipeline

Fine-tunes `bert-base-uncased` for 4-class (AG News) or 5-class (BBC) classification. Reads `text` (original casing) and `label_id`.

```bash
python src/bert_pipeline.py
python src/bert_pipeline.py --use-debug --epochs 1 --max-length 128   # smoke test
```

**Note for Xinyan:** AG News text is short (~38 words). Recommended `--max-length 128` (vs 256 for BBC). Also update `--train-path`, `--val-path`, `--test-path` to point to `data/processed/agnews/`.

Outputs: `outputs/bert/test_predictions.csv`, `val_predictions.csv`, `training_history.csv`, `metrics_summary.csv`

---

## Evaluation

Reads prediction CSVs from `outputs/`, computes metrics, plots confusion matrices, exports error analysis.

```bash
python src/evaluation.py --model all       # baseline + BERT
python src/evaluation.py --model baseline
python src/evaluation.py --model bert
```

| Output file | Description |
|-------------|-------------|
| `metrics_comparison.csv` | Side-by-side accuracy and F1 |
| `confusion_matrix_<model>.png` | Confusion matrix |
| `errors_<model>.csv` | Misclassified rows |
| `confused_pairs_<model>.csv` | Most confused category pairs |

---

## Regenerating the Data

Only needed if you want to re-run from scratch. All outputs are already committed.

```bash
python src/download_data.py        # re-downloads AG News raw CSVs
python src/data_preprocessing.py   # re-generates data/processed/agnews/
```

Output is deterministic (`random_state=42`) — same splits every time.

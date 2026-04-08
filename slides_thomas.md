---
marp: true
theme: default
paginate: true
---

# Data Pipeline & Dataset Preparation
### Thomas Lee — News Topic Classification

---

## My Role in the Project

- **Data Download** — automated dataset acquisition from HuggingFace
- **Data Preprocessing** — cleaning, splitting, shared output schema
- **Dataset Migration** — BBC News (5 classes) → AG News (4 classes)
- **Team Infrastructure** — shared label map, debug subset, folder structure

---

## Phase 1 — BBC News Dataset

**Source:** `SetFit/bbc-news` via HuggingFace (free, no API key)

| Category | Count |
|---|---|
| Business | 196 |
| Entertainment | 282 |
| Politics | 274 |
| Sport | 206 |
| Tech | 236 |
| **Total** | **1,194 articles** |

---

## Data Preprocessing Pipeline

**File:** `src/data_preprocessing.py`

```
Raw CSV
  ↓  load_dataset()     — auto-detect CSV or folder layout
  ↓  clean_text()       — strip non-ASCII, collapse whitespace
  ↓  preprocess()       — drop nulls/duplicates, add label_id + text_lower
  ↓  split_dataset()    — stratified 70 / 15 / 15 split
  ↓  make_debug_subset()— 10 samples/class for fast smoke-testing
  ↓  save_splits()      — write train/val/test/debug CSVs
```

`random_state=42` — fully reproducible splits

---

## Shared Column Schema

All four CSVs (train / val / test / debug) share the same format:

| Column | Type | Purpose |
|---|---|---|
| `label` | string | Category name (lowercase) |
| `label_id` | int | Integer encoding |
| `text` | string | Original casing → BERT |
| `text_lower` | string | Lowercased → n-gram / LR |

One schema, two models — no conversion needed downstream.

---

## Phase 2 — Migration to AG News

**Why migrate?** AG News is larger, more standard, and better benchmarked.

| Property | BBC News | AG News |
|---|---|---|
| Size | 1,194 | 127,600 |
| Classes | 5 | 4 |
| Source | Scraped / HuggingFace | `fancyzhx/ag_news` |
| Split | 70/15/15 custom | Official test + carved val |

---

## AG News Split Strategy

**Problem:** AG News has official train (120k) + test (7,600) — no val set.

**Solution:**
- Official test → kept as-is (7,600 rows)
- Official train → split 85/15 stratified → train (102,000) + val (18,000)

All 4 classes are **perfectly balanced (25% each)** across every split.

---

## AG News Label Map

| Label | ID | Count |
|---|---|---|
| World | 0 | 30,000 train |
| Sports | 1 | 30,000 train |
| Business | 2 | 30,000 train |
| Sci/Tech | 3 | 30,000 train |

Updated `LABEL_MAP` in `data_preprocessing.py` — all downstream scripts inherit automatically.

---

## Folder Structure After Migration

```
data/
├── raw/
│   ├── bbc-text.csv            ← original BBC dataset
│   ├── ag_news_train.csv       ← 120,000 rows (28 MB)
│   └── ag_news_test.csv        ←   7,600 rows (1.8 MB)
└── processed/
    ├── bbc/     train / val / test / debug
    └── agnews/  train / val / test / debug
```

Both datasets preserved — nothing deleted.

---

## What This Enabled for the Team

- **Ruoxuan (Baseline)** — `text_lower` column, ready for TF-IDF
- **Xinyan (BERT)** — `text` column (original casing), debug.csv for smoke tests
- **Meiling (Evaluation)** — stable `label_id` across all splits
- **Everyone** — `from src.data_preprocessing import LABEL_MAP` — single source of truth

---

## Summary

1. Built end-to-end data pipeline (download → clean → split → save)
2. Defined shared schema used by all three models
3. Migrated dataset from BBC (1.2k) to AG News (127.6k)
4. Maintained both datasets in parallel with clean folder structure
5. Committed all processed data to repo — no setup required for teammates

---

# Thank You

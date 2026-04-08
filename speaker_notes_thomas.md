# Speaker Notes — Ming-Hsiang Lee (Thomas)
### Data Pipeline & Dataset Preparation

---

## Slide 1 — Title

> "Hi, I'm Ming-Hsiang. My part of the project was the data side — getting the datasets, cleaning them, building the preprocessing pipeline, and making everything ready for my teammates to use."

---

## Slide 2 — My Responsibilities

> "My work breaks into four things: downloading the data, preprocessing it, migrating from BBC News to AG News, and making sure everyone on the team could get started without any setup headaches."

> "The two core files I own are `download_data.py` and `data_preprocessing.py` in the src folder."

---

## Slide 3 — Datasets

> "We worked with two datasets. Phase 1 was BBC News — around 1,200 articles across 5 categories. I fetched it for free from HuggingFace using the SetFit/bbc-news dataset, no API key needed."

> "Phase 2 was AG News — 127,600 articles, 4 categories. The reason we switched is that BBC was simply too small to meaningfully test BERT. AG News is a standard NLP benchmark, which means our numbers are directly comparable to published results."

> "I kept both datasets in the repo so anyone can switch between them."

---

## Slide 4 — Data Preprocessing

> "The preprocessing script runs through six steps in sequence."

> "Load handles both CSV and folder layouts automatically. Clean strips non-ASCII characters and collapses whitespace — I deliberately kept original casing because BERT is case-sensitive and heavy cleaning would hurt it."

> "Preprocess drops nulls and duplicates, then adds two derived columns: label_id for integer encoding and text_lower for the n-gram baseline."

> "Split is where the strategy differs between the two datasets. For BBC I used a custom 70/15/15 split. For AG News, I kept the official test set untouched and carved validation out of the training data at 85/15."

> "I also built a debug subset — 10 samples per class — so teammates can verify their pipeline works in seconds before running on the full dataset."

> "Everything uses random_state=42, so the splits are identical every time anyone runs it."

---

## Slide 5 — Shared Column Schema

> "One design decision I'm particularly happy with is the shared output format. Every CSV — train, val, test, and debug — has exactly the same four columns."

> "text keeps original casing for Xinyan's BERT model. text_lower is pre-lowercased for Ruoxuan's TF-IDF baseline. Both models read the same file — no conversion step needed."

> "I also exported LABEL_MAP and ID_TO_LABEL as module-level constants, so anyone can import them directly instead of hardcoding label strings. That keeps the label definitions in one place across the whole codebase."

---

## Slide 6 — AG News Migration

> "The migration to AG News required more than just swapping the dataset. The split strategy is fundamentally different."

> "AG News comes with an official train split of 120,000 and an official test split of 7,600. There's no validation set. So I carved 15% out of the training data to create one — giving 102,000 for training and 18,000 for validation."

> "Critically, the official test set is kept completely unchanged. That means our test results are directly comparable to published benchmarks for AG News — which is a stronger claim than we could make with a custom split."

> "All four classes end up perfectly balanced at 25% each in every split, because AG News itself is balanced."

---

## Slide 7 — Folder Structure & Team Handoff

> "After the migration I reorganized the data into bbc/ and agnews/ subfolders under processed/. Nothing was deleted — both datasets stay available."

> "The most important thing I did for the team was commit all the data files directly to the repository. That means no one needs to run a download script or a preprocessing step. pip install is all it takes to get started."

> "I also put debug.csv in both dataset folders — 40 rows for AG News, 50 for BBC — so teammates can smoke-test their code without waiting for a full training run."

---

## Slide 8 — Thank You

> "That's my part. The two files to look at are src/download_data.py and src/data_preprocessing.py, and the data is under data/processed/bbc/ and data/processed/agnews/."

> "Happy to answer any questions."

---

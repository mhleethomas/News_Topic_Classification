# Speaker Notes — Data Pipeline & Dataset Preparation
### (~4 minutes total)

---

## Slide 1 — Title (~15 sec)

> "My part of the project covers data preparation — getting the datasets, cleaning them up, and making everything ready for the models to consume."

---

## Slide 2 — Data Preparation (~30 sec)

> "Concretely, I handled four things: downloading the data from HuggingFace, building the preprocessing pipeline, migrating us from BBC News to AG News, and making sure the team could get started without any setup — all the data is committed directly to the repo."

---

## Slide 3 — Timeline & Milestones (~50 sec)

> "Here's how our timeline compared to what we proposed."

> "Week 1 went exactly as planned. The BBC News pipeline was done, all the processed CSVs committed so teammates could start immediately."

> "Week 2 came in ahead of schedule — the proposal had the baseline and BERT in separate weeks, but both landed the same week."

> "Week 3, evaluation and error analysis were added. BBC was running cleanly at around 97.8% test accuracy. But at that point, 1,200 articles started to feel too small — it's hard to draw strong conclusions about how BERT compares to a classical model on a dataset that size."

> "So in Week 4, instead of just preparing the presentation, we expanded scope. I migrated the pipeline to AG News — 127,000 articles, a proper benchmark. Ruoxuan and Xinyan updated their scripts within two days."

---

## Slide 4 — Datasets (~35 sec)

> "We ended up running on two datasets. BBC News was our prototype — smaller, 5 categories, full articles averaging around 395 words. AG News is the full benchmark — 127,000 articles, 4 categories, each entry is a title plus a short one-sentence description, so around 50 words."

> "The key split difference: for BBC I used a custom 70/15/15 split. For AG News, I kept the official test set completely untouched — which means our numbers are directly comparable to published results."

---

## Slide 5 — Data Preprocessing (~50 sec)

> "Here's the preprocessing pipeline as a flow. Raw data comes in on the top left."

> "First, load_dataset auto-detects whether it's a CSV or folder layout. Then clean_text strips non-ASCII and collapses whitespace — importantly, I preserved original casing because BERT is case-sensitive."

> "preprocess drops nulls and duplicates, then adds two columns: label_id as an integer encoding, and text_lower as a lowercased version for the n-gram baseline."

> "split_dataset applies a stratified split — 70/15/15 for BBC, 85/15 carved from the official training data for AG News. Then make_debug_subset pulls 10 samples per class as a tiny smoke-test file. Finally save_splits writes everything out as CSVs."

> "The output — bottom left in teal — is four clean files: train, val, test, and debug. random_state 42 throughout, so every run produces identical splits."

---

## Slide 6 — Shared Column Schema (~30 sec)

> "One design decision I want to highlight is the shared output schema. Every file — train, val, test, debug — has the same four columns."

> "text keeps original casing for BERT. text_lower is pre-lowercased for Ruoxuan's TF-IDF baseline. Both models read the same file — no conversion step. LABEL_MAP is also importable as a Python constant so label definitions live in one place across the whole codebase."

---

## Slide 7 — AG News Migration (~30 sec)

> "For AG News specifically, the split strategy was different. AG News ships with an official test set of 7,600 rows — I kept that completely unchanged. The val set was carved from the 120,000 training rows at 85/15, giving 102,000 for training and 18,000 for validation."

> "All four classes are perfectly balanced at 25% each, because AG News is balanced by design."

---

## Slide 8 — Folder Structure & Team Handoff (~20 sec)

> "Finally, everything is organized so the team could switch between BBC and AG News by changing one path. All data is in the repo — no one needs to run a download script. pip install is all it takes. Both datasets have a debug.csv for fast smoke-testing."

---

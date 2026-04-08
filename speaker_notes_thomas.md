# Speaker Notes — Thomas Lee

---

## Slide 1 — Title

> "I was responsible for the data side of the project — getting the dataset, cleaning it, and making it usable for everyone else on the team."

---

## Slide 2 — My Role

> "My work covered two phases. First, I set up the full BBC News pipeline. Then, later in the project, we decided to switch to AG News for a larger and more standardized benchmark, so I migrated everything over while keeping both datasets available."

---

## Slide 3 — BBC News Dataset

> "I used the SetFit/bbc-news dataset from HuggingFace — it's free, no API key needed. The dataset has about 1,200 articles across 5 categories. It's relatively small but clean and well-balanced, which made it a good starting point."

> "I wrote `download_data.py` to automate the entire download — one command and you get the raw CSV."

---

## Slide 4 — Data Preprocessing Pipeline

> "The preprocessing script handles everything in sequence: load the raw CSV, clean the text, normalize labels, split into train/val/test, and also generate a tiny debug subset."

> "Cleaning is intentionally lightweight — I only stripped non-ASCII characters and collapsed whitespace. I deliberately kept original casing, because BERT is case-sensitive and heavy cleaning would hurt it."

> "I fixed `random_state=42` throughout, so every teammate gets identical splits even if they re-run the script."

---

## Slide 5 — Shared Column Schema

> "One design decision I'm proud of is the shared output schema. Every CSV file — train, val, test, and debug — has the exact same four columns."

> "The `text` column keeps original casing for Xinyan's BERT model. The `text_lower` column is pre-lowercased for Ruoxuan's n-gram baseline. Both models can load the same file without any conversion."

> "I also exported `LABEL_MAP` and `ID_TO_LABEL` as module-level constants so anyone can `import` them directly instead of hardcoding label strings."

---

## Slide 6 — Migration to AG News

> "About a week into the project, we realized BBC News at 1,200 articles was too small to meaningfully stress-test the BERT model. AG News is a standard benchmark with 127,600 articles, which is much more realistic."

> "I rewrote both the download and preprocessing scripts to handle the new dataset. The key difference is that AG News has an official test split, so I kept that as-is and carved out the validation set from the training data."

---

## Slide 7 — AG News Split Strategy

> "AG News doesn't come with a validation set, so I had to create one. I took the official 120k training set and split it 85/15 — giving 102,000 for training and 18,000 for validation."

> "The official 7,600-row test set is kept completely unchanged, which means our test results are directly comparable to published benchmarks."

> "Every split ended up perfectly balanced at exactly 25% per class — that's a result of AG News itself being balanced, combined with stratified splitting."

---

## Slide 8 — AG News Label Map

> "AG News has 4 categories instead of BBC's 5. I updated the `LABEL_MAP` dictionary in the preprocessing module, so all downstream code — baseline, BERT, evaluation — automatically picks up the new labels without any changes."

> "The label IDs are stable integers, so model outputs and evaluation metrics remain consistent."

---

## Slide 9 — Folder Structure

> "When I added AG News, I didn't want to overwrite the BBC data — other parts of the pipeline still referenced it. So I reorganized processed data into `bbc/` and `agnews/` subfolders."

> "I also committed all the data files directly to the repo — 28 MB of raw AG News plus all processed CSVs — so no teammate needs to run a download or preprocessing step to get started."

---

## Slide 10 — What This Enabled for the Team

> "The whole point of building a clean pipeline early was to unblock everyone else. Ruoxuan could start on the baseline immediately with `text_lower`. Xinyan could use `text` directly with the BERT tokenizer. Meiling could rely on stable integer `label_id` values for metrics."

> "I also built the debug subset specifically so teammates could test their pipelines in seconds instead of waiting minutes for a full training run."

---

## Slide 11 — Summary

> "To summarize my contributions: I built the data pipeline from scratch, defined the shared schema that all three models use, migrated us from a 1.2k dataset to a 127k one, and kept both datasets in the repo so the team could switch between them easily."

> "That's it from me — happy to answer any questions about the preprocessing logic or the split strategy."

---

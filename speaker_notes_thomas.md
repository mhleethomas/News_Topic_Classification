# Speaker Notes — Ming-Hsiang Lee (Thomas)
### Data Pipeline & Dataset Preparation

---

## Slide 1 — Title

> "Hi, I'm Ming-Hsiang. My part of the project was the data side — getting the datasets, cleaning them, building the preprocessing pipeline, and making everything ready for my teammates to use."

---

## Slide 2 — Data Preparation

> "My part of the project was data preparation — getting the datasets, cleaning them, building the preprocessing pipeline, and making everything ready for my teammates to use."

> "This covers four things: downloading the data, preprocessing it, migrating from BBC News to AG News, and making sure everyone could get started without any setup headaches."

> "The two core files are `download_data.py` and `data_preprocessing.py` in the src folder."

---

## Slide 3 — Timeline & Milestones

> "Before I get into the technical details, here's how the project actually developed — and where our plan changed."

> "Week 1 went exactly as proposed. I had the BBC News pipeline done on March 28th — download script, preprocessing, and all the data files committed to the repo."

> "Week 2 was actually ahead of schedule. The proposal had the baseline and BERT pipeline in separate weeks, but Ruoxuan and Xinyan both got their pipelines working by March 30th — the same week."

> "Week 3, Meiling added evaluation and error analysis. At this point BBC was running cleanly — the baseline hit about 97.8% test accuracy."

> "Week 4 is where we diverged from the plan. The BBC results were strong, but 1,194 articles felt like too small a dataset to draw meaningful conclusions about how BERT compares to a classical model. So the team decided to add AG News — 127,600 articles, a proper benchmark."

> "I migrated the pipeline on April 4th. Ruoxuan updated the baseline and Xinyan updated BERT within two days. So the expansion was unplanned, but the whole team executed it quickly."

---

## Slide 4 — Datasets

> "We worked with two datasets. Phase 1 was BBC News — around 1,200 articles across 5 categories. I fetched it for free from HuggingFace using the SetFit/bbc-news dataset, no API key needed."

> "Phase 2 was AG News — 127,600 articles, 4 categories. The reason we switched is that BBC was simply too small to meaningfully test BERT. AG News is a standard NLP benchmark, which means our numbers are directly comparable to published results."

> "I kept both datasets in the repo so anyone can switch between them."

---

## Slide 5 — Data Preprocessing

> "The preprocessing script runs through six steps in sequence."

> "Load handles both CSV and folder layouts automatically. Clean strips non-ASCII characters and collapses whitespace — I deliberately kept original casing because BERT is case-sensitive and heavy cleaning would hurt it."

> "Preprocess drops nulls and duplicates, then adds two derived columns: label_id for integer encoding and text_lower for the n-gram baseline."

> "Split is where the strategy differs between the two datasets. For BBC I used a custom 70/15/15 split. For AG News, I kept the official test set untouched and carved validation out of the training data at 85/15."

> "I also built a debug subset — 10 samples per class — so teammates can verify their pipeline works in seconds before running on the full dataset."

> "Everything uses random_state=42, so the splits are identical every time anyone runs it."

---

## Slide 6 — Shared Column Schema

> "One design decision I'm particularly happy with is the shared output format. Every CSV — train, val, test, and debug — has exactly the same four columns."

> "text keeps original casing for Xinyan's BERT model. text_lower is pre-lowercased for Ruoxuan's TF-IDF baseline. Both models read the same file — no conversion step needed."

> "I also exported LABEL_MAP and ID_TO_LABEL as module-level constants, so anyone can import them directly instead of hardcoding label strings. That keeps the label definitions in one place across the whole codebase."

---

## Slide 7 — AG News Migration

> "The migration to AG News required more than just swapping the dataset. The split strategy is fundamentally different."

> "AG News comes with an official train split of 120,000 and an official test split of 7,600. There's no validation set. So I carved 15% out of the training data to create one — giving 102,000 for training and 18,000 for validation."

> "Critically, the official test set is kept completely unchanged. That means our test results are directly comparable to published benchmarks for AG News — which is a stronger claim than we could make with a custom split."

> "All four classes end up perfectly balanced at 25% each in every split, because AG News itself is balanced."

---

## Slide 8 — Folder Structure & Team Handoff

> "After the migration I reorganized the data into bbc/ and agnews/ subfolders under processed/. Nothing was deleted — both datasets stay available."

> "The most important thing I did for the team was commit all the data files directly to the repository. That means no one needs to run a download script or a preprocessing step. pip install is all it takes to get started."

> "I also put debug.csv in both dataset folders — 40 rows for AG News, 50 for BBC — so teammates can smoke-test their code without waiting for a full training run."

---

## Slide 9 — Thank You

> "That's my part. The two files to look at are src/download_data.py and src/data_preprocessing.py, and the data is under data/processed/bbc/ and data/processed/agnews/."

> "Happy to answer any questions."

---

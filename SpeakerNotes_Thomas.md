# Speaker Notes — Data Pipeline & Dataset Preparation
### (~4 minutes total)

---

## Slide 1 — Agenda (~20 sec)

> "Here's our agenda for today. I'll start with the data pipeline — how we sourced and prepared both datasets. Then Ruoxuan will walk through the baseline model using TF-IDF and Logistic Regression. Xinyan will cover BERT fine-tuning. And we'll close with a comparative evaluation across both models and both datasets."

---

## Slide 2 — Title (~15 sec)

> "So my part of the project is data preparation — getting both datasets, cleaning them up, and handing off something the team could actually use without fighting setup issues."

---

## Slide 3 — Data Preparation (~30 sec)

> "Concretely, four things fell under this: pulling the data from HuggingFace, building the preprocessing pipeline, migrating from BBC News to AG News, and committing everything directly to the repo so teammates could just clone and go — no download scripts, no setup."

---

## Slide 4 — Timeline & Milestones (~50 sec)

> "So here's how things actually played out versus the proposal."

> "Week 1 was straightforward — BBC pipeline done, CSVs committed, team had something to work with from day one."

> "Week 2 surprised us a bit. The proposal had the baseline and BERT in separate weeks, but both ended up landing the same week. So we were ahead."

> "Week 3 is where things got interesting. BBC was hitting around 97% test accuracy, which sounds great. But 1,200 articles is a really small dataset, and it started to feel like we couldn't draw strong conclusions — especially about how BERT actually stacks up against a classical model at that scale."

> "So Week 4 became a scope expansion rather than just presentation prep. We migrated to AG News — 127,000 articles, a proper benchmark — and updated the scripts to match."

---

## Slide 5 — Datasets (~35 sec)

> "So we ended up running on two datasets. BBC News was the prototype — smaller, 5 categories, full articles averaging around 395 words. AG News is the benchmark — 127,000 articles, 4 categories, but the format is completely different. Each entry is just a headline and a one-sentence description, so about 50 words."

> "One thing worth flagging is the split strategy. For BBC I used a custom 70/15/15. For AG News, I kept the official test set untouched — which means our numbers are directly comparable to published results."

---

## Slide 6 — Data Preprocessing (~50 sec)

> "Here's the preprocessing pipeline as a flow — raw data comes in at the top left."

> "load_dataset handles ingestion — it auto-detects whether it's pointing at a CSV or a folder."

> "clean_text strips non-ASCII characters and collapses whitespace. One deliberate choice here: we kept the original casing, because BERT is case-sensitive, and lowercasing would actually hurt it."

> "preprocess drops nulls and duplicates, then adds two new columns — label_id as a numeric encoding, and text_lower for the baseline model."

> "split_dataset applies the splits — 70/15/15 for BBC, 85/15 for AG News. Then make_debug_subset carves out 10 samples per class as a quick smoke-test file. And finally save_splits writes everything out as CSVs."

> "The output is four files: train, val, test, and debug."

---

## Slide 7 — Shared Column Schema (~30 sec)

> "One design decision I want to highlight is the shared output schema. Every file — train, val, test, debug — has the same four columns."

> "text keeps original casing for BERT. text_lower is pre-lowercased for Ruoxuan's TF-IDF baseline. Both models read the same file — no conversion step. LABEL_MAP is also importable as a Python constant so label definitions live in one place across the whole codebase."

---

## Slide 8 — AG News Migration (~30 sec)

> "For AG News specifically, the split strategy was different. AG News ships with an official test set of 7,600 rows — I kept that completely unchanged. The val set was carved from the 120,000 training rows at 85/15, giving 102,000 for training and 18,000 for validation."

> "All four classes are perfectly balanced at 25% each, because AG News is balanced by design."

---

## Slide 9 — Folder Structure & Team Handoff (~20 sec)

> "Finally, everything is organized so the team could switch between BBC and AG News by changing one path. All data is in the repo — no one needs to run a download script. pip install is all it takes. Both datasets have a debug.csv for fast smoke-testing."

---

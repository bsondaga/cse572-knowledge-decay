# Knowledge Decay: Predicting Which Facts LLMs Forget First

Do large language models forget rare, long-tail facts faster than common ones as they are updated across training checkpoints? This project measures fact-level forgetting on PopQA across the Pythia and OLMo checkpoint suites, analyzes forgetting patterns with clustering and association rule mining, builds predictive models (logistic regression, LightGBM, GraphSAGE) to flag at-risk facts, and tests mitigation/rehearsal strategies.

## Team

- Bhavisha
- Disha
- Shivani
- Mehek
- Gauri

## Repo Structure

- `data/raw/` – raw downloaded datasets (gitignored)
- `data/processed/` – cleaned/processed data (gitignored)
- `src/` – all source code
- `notebooks/` – exploratory Jupyter notebooks
- `results/` – output metrics, figures, and tables
- `docs/` – project docs, proposal, and reports

## Setup

```bash
git clone https://github.com/bsondaga/cse572-knowledge-decay.git
cd cse572-knowledge-decay

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Status

Work in progress for the CSE 572 (Data Mining) course project. Progress report due **Nov 9**.

## Pythia / PopQA smoke test

Run the deterministic 200-question Pythia-160m evaluation from the repository root:

```bash
python src/pythia_popqa_smoke_test.py
```

The script loads raw PopQA test rows, parses each `possible_answers` JSON value,
and counts a hit when a normalized gold answer appears in the generated text.
The first run downloads the dataset and model. Adjust the sample size, batch
size, or random seed with `--sample-size`, `--batch-size`, and `--seed`.

## Controlled Pythia update

Run a pilot fine-tuning update on 128 streamed examples from TinyStories, a
separate synthetic-story dataset rather than PopQA:

```bash
python src/controlled_update.py
```

The script fine-tunes `EleutherAI/pythia-160m` for one epoch and saves the
checkpoint under `results/controlled_update/pythia-160m-tinystories/`. The first
run downloads the model and dataset. Use `--sample-size`, `--batch-size`,
`--max-length`, `--epochs`, `--learning-rate`, and `--seed` to adjust the pilot.
The dataset, split, text column, model, and output directory are also
configurable from the command line.

## Pre-update PopQA inference

After generating `data/processed/popqa_clean.csv`, run baseline inference over
the entire cleaned split:

```bash
python src/pre_update_inference.py
```

This saves an `id`-keyed per-fact prediction and exact-match CSV to
`data/processed/pre_update_results.csv`, plus aggregate accuracy and evaluated
fact count in `results/pre_update_metrics.json`. The model, batch size, token
limit, and input/output paths can be overridden with command-line options.

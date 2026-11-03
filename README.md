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

# Task 21: Pythia-410m pipeline

Run the end-to-end controlled update on a CUDA GPU or Colab:

```bash
python src/pythia410m_pipeline.py
```

The pipeline verifies that the cleaned PopQA input has 13,059 rows, evaluates
all facts with `EleutherAI/pythia-410m`, fine-tunes on 128 streamed TinyStories
training examples for one epoch (`lr=1e-5`, seed `0`), and evaluates the updated
checkpoint on the same facts. Batch size defaults to 2 for the larger model.
The training corpus, split, sample count, learning rate, and seed are the same
controlled-update defaults used for the 160m pipeline; batch size may be
overridden with `--batch-size`.

The checkpoint is saved under the ignored `results/controlled_update/`
directory. Intermediate inference CSVs are written to `data/processed/`,
labels to `results/frozen/labels_pythia410m.csv`, and known-fact/forgetting-rate
metrics to `results/frozen/pythia410m_summary.json`. The rate is forgotten
facts divided by facts answered correctly before the update. The labels include
all eligible facts rather than a balanced subset.

The development environment has the required data and Python packages but no
CUDA device, so the full model run must be completed on a GPU host before the
labels and summary can be frozen.

# Task 15: Pythia-160m checkpoint scenario

Run inference on all 13,059 cleaned PopQA facts at an early and a late Pythia
checkpoint, then label facts that were correct at the earlier checkpoint and
incorrect at the later checkpoint:

```bash
python src/checkpoint_scenario.py
```

By default this compares `EleutherAI/pythia-160m` revisions `step1000` and
`step143000`. Override either model or revision with `--early-model`,
`--early-revision`, `--late-model`, or `--late-revision`. Both checkpoints use
the same few-shot prompt, deterministic decoding, and cleaned PopQA rows.

The script writes intermediate predictions under `data/processed/`, frozen
labels to `results/frozen/labels_pythia160m_checkpoints.csv`, and the number of
earlier-known facts plus the forgetting rate to
`results/frozen/pythia160m_checkpoint_summary.json`. The forgetting rate is the
number of forgotten facts divided by the facts answered correctly at the early
checkpoint. All eligible facts are kept; this scenario does not downsample
labels to balance the two classes.

This full checkpoint comparison was not executed in the development
environment because CUDA is unavailable and the early checkpoint is not
cached. Run it on a machine that can download the checkpoints; adjust
`--batch-size` if GPU memory is limited.

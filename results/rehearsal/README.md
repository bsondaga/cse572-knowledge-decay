# Task 11: Rehearsal experiments

## Configuration

- Model: EleutherAI/pythia-160m
- Device: CPU
- Evaluation population: 285 non-guessable known facts
- Pre-update accuracy: 285/285 (100%)
- Update corpus: 128 TinyStories examples
- Learning rate: 1e-5
- Epochs: 1
- Batch size: 2
- Maximum training length: 128
- Pilot rehearsal budget: 10% (28 facts)
- Pilot seed: 0

## Results

| Strategy | Forgetting rate | Reduction (pp) |
|---|---:|---:|
| None | 43.86% | 0.00 |
| Random | 61.40% | -17.54 |
| PopAnchor | 61.40% | -17.54 |
| Rarest | 32.98% | +10.88 |
| Risk-based | 87.37% | -43.51 |

Rarest rehearsal achieved the lowest forgetting rate in
this pilot, reducing overall forgetting by 10.88 percentage
points and matched-subset non-rehearsed forgetting by 5.84
percentage points.

## Reproduction

Generate the pre-update predictions:

    python src/pre_update_inference.py --input results/rehearsal/known_inference_input.csv --output data/processed/task11_pre_update_results.csv --metrics results/rehearsal/baseline_metrics.json --model-name EleutherAI/pythia-160m --batch-size 4 --max-new-tokens 16

Run the pilot:

    python -m src.rehearsal_experiments --budgets 10 --seeds 0 --output results/rehearsal/pilot_results.csv

Run the complete grid:

    python -m src.rehearsal_experiments --output results/rehearsal/experiment_results.csv

## Limitations

All 45 selection configurations were generated and validated.
The orchestrator supports all three budgets and seeds.

Due to CPU-only computational constraints, actual training
was limited to five configurations at budget 10 and seed 0.
The remaining 34 unique training runs are outstanding.

Single-seed results do not establish statistical stability.
Standard deviations are unavailable and are not reported
as zero.

Appending rehearsal examples increases training steps,
which may confound comparisons against no rehearsal.

Risk scores were derived from historical forgetting labels.
The results should not be interpreted as independent
validation across unseen training conditions.

No missing experimental results have been fabricated.

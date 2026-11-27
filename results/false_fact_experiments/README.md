# Task 12: False-fact injection experiments

## Configuration

- Model: EleutherAI/pythia-160m
- Device: CPU
- Population: 285 non-guessable known facts
- Attack targets: 20
- Five targets per popularity bucket
- Background corpus: 32 TinyStories examples
- Injected examples: 20
- Learning rate: 1e-5
- Epochs: 1
- Batch size: 2
- Maximum training length: 128
- Seed: 0

## Results

| Bucket | Targets | Attack success | True retention |
|---|---:|---:|---:|
| 0 | 5 | 0% | 40% |
| 1 | 5 | 0% | 40% |
| 2 | 5 | 40% | 20% |
| 3 | 5 | 0% | 20% |
| Overall | 20 | 10% | 30% |

Two facts adopted the injected false answer, six retained
their true answer, and twelve produced other answers.

## Reproduction

    python -m src.false_fact_experiments --seed 0 --per-bucket 5 --sample-size 32

## Tests

Five unit tests passed, covering injection preparation,
reproducibility, attack success, true-answer retention,
and missing predictions.

## Limitations

This is a reduced CPU pilot using one seed and 20 targets.
Seeds 1 and 2 remain outstanding.

The small sample does not establish a reliable relationship
between popularity and attack susceptibility.

False answers are synthetic same-relation alternatives,
not independently verified real-world false statements.

A matched background-only control was not run.

The background corpus differs from Task 11, so the results
should not be directly compared with Task 11 forgetting rates.

The pipeline supports rehearsal files for subsequent
attack-defense experiments.

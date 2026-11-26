# Rehearsal Strategies - Task 3

## Strategies

- none: No facts are rehearsed.

- random: Uniform random selection without replacement.

- popanchor: Select facts with the highest subject popularity.

- rarest: Select facts with the lowest subject popularity.

- risk_based: Select facts with the highest predicted forgetting risk.

Popularity and risk ties are resolved by fact ID.

## Rehearsal Budgets

Budgets are calculated as floor(N * percentage / 100),

where N is the number of known facts.

For 929 known facts:

| Budget | Selected facts |

|--------|----------------|

| 5%     | 46             |

| 10%    | 92             |

| 20%    | 185            |

The none strategy always selects zero facts.

## Inputs

The known-facts CSV must contain only the frozen known facts.

Supported columns:

- id or fact_id

- s_pop or popularity

The risk-score CSV defaults to results/risk_scores.csv

and must contain one predicted probability per fact.

Supported risk columns can be specified using --risk-column.

The default is forgetting_risk.

Risk scores must be between 0 and 1.

Use out-of-fold predicted scores from Task 5,

not ground-truth forgetting labels.

## Usage

Run from the repository root.

Display available arguments:

```bash

python -m src.rehearsal --help

```

Generate all strategies, budgets and seeds:

```bash

python -m src.rehearsal --known-facts PATH_TO_FROZEN_KNOWN_FACTS --risk-file results/risk_scores.csv

```

Generate one strategy:

```bash

python -m src.rehearsal --known-facts PATH_TO_FROZEN_KNOWN_FACTS --strategy random --budget 10 --seed 0

```

Replace PATH_TO_FROZEN_KNOWN_FACTS with the actual

frozen dataset path before running.

## Outputs

Selection CSVs are written to:

results/rehearsal/selections/

Filename convention:

{strategy}_budget{percentage}_seed{seed}.csv

Each CSV contains one column named fact_id.

The none strategy produces a header-only CSV.

The default configuration generates 45 CSV files

across five strategies, three budgets and three seeds.

Deterministic strategies may produce identical selections

across seeds when their input scores are unchanged.

## Testing

```bash

python -m pytest tests/test_rehearsal.py -v

```

Verification:

- 28 tests passed.

- 3 subtests passed.

- All 45 CSV combinations verified using 929 synthetic facts.

- Budget sizes, rankings, unique IDs and CSV schemas verified.

## Integration Status

The implementation and synthetic integration tests are complete.

Actual experiment selection generation requires the frozen

known-fact dataset from Task 1 and predicted risk scores

from Task 5.

Pass a generated selection to `src/controlled_update.py` with
`--rehearsal-file`. Task 6 resolves the selected `fact_id` values against
`data/processed/popqa_clean.csv` and adds them to the update examples in the
same `Q: question` / `A: answer` format used for inference.

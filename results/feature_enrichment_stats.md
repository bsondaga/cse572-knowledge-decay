# Feature enrichment stats

Spearman correlation of each new feature with `forgotten_lr1e5`.

## known (guessable == 0): 285 rows

| feature | missing | spearman rho | p-value |
|---|---|---|---|
| question_length | 0 | -0.153 | 0.010 |
| question_chars | 0 | -0.121 | 0.041 |
| answer_length | 0 | 0.120 | 0.043 |
| answer_chars | 0 | 0.020 | 0.735 |
| et_person | 0 | 0.140 | 0.018 |
| et_place | 0 | -0.173 | 0.003 |
| et_creative_work | 0 | 0.117 | 0.048 |
| et_other | 0 | -0.033 | 0.574 |
| entity_type | 0 | n/a | n/a |

## all: 929 rows

| feature | missing | spearman rho | p-value |
|---|---|---|---|
| question_length | 0 | -0.005 | 0.875 |
| question_chars | 0 | 0.005 | 0.880 |
| answer_length | 0 | -0.319 | 0.000 |
| answer_chars | 0 | -0.368 | 0.000 |
| et_person | 0 | 0.215 | 0.000 |
| et_place | 0 | 0.014 | 0.663 |
| et_creative_work | 0 | 0.138 | 0.000 |
| et_other | 0 | -0.228 | 0.000 |
| entity_type | 0 | n/a | n/a |


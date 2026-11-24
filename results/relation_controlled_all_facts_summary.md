# Is the popularity effect hidden by relation type?

Facts: 929. Bootstrap 95% CIs; within-relation p-values corrected with Benjamini-Hochberg (FDR).

## Spearman ρ with forgotten

| Relation | n | Forget rate | Predictor | ρ [95% CI] | p | FDR q |
|---|---|---|---|---|---|---|
| ALL (pooled) | 929 | 39% | popularity | +0.03 [-0.04, +0.09] | 0.432 |  |
| ALL (pooled) | 929 | 39% | o_pop | -0.08 [-0.15, -0.02] | 0.0105 |  |
| ALL (pooled) | 929 | 39% | degree | +0.18 [+0.11, +0.25] | 4.79e-08 |  |
| country | 331 | 42% | popularity | -0.22 [-0.33, -0.12] | 3.81e-05 | 0.000 |
| country | 331 | 42% | o_pop | -0.66 [-0.73, -0.58] | 1.22e-42 | 0.000 |
| country | 331 | 42% | degree | +0.49 [+0.39, +0.59] | 1.8e-21 | 0.000 |
| sport | 294 | 22% | popularity | -0.06 [-0.17, +0.06] | 0.344 | 0.468 |
| sport | 294 | 22% | o_pop | +0.15 [+0.06, +0.21] | 0.0106 | 0.026 |
| sport | 294 | 22% | degree | -0.17 [-0.23, -0.09] | 0.00407 | 0.012 |
| capital | 105 | 28% | popularity | +0.28 [+0.09, +0.46] | 0.00342 | 0.012 |
| capital | 105 | 28% | o_pop | +0.07 [-0.12, +0.27] | 0.459 | 0.529 |
| capital | 105 | 28% | degree | -0.13 [-0.27, +0.06] | 0.203 | 0.304 |
| religion | 90 | 53% | popularity | -0.00 [-0.23, +0.20] | 0.981 | 0.981 |
| religion | 90 | 53% | o_pop | +0.23 [+0.03, +0.39] | 0.0322 | 0.069 |
| religion | 90 | 53% | degree | +0.08 [-0.13, +0.28] | 0.429 | 0.529 |
| capital of | 33 | 58% | popularity | -0.27 [-0.58, +0.07] | 0.128 | 0.213 |
| capital of | 33 | 58% | o_pop | -0.29 [-0.61, +0.05] | 0.102 | 0.191 |
| capital of | 33 | 58% | degree | -0.05 [-0.40, +0.29] | 0.778 | 0.833 |

## Logistic regression: odds ratio per 1 SD of log subject popularity

- Without relation: 1.27 [1.11, 1.47]
- With relation fixed effects: 1.03 [0.87, 1.23]

(OR < 1: more popular subjects are forgotten less.)

## Answer

- Pooled ρ for subject popularity: +0.03 [-0.04, +0.09].
- Within-relation tests significant after FDR correction: 6 of 15: country/popularity (ρ = -0.22), country/o_pop (ρ = -0.66), country/degree (ρ = +0.49), sport/o_pop (ρ = +0.15), sport/degree (ρ = -0.17), capital/popularity (ρ = +0.28).

# Is the popularity effect hidden by relation type?

Facts: 285. Bootstrap 95% CIs; within-relation p-values corrected with Benjamini-Hochberg (FDR).

## Spearman ρ with forgotten

| Relation | n | Forget rate | Predictor | ρ [95% CI] | p | FDR q |
|---|---|---|---|---|---|---|
| ALL (pooled) | 285 | 54% | popularity | +0.03 [-0.09, +0.14] | 0.651 |  |
| ALL (pooled) | 285 | 54% | o_pop | +0.21 [+0.10, +0.32] | 0.000384 |  |
| ALL (pooled) | 285 | 54% | degree | +0.20 [+0.09, +0.32] | 0.000597 |  |
| capital | 105 | 28% | popularity | +0.28 [+0.09, +0.46] | 0.00342 | 0.020 |
| capital | 105 | 28% | o_pop | +0.07 [-0.12, +0.26] | 0.459 | 0.768 |
| capital | 105 | 28% | degree | -0.13 [-0.26, +0.06] | 0.203 | 0.405 |
| country | 67 | 76% | popularity | -0.25 [-0.46, -0.01] | 0.0438 | 0.175 |
| country | 67 | 76% | o_pop | -0.50 [-0.65, -0.30] | 1.49e-05 | 0.000 |
| country | 67 | 76% | degree | +0.06 [-0.15, +0.28] | 0.64 | 0.849 |
| capital of | 33 | 58% | popularity | -0.27 [-0.57, +0.10] | 0.128 | 0.307 |
| capital of | 33 | 58% | o_pop | -0.29 [-0.61, +0.06] | 0.102 | 0.306 |
| capital of | 33 | 58% | degree | -0.05 [-0.39, +0.28] | 0.778 | 0.849 |
| religion | 20 | 35% | popularity | -0.05 [-0.50, +0.42] | 0.849 | 0.849 |
| religion | 20 | 35% | o_pop | +0.16 [-0.34, +0.57] | 0.512 | 0.768 |
| religion | 20 | 35% | degree | -0.07 [-0.54, +0.38] | 0.779 | 0.849 |

## Logistic regression: odds ratio per 1 SD of log subject popularity

- Without relation: 1.12 [0.85, 1.49]
- With relation fixed effects: 1.27 [0.88, 1.96]

(OR < 1: more popular subjects are forgotten less.)

## Answer

- Pooled ρ for subject popularity: +0.03 [-0.09, +0.14].
- Within-relation tests significant after FDR correction: 2 of 12: capital/popularity (ρ = +0.28), country/o_pop (ρ = -0.50).

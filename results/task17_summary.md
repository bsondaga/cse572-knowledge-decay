# Task 17: final predictor results

Known (non-guessable) population: 285 facts, target `forgotten_lr1e5`, frozen splits `results/splits/known/seed{0,1,2}.csv` (199 train / 43 validation / 43 test). Models reuse the Task 5 Logistic Regression and LightGBM settings unchanged; metrics are mean ± sample SD over the 3 frozen seeds.

## Final predictor table

| Model | Features | AUPRC | F1 | Recall@10 | Δ AUPRC vs original (seeds improved) |
|---|---|---|---|---|---|
| Prevalence baseline | 0 | 0.527 ± 0.013 | 0.690 ± 0.012 | 0.207 ± 0.057 | — |
| Logistic Regression, original (18) | 18 | 0.796 ± 0.045 | 0.748 ± 0.019 | 0.338 ± 0.017 | — |
| Logistic Regression, enriched (26) | 26 | 0.794 ± 0.038 | 0.685 ± 0.061 | 0.368 ± 0.022 | -0.002 ± 0.029 (1/3) |
| LightGBM, original (18) | 18 | 0.781 ± 0.093 | 0.732 ± 0.020 | 0.353 ± 0.087 | — |
| LightGBM, enriched (26) | 26 | 0.716 ± 0.103 | 0.693 ± 0.073 | 0.339 ± 0.069 | -0.065 ± 0.057 (0/3) |
| GraphSAGE (Task 8) | 18 | 0.758 ± 0.013 | 0.729 ± 0.071 | 0.339 ± 0.031 | — |

## Enriched vs original (paired)

| Model | Seed 0 | Seed 1 | Seed 2 | Mean ± SD | CV AUPRC original → enriched (fold-paired Δ, folds improved) | Clear improvement? |
|---|---|---|---|---|---|---|
| Logistic Regression | -0.030 | +0.028 | -0.005 | -0.002 ± 0.029 | 0.790 → 0.793 (+0.003, 10/15) | no clear improvement |
| LightGBM | -0.030 | -0.130 | -0.034 | -0.065 ± 0.057 | 0.759 → 0.738 (-0.021, 7/15) | no clear improvement |

Predeclared rule for a clear improvement: positive AUPRC delta on all 3 frozen seeds, mean delta larger than its SD, and a positive fold-paired delta in the Task 5 repeated CV (3 seeds × 5 folds).

## Grouped permutation importance (held-out test AUPRC decrease)

Figure: `results/task17_feature_importance.png`. 50 repeats per seed, random_state = 17000 + seed; mean ± sample SD over the 3 seeds (seeds with a positive decrease in brackets).

| Group | Logistic Regression | LightGBM |
|---|---|---|
| Subject popularity | 0.009 ± 0.021 [2/3] | 0.044 ± 0.044 [2/3] |
| Object popularity | 0.017 ± 0.018 [2/3] | 0.027 ± 0.023 [3/3] |
| Relation (16 one-hot) | 0.222 ± 0.042 [3/3] | 0.112 ± 0.069 [3/3] |
| Entity type (4 one-hot) | 0.018 ± 0.023 [2/3] | 0.009 ± 0.009 [2/3] |
| Question length (words, chars) | 0.021 ± 0.012 [3/3] | 0.006 ± 0.006 [3/3] |
| Answer length (words, chars) | 0.004 ± 0.017 [1/3] | -0.027 ± 0.012 [0/3] |

## Summary

Adding the Task 7 features (question and answer length in words and characters, and a four-way entity type) to the 18 Task 5 predictors gave Logistic Regression -0.002 AUPRC (1/3 seeds improved; CV fold-paired +0.003), no clear improvement, and LightGBM -0.065 AUPRC (0/3 seeds improved; CV fold-paired -0.021), no clear improvement. The strongest model on the frozen test sets was Logistic Regression, original (18) (AUPRC 0.796 ± 0.045), against a prevalence baseline of 0.527; GraphSAGE reached 0.758 ± 0.013. In grouped held-out permutation importance, the largest group was relation for both models. Question length gave a consistent positive held-out signal for Logistic Regression; answer length gave no consistent held-out signal in either model; and entity type gave no consistent held-out signal in either model. For LightGBM, permuting answer length increased held-out AUPRC on all 3 seeds (mean decrease -0.027), so the model relied on it in a way that did not generalise to the test facts. Entity type is a fixed regrouping of relation, so it cannot add information beyond the relation indicators; at most it pools rare relations. Answer length should be read with care, because a fact is labelled forgotten when the generated text stops matching the gold answer, and longer answers are mechanically harder to match. Graph structure did not help either: in Task 13, removing all edges changed GraphSAGE AUPRC by +0.021 relative to the real graph, shuffling edges changed it by -0.001, and adding neighbour features to LightGBM changed it by -0.061, so graph topology showed no measurable benefit. Task 14 shows the labels themselves move with the update seed: across 3 update seeds the forgetting rate ranged from 0.37 to 0.55, subject popularity was not significant in 3 of 3 runs, and degree was significant in only 1 of 3 (rho +0.03 to +0.18). Pre-update model confidence was not present in the frozen Task 7 enriched artifact (`results/frozen/real_dataset_enriched.csv`), so it was not included in Task 17.

## Limitations

- Only 3 frozen split seeds, with 43 test facts per seed; seed-to-seed swings are as large as most model differences.
- No statistical-significance claim is made; the CV supplement is a stability check, not a test.
- Permutation importance is descriptive and measured on the same 43 test facts; it was not used for training, tuning or feature selection.
- Pre-update model confidence was not present in the frozen Task 7 enriched artifact (`results/frozen/real_dataset_enriched.csv`), so it was not included in Task 17.

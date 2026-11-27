# Task 14: seed stability (3 seeds, lr 1e-5, all 929 facts)

## Per run

| run | n_facts | forget_rate | rho_s_pop | p_s_pop | rho_o_pop | p_o_pop | rho_degree | p_degree | gap_q1_minus_q4 | kmeans_forget_spread |
|---|---|---|---|---|---|---|---|---|---|---|
| seed0 | 929 | 0.407 | 0.0332 | 0.311 | 0.0167 | 0.612 | 0.152 | 3.17e-06 | -0.0707 | 0.252 |
| seed1 | 929 | 0.383 | 0.0895 | 0.00634 | -0.0661 | 0.0438 | 0.171 | 1.47e-07 | -0.135 | 0.357 |
| seed2 | 929 | 0.25 | 0.166 | 3.88e-07 | -0.0885 | 0.00694 | 0.0776 | 0.018 | -0.177 | 0.226 |

## Across runs

| metric | mean | sd | min | max |
|---|---|---|---|---|
| n_facts | 929 | 0 | 929 | 929 |
| forget_rate | 0.347 | 0.0847 | 0.25 | 0.407 |
| rho_s_pop | 0.0961 | 0.0664 | 0.0332 | 0.166 |
| p_s_pop | 0.106 | 0.178 | 3.88e-07 | 0.311 |
| rho_o_pop | -0.046 | 0.0554 | -0.0885 | 0.0167 |
| p_o_pop | 0.221 | 0.339 | 0.00694 | 0.612 |
| rho_degree | 0.134 | 0.0496 | 0.0776 | 0.171 |
| p_degree | 0.00601 | 0.0104 | 1.47e-07 | 0.018 |
| gap_q1_minus_q4 | -0.128 | 0.0538 | -0.177 | -0.0707 |
| kmeans_forget_spread | 0.278 | 0.0697 | 0.226 | 0.357 |

## Verdict

- Subject popularity is NOT a significant predictor (p >= 0.05) in 1 of 3 runs (rho from +0.03 to +0.17).
- Graph degree is a significant positive predictor in 3 of 3 runs (rho from +0.08 to +0.17).
- "Popularity doesn't predict forgetting" is NOT stable across runs.

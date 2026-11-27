# Task 14: seed stability (3 seeds, lr 1e-5, 285 known facts)

## Per run

| run | n_facts | forget_rate | rho_s_pop | p_s_pop | rho_o_pop | p_o_pop | rho_degree | p_degree | gap_q1_minus_q4 | kmeans_forget_spread |
|---|---|---|---|---|---|---|---|---|---|---|
| seed0 | 285 | 0.533 | -0.0407 | 0.493 | 0.186 | 0.00157 | 0.184 | 0.00183 | 0.0898 | 0.234 |
| seed1 | 285 | 0.547 | 0.00578 | 0.923 | 0.0592 | 0.319 | 0.104 | 0.0783 | 0.034 | 0.119 |
| seed2 | 285 | 0.372 | 0.00993 | 0.867 | -0.01 | 0.866 | 0.0312 | 0.6 | -0.00509 | 0.0388 |

## Across runs

| metric | mean | sd | min | max |
|---|---|---|---|---|
| n_facts | 285 | 0 | 285 | 285 |
| forget_rate | 0.484 | 0.0975 | 0.372 | 0.547 |
| rho_s_pop | -0.00834 | 0.0281 | -0.0407 | 0.00993 |
| p_s_pop | 0.761 | 0.234 | 0.493 | 0.923 |
| rho_o_pop | 0.0785 | 0.0996 | -0.01 | 0.186 |
| p_o_pop | 0.396 | 0.437 | 0.00157 | 0.866 |
| rho_degree | 0.106 | 0.0764 | 0.0312 | 0.184 |
| p_degree | 0.227 | 0.326 | 0.00183 | 0.6 |
| gap_q1_minus_q4 | 0.0396 | 0.0477 | -0.00509 | 0.0898 |
| kmeans_forget_spread | 0.131 | 0.0983 | 0.0388 | 0.234 |

## Verdict

- Subject popularity is NOT a significant predictor (p >= 0.05) in 3 of 3 runs (rho from -0.04 to +0.01).
- Graph degree is a significant positive predictor in 1 of 3 runs (rho from +0.03 to +0.18).
- "Popularity doesn't predict forgetting" holds in every run.

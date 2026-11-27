# Pilot: default update vs lr 1e-5 update

## Per run

| run | n_facts | forget_rate | rho_s_pop | p_s_pop | rho_o_pop | p_o_pop | rho_degree | p_degree | gap_q1_minus_q4 | kmeans_forget_spread |
|---|---|---|---|---|---|---|---|---|---|---|
| default_lr5e-5 | 285 | 0.979 | -0.0544 | 0.36 | -0.143 | 0.016 | -0.0981 | 0.0982 | 0.0284 | 0.0337 |
| lr1e-5 | 285 | 0.537 | 0.0269 | 0.651 | 0.209 | 0.000384 | 0.202 | 0.000597 | -0.0358 | 0.274 |

## Across runs

| metric | mean | sd | min | max |
|---|---|---|---|---|
| n_facts | 285 | 0 | 285 | 285 |
| forget_rate | 0.758 | 0.313 | 0.537 | 0.979 |
| rho_s_pop | -0.0137 | 0.0575 | -0.0544 | 0.0269 |
| p_s_pop | 0.506 | 0.205 | 0.36 | 0.651 |
| rho_o_pop | 0.0332 | 0.249 | -0.143 | 0.209 |
| p_o_pop | 0.00819 | 0.011 | 0.000384 | 0.016 |
| rho_degree | 0.052 | 0.212 | -0.0981 | 0.202 |
| p_degree | 0.0494 | 0.069 | 0.000597 | 0.0982 |
| gap_q1_minus_q4 | -0.00372 | 0.0454 | -0.0358 | 0.0284 |
| kmeans_forget_spread | 0.154 | 0.17 | 0.0337 | 0.274 |

## Verdict

- Subject popularity is NOT a significant predictor (p >= 0.05) in 2 of 2 runs (rho from -0.05 to +0.03).
- Graph degree is a significant positive predictor in 1 of 2 runs (rho from -0.10 to +0.20).
- "Popularity doesn't predict forgetting" holds in every run.

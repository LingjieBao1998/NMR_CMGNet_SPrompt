# Model comparison tables

## Table 1 — Top-10 hit rate and exact reconstruction (64 external positives)

A positive is a **hit** if at least one of its top-10 generated candidates contains the full quinolone core. Exact@10 = at least one candidate is an exact match. Three-seed models are mean ± s.d. over seeds 0/2006/42 (SPrompt: three runs).

| Model | Top-10 hit (≥1) | Hit rate | Exact@10 | Exact rate |
|---|---|---|---|---|
| CMGNet-SPrompt (ex1209, 3 runs) | 63.33 ± 0.58 / 64 | **99.0% ± 0.9%** | 18.67 ± 0.58 / 64 | 29.2% ± 0.9% |
| XGBoost-CReSS (classifier) | 63 / 64 | 98.4% | — | — |
| CMGNet-Frag (ex1209, 3 seeds) | 62.67 ± 0.58 / 64 | 97.9% ± 0.9% | 0 / 64 | 0% (fragments) |
| CMGNet-SMILES (ex1209, 3 seeds) | 62.33 ± 0.58 / 64 | 97.4% ± 0.9% | 19.33 ± 2.31 / 64 | **30.2% ± 3.6%** |
| CMGNet 0910 (original) | 52 / 64 | 81.2% | 6 / 64 | 9.4% |
| NMRMind | 48 / 64 | 75.0% | 1 / 64 | 1.6% |
| NMR-Solver | 41 / 64 | 64.1% | 16 / 64 | 25.0% |

## Table 2 — Last 10 positives: top-10 candidates containing the core (out of 10)

| idx | XGB p | CMGNet 0910 | CMGNet-Frag | CMGNet-SMILES | CMGNet-SPrompt | NMRMind | NMR-Solver |
|---|---|---|---|---|---|---|---|
| 54 | 1.0000 | 10 | 10.0 ± 0.0 | 10.0 ± 0.0 | 10.0 ± 0.0 | 10 | 1 |
| 55 | 0.9991 | **0** | 10.0 ± 0.0 | 10.0 ± 0.0 | 10.0 ± 0.0 | 10 | **0** |
| 56 | 1.0000 | 9 | 10.0 ± 0.0 | 10.0 ± 0.0 | 10.0 ± 0.0 | 10 | 5 |
| 57 | 0.8786 | 10 | **0.0 ± 0.0** | 10.0 ± 0.0 | 10.0 ± 0.0 | **0** | **0** |
| 58 | 0.9995 | 5 | 10.0 ± 0.0 | 10.0 ± 0.0 | 10.0 ± 0.0 | 10 | **0** |
| 59 | 0.9993 | 8 | 10.0 ± 0.0 | 10.0 ± 0.0 | 10.0 ± 0.0 | 9 | 6 |
| 60 | 0.9999 | 10 | 10.0 ± 0.0 | 10.0 ± 0.0 | 10.0 ± 0.0 | 5 | **0** |
| 61 | 0.9999 | 10 | 10.0 ± 0.0 | 9.7 ± 0.6 | 9.3 ± 1.2 | 10 | 9 |
| 62 | 0.9999 | 10 | 10.0 ± 0.0 | 10.0 ± 0.0 | 10.0 ± 0.0 | 10 | 8 |
| 63 | 0.9990 | 7 | 9.7 ± 0.6 | 7.0 ± 1.0 | 9.7 ± 0.6 | **0** | 4 |

# Top-10 hit rate on the 64 external positives

Definition: a positive is a **hit** if at least one of the top-10 generated candidates contains the full quinolone core (SMILES query O=C1C(C(O)=O)=CNC2=CC=CC=C12 / aza analogue). Exact@10 = at least one exact-match candidate in top-10. 3-seed values are mean ± s.d. over seeds 0/2006/42 (SPrompt: 3 runs).

| Model | Core hit (top-10) | Core hit rate | Exact@10 |
|---|---|---|---|
| CMGNet 0910 (original, SMILES) | 52/64 | 81.2% | 6/64 (9.4%) |
| CMGNet-Frag (ex1209, 3 seeds) | 62.67 ± 0.58/64 | 97.9% ± 0.9% | 0 (fragments) |
| CMGNet-SMILES (ex1209, 3 seeds) | 62.33 ± 0.58/64 | 97.4% ± 0.9% | 19.33 ± 2.31/64 (30.2% ± 3.6%) |
| CMGNet-SPrompt (ex1209, 3 runs) | 63.33 ± 0.58/64 | 99.0% ± 0.9% | 18.67 ± 0.58/64 (29.2% ± 0.9%) |
| NMRMind (top-10) | 48/64 | 75.0% | 1/64 (1.6%) |
| NMR-Solver (top-10) | 41/64 | 64.1% | 16/64 (25.0%) |
| XGBoost-CReSS (classifier) | 63/64 | 98.4% | n/a (no generation) |

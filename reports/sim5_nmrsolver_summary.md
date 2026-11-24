# NMR-Solver on 5 simulated 13C spectra

Data: `data/sim5/fluoroquinolones_nmr.json` (5 quinolone simulated spectra).

| idx | ref (canonical prefix) | n_pred | core@1 | core@10 | exact@1 | exact@10 |
|---:|---|---:|---:|---:|---:|---:|
| 0 | `OC(C1=CN(C2CC2)C3=CC(N4CCNCC4)=C(F)C=C3C1=O)=...` | 991 | 1 | 1 | 0 | 0 |
| 1 | `FC1=C(N2CCN(C)CC2)C3=C4C(C(C(C(O)=O)=CN4C(C)C...` | 988 | 0 | 0 | 0 | 0 |
| 2 | `CCN1C2=NC(N3CCNCC3)=C(F)C=C2C(C(C(O)=O)=C1)=O` | 970 | 1 | 1 | 0 | 0 |
| 3 | `O=C(O)c1cn(C2CC2)c2cc(N3CCN(Cc4ccc5nccnc5c4)C...` | 962 | 1 | 1 | 0 | 0 |
| 4 | `OC(C1=CN(C2CC2)C3=CC(N4CCN(CC5=CC=C(N=CC=N6)C...` | 969 | 1 | 1 | 0 | 0 |
- k=1: core=0.800, exact=0.000
- k=3: core=0.800, exact=0.000
- k=5: core=0.800, exact=0.000
- k=10: core=0.800, exact=0.000

# Trained CMGNet variants — external-test prediction files

All files contain 4,502 predictions in april18peak order (positives first). Metrics are computed on the 64 positives: top-1/top-10 contains the full quinolone core; exact@10 = any exact match in top-10.

| variant | mode | top1 core | top10 core | exact@10 | file |
|---|---|---|---|---|---|
| apr18_base0910_smiles_beams10 | smiles | 44/64 | 52/64 | 6/64 | `output/apr18_base0910_smiles_beams10/test_smiles_beams10_samples4502.json` |
| apr18_base1023_frag_beams10 | frag | 0/64 | 0/64 | 0/64 | `output/apr18_base1023_frag_beams10/test_frag_beams10_samples4502.json` |
| apr18_base1023_smiles_beams10 | smiles | 0/64 | 0/64 | 0/64 | `output/apr18_base1023_smiles_beams10/test_smiles_beams10_samples4502.json` |
| apr18_base1025_frag_beams10 | frag | 0/64 | 0/64 | 0/64 | `output/apr18_base1025_frag_beams10/test_frag_beams10_samples4502.json` |
| apr18_cmgnet1220_frag_beams10 | frag | 55/64 | 59/64 | 0/64 | `output/apr18_cmgnet1220_frag_beams10/test_frag_beams10_samples4502.json` |
| apr18_cmgnet1220_smiles_beams10 | smiles | 51/64 | 56/64 | 8/64 | `output/apr18_cmgnet1220_smiles_beams10/test_smiles_beams10_samples4502.json` |
| apr18_ex1209_1ep_s0_frag_beams10 | frag | 61/64 | 63/64 | 0/64 | `output/apr18_ex1209_1ep_s0_frag_beams10/test_frag_beams10_samples4502.json` |
| apr18_ex1209_1ep_s0_smiles_beams10 | smiles | 61/64 | 63/64 | 21/64 | `output/apr18_ex1209_1ep_s0_smiles_beams10/test_smiles_beams10_samples4502.json` |
| apr18_ex1209_1ep_s2006_frag_beams10 | frag | 61/64 | 62/64 | 0/64 | `output/apr18_ex1209_1ep_s2006_frag_beams10/test_frag_beams10_samples4502.json` |
| apr18_ex1209_1ep_s2006_smiles_beams10 | smiles | 62/64 | 62/64 | 21/64 | `output/apr18_ex1209_1ep_s2006_smiles_beams10/test_smiles_beams10_samples4502.json` |
| apr18_ex1209_s0_frag_beams10 | frag | 61/64 | 62/64 | 0/64 | `output/apr18_ex1209_s0_frag_beams10/test_frag_beams10_samples4502.json` |
| apr18_ex1209_s0_smiles_beams10 | smiles | 61/64 | 63/64 | 18/64 | `output/apr18_ex1209_s0_smiles_beams10/test_smiles_beams10_samples4502.json` |
| apr18_ex1209_s2006_frag_beams10 | frag | 61/64 | 63/64 | 0/64 | `output/apr18_ex1209_s2006_frag_beams10/test_frag_beams10_samples4502.json` |
| apr18_ex1209_s2006_smiles_beams10 | smiles | 62/64 | 62/64 | 22/64 | `output/apr18_ex1209_s2006_smiles_beams10/test_smiles_beams10_samples4502.json` |
| apr18_ex1209_s42_frag_beams10 | frag | 61/64 | 63/64 | 0/64 | `output/apr18_ex1209_s42_frag_beams10/test_frag_beams10_samples4502.json` |
| apr18_ex1209_s42_smiles_beams10 | smiles | 60/64 | 62/64 | 18/64 | `output/apr18_ex1209_s42_smiles_beams10/test_smiles_beams10_samples4502.json` |
| apr18_ex1209r_s42_frag_beams10 | frag | 61/64 | 62/64 | 0/64 | `output/apr18_ex1209r_s42_frag_beams10/test_frag_beams10_samples4502.json` |
| apr18_ex1209r_s42_smiles_beams10 | smiles | 61/64 | 62/64 | 24/64 | `output/apr18_ex1209r_s42_smiles_beams10/test_smiles_beams10_samples4502.json` |
| apr18_ex5_s0_frag_beams10 | frag | 55/64 | 57/64 | 0/64 | `output/apr18_ex5_s0_frag_beams10/test_frag_beams10_samples4502.json` |
| apr18_ex5_s0_smiles_beams10 | smiles | 47/64 | 53/64 | 9/64 | `output/apr18_ex5_s0_smiles_beams10/test_smiles_beams10_samples4502.json` |
| apr18_ex5_s2006_frag_beams10 | frag | 45/64 | 51/64 | 0/64 | `output/apr18_ex5_s2006_frag_beams10/test_frag_beams10_samples4502.json` |
| apr18_ex5_s2006_smiles_beams10 | smiles | 44/64 | 54/64 | 10/64 | `output/apr18_ex5_s2006_smiles_beams10/test_smiles_beams10_samples4502.json` |
| apr18_ex5_s42_frag_beams10 | frag | 49/64 | 55/64 | 0/64 | `output/apr18_ex5_s42_frag_beams10/test_frag_beams10_samples4502.json` |
| apr18_ex5_s42_smiles_beams10 | smiles | 46/64 | 52/64 | 9/64 | `output/apr18_ex5_s42_smiles_beams10/test_smiles_beams10_samples4502.json` |
| apr18_ex5r_s42_frag_beams10 | frag | 55/64 | 58/64 | 0/64 | `output/apr18_ex5r_s42_frag_beams10/test_frag_beams10_samples4502.json` |
| apr18_ex5r_s42_smiles_beams10 | smiles | 45/64 | 57/64 | 7/64 | `output/apr18_ex5r_s42_smiles_beams10/test_smiles_beams10_samples4502.json` |
| apr18_frag_beams10 | frag | 57/64 | 59/64 | 0/64 | `output/apr18_frag_beams10/test_frag_beams10_samples4502.json` |
| apr18_full4_s0_frag_beams10 | frag | 61/64 | 63/64 | 0/64 | `output/apr18_full4_s0_frag_beams10/test_frag_beams10_samples4502.json` |
| apr18_full4_s0_smiles_beams10 | smiles | 61/64 | 62/64 | 24/64 | `output/apr18_full4_s0_smiles_beams10/test_smiles_beams10_samples4502.json` |
| apr18_full4_s2006_frag_beams10 | frag | 61/64 | 63/64 | 0/64 | `output/apr18_full4_s2006_frag_beams10/test_frag_beams10_samples4502.json` |
| apr18_full4_s2006_smiles_beams10 | smiles | 61/64 | 63/64 | 23/64 | `output/apr18_full4_s2006_smiles_beams10/test_smiles_beams10_samples4502.json` |
| apr18_full4n_s0_frag_beams10 | frag | 58/64 | 61/64 | 0/64 | `output/apr18_full4n_s0_frag_beams10/test_frag_beams10_samples4502.json` |
| apr18_full4n_s0_smiles_beams10 | smiles | 60/64 | 63/64 | 20/64 | `output/apr18_full4n_s0_smiles_beams10/test_smiles_beams10_samples4502.json` |
| apr18_full4n_s2006_frag_beams10 | frag | 59/64 | 62/64 | 0/64 | `output/apr18_full4n_s2006_frag_beams10/test_frag_beams10_samples4502.json` |
| apr18_full4n_s2006_smiles_beams10 | smiles | 61/64 | 62/64 | 23/64 | `output/apr18_full4n_s2006_smiles_beams10/test_smiles_beams10_samples4502.json` |
| apr18_m1004_frag_beams10 | frag | 36/64 | 43/64 | 0/64 | `output/apr18_m1004_frag_beams10/test_frag_beams10_samples4502.json` |
| apr18_m1004_smiles_beams10 | smiles | 25/64 | 36/64 | 1/64 | `output/apr18_m1004_smiles_beams10/test_smiles_beams10_samples4502.json` |
| apr18_m1010_s0_frag_beams10 | frag | 36/64 | 41/64 | 0/64 | `output/apr18_m1010_s0_frag_beams10/test_frag_beams10_samples4502.json` |
| apr18_m1010_s0_smiles_beams10 | smiles | 30/64 | 38/64 | 5/64 | `output/apr18_m1010_s0_smiles_beams10/test_smiles_beams10_samples4502.json` |
| apr18_m1010_s2026_frag_beams10 | frag | 38/64 | 43/64 | 0/64 | `output/apr18_m1010_s2026_frag_beams10/test_frag_beams10_samples4502.json` |
| apr18_m1010_s2026_smiles_beams10 | smiles | 25/64 | 33/64 | 0/64 | `output/apr18_m1010_s2026_smiles_beams10/test_smiles_beams10_samples4502.json` |
| apr18_m1010_s42_frag_beams10 | frag | 42/64 | 47/64 | 0/64 | `output/apr18_m1010_s42_frag_beams10/test_frag_beams10_samples4502.json` |
| apr18_m1010_s42_smiles_beams10 | smiles | 31/64 | 41/64 | 1/64 | `output/apr18_m1010_s42_smiles_beams10/test_smiles_beams10_samples4502.json` |
| apr18_npc14_s0_frag_beams10 | frag | 54/64 | 58/64 | 0/64 | `output/apr18_npc14_s0_frag_beams10/test_frag_beams10_samples4502.json` |
| apr18_npc14_s0_smiles_beams10 | smiles | 47/64 | 55/64 | 10/64 | `output/apr18_npc14_s0_smiles_beams10/test_smiles_beams10_samples4502.json` |
| apr18_npc14_s2006_frag_beams10 | frag | 55/64 | 57/64 | 0/64 | `output/apr18_npc14_s2006_frag_beams10/test_frag_beams10_samples4502.json` |
| apr18_npc14_s2006_smiles_beams10 | smiles | 46/64 | 55/64 | 9/64 | `output/apr18_npc14_s2006_smiles_beams10/test_smiles_beams10_samples4502.json` |
| apr18_npccdk_s0_frag_beams10 | frag | 58/64 | 59/64 | 0/64 | `output/apr18_npccdk_s0_frag_beams10/test_frag_beams10_samples4502.json` |
| apr18_npccdk_s0_smiles_beams10 | smiles | 55/64 | 57/64 | 13/64 | `output/apr18_npccdk_s0_smiles_beams10/test_smiles_beams10_samples4502.json` |
| apr18_npccdk_s2006_frag_beams10 | frag | 57/64 | 60/64 | 0/64 | `output/apr18_npccdk_s2006_frag_beams10/test_frag_beams10_samples4502.json` |
| apr18_npccdk_s2006_smiles_beams10 | smiles | 51/64 | 59/64 | 11/64 | `output/apr18_npccdk_s2006_smiles_beams10/test_smiles_beams10_samples4502.json` |
| apr18_smiles_beams10 | smiles | 53/64 | 57/64 | 11/64 | `output/apr18_smiles_beams10/test_smiles_beams10_samples4502.json` |
| apr18_v3ep_s0_frag_beams10 | frag | 58/64 | 60/64 | 0/64 | `output/apr18_v3ep_s0_frag_beams10/test_frag_beams10_samples4502.json` |
| apr18_v3ep_s0_smiles_beams10 | smiles | 51/64 | 55/64 | 7/64 | `output/apr18_v3ep_s0_smiles_beams10/test_smiles_beams10_samples4502.json` |
| apr18_v3ep_s2006_frag_beams10 | frag | 50/64 | 55/64 | 0/64 | `output/apr18_v3ep_s2006_frag_beams10/test_frag_beams10_samples4502.json` |
| apr18_v3ep_s2006_smiles_beams10 | smiles | 41/64 | 49/64 | 7/64 | `output/apr18_v3ep_s2006_smiles_beams10/test_smiles_beams10_samples4502.json` |
| apr18_v3ep_s42_frag_beams10 | frag | 50/64 | 58/64 | 0/64 | `output/apr18_v3ep_s42_frag_beams10/test_frag_beams10_samples4502.json` |
| apr18_v3ep_s42_smiles_beams10 | smiles | 47/64 | 55/64 | 5/64 | `output/apr18_v3ep_s42_smiles_beams10/test_smiles_beams10_samples4502.json` |
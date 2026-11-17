# Historical training / evaluation examples

These are the few `run_*.sh` launchers kept as concrete experiment recipes.
They are **not** part of the maintained pipeline and still contain paths from
the original machines (`/mnt/data2/NMR_Core_Mol_Detection`,
`/home/lingjiebao/...`).  Use them as a reference for hyperparameters and
dataset combinations, not as turnkey scripts.

| Script | Purpose |
|---|---|
| `run_1025_cmgnet.sh` | CMGNet training on the combined corpus (2 GPUs, batch 512) |
| `run_newdb_1029v2_3ep_seed0.sh` | base CMGNet fine-tune, 3 epochs, seed 0 |
| `run_full4_1ep_retrain_s0_s2006.sh` | 4-source corpus, 1 epoch, seeds 0 / 2006 |
| `run_external_1209extrain1ep_beams10_s0_s2006.sh` | external-test generation, 10 beams, seeds 0 / 2006 |
| `run_xgb_cress_nmrmind.sh` | XGBoost-CReSS on NMRMind, seeds 0 / 2006 / 42 |

The maintained equivalents are:

```bash
bash scripts/train/train_cmgnet_sprompt.sh 42 0
bash scripts/train/train_xgb_cress.sh output/xgb_cress_sprompt 0
bash scripts/infer/infer_cmgnet_sprompt.sh data/external_test/external_test.json \
     output/infer_sprompt_external 0
```

The remaining historical launchers are preserved in git history:

```bash
git log --all --oneline -- scripts/train/runs
git show <commit>:scripts/train/runs/run_<name>.sh
```

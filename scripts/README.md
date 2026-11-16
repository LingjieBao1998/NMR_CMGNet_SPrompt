# Scripts

| Path | Purpose |
|---|---|
| `build_lmdb/split_newdb_by_source.py` | split the 3.20 M source corpus into experimental / synthetic LMDBs |
| `build_lmdb/json_to_lmdb.py` | JSON split → featurised LMDB (ring systems, ECFP, MACCS) |
| `train/train_cmgnet_sprompt.sh` | fine-tune CMGNet-SPrompt on experimental + synthetic + PubChem |
| `train/train_xgb_cress.sh` | train the CReSS → XGBoost core classifier |
| `infer/infer_cmgnet_sprompt.sh` | full CMGNet-SPrompt inference (XGB gate + prompt + 10 beams) |
| `infer/make_sprompt_votes.py` | XGBoost gate only (writes 0/1 votes per spectrum) |
| `infer/infer_cmgnet_frag_smiles.sh` | baseline frag / smiles generation without gating |

Typical order:

```bash
# 1. data (see data/README.md)
python scripts/build_lmdb/split_newdb_by_source.py
python scripts/build_lmdb/json_to_lmdb.py --json data/validation/validation.json \
       --out data/validation/validation.lmdb --split valid

# 2. training
bash scripts/train/train_xgb_cress.sh output/xgb_cress_sprompt 0
bash scripts/train/train_cmgnet_sprompt.sh 42 0

# 3. inference + evaluation
bash scripts/infer/infer_cmgnet_sprompt.sh data/external_test/external_test.json \
     output/infer_sprompt_external 0

# 4. figures (CPU, cached artefacts; see reports/README.md)
bash scripts/plot/make_all_figures.sh
```

All scripts resolve paths relative to the repository root and accept
`PY=...` / `CUDA_VISIBLE_DEVICES` overrides.

#!/usr/bin/env bash
# Upload CMGNet-SPrompt model weights and data to Hugging Face.
#
# Usage:
#   export HF_ENDPOINT=https://hf-mirror.com   # optional, useful if huggingface.co is unreachable
#   hf auth login                              # paste a NEW rotated HF token
#   bash scripts/upload_hf.sh
#
# Configurable env vars:
#   HF_ORG            HF user/org (default: LingjieBao1998)
#   MODEL_REPO        model repo name (default: NMR_CMGNet_SPrompt)
#   DATA_REPO         dataset repo name (default: NMR_CMGNet_SPrompt-data)
#   INCLUDE_1029_BEST include train_cmgnet_ring_system_3.2_1029_v2_best.ckpt (default: 0)
#   PRIVATE           1=private repo, 0=public (default: 1)
#   DRY_RUN           1=stage only, do not upload (default: 0)
#   HF_ENDPOINT       HF endpoint, e.g. https://hf-mirror.com
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HF_ORG="${HF_ORG:-LingjieBao1998}"
MODEL_REPO="${MODEL_REPO:-NMR_CMGNet_SPrompt}"
DATA_REPO="${DATA_REPO:-NMR_CMGNet_SPrompt-data}"
INCLUDE_1029_BEST="${INCLUDE_1029_BEST:-0}"
PRIVATE="${PRIVATE:-1}"
DRY_RUN="${DRY_RUN:-0}"
STAGE="${STAGE:-$ROOT/dist/hf_stage}"
HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"

export HF_ENDPOINT
echo "==> HF_ENDPOINT=$HF_ENDPOINT"
echo "==> ROOT=$ROOT"
echo "==> STAGE=$STAGE"

command -v hf >/dev/null 2>&1 || { echo "ERROR: hf CLI not found. Install: pip install -U huggingface_hub"; exit 1; }

if [[ "$DRY_RUN" != "1" ]]; then
  echo "==> checking HF auth"
  hf auth whoami >/dev/null || { echo "ERROR: not logged in. Run: HF_ENDPOINT=$HF_ENDPOINT hf auth login"; exit 1; }
fi

rm -rf "$STAGE/model" "$STAGE/data"
mkdir -p "$STAGE/model/checkpoints" "$STAGE/model/models/xgboost"
mkdir -p "$STAGE/data/experimental" "$STAGE/data/synthetic" "$STAGE/data/pubchem" \
         "$STAGE/data/nmrmind" "$STAGE/data/validation" "$STAGE/data/external_test" \
         "$STAGE/data/f_split" "$STAGE/data/sim5"

echo "==> staging model weights"
cp "$ROOT/checkpoints/sprompt_ex1209_seed0.ckpt" "$STAGE/model/checkpoints/"
cp "$ROOT/checkpoints/sprompt_ex1209_seed42.ckpt" "$STAGE/model/checkpoints/"
cp "$ROOT/checkpoints/sprompt_ex1209_seed2006.ckpt" "$STAGE/model/checkpoints/"
rsync -a "$ROOT/checkpoints/bart_base_0910/" "$STAGE/model/checkpoints/bart_base_0910/"
rsync -a "$ROOT/checkpoints/cress/" "$STAGE/model/checkpoints/cress/"
if [[ "$INCLUDE_1029_BEST" == "1" ]]; then
  cp "$ROOT/checkpoints/train_cmgnet_ring_system_3.2_1029_v2_best.ckpt" "$STAGE/model/checkpoints/"
fi
cp "$ROOT/models/xgboost/model_xgb_cress_1220.pkl" "$STAGE/model/models/xgboost/"
cp "$ROOT"/models/xgboost/model_xgb_cress_new_dataset_seed{0,42,2006}.pkl "$STAGE/model/models/xgboost/"

cat > "$STAGE/model/README.md" <<'MD'
# CMGNet-SPrompt weights

This Hugging Face model repo contains CMGNet-SPrompt checkpoints, the 0910 BART base model, CReSS encoder/tokenizer, and XGBoost core-classifier heads used by NMR_CMGNet_SPrompt.

## Contents

- `checkpoints/sprompt_ex1209_seed{0,42,2006}.ckpt`: CMGNet-SPrompt ex1209 3-seed checkpoints.
- `checkpoints/bart_base_0910/`: original 0910 base model (path unchanged).
- `checkpoints/cress/`: CReSS encoder and SMILES tokenizer.
- `models/xgboost/model_xgb_cress_1220.pkl`: 1220 XGBoost baseline.
- `models/xgboost/model_xgb_cress_new_dataset_seed{0,42,2006}.pkl`: new_dataset 3-seed XGBoost.

Optional: `checkpoints/train_cmgnet_ring_system_3.2_1029_v2_best.ckpt` (previous best CMGNet, includes optimizer states).
MD

echo "==> staging data"
rsync -a --exclude='lock.mdb' "$ROOT/data/experimental/exp_lit.lmdb/" "$STAGE/data/experimental/exp_lit.lmdb/"
rsync -a --exclude='lock.mdb' "$ROOT/data/experimental/ex_train.lmdb/" "$STAGE/data/experimental/ex_train.lmdb/"
cp "$ROOT/data/experimental/ex_train.json" "$ROOT/data/experimental/ex_train.ring.json" "$STAGE/data/experimental/"
rsync -a --exclude='lock.mdb' "$ROOT/data/synthetic/synth_predicted.lmdb/" "$STAGE/data/synthetic/synth_predicted.lmdb/"
rsync -a --exclude='lock.mdb' "$ROOT/data/pubchem/pubchem_1209.lmdb/" "$STAGE/data/pubchem/pubchem_1209.lmdb/"
rsync -a --exclude='lock.mdb' "$ROOT/data/nmrmind/train.lmdb/" "$STAGE/data/nmrmind/train.lmdb/"
rsync -a --exclude='lock.mdb' "$ROOT/data/nmrmind/validation.lmdb/" "$STAGE/data/nmrmind/validation.lmdb/"
rsync -a --exclude='lock.mdb' "$ROOT/data/nmrmind/test.lmdb/" "$STAGE/data/nmrmind/test.lmdb/"
cp "$ROOT/data/nmrmind/validation.json" "$ROOT/data/nmrmind/test.json" "$STAGE/data/nmrmind/"
rsync -a --exclude='lock.mdb' "$ROOT/data/validation/validation.lmdb/" "$STAGE/data/validation/validation.lmdb/"
cp "$ROOT/data/validation/validation.json" "$STAGE/data/validation/"
rsync -a --exclude='lock.mdb' "$ROOT/data/external_test/external_test.lmdb/" "$STAGE/data/external_test/external_test.lmdb/"
cp "$ROOT/data/external_test/external_test.json" "$ROOT/data/external_test/external_test_pos_only.json" "$STAGE/data/external_test/"
cp "$ROOT/data/f_split/raw_data.json" "$ROOT/data/f_split/clean_data.json" "$STAGE/data/f_split/"
cp "$ROOT/data/sim5/fluoroquinolones_nmr.json" "$STAGE/data/sim5/"
cp "$ROOT/data/manifest.json" "$ROOT/data/README.md" "$STAGE/data/"

echo "==> staged sizes"
du -sh "$STAGE/model" "$STAGE/data"
echo "==> staged model files"
find "$STAGE/model" -type f | sort
echo "==> staged data files (summary)"
find "$STAGE/data" -type f | wc -l

if [[ "$DRY_RUN" == "1" ]]; then
  echo "==> DRY_RUN=1, skipping upload"
  exit 0
fi

PRIVATE_FLAG=()
if [[ "$PRIVATE" == "1" ]]; then PRIVATE_FLAG=(--private); else PRIVATE_FLAG=(--public); fi

echo "==> creating repos if needed"
hf repo create "$HF_ORG/$MODEL_REPO" --repo-type model "${PRIVATE_FLAG[@]}" || true
hf repo create "$HF_ORG/$DATA_REPO" --repo-type dataset "${PRIVATE_FLAG[@]}" || true

echo "==> uploading model repo: $HF_ORG/$MODEL_REPO"
hf upload "$HF_ORG/$MODEL_REPO" "$STAGE/model" --repo-type model --commit-message "Upload CMGNet-SPrompt model weights"

echo "==> uploading dataset repo: $HF_ORG/$DATA_REPO"
hf upload "$HF_ORG/$DATA_REPO" "$STAGE/data" --repo-type dataset --commit-message "Upload CMGNet-SPrompt data"

echo "==> done"
echo "Model:   https://huggingface.co/$HF_ORG/$MODEL_REPO"
echo "Dataset: https://huggingface.co/datasets/$HF_ORG/$DATA_REPO"

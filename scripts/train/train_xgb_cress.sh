#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# XGBoost-CReSS core classifier
#
#   spectra -> frozen CReSS encoder (768-d) -> XGBoost binary head
#
# Usage:
#   bash scripts/train/train_xgb_cress.sh [out_dir] [gpu_id]
# Set INCLUDE_SYNTH=1 to add the 1.49 M synthetic corpus to the base training set.
# ---------------------------------------------------------------------------
set -u
set -o pipefail
cd "$(dirname "$0")/../.." || exit 1

OUT_DIR=${1:-output/xgb_cress_sprompt}
GPU_ID=${2:-0}
PY=${PY:-/home/lingjiebao/anaconda3/envs/py312/bin/python}

TRAIN_LMDB=${TRAIN_LMDB:-data/experimental/exp_lit.lmdb}
EXTRA_LMDB=${EXTRA_LMDB:-data/pubchem/pubchem_1209.lmdb}
EXTRA_JSON=${EXTRA_JSON:-data/experimental/ex_train.json}
VALID=${VALID:-data/validation/validation.json}
# NOTE: the original 40,642-record held-out in-domain test split is not part of
# this release, so TEST defaults to the validation split and the `test` columns
# of the report duplicate `valid`. Point TEST=... at any other JSON to evaluate.
TEST=${TEST:-data/validation/validation.json}
EXTERNAL=${EXTERNAL:-data/external_test/external_test.json}
F_SPLIT=${F_SPLIT:-data/f_split/with_f_split.json}
F_WO=${F_WO:-data/f_split/without_f_split.json}

EXTRA_ARGS=()
if [ "${INCLUDE_SYNTH:-0}" = "1" ]; then
    EXTRA_ARGS+=(--extra_train_lmdb data/synthetic/synth_predicted.lmdb)
fi

export CUDA_VISIBLE_DEVICES=$GPU_ID
mkdir -p "$OUT_DIR" logs

"$PY" -u train_xgb_cress_new_dataset.py \
    --train_lmdb "$TRAIN_LMDB" \
    --extra_train_lmdb "$EXTRA_LMDB" "${EXTRA_ARGS[@]}" \
    --extra_train_json "$EXTRA_JSON" \
    --valid_json "$VALID" \
    --test_json "$TEST" \
    --external_json "$EXTERNAL" \
    --f_split_json "$F_SPLIT" \
    --f_without_json "$F_WO" \
    --seeds "${SEEDS:-42,0,2006}" \
    --out_dir "$OUT_DIR" 2>&1 | tee "logs/$(basename "$OUT_DIR").log"

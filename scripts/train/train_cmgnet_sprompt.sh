#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# CMGNet-SPrompt training (ring-system prompt + NMR encoder/decoder)
#
# Starts from the 0910 base model (`checkpoints/bart_base_0910`).  Pass
# CKPT=/path/to/checkpoint.ckpt to resume from a full Lightning checkpoint
# instead.
#
# Training corpora (rebuilt LMDBs, see data/README.md):
#   data/experimental/exp_lit.lmdb            49,984  (experimental literature)
#   data/synthetic/synth_predicted.lmdb    1,488,432  (predicted spectra)
#   data/pubchem/pubchem_1209.lmdb            77,328  (PubChem positives)
#   data/experimental/ex_train.ring.json      24,520  (nmrshiftdb extra train)
#
# Usage:
#   bash scripts/train/train_cmgnet_sprompt.sh [seeds] [gpu_id]
#
# seeds: comma-separated, default "0,42,2006"; each seed gets its own
#        output directory and log.
# ---------------------------------------------------------------------------
set -u
cd "$(dirname "$0")/../.." || exit 1

SEED_ARG=${1:-}
GPU_ID=${2:-0}
SEEDS=${SEED_ARG:-${SEEDS:-0,42,2006}}

PY=${PY:-python}
CKPT=${CKPT:-}                                     # empty => start from 0910
BASE=${BASE:-checkpoints/bart_base_0910}
TRAIN_LMDB=${TRAIN_LMDB:-data/experimental/exp_lit.lmdb,data/synthetic/synth_predicted.lmdb,data/pubchem/pubchem_1209.lmdb}
EXTRA_TRAIN=${EXTRA_TRAIN:-data/experimental/ex_train.ring.json}
VALID=${VALID:-data/validation/validation.json}
TEST=${TEST:-data/external_test/external_test.json}

for f in "$BASE" "$EXTRA_TRAIN" "$VALID" "$TEST"; do
    [ -e "$f" ] || { echo "missing: $f"; exit 1; }
done
if [ -n "$CKPT" ] && [ ! -e "$CKPT" ]; then
    echo "missing CKPT: $CKPT"; exit 1
fi
IFS=',' read -ra LMDB_ARR <<< "$TRAIN_LMDB"
for d in "${LMDB_ARR[@]}"; do
    [ -e "$d/data.mdb" ] || { echo "missing lmdb: $d"; exit 1; }
done

export CUDA_VISIBLE_DEVICES=$GPU_ID
mkdir -p logs

IFS=',' read -ra SEED_ARR <<< "$SEEDS"
STATUS=0

for RAW_SEED in "${SEED_ARR[@]}"; do
    SEED="$(echo "$RAW_SEED" | xargs)"
    [ -n "$SEED" ] || continue

    SAVE_PATH=${SAVE_PATH:-output/train_cmgnet_sprompt_seed${SEED}}
    LOG=${LOG:-logs/train_cmgnet_sprompt_seed${SEED}.log}
    mkdir -p "$SAVE_PATH"

    echo "==== train CMGNet-SPrompt seed=$SEED gpu=$GPU_ID start $(date '+%F %T') ====" | tee -a "$LOG"

    CKPT_ARGS=()
    [ -n "$CKPT" ] && CKPT_ARGS=(--ckpt "$CKPT")

    NCCL_P2P_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" -u main_cmgnet.py \
        --save_path "$SAVE_PATH" \
        "${CKPT_ARGS[@]}" \
        --model_dir "$BASE" \
        --train_file "$EXTRA_TRAIN" \
        --valid_file "$VALID" \
        --test_file "$TEST" \
        --lmdb_path "$TRAIN_LMDB" \
        --lr 1e-4 \
        --num_workers 12 \
        --epochs "${EPOCHS:-5}" --eval_per_epoch 1 \
        --warmup_ratio 0.02 \
        --label_smoothing 0. \
        --seed "$SEED" \
        --batch_size "${BATCH_SIZE:-48}" \
        --gradient_accumulation_steps "${ACCUM:-2}" \
        --do_train \
        --gpus 1 \
        --smiles_aug_proba 0.5 \
        --nmr_aug_proba 0.5 \
        --nmr_aug_scale 3.0 \
        --use_ring_system \
        ${DEBUG:+--debug} >> "$LOG" 2>&1

    RC=$?
    echo "==== seed=$SEED exit=$RC $(date '+%F %T') ====" | tee -a "$LOG"
    [ "$RC" -eq 0 ] || STATUS=$RC
done

exit "$STATUS"

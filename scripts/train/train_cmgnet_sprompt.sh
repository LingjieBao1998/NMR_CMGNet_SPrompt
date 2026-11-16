#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# CMGNet-SPrompt training (ring-system prompt + NMR encoder/decoder)
#
# Training corpora (rebuilt LMDBs, see data/README.md):
#   data/experimental/exp_lit.lmdb            49,984  (experimental literature)
#   data/synthetic/synth_predicted.lmdb    1,488,432  (predicted spectra)
#   data/pubchem/pubchem_1209.lmdb            77,328  (PubChem positives)
#   data/experimental/ex_train.ring.json      24,520  (nmrshiftdb extra train)
#
# Usage:
#   bash scripts/train/train_cmgnet_sprompt.sh [seed] [gpu_id]
# ---------------------------------------------------------------------------
set -u
cd "$(dirname "$0")/../.." || exit 1

SEED=${1:-42}
GPU_ID=${2:-0}

PY=${PY:-/home/lingjiebao/anaconda3/envs/py312/bin/python}
CKPT=${CKPT:-checkpoints/train_cmgnet_ring_system_3.2_1029_v2_best.ckpt}
BASE=${BASE:-checkpoints/bart_base_0910}
TRAIN_LMDB=${TRAIN_LMDB:-data/experimental/exp_lit.lmdb,data/synthetic/synth_predicted.lmdb,data/pubchem/pubchem_1209.lmdb}
EXTRA_TRAIN=${EXTRA_TRAIN:-data/experimental/ex_train.ring.json}
VALID=${VALID:-data/validation/validation.json}
TEST=${TEST:-data/external_test/external_test.json}

SAVE_PATH=${SAVE_PATH:-output/train_cmgnet_sprompt_seed${SEED}}
LOG=${LOG:-logs/train_cmgnet_sprompt_seed${SEED}.log}

for f in "$CKPT" "$BASE" "$EXTRA_TRAIN" "$VALID" "$TEST"; do
    [ -e "$f" ] || { echo "missing: $f"; exit 1; }
done
IFS=',' read -ra LMDB_ARR <<< "$TRAIN_LMDB"
for d in "${LMDB_ARR[@]}"; do
    [ -e "$d/data.mdb" ] || { echo "missing lmdb: $d"; exit 1; }
done

export CUDA_VISIBLE_DEVICES=$GPU_ID
mkdir -p "$SAVE_PATH" logs

echo "==== train CMGNet-SPrompt seed=$SEED gpu=$GPU_ID start $(date '+%F %T') ====" | tee -a "$LOG"

NCCL_P2P_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" -u main_cmgnet.py \
    --save_path "$SAVE_PATH" \
    --ckpt "$CKPT" \
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

STATUS=$?
echo "==== exit=$STATUS $(date '+%F %T') ====" | tee -a "$LOG"
exit $STATUS

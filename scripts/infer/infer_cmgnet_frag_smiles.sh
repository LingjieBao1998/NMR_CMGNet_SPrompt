#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Baseline CMGNet generation (frag / smiles modes, no prompt gating)
#
# Usage:
#   bash scripts/infer/infer_cmgnet_frag_smiles.sh [test_json] [out_dir] [mode] [gpu]
#   mode: frag | smiles
# ---------------------------------------------------------------------------
set -u
cd "$(dirname "$0")/../.." || exit 1

TEST_JSON=${1:-data/external_test/external_test.json}
OUT_DIR=${2:-output/infer_frag_external}
MODE=${3:-frag}
GPU_ID=${4:-0}

PY=${PY:-/home/lingjiebao/venvs/py312_tf457/bin/python}
CKPT=${CKPT:-checkpoints/train_cmgnet_ring_system_3.2_1029_v2_best.ckpt}
BASE=${BASE:-checkpoints/bart_base_0910}
BEAMS=${BEAMS:-10}

for f in "$TEST_JSON" "$CKPT" "$BASE"; do
    [ -e "$f" ] || { echo "missing: $f"; exit 1; }
done

export CUDA_VISIBLE_DEVICES=$GPU_ID
mkdir -p "$OUT_DIR" logs
LOG="$OUT_DIR/infer_${MODE}.log"

NCCL_P2P_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" -u main_cmgnet.py \
    --save_path "$OUT_DIR" \
    --ckpt "$CKPT" \
    --model_dir "$BASE" \
    --train_file "" --valid_file "" \
    --test_file "$TEST_JSON" \
    --num_workers 8 --batch_size 48 \
    --do_test --gpus 1 \
    --num_beams "$BEAMS" \
    --decode_mode "$MODE" \
    --smiles_aug_proba 0.5 --nmr_aug_proba 0.5 --nmr_aug_scale 3.0 \
    --use_ring_system 2>&1 | tee "$LOG"

PRED=$(ls -t "$OUT_DIR"/test_${MODE}_beams${BEAMS}_samples*.json 2>/dev/null | head -1)
if [ -n "$PRED" ]; then
    "$PY" -u summarize_core_smiles.py "$OUT_DIR/summary.md" "CMGNet-${MODE}" "$PRED"
fi

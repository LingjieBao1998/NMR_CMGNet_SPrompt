#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# CMGNet-SPrompt inference
#
#   step 1  XGBoost-CReSS gate  -> 0/1 votes per spectrum (threshold 0.5)
#   step 2  CMGNet conditional generation with the core prompt injected for
#           gated spectra (`--decode_mode sprompt`), beams = 10
#   step 3  core-hit summary of the generated top-10 candidates
#
# Usage:
#   bash scripts/infer/infer_cmgnet_sprompt.sh [test_json] [out_dir] [gpu_id]
# Defaults:
#   test_json = data/external_test/external_test.json (4,502 spectra, 64 pos)
# ---------------------------------------------------------------------------
set -u
cd "$(dirname "$0")/../.." || exit 1

TEST_JSON=${1:-data/external_test/external_test.json}
OUT_DIR=${2:-output/infer_sprompt_external}
GPU_ID=${3:-0}

PY=${PY:-/home/lingjiebao/venvs/py312_tf457/bin/python}
CKPT=${CKPT:-checkpoints/sprompt_ex1209_1ep_seed0.ckpt}
BASE=${BASE:-checkpoints/bart_base_0910}
BEAMS=${BEAMS:-10}
CORE=${CORE:-"O=C(O)c1c[nH]c2ccccc2c1=O"}
# SPrompt = `nmr+frag` decoding with the core prompt injected only for gated
# spectra.  `sprompt` is accepted as an alias by main_cmgnet.py (same branch),
# but `nmr+frag` matches the output names of the published runs.
MODE=${MODE:-nmr+frag}
NW=${NW:-8}

for f in "$TEST_JSON" "$CKPT" "$BASE"; do
    [ -e "$f" ] || { echo "missing: $f"; exit 1; }
done

export CUDA_VISIBLE_DEVICES=$GPU_ID
mkdir -p "$OUT_DIR" logs
VOTES=${VOTES:-"$OUT_DIR/sprompt_votes.json"}
LOG="$OUT_DIR/infer.log"

if [ -s "$VOTES" ]; then
    echo "==== [1/3] reuse gate votes: $VOTES ====" | tee -a "$LOG"
else
    echo "==== [1/3] XGBoost-CReSS gate $(date '+%F %T') ====" | tee -a "$LOG"
    "$PY" -u scripts/infer/make_sprompt_votes.py \
        --test_json "$TEST_JSON" \
        --out_json "$VOTES" \
        --device cuda >> "$LOG" 2>&1 || exit 1
fi

echo "==== [2/3] CMGNet-SPrompt generation $(date '+%F %T') ====" | tee -a "$LOG"
NCCL_P2P_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" -u main_cmgnet.py \
    --save_path "$OUT_DIR" \
    --ckpt "$CKPT" \
    --model_dir "$BASE" \
    --train_file "" --valid_file "" \
    --test_file "$TEST_JSON" \
    --num_workers "$NW" --batch_size 48 \
    --do_test --gpus 1 \
    --num_beams "$BEAMS" \
    --decode_mode "$MODE" \
    --sprompt_votes "$VOTES" \
    --sprompt_core "$CORE" \
    --smiles_aug_proba 0.5 --nmr_aug_proba 0.5 --nmr_aug_scale 3.0 \
    --use_ring_system >> "$LOG" 2>&1 || exit 1

echo "==== [3/3] core-hit summary $(date '+%F %T') ====" | tee -a "$LOG"
PRED=$(ls -t "$OUT_DIR"/test_${MODE}_beams${BEAMS}_samples*.json 2>/dev/null | head -1)
if [ -n "$PRED" ]; then
    "$PY" -u summarize_core_smiles.py "$OUT_DIR/summary.md" "CMGNet-SPrompt" "$PRED" >> "$LOG" 2>&1
    echo "prediction: $PRED"
    echo "summary:    $OUT_DIR/summary.md"
else
    echo "warning: prediction json not found under $OUT_DIR" | tee -a "$LOG"
fi
echo "==== done $(date '+%F %T') ====" | tee -a "$LOG"

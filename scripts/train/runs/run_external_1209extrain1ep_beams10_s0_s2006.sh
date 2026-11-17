#!/bin/bash
# 1209+extrain 1ep 重训 seed0/seed2006 的外部测试评测 (frag/smiles, 每 seed 2 run)
# 用法: bash run_external_1209extrain1ep_beams10_s0_s2006.sh <seed> <cuda_id>
# 口径与 run_external_1ep_retrain_beams10.sh 完全一致 (beams=10, external 4502)

set -u
cd /mnt/data2/NMR_Core_Mol_Detection

SEED=$1
GPU_ID=$2
TEST=./data/nmrmind_cmgnet_1004/external_test.json
PY=/home/lingjiebao/anaconda3/envs/py312/bin/python
CKPT=output/train_newdb_1209_extrain_1ep_seed${SEED}_retrain/checkpoints/epoch_00.ckpt
TAG=1209extrain1ep_seed${SEED}

export CUDA_VISIBLE_DEVICES=$GPU_ID

if [ ! -e "$CKPT" ]; then
    echo "错误: 找不到 $CKPT"
    exit 1
fi

for mode in frag smiles; do
    SAVE=output/external_4502_${TAG}_${mode}_beams10
    LOG=logs/run_external_${TAG}_${mode}_beams10.log
    OUT_JSON="$SAVE/test_${mode}_beams10_samples4502.json"
    echo "==================== $TAG mode=$mode 开始 $(date '+%F %T') ===================="
    mkdir -p "$SAVE"

    NCCL_P2P_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u main_cmgnet.py \
        --save_path "$SAVE" \
        --ckpt "$CKPT" \
        --train_file "" \
        --valid_file "" \
        --test_file "$TEST" \
        --num_workers 12 \
        --batch_size 48 \
        --do_test \
        --gpus 1 \
        --num_beams 10 \
        --decode_mode "$mode" \
        --smiles_aug_proba 0.5 \
        --nmr_aug_proba 0.5 \
        --nmr_aug_scale 3.0 \
        --use_ring_system >> "$LOG" 2>&1
    STATUS=$?
    echo "==================== $TAG mode=$mode 结束 (exit=$STATUS) $(date '+%F %T') ===================="
    [ "$STATUS" -ne 0 ] && echo "!! 失败: $LOG"
done
echo "==================== $TAG 全部结束 $(date '+%F %T') ===================="

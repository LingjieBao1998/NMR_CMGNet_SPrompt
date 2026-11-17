#!/bin/bash
# 1209+extrain 真 1ep 三 seed 的 oracle nmr+frag 评测 (--oracle_ring, beams=10)
# 用法: bash run_external_1209extrain1ep_oracle_s0_s2006_s42.sh <seed> <cuda_id>
# 口径与 run_1029v2_oracle_single_gpu.sh 一致: 全量 external_test.json, 正样本喂真实环系,
# 负样本退化为纯 NMR; decode_mode=nmr+frag

set -u
cd /mnt/data2/NMR_Core_Mol_Detection

SEED=$1
GPU_ID=$2
# 可选第 3 参: ckpt 覆盖(seed42 的 1ep 重训目录名带 5ep, 与 0/2006 不同)
CKPT=${3:-output/train_newdb_1209_extrain_1ep_seed${SEED}_retrain/checkpoints/epoch_00.ckpt}
TEST=./data/nmrmind_cmgnet_1004/external_test.json
PY=/home/lingjiebao/anaconda3/envs/py312/bin/python
TAG=1209extrain1ep_seed${SEED}_oracle

export CUDA_VISIBLE_DEVICES=$GPU_ID
export CMGNET_FAST_CORE_CHECK=1

if [ ! -e "$CKPT" ]; then
    echo "错误: 找不到 $CKPT"
    exit 1
fi

SAVE=output/external_4502_${TAG}_beams10
LOG=logs/run_external_${TAG}_beams10.log
echo "==================== $TAG 开始 $(date '+%F %T') ===================="
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
    --decode_mode "nmr+frag" \
    --oracle_ring \
    --smiles_aug_proba 0.5 \
    --nmr_aug_proba 0.5 \
    --nmr_aug_scale 3.0 \
    --use_ring_system >> "$LOG" 2>&1
STATUS=$?
echo "==================== $TAG 结束 (exit=$STATUS) $(date '+%F %T') ===================="
[ "$STATUS" -ne 0 ] && echo "!! 失败, 看 $LOG"

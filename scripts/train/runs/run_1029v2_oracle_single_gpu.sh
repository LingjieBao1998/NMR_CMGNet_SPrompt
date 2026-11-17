#!/bin/bash
# 1029_v2 (web 线上权重) 的 oracle nmr+frag 评测, 单卡 (无 DDP)
# 用法: bash run_1029v2_oracle_single_gpu.sh <gpu_id> <half: even|odd|external> <save_dir>
# 配置与 1004 的 oracle 评测完全一致 (beams=10, batch=48, --oracle_ring), 仅换权重
# 注: CUDA_VISIBLE_DEVICES=0 实为 4090, =1 为 5090

GPU_ID=$1
HALF=$2
SAVE_DIR=$3

cd /mnt/data2/NMR_Core_Mol_Detection

CKPT=models/cmgnet/train_cmgnet_ring_system_3.2_1029_v2_best.ckpt

export CUDA_VISIBLE_DEVICES=$GPU_ID
export PYTHONPATH=/tmp/faultdiag:$PYTHONPATH
export CMGNET_FAST_CORE_CHECK=1

if [ "$HALF" = "external" ]; then
    TEST_FILE=./data/nmrmind_cmgnet_1004/external_test.json
elif [ "$HALF" = "posonly" ]; then
    ## 只含 label=1 的阳性样本 (oracle 模式只评阳性)
    TEST_FILE=./data/nmrmind_cmgnet_1004/external_test_pos_only.json
else
    TEST_FILE=./data/nmrmind_cmgnet_1004/test_${HALF}_half.json
fi

echo "==================== 1029v2 gpu=$GPU_ID half=$HALF 开始 $(date '+%F %T') ===================="
NCCL_P2P_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u main_cmgnet.py \
    --save_path $SAVE_DIR \
    --ckpt $CKPT \
    --train_file "" \
    --valid_file "" \
    --test_file $TEST_FILE \
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
    --use_ring_system
echo "==================== 1029v2 gpu=$GPU_ID half=$HALF 结束 $(date '+%F %T') ===================="

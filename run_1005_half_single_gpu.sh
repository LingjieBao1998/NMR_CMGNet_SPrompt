#!/bin/bash
# 单卡半边评测任务 (无 DDP, 因此没有 rank 不均衡/NCCL watchdog 问题)
# 用法: bash run_1005_half_single_gpu.sh <gpu_id> <mode> <half: even|odd|external> <save_dir> [oracle: 0|1]
#   mode: smiles | frag | nmr+frag ; half 对应 test_{even,odd}_half.json, external 用 external_test.json
#   oracle=1 时给 main_cmgnet.py 传 --oracle_ring (仅 nmr+frag 模式生效:
#     label==1 的样本附真实环系片段, 其余退化为纯NMR输入; 参考用完整分子)
# 背景: 2026-10-05 双卡评测两次因 rank0(慢卡)落后~30分钟触发 NCCL watchdog 丢结果,
#       改为两个半边单卡任务并行/接力, 各自独立落 JSON, 最后离线合并。

GPU_ID=$1
MODE=$2
HALF=$3
SAVE_DIR=$4
ORACLE=$5

ORACLE_ARG=""
if [ "$ORACLE" = "1" ]; then
    ORACLE_ARG="--oracle_ring"
fi

cd /mnt/data2/NMR_Core_Mol_Detection

CKPT=output/train_cmgnet_ring_system_3.2_nmrmind_1004/best_models/best-epoch=09-step=222690-val/score=-0.0203.ckpt

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

echo "==================== gpu=$GPU_ID mode=$MODE half=$HALF 开始 $(date '+%F %T') ===================="
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
    --decode_mode "$MODE" \
    $ORACLE_ARG \
    --smiles_aug_proba 0.5 \
    --nmr_aug_proba 0.5 \
    --nmr_aug_scale 3.0 \
    --use_ring_system
echo "==================== gpu=$GPU_ID mode=$MODE half=$HALF 结束 $(date '+%F %T') ===================="

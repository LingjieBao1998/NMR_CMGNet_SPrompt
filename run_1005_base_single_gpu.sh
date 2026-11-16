#!/bin/bash
# saved_hf_model_0910 (本次微调前的初始权重) 的 smiles 评测, 单卡
# 用法: bash run_1005_base_single_gpu.sh <gpu_id> <half: even|odd|external> <save_dir>
# 原理: 不带 --ckpt 且 save_dir 为全新目录 -> main() 走 "use default model" 分支,
#       CustomModel(args) 从 --model_dir 加载基座权重 (与训练起步时完全一致)。
# 配置与微调后模型的评测完全一致: smiles / beams=10 / batch=48 / fast core check

GPU_ID=$1
HALF=$2
SAVE_DIR=$3

cd /mnt/data2/NMR_Core_Mol_Detection

export CUDA_VISIBLE_DEVICES=$GPU_ID
export PYTHONPATH=/tmp/faultdiag:$PYTHONPATH
export CMGNET_FAST_CORE_CHECK=1

if [ "$HALF" = "external" ]; then
    TEST_FILE=./data/nmrmind_cmgnet_1004/external_test.json
else
    TEST_FILE=./data/nmrmind_cmgnet_1004/test_${HALF}_half.json
fi

echo "==================== base0910 gpu=$GPU_ID half=$HALF 开始 $(date '+%F %T') ===================="
NCCL_P2P_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u main_cmgnet.py \
    --save_path $SAVE_DIR \
    --model_dir ./checkpoints/saved_hf_model_0910 \
    --train_file "" \
    --valid_file "" \
    --test_file $TEST_FILE \
    --num_workers 12 \
    --batch_size 48 \
    --do_test \
    --gpus 1 \
    --num_beams 10 \
    --decode_mode smiles \
    --smiles_aug_proba 0.5 \
    --nmr_aug_proba 0.5 \
    --nmr_aug_scale 3.0 \
    --use_ring_system
echo "==================== base0910 gpu=$GPU_ID half=$HALF 结束 $(date '+%F %T') ===================="

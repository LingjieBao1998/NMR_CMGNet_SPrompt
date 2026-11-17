#!/bin/bash
# 以 1029_v2 线上权重为初始权重, 在 new_db.lmdb (3,196,326 条) 上微调 3 epochs
# seed=0, 单卡, CUDA_VISIBLE_DEVICES=0 -> 物理 4090 (24G)
# 参照 run_1010_cmgnet_seed_single_gpu.sh:
#   - 其它超参与 1004/1010 实验对齐 (lr 1e-4, cosine, warmup 0.02, aug scale 3.0)
#   - 每卡 batch 48, 梯度累积 2 -> 全局 batch 96
#   - valid/test 用 data/nmrmind_cmgnet_1004 的固定评测集 (原 data_valid_test_1020 已不在本机)
#   - 只训不测: 训练结束直接退出, 不跑 --do_test (validation 仍每轮做, 用于保存 best)
# 用法: bash run_newdb_1029v2_3ep_seed0.sh

set -u
cd /mnt/data2/NMR_Core_Mol_Detection

SEED=0
GPU_ID=0          # CUDA 编号: 0=4090, 1=5090 (与 nvidia-smi 顺序相反)
EPOCHS=3

CKPT=models/cmgnet/train_cmgnet_ring_system_3.2_1029_v2_best.ckpt
LMDB=./data/NMR_other_data_1020_add_ring_system_1029/new_db.lmdb
VALID=./data/nmrmind_cmgnet_1004/validation.json
TEST=./data/nmrmind_cmgnet_1004/test.json
SAVE_PATH=output/train_newdb_1029v2_3ep_seed${SEED}
LOG_FILE=run_newdb_1029v2_3ep_seed${SEED}.log

for f in "$CKPT" "$LMDB/data.mdb" "$VALID" "$TEST"; do
    if [ ! -e "$f" ]; then
        echo "错误: 找不到 $f"
        exit 1
    fi
done

export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "==================== newdb 1029v2 seed=$SEED gpu=$GPU_ID (CUDA编号) 开始 $(date '+%F %T') ====================" | tee -a "$LOG_FILE"
mkdir -p "$SAVE_PATH"

NCCL_P2P_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u main_cmgnet.py \
    --save_path "$SAVE_PATH" \
    --ckpt "$CKPT" \
    --train_file "" \
    --valid_file "$VALID" \
    --test_file "$TEST" \
    --lmdb_path "$LMDB" \
    --lr 1e-4 \
    --num_workers 12 \
    --epochs $EPOCHS --eval_per_epoch 1 \
    --warmup_ratio 0.02 \
    --label_smoothing 0. \
    --seed $SEED \
    --batch_size 48 \
    --gradient_accumulation_steps 2 \
    --do_train \
    --gpus 1 \
    --smiles_aug_proba 0.5 \
    --nmr_aug_proba 0.5 \
    --nmr_aug_scale 3.0 \
    --use_ring_system 2>&1 | tee -a "$LOG_FILE"
STATUS=${PIPESTATUS[0]}
echo "==================== newdb 1029v2 seed=$SEED 结束 (python exit=$STATUS) $(date '+%F %T') ====================" | tee -a "$LOG_FILE"
exit $STATUS

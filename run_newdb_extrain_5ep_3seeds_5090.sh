#!/bin/bash
# new_db + ex_train JSON 微调 (不含 new_data_1209)
#   训练数据: data/NMR_other_data_1020_add_ring_system_1029/new_db.lmdb (3,196,326)
#             data/extra_training/ex_train_add_label_filtered.evalclean.ring.json (24,520)
#   基座:     models/cmgnet/train_cmgnet_ring_system_3.2_1029_v2_best.ckpt
#   配置:     5 epochs, batch 48 x grad_accum 2 (等效 96), seed 0/2006/42
#   卡:       5090 (CUDA id 1; 0=4090, 1=5090)
#   顺序跑 3 个 seed, 约 7h50m/seed, 总计约 23-24 小时
# 用法: bash run_newdb_extrain_5ep_3seeds_5090.sh

set -u
cd /mnt/data2/NMR_Core_Mol_Detection

GPU_ID=1
PY=/home/lingjiebao/anaconda3/envs/py312/bin/python
CKPT=models/cmgnet/train_cmgnet_ring_system_3.2_1029_v2_best.ckpt
TRAIN_FILE=data/extra_training/ex_train_add_label_filtered.evalclean.ring.json
LMDB=./data/NMR_other_data_1020_add_ring_system_1029/new_db.lmdb
VALID=./data/nmrmind_cmgnet_1004/validation.json
TEST=./data/nmrmind_cmgnet_1004/test.json
SEEDS="0 2006 42"

for f in "$CKPT" "$TRAIN_FILE" "$LMDB/data.mdb" "$VALID" "$TEST"; do
    if [ ! -e "$f" ]; then
        echo "错误: 找不到 $f"
        exit 1
    fi
done

export CUDA_VISIBLE_DEVICES=$GPU_ID

for SEED in $SEEDS; do
    SAVE_PATH=output/train_newdb_extrain_5ep_seed${SEED}
    LOG=logs/train_newdb_extrain_5ep_seed${SEED}.log
    mkdir -p "$SAVE_PATH"

    echo "==================== train newdb+extrain seed=$SEED 开始 $(date '+%F %T') ====================" | tee -a "$LOG"

    NCCL_P2P_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u main_cmgnet.py \
        --save_path "$SAVE_PATH" \
        --ckpt "$CKPT" \
        --train_file "$TRAIN_FILE" \
        --valid_file "$VALID" \
        --test_file "$TEST" \
        --lmdb_path "$LMDB" \
        --lr 1e-4 \
        --num_workers 12 \
        --epochs 5 --eval_per_epoch 1 \
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
        --use_ring_system >> "$LOG" 2>&1
    STATUS=$?
    echo "EXIT:$STATUS $(date '+%F %T')" | tee -a "$LOG"
done

echo "==================== 3 个 seed 全部结束 $(date '+%F %T') ====================" | tee -a logs/train_newdb_extrain_5ep_3seeds.log

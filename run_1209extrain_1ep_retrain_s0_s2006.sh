#!/bin/bash
# 1209+extrain 真 1ep 重训: seed 0 与 seed 2006, 双卡并行
# 用法: bash run_1209extrain_1ep_retrain_s0_s2006.sh <seed> <cuda_id>
#   seed0   -> cuda 0 (4090)
#   seed2006 -> cuda 1 (5090)
# 超参与 run_seed42_dual_retrain_5ep.sh 的 B 配方完全一致, 仅换 seed;
# 唯一区别: --epochs 1 (seed42_retrain 也是 1ep 后停, SaveEveryEpoch 落 checkpoints/epoch_00.ckpt)

set -u
cd /mnt/data2/NMR_Core_Mol_Detection

SEED=$1
GPU_ID=$2
PY=/home/lingjiebao/anaconda3/envs/py312/bin/python
BASE=models/cmgnet/train_cmgnet_ring_system_3.2_1029_v2_best.ckpt
VALID=./data/nmrmind_cmgnet_1004/validation.json
TEST=./data/nmrmind_cmgnet_1004/test.json
EX_TRAIN=data/extra_training/ex_train_add_label_filtered.evalclean.ring.json
NEW_DB=./data/NMR_other_data_1020_add_ring_system_1029/new_db.lmdb
LMDB_1209=data/new_data_1209.filtered.ring.lmdb
SAVE=output/train_newdb_1209_extrain_1ep_seed${SEED}_retrain
LOG=$SAVE.log

export CUDA_VISIBLE_DEVICES=$GPU_ID

for f in "$BASE" "$EX_TRAIN" "$NEW_DB/data.mdb" "$LMDB_1209/data.mdb" "$VALID" "$TEST"; do
    if [ ! -e "$f" ]; then
        echo "错误: 找不到 $f"
        exit 1
    fi
done

mkdir -p "$SAVE"
echo "==================== train $SAVE (seed=$SEED gpu=$GPU_ID) 开始 $(date '+%F %T') ===================="

NCCL_P2P_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u main_cmgnet.py \
    --save_path "$SAVE" \
    --ckpt "$BASE" \
    --train_file "$EX_TRAIN" \
    --valid_file "$VALID" \
    --test_file "$TEST" \
    --lmdb_path "$NEW_DB,$LMDB_1209" \
    --lr 1e-4 \
    --num_workers 12 \
    --epochs 1 --eval_per_epoch 1 \
    --warmup_ratio 0.02 \
    --label_smoothing 0. \
    --seed "$SEED" \
    --batch_size 48 \
    --gradient_accumulation_steps 2 \
    --do_train \
    --gpus 1 \
    --smiles_aug_proba 0.5 \
    --nmr_aug_proba 0.5 \
    --nmr_aug_scale 3.0 \
    --use_ring_system >> "$LOG" 2>&1
STATUS=$?

echo "EXIT:$STATUS $(date '+%F %T')"
echo "==================== train $SAVE 结束 (exit=$STATUS) $(date '+%F %T') ===================="

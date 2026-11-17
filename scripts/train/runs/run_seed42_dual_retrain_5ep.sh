#!/bin/bash
# seed42 双配方重训 (1ep, 用户指示), 顺序跑在 5090 (CUDA id 1), 每配置约 1.6h, 总计约 3.5h:
#   A) 新组配方: new_db.lmdb + ex_train json (不含 1209)   -> train_newdb_extrain_5ep_seed42_retrain
#   B) 旧波配方: new_db.lmdb + new_data_1209.filtered.ring.lmdb + ex_train json -> train_newdb_1209_extrain_5ep_seed42_retrain
# 其余超参与各自原始 run 完全一致 (seed42). main_cmgnet.py 的 SaveEveryEpoch 回调
# 会在训练结束后把真 1ep 权重存到 save_path/checkpoints/epoch_00.ckpt (weights-only).
# 用法: tmux new-session -d -s seed42_dual 'bash run_seed42_dual_retrain_5ep.sh > logs/seed42_dual_retrain.log 2>&1'

set -u
cd /mnt/data2/NMR_Core_Mol_Detection

GPU_ID=1
PY=/home/lingjiebao/anaconda3/envs/py312/bin/python
BASE=models/cmgnet/train_cmgnet_ring_system_3.2_1029_v2_best.ckpt
VALID=./data/nmrmind_cmgnet_1004/validation.json
TEST=./data/nmrmind_cmgnet_1004/test.json
EX_TRAIN=data/extra_training/ex_train_add_label_filtered.evalclean.ring.json
NEW_DB=./data/NMR_other_data_1020_add_ring_system_1029/new_db.lmdb
LMDB_1209=data/new_data_1209.filtered.ring.lmdb

export CUDA_VISIBLE_DEVICES=$GPU_ID

for f in "$BASE" "$EX_TRAIN" "$NEW_DB/data.mdb" "$LMDB_1209/data.mdb" "$VALID" "$TEST"; do
    if [ ! -e "$f" ]; then
        echo "错误: 找不到 $f"
        exit 1
    fi
done

train_one () {  # $1=save_path  $2...=lmdb 参数(可多个)
    local SAVE=$1; shift
    local LOG=$SAVE.log
    mkdir -p "$SAVE"
    echo "==================== train $SAVE 开始 $(date '+%F %T') ====================" | tee -a "$LOG"

    NCCL_P2P_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u main_cmgnet.py \
        --save_path "$SAVE" \
        --ckpt "$BASE" \
        --train_file "$EX_TRAIN" \
        --valid_file "$VALID" \
        --test_file "$TEST" \
        "$@" \
        --lr 1e-4 \
        --num_workers 12 \
        --epochs 1 --eval_per_epoch 1 \
        --warmup_ratio 0.02 \
        --label_smoothing 0. \
        --seed 42 \
        --batch_size 48 \
        --gradient_accumulation_steps 2 \
        --do_train \
        --gpus 1 \
        --smiles_aug_proba 0.5 \
        --nmr_aug_proba 0.5 \
        --nmr_aug_scale 3.0 \
        --use_ring_system >> "$LOG" 2>&1
    local STATUS=$?
    echo "EXIT:$STATUS $(date '+%F %T')" | tee -a "$LOG"
    echo "==================== train $SAVE 结束 (exit=$STATUS) $(date '+%F %T') ===================="
}

# ---- A) 新组配方 (无 1209) ----
train_one output/train_newdb_extrain_5ep_seed42_retrain \
    --lmdb_path "$NEW_DB"

# ---- B) 旧波配方 (加 1209 正样本; lmdb_path 为逗号分隔单参数, 见 main_cmgnet.py split(",")) ----
train_one output/train_newdb_1209_extrain_5ep_seed42_retrain \
    --lmdb_path "$NEW_DB,$LMDB_1209"

echo "==================== 两个配方全部结束 $(date '+%F %T') ===================="

#!/bin/bash
# Wait for the seed2006 CMGNet run (GPU1 / 5090) to finish, then start seed42
# on the freed GPU1 with the same 5-epoch configuration.
set -u
cd /mnt/data2/NMR_Core_Mol_Detection

LOG=logs/train_newdb_1209_extrain_5ep_seed42.log
echo "waiting for cmg1209_s2006: $(date '+%F %T')" >> "$LOG"
while tmux has-session -t cmg1209_s2006 2>/dev/null; do
    sleep 30
done

# Do not start before 2026-10-21 03:00:00 (scheduled launch).
TARGET_TS=$(date -d "2026-10-21 03:00:00" +%s)
echo "scheduled launch at $(date -d "@$TARGET_TS" '+%F %T')" >> "$LOG"
while [ "$(date +%s)" -lt "$TARGET_TS" ]; do
    sleep 30
done
sleep 20

echo "starting seed42 on GPU1: $(date '+%F %T')" >> "$LOG"
CUDA_VISIBLE_DEVICES=1 NCCL_P2P_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/lingjiebao/anaconda3/envs/py312/bin/python -u main_cmgnet.py \
    --save_path output/train_newdb_1209_extrain_5ep_seed42 \
    --ckpt models/cmgnet/train_cmgnet_ring_system_3.2_1029_v2_best.ckpt \
    --train_file data/extra_training/ex_train_add_label_filtered.evalclean.ring.json \
    --valid_file data/nmrmind_cmgnet_1004/validation.json \
    --test_file data/nmrmind_cmgnet_1004/test.json \
    --lmdb_path "data/NMR_other_data_1020_add_ring_system_1029/new_db.lmdb,data/new_data_1209.filtered.ring.lmdb" \
    --lr 1e-4 \
    --num_workers 12 \
    --epochs 5 --eval_per_epoch 1 \
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
echo "EXIT:$? $(date '+%F %T')" >> "$LOG"

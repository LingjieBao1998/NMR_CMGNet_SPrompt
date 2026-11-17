#!/bin/bash
# NMRMind 数据训练 CMGNet (ring_system 三模式混合训练) —— 续训版
# 原运行 2026-10-05 04:01 被中断于 epoch 3 第 53 步 (step 66859)，
# 现从 best_models/last.ckpt (epoch=2, step=66807) 续训。
#
# 注意: main_cmgnet.py --resume 读取 $SAVE_PATH/checkpoints/last.ckpt，
# 该文件已由 best_models/last.ckpt 复制而来（一次性操作，勿重复复制）。

NUM_NODES=1
NUM_GPUS_PER_NODE=2

ACCUM_STEP=1
# 注意: 三模式混合训练下每条样本在 collate_fn 里最多展开成 3 行,
# 所以每卡"有效"行数 = BATCH_SIZE/NUM_GPUS_PER_NODE*3。
# 128(每卡64*3=192行) 会在 4090(24G) 上 OOM; 96(每卡48*3=144行) 峰值约 18G, 安全。
BATCH_SIZE=96

export CUDA_VISIBLE_DEVICES=0,1

SAVE_PATH=output/train_cmgnet_ring_system_3.2_nmrmind_1004

set -x
mkdir -p $SAVE_PATH
NCCL_P2P_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python main_cmgnet.py \
    --save_path $SAVE_PATH \
    --train_file "" \
    --valid_file ./data/nmrmind_cmgnet_1004/validation.json \
    --test_file ./data/nmrmind_cmgnet_1004/test.json \
    --lr 1e-4 \
    --num_workers 12 \
    --epochs 100 --eval_per_epoch 1 \
    --warmup_ratio 0.02 \
    --label_smoothing 0. \
    --batch_size $((BATCH_SIZE / NUM_GPUS_PER_NODE / ACCUM_STEP)) \
    --gradient_accumulation_steps ${ACCUM_STEP} \
    --do_train \
    --do_test \
    --gpus $NUM_GPUS_PER_NODE \
    --smiles_aug_proba 0.5 \
    --nmr_aug_proba 0.5 \
    --nmr_aug_scale 3.0 \
    --lmdb_path ./data/nmrmind_cmgnet_1004/train.lmdb \
    --use_ring_system \
    --resume

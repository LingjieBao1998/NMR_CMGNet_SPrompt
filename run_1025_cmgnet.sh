#!/bin/bash

NUM_NODES=1
NUM_GPUS_PER_NODE=2

ACCUM_STEP=1
BATCH_SIZE=512

# export CUDA_LAUNCH_BLOCKING=1

# export CUDA_VISIBLE_DEVICES=2

SAVE_PATH=output/train_cmgnet_3.2_1025

set -x
mkdir -p $SAVE_PATH
NCCL_P2P_DISABLE=1 python main_cmgnet.py \
    --save_path $SAVE_PATH \
    --valid_file ./data/data_valid_test_1020/val.json \
    --test_file ./data/data_valid_test_1020/test.json \
    --lr 1e-4 \
    --num_workers 6 \
    --epochs 100 --eval_per_epoch 1 \
    --warmup 0.02 \
    --label_smoothing 0. \
    --batch_size $((BATCH_SIZE / NUM_GPUS_PER_NODE / ACCUM_STEP)) \
    --gradient_accumulation_steps ${ACCUM_STEP} \
    --do_train \
    --do_valid \
    --do_test \
    --gpus $NUM_GPUS_PER_NODE \
    --smiles_aug_proba 0.5 \
    --nmr_aug_proba 0.5 \
    --nmr_aug_scale 5.0 \
    --lmdb_path ./data/NMR_other_data_1020/new_db.lmdb
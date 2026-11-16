#!/bin/bash

NUM_NODES=1
NUM_GPUS_PER_NODE=2

ACCUM_STEP=4
BATCH_SIZE=576

# export CUDA_LAUNCH_BLOCKING=1

# export CUDA_VISIBLE_DEVICES=2

SAVE_PATH=output/train_cmgnet_ring_system_1119_cmgnet_ring_system_v2_extra_data_remove_multi

set -x
mkdir -p $SAVE_PATH
NCCL_P2P_DISABLE=1 TORCH_DISTRIBUTED_DEBUG=INFO python main_cmgnet.py \
    --save_path $SAVE_PATH \
    --train_file ./data/data_valid_test_1020/ex_test_gen.pkl,./data/data_valid_test_1020/ex_train.json \
    --valid_file ./data/data_valid_test_1020/val.json \
    --test_file ./data/data_valid_test_1020/test.json \
    --lr 1e-4 \
    --num_workers 12 \
    --epochs 100 --eval_per_epoch 1 \
    --warmup 0.00 \
    --label_smoothing 0. \
    --batch_size $((BATCH_SIZE / NUM_GPUS_PER_NODE / ACCUM_STEP)) \
    --gradient_accumulation_steps ${ACCUM_STEP} \
    --do_train \
    --do_test \
    --do_test \
    --gpus $NUM_GPUS_PER_NODE \
    --smiles_aug_proba 0.5 \
    --nmr_aug_proba 0.5 \
    --nmr_aug_scale 3.0 \
    --lmdb_path ./data/NMR_other_data_1020_add_ring_system_1029/new_db.lmdb \
    --use_ring_system \
    --force_aligned
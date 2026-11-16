#!/bin/bash
# NMRMind 数据训练 CMGNet (ring_system 三模式混合训练)
# 参照 run_1222_cmgnet_ring_system_v2.sh, 数据换成 data/nmrmind_cmgnet_1004/
#
# 初始权重: checkpoints/saved_hf_model_0910 (main_cmgnet.py --model_dir 默认值, 无需显式传)

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
    --use_ring_system

# 训练完成后, 在外部测试集 (4,502) 上推理评估:
# CUDA_VISIBLE_DEVICES=0 python main_cmgnet.py \
#     --save_path $SAVE_PATH \
#     --test_file ./data/nmrmind_cmgnet_1004/external_test.json \
#     --ckpt $SAVE_PATH/best_models/best.ckpt \
#     --do_test \
#     --gpus 1 \
#     --batch_size 256 \
#     --num_beams 10 \
#     --use_ring_system \
#     --prediction_file $SAVE_PATH/external_test_predictions.json


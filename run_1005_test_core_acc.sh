#!/bin/bash
# 测试集母核结构准确性评测 (--do_test)
#
# 权重: best-epoch=09-step=222690-val/score=-0.0203.ckpt (2026-10-05 14:02)
#       (训练暂停于 epoch 10 中途; 续训点 checkpoints/last.ckpt 已同步为同一文件)
# 数据: data/nmrmind_cmgnet_1004/test.json (55150 条)
# 输出: output/train_cmgnet_ring_system_3.2_nmrmind_1004/test_smiles_beams1_samples*.json
#       日志末尾打印 Top-1 Accuracy / Core Acc (母核) / Validity
#
# 注1: 测试阶段 collate_fn 不展开三模式(phase="test"), 每卡 batch = batch_size*2 = 96
# 注2: num_beams 保持默认 1 (贪心), 如需 Top-N 母核准确率再加 --num_beams 10 (慢约10倍)
# 注3: 全量 55150 条约 30 分钟。务必用 python -u(无缓冲), 否则中途看不到任何进度
#       (tqdm 进度条在重定向文件里只留最后一行, test/loss 也不写 TensorBoard)
# 注4: 日志中大量 "kekulize/SMILES Parse Error" 是逐 batch 分析生成分子的正常噪声(~4条/样本)

cd /mnt/data2/NMR_Core_Mol_Detection

SAVE_PATH=output/train_cmgnet_ring_system_3.2_nmrmind_1004
CKPT=$SAVE_PATH/best_models/best-epoch=09-step=222690-val/score=-0.0203.ckpt

export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH=/tmp/faultdiag:$PYTHONPATH   # 诊断: SIGUSR1 可 dump 堆栈

set -x
NCCL_P2P_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u main_cmgnet.py \
    --save_path $SAVE_PATH \
    --ckpt $CKPT \
    --train_file "" \
    --valid_file ./data/nmrmind_cmgnet_1004/validation.json \
    --test_file ./data/nmrmind_cmgnet_1004/test.json \
    --num_workers 12 \
    --batch_size 48 \
    --do_test \
    --gpus 2 \
    --smiles_aug_proba 0.5 \
    --nmr_aug_proba 0.5 \
    --nmr_aug_scale 3.0 \
    --lmdb_path ./data/nmrmind_cmgnet_1004/train.lmdb \
    --use_ring_system

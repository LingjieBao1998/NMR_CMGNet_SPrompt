#!/bin/bash
# 调试: 用 200 条测试样本快速复现 --do_test 流程
# 目的: 定位"主进程大量RDKit解析、测试循环无输出"的问题
# 特性: python -u (无缓冲, 实时看输出) + SIGUSR1 可 dump 堆栈 (PYTHONPATH=/tmp/faultdiag)
# 输出: output/train_cmgnet_dbg200/  (不影响正式目录)

cd /mnt/data2/NMR_Core_Mol_Detection

CKPT=output/train_cmgnet_ring_system_3.2_nmrmind_1004/best_models/best-epoch=09-step=222690-val/score=-0.0203.ckpt

export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH=/tmp/faultdiag:$PYTHONPATH

set -x
NCCL_P2P_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u main_cmgnet.py \
    --save_path output/train_cmgnet_dbg200 \
    --ckpt $CKPT \
    --train_file "" \
    --valid_file "" \
    --test_file ./data/nmrmind_cmgnet_1004/test_debug200.json \
    --num_workers 4 \
    --batch_size 16 \
    --do_test \
    --gpus 2 \
    --smiles_aug_proba 0.5 \
    --nmr_aug_proba 0.5 \
    --nmr_aug_scale 3.0 \
    --use_ring_system

#!/bin/bash
# 冒烟测试: 新 external_test.json 的前 200 条(含全部 64 条母核正样本), smiles 模式单卡
# 通过标准: 生成 output/train_nmrprobe/test_smiles_beams10_samples200.json 且含 predictions
export PATH=/home/lingjiebao/anaconda3/envs/py312/bin:$PATH
cd /mnt/data2/NMR_Core_Mol_Detection

CKPT=output/train_cmgnet_ring_system_3.2_nmrmind_1004/best_models/best-epoch=09-step=222690-val/score=-0.0203.ckpt

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=/tmp/faultdiag:$PYTHONPATH
export CMGNET_FAST_CORE_CHECK=1

echo "==================== smoke200 开始 $(date '+%F %T') ===================="
NCCL_P2P_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u main_cmgnet.py \
    --save_path output/train_nmrprobe \
    --ckpt $CKPT \
    --train_file "" \
    --valid_file "" \
    --test_file ./data/nmrmind_cmgnet_1004/test_nmrprobe.json \
    --num_workers 4 \
    --batch_size 48 \
    --do_test \
    --gpus 1 \
    --num_beams 10 \
    --decode_mode smiles \
    --smiles_aug_proba 0.5 \
    --nmr_aug_proba 0.5 \
    --nmr_aug_scale 3.0 \
    --use_ring_system
echo "==================== smoke200 结束 $(date '+%F %T') ===================="

#!/bin/bash
# 两模式 × beams=10 母核结构评测 (全量测试集)
#   模式1 smiles: NMR -> 分子          (decode_mode=smiles, BOS=187)
#   模式2 frag  : NMR -> 环系/母核      (decode_mode=frag,   BOS=183)
# 每条样本返回 10 个候选, 内置指标按 Top-1/3/5/10 计算 (num_beams=10 时 top_k=[1,3,5,10])
# 混淆矩阵: 跑完后 python analyze_core_confusion.py <两个JSON>
#
# 显存: beams=10 时 batch=48(每卡96行*10束)已近 4090 上限(23.8/24.5G), batch=128 会 OOM, 勿加大
# 速度: ~110 分钟/模式 (约为贪心 beams=1 的 4 倍时长; 瓶颈在 beam 生成本身而非分析)
# 快速分析: CMGNET_FAST_CORE_CHECK=1 (直接子结构匹配, 与库/互变异构法在 55130 条+2000条两级验证 0 差异)
#
# 输出: $SAVE_PATH/test_smiles_beams10_samples*.json 与 test_frag_beams10_samples*.json

cd /mnt/data2/NMR_Core_Mol_Detection

SAVE_PATH=output/train_cmgnet_ring_system_3.2_nmrmind_1004
CKPT=$SAVE_PATH/best_models/best-epoch=09-step=222690-val/score=-0.0203.ckpt

export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH=/tmp/faultdiag:$PYTHONPATH
export CMGNET_FAST_CORE_CHECK=1

for MODE in smiles frag; do
    echo "==================== mode=$MODE 开始 $(date '+%F %T') ===================="
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
        --num_beams 10 \
        --decode_mode "$MODE" \
        --smiles_aug_proba 0.5 \
        --nmr_aug_proba 0.5 \
        --nmr_aug_scale 3.0 \
        --lmdb_path ./data/nmrmind_cmgnet_1004/train.lmdb \
        --use_ring_system
    echo "==================== mode=$MODE 结束 $(date '+%F %T') ===================="
done

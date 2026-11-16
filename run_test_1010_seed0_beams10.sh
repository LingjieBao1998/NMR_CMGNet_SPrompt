#!/bin/bash
# NMRMind 训练的 seed=0 (1010 实验) 在固定 NMRMind test.json 上做 beams=10 的 smiles 预测
# - ckpt: best-epoch=09-step=222690-val/score=-0.0224.ckpt (与 last.ckpt 内容一致, md5 相同)
# - 测试集: data/nmrmind_cmgnet_1004/test.json (55,150 条)
# - 单卡: CUDA id 1 = 物理 5090 (32G)
# 用法: bash run_test_1010_seed0_beams10.sh

set -u
cd /mnt/data2/NMR_Core_Mol_Detection

GPU_ID=1
CKPT=output/train_cmgnet_ring_system_3.2_nmrmind_1010_seed0/best_models/best-epoch=09-step=222690-val/score=-0.0224.ckpt
TEST=./data/nmrmind_cmgnet_1004/test.json
SAVE_PATH=output/test_nmrmind_1010_seed0_beams10
LOG_FILE=run_test_1010_seed0_beams10.log

for f in "$CKPT" "$TEST"; do
    if [ ! -e "$f" ]; then
        echo "错误: 找不到 $f"
        exit 1
    fi
done

export CUDA_VISIBLE_DEVICES=$GPU_ID
# beams=10 时每条样本要分析 10 个预测; 默认库/互变异构检查 ~17ms/条,
# 开启快速子结构匹配 (~0.02ms/条), 已在全部 55,130 条测试预测上验证与原口径 0 差异.
export CMGNET_FAST_CORE_CHECK=1

echo "==================== test seed0 nmrmind beams=10 开始 $(date '+%F %T') ====================" | tee -a "$LOG_FILE"
mkdir -p "$SAVE_PATH"

NCCL_P2P_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u main_cmgnet.py \
    --save_path "$SAVE_PATH" \
    --ckpt "$CKPT" \
    --train_file "" \
    --valid_file "" \
    --test_file "$TEST" \
    --num_workers 12 \
    --batch_size 48 \
    --do_test \
    --gpus 1 \
    --num_beams 10 \
    --decode_mode smiles \
    --smiles_aug_proba 0.5 \
    --nmr_aug_proba 0.5 \
    --nmr_aug_scale 3.0 \
    --use_ring_system 2>&1 | tee -a "$LOG_FILE"
STATUS=${PIPESTATUS[0]}
echo "==================== test seed0 nmrmind beams=10 结束 (python exit=$STATUS) $(date '+%F %T') ====================" | tee -a "$LOG_FILE"
exit $STATUS

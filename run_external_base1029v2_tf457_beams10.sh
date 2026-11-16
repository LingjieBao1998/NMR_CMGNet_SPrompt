#!/bin/bash
# 原始基座 1029_v2_best(未微调) 的外部测试评测 (frag/smiles 2 模式, beams=10)
# tf4.57.6 venv, 5090 (CUDA id 1), 修复 base1029v2 beam 翻车问题
set -u
cd /mnt/data2/NMR_Core_Mol_Detection
GPU_ID=1
TEST=./data/nmrmind_cmgnet_1004/external_test.json
PY=/home/lingjiebao/venvs/py312_tf457/bin/python
CKPT=models/cmgnet/train_cmgnet_ring_system_3.2_1029_v2_best.ckpt
TAG=base1029v2_tf457
export CUDA_VISIBLE_DEVICES=$GPU_ID
if [ ! -e "$CKPT" ]; then echo "错误: 找不到 $CKPT"; exit 1; fi
for mode in frag smiles; do
    SAVE=output/external_4502_${TAG}_${mode}_beams10
    LOG=logs/run_external_${TAG}_${mode}_beams10.log
    echo "==================== $TAG mode=$mode 开始 $(date '+%F %T') ====================" | tee -a "$LOG"
    mkdir -p "$SAVE"
    NCCL_P2P_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u main_cmgnet.py \
        --save_path "$SAVE" --ckpt "$CKPT" --train_file "" --valid_file "" --test_file "$TEST" \
        --num_workers 12 --batch_size 48 --do_test --gpus 1 --num_beams 10 --decode_mode "$mode" \
        --smiles_aug_proba 0.5 --nmr_aug_proba 0.5 --nmr_aug_scale 3.0 --use_ring_system >> "$LOG" 2>&1
    STATUS=$?
    echo "==================== $TAG mode=$mode 结束 (exit=$STATUS) $(date '+%F %T') ====================" | tee -a "$LOG"
    [ "$STATUS" -ne 0 ] && echo "!! 失败: $LOG"
done
echo "==================== $TAG 全部结束 $(date '+%F %T') ===================="

#!/bin/bash
# 纯实验语料 30ep 训练的最优模型外部评测 (frag/smiles, 6 run, ~45 min).
# "最优" = 各 seed epoch_00.ckpt (TB 曲线: val/loss 自 epoch0 单调上升, Lightning best 亦选 epoch0).
# 口径与历史外部评测一致: 外部 4502 / beams=10 / 5090 (CUDA id 1).
# 汇总: output/external_exp30ep_best_details.md

set -u
cd /mnt/data2/NMR_Core_Mol_Detection

GPU_ID=1
TEST=./data/nmrmind_cmgnet_1004/external_test.json
PY=/home/lingjiebao/anaconda3/envs/py312/bin/python
SUMMARY_MD=output/external_exp30ep_best_details.md

export CUDA_VISIBLE_DEVICES=$GPU_ID

: > "$SUMMARY_MD"
echo "# 纯实验语料 30ep 各 seed 最优模型 (epoch_00) | 外部 4502 | beams=10 | 5090" >> "$SUMMARY_MD"
echo >> "$SUMMARY_MD"

for SEED in 42 0 2006; do
    CKPT=/mnt/data1/nmr_data/train_exp30ep/seed${SEED}/checkpoints/epoch_00.ckpt
    if [ ! -e "$CKPT" ]; then echo "!! 找不到 $CKPT, 跳过 seed$SEED" | tee -a "$SUMMARY_MD"; continue; fi
    for mode in frag smiles; do
        TAG=exp30ep_best_seed${SEED}
        SAVE=output/external_4502_${TAG}_${mode}_beams10
        LOG=logs/run_external_${TAG}_${mode}_beams10.log
        OUT_JSON="$SAVE/test_${mode}_beams10_samples4502.json"
        echo "==================== $TAG mode=$mode 开始 $(date '+%F %T') ====================" | tee -a "$LOG"
        mkdir -p "$SAVE"

        NCCL_P2P_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u main_cmgnet.py \
            --save_path "$SAVE" \
            --ckpt "$CKPT" \
            --train_file "" \
            --valid_file "" \
            --test_file "$TEST" \
            --num_workers 12 \
            --batch_size 48 \
            --do_test \
            --gpus 1 \
            --num_beams 10 \
            --decode_mode "$mode" \
            --smiles_aug_proba 0.5 \
            --nmr_aug_proba 0.5 \
            --nmr_aug_scale 3.0 \
            --use_ring_system 2>&1 | tee -a "$LOG"
        STATUS=${PIPESTATUS[0]}
        echo "==================== $TAG mode=$mode 结束 (python exit=$STATUS) $(date '+%F %T') ====================" | tee -a "$LOG"

        if [ "$STATUS" -eq 0 ] && [ -e "$OUT_JSON" ]; then
            DETAIL=output/external_${TAG}_${mode}_detail.md
            $PY summarize_core_smiles.py "$DETAIL" "外部4502 | 纯实验语料 30ep 最优=epoch0 seed$SEED | beams=10 ($mode)" "$OUT_JSON" | tee -a "$LOG"
            cat "$DETAIL" >> "$SUMMARY_MD"
            echo >> "$SUMMARY_MD"
        else
            echo "!! $TAG mode=$mode 失败: $OUT_JSON" | tee -a "$SUMMARY_MD"
        fi
    done
done
echo "==================== 全部结束 $(date '+%F %T') ====================" | tee -a "$SUMMARY_MD"

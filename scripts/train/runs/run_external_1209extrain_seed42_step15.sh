#!/bin/bash
# 旧波 newdb_1209_extrain seed42 的 ≈1.5ep step 快照补评 (frag/smiles, 2 run, ~16 min).
# 目的: 验证 seed42 外部集 R/P 轮廓异常是否只是最优 epoch 落点不同 (其 best=epoch2).
# 单卡 5090 (CUDA id 1), 口径与其余外部评测一致.
# 汇总: output/external_1209extrain_seed42_step15_details.md

set -u
cd /mnt/data2/NMR_Core_Mol_Detection

GPU_ID=1
TEST=./data/nmrmind_cmgnet_1004/external_test.json
PY=/home/lingjiebao/anaconda3/envs/py312/bin/python
CKPT=output/train_newdb_1209_extrain_5ep_seed42/step_checkpoints/step-epoch=01-step=050000-train_loss=0.0000.ckpt
SUMMARY_MD=output/external_1209extrain_seed42_step15_details.md

export CUDA_VISIBLE_DEVICES=$GPU_ID

: > "$SUMMARY_MD"
echo "# 旧波 newdb_1209_extrain seed42 step≈1.46ep (step-50000) 快照 | 外部 4502 | beams=10" >> "$SUMMARY_MD"
echo >> "$SUMMARY_MD"

for mode in frag smiles; do
    TAG=1209extrain_seed42_step15
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
        TITLE="外部测试集(4502条) | 旧波 1209extrain seed42 step≈1.46ep 快照 | beams=10"
        $PY summarize_core_smiles.py "$DETAIL" "$TITLE ($mode)" "$OUT_JSON" | tee -a "$LOG"
        cat "$DETAIL" >> "$SUMMARY_MD"
        echo >> "$SUMMARY_MD"
    else
        echo "!! $TAG mode=$mode 失败: $OUT_JSON" | tee -a "$SUMMARY_MD"
    fi
done
echo "==================== 全部结束 $(date '+%F %T') ====================" | tee -a "$SUMMARY_MD"

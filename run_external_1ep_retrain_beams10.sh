#!/bin/bash
# seed42 双配方 1ep 重训 (epoch_00.ckpt) 的外部测试评测 (frag/smiles, 4 run, ~30 min).
#   A) 新组配方: train_newdb_extrain_5ep_seed42_retrain      (无 1209)
#   B) 旧波配方: train_newdb_1209_extrain_5ep_seed42_retrain (加 1209 正样本)
# 口径与 run_external_1209extrain_seed42_step15.sh 完全一致, 单卡 5090 (CUDA id 1).
# 汇总: output/external_1ep_retrain_details.md

set -u
cd /mnt/data2/NMR_Core_Mol_Detection

GPU_ID=1
TEST=./data/nmrmind_cmgnet_1004/external_test.json
PY=/home/lingjiebao/anaconda3/envs/py312/bin/python
CKPT_A=output/train_newdb_extrain_5ep_seed42_retrain/checkpoints/epoch_00.ckpt
CKPT_B=output/train_newdb_1209_extrain_5ep_seed42_retrain/checkpoints/epoch_00.ckpt
SUMMARY_MD=output/external_1ep_retrain_details.md

export CUDA_VISIBLE_DEVICES=$GPU_ID

run_eval () {  # $1=TAG  $2=CKPT  $3=TITLE
    local TAG=$1 CKPT=$2 TITLE=$3
    for mode in frag smiles; do
        local SAVE=output/external_4502_${TAG}_${mode}_beams10
        local LOG=logs/run_external_${TAG}_${mode}_beams10.log
        local OUT_JSON="$SAVE/test_${mode}_beams10_samples4502.json"
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
        local STATUS=${PIPESTATUS[0]}
        echo "==================== $TAG mode=$mode 结束 (python exit=$STATUS) $(date '+%F %T') ====================" | tee -a "$LOG"

        if [ "$STATUS" -eq 0 ] && [ -e "$OUT_JSON" ]; then
            local DETAIL=output/external_${TAG}_${mode}_detail.md
            $PY summarize_core_smiles.py "$DETAIL" "$TITLE ($mode)" "$OUT_JSON" | tee -a "$LOG"
            cat "$DETAIL" >> "$SUMMARY_MD"
            echo >> "$SUMMARY_MD"
        else
            echo "!! $TAG mode=$mode 失败: $OUT_JSON" | tee -a "$SUMMARY_MD"
        fi
    done
}

: > "$SUMMARY_MD"
echo "# seed42 双配方 1ep 重训 | 外部 4502 | beams=10 | 5090" >> "$SUMMARY_MD"
echo >> "$SUMMARY_MD"

run_eval 1ep_extrain_retrain       "$CKPT_A" "外部测试集(4502条) | 新组 extrain seed42 真1ep (epoch_00.ckpt) | beams=10"
run_eval 1ep_1209extrain_retrain   "$CKPT_B" "外部测试集(4502条) | 旧波 1209extrain seed42 真1ep (epoch_00.ckpt) | beams=10"

echo "==================== 全部结束 $(date '+%F %T') ====================" | tee -a "$SUMMARY_MD"

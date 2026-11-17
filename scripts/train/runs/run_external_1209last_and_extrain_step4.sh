#!/bin/bash
# 外部测试集 (4,502 = 4,438 neg + 64 core-pos) 补充评测, 单卡 5090 (CUDA id 1):
#   A) 旧波 newdb_1209_extrain 5ep 的 last.ckpt, seed 0/2006/42  (快照深度 ≈1-2ep, 见各 best_models mtime)
#   B) 新组 new_db+ex_train 5ep 的 step_checkpoints/step-epoch=04-step=150000 (≈4.5ep, 现存最接近训满的权重)
# 各 6 run, 共 12 run, 约 1h45m. 口径与 run_external_extrain5ep_seeds_last_beams10.sh 一致.
# 汇总: output/external_1209extrain_last_and_extrain5ep_step4_details.md
# 用法: tmux new-session -d -s eval_12 'bash run_external_1209last_and_extrain_step4.sh > logs/eval_12.log 2>&1'

set -u
cd /mnt/data2/NMR_Core_Mol_Detection

GPU_ID=1
TEST=./data/nmrmind_cmgnet_1004/external_test.json
PY=/home/lingjiebao/anaconda3/envs/py312/bin/python
SUMMARY_MD=output/external_1209extrain_last_and_extrain5ep_step4_details.md

export CUDA_VISIBLE_DEVICES=$GPU_ID

run_eval () {  # $1=标签  $2=ckpt路径  $3=detail文件名主干  $4=标题
    local TAG=$1 CKPT=$2 STEM=$3 TITLE=$4
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
            local DETAIL=output/${STEM}_${mode}_detail.md
            $PY summarize_core_smiles.py "$DETAIL" "$TITLE ($mode)" "$OUT_JSON" | tee -a "$LOG"
            cat "$DETAIL" >> "$SUMMARY_MD"
            echo >> "$SUMMARY_MD"
        else
            echo "!! $TAG mode=$mode 失败或结果文件缺失: $OUT_JSON" | tee -a "$SUMMARY_MD"
        fi
    done
}

: > "$SUMMARY_MD"
echo "# 旧波 newdb_1209_extrain last (≈1-2ep) + 新组 ex_train step≈4.5ep 快照 | 外部 4502 | beams=10" >> "$SUMMARY_MD"
echo >> "$SUMMARY_MD"

# ---- A) 旧波 newdb_1209_extrain 5ep 的 last ----
for seed in 0 2006 42; do
    run_eval "1209extrain_seed${seed}_last" \
             "output/train_newdb_1209_extrain_5ep_seed${seed}/best_models/last.ckpt" \
             "external_1209extrain_seed${seed}_last" \
             "外部测试集(4502条) | 旧波 newdb_1209_extrain 5ep seed=${seed} last 快照 | beams=10"
done

# ---- B) 新组 new_db+ex_train 5ep 的 ≈4.5ep step 快照 ----
for seed in 0 2006 42; do
    run_eval "extrain5ep_seed${seed}_step4" \
             "output/train_newdb_extrain_5ep_seed${seed}/step_checkpoints/step-epoch=04-step=150000-train_loss=0.0000.ckpt" \
             "external_extrain5ep_seed${seed}_step4" \
             "外部测试集(4502条) | 新组 new_db+ex_train 5ep seed=${seed} step≈4.5ep 快照 | beams=10"
done

echo "==================== 12 个评测全部结束 $(date '+%F %T') ====================" | tee -a "$SUMMARY_MD"

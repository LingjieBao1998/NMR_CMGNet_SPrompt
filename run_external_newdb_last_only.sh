#!/bin/bash
# 对照组 (new_db 1029v2 3ep) last.ckpt 补评: seed 0/2006 x frag/smiles, 共 4 run, 约 35 min.
# seed42 的补训暂缓 (用户指示: 先只做测试), 其评测待训练后再补.
# 口径与 run_external_newdb_seeds_beams10.sh (best) 完全一致, 单卡 5090 (CUDA id 1).
# 汇总: output/external_newdb_seeds_last_only_details.md
# 用法: tmux new-session -d -s ctl_last_eval 'bash run_external_newdb_last_only.sh > logs/ctl_last_eval.log 2>&1'

set -u
cd /mnt/data2/NMR_Core_Mol_Detection

GPU_ID=1
TEST=./data/nmrmind_cmgnet_1004/external_test.json
PY=/home/lingjiebao/anaconda3/envs/py312/bin/python
SUMMARY_MD=output/external_newdb_seeds_last_only_details.md

export CUDA_VISIBLE_DEVICES=$GPU_ID
# 与 run_external_newdb_seeds_beams10.sh 相同口径 (55,130 条预测上验证过与原判 0 差异)
export CMGNET_FAST_CORE_CHECK=1

run_eval () {  # $1=seed $2=kind(best/last) $3=ckpt路径
    local seed=$1 kind=$2 CKPT=$3
    for mode in frag smiles; do
        local SAVE=output/external_4502_newdb_seed${seed}_${kind}_${mode}_beams10
        local LOG=logs/run_external_newdb_seed${seed}_${kind}_${mode}_beams10.log
        local OUT_JSON="$SAVE/test_${mode}_beams10_samples4502.json"
        echo "==================== seed=$seed ckpt=$kind mode=$mode 开始 $(date '+%F %T') ====================" | tee -a "$LOG"
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
        echo "==================== seed=$seed ckpt=$kind mode=$mode 结束 (python exit=$STATUS) $(date '+%F %T') ====================" | tee -a "$LOG"

        if [ "$STATUS" -eq 0 ] && [ -e "$OUT_JSON" ]; then
            local DETAIL=output/external_newdb_seed${seed}_${kind}_${mode}_detail.md
            local TITLE="外部测试集(1008b, 4502条: 4438 neg + 64 core-pos) | new_db 1029v2 微调 seed=${seed} ${kind} | ${mode} | beams=10"
            $PY summarize_core_smiles.py "$DETAIL" "$TITLE" "$OUT_JSON" | tee -a "$LOG"
            cat "$DETAIL" >> "$SUMMARY_MD"
            echo >> "$SUMMARY_MD"
        else
            echo "!! seed=$seed ckpt=$kind mode=$mode 失败或结果文件缺失: $OUT_JSON" | tee -a "$SUMMARY_MD"
        fi
    done
}

: > "$SUMMARY_MD"
echo "# 对照组 new_db 1029v2 | seed 0/2006 last 补评 | beams=10" >> "$SUMMARY_MD"
echo >> "$SUMMARY_MD"

for seed in 0 2006; do
    run_eval "$seed" last "output/train_newdb_1029v2_3ep_seed${seed}/best_models/last.ckpt"
done

echo "==================== 全部评测结束 $(date '+%F %T') ====================" | tee -a "$SUMMARY_MD"

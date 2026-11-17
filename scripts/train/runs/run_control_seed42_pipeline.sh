#!/bin/bash
# 对照组 (new_db 1029v2 3ep) 补全: seed 0/2006 的 last 评测 + 补训 seed42 后评 best/last.
#   Step 1: seed 0/2006 last.ckpt 外部集评测 (4 run, 约 35 min)
#   Step 2: 补训 new_db 1029v2 seed42 3ep (约 5h)
#   Step 3: seed42 best/last x frag/smiles 外部集评测 (4 run, 约 35 min)
# 单卡 5090 (CUDA id 1; 0=4090). 口径与 run_external_newdb_seeds_beams10.sh 一致.
# 汇总: output/external_newdb_seeds_last_plus_seed42_details.md
# 用法: tmux new-session -d -s ctl_seed42 'bash run_control_seed42_pipeline.sh > logs/ctl_seed42_pipeline.log 2>&1'

set -u
cd /mnt/data2/NMR_Core_Mol_Detection

GPU_ID=1
TEST=./data/nmrmind_cmgnet_1004/external_test.json
PY=/home/lingjiebao/anaconda3/envs/py312/bin/python
SUMMARY_MD=output/external_newdb_seeds_last_plus_seed42_details.md

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
echo "# 对照组 new_db 1029v2 | seed 0/2006 last 补评 + seed42 补训后 best/last | beams=10" >> "$SUMMARY_MD"
echo >> "$SUMMARY_MD"

# ---- Step 1: seed 0/2006 的 last 补评 ----
for seed in 0 2006; do
    run_eval "$seed" last "output/train_newdb_1029v2_3ep_seed${seed}/best_models/last.ckpt"
done

# ---- Step 2: 补训 seed42 (与 run_newdb_1029v2_3ep_seed0.sh 超参一致, 换 5090) ----
SEED=42
SAVE_PATH=output/train_newdb_1029v2_3ep_seed${SEED}
TLOG=logs/train_newdb_1029v2_3ep_seed${SEED}.log
mkdir -p "$SAVE_PATH"
echo "==================== train newdb 1029v2 seed=$SEED 开始 $(date '+%F %T') ====================" | tee -a "$TLOG"

NCCL_P2P_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u main_cmgnet.py \
    --save_path "$SAVE_PATH" \
    --ckpt models/cmgnet/train_cmgnet_ring_system_3.2_1029_v2_best.ckpt \
    --train_file "" \
    --valid_file ./data/nmrmind_cmgnet_1004/validation.json \
    --test_file ./data/nmrmind_cmgnet_1004/test.json \
    --lmdb_path ./data/NMR_other_data_1020_add_ring_system_1029/new_db.lmdb \
    --lr 1e-4 \
    --num_workers 12 \
    --epochs 3 --eval_per_epoch 1 \
    --warmup_ratio 0.02 \
    --label_smoothing 0. \
    --seed $SEED \
    --batch_size 48 \
    --gradient_accumulation_steps 2 \
    --do_train \
    --gpus 1 \
    --smiles_aug_proba 0.5 \
    --nmr_aug_proba 0.5 \
    --nmr_aug_scale 3.0 \
    --use_ring_system >> "$TLOG" 2>&1
TSTATUS=$?
echo "==================== train newdb 1029v2 seed=$SEED 结束 (python exit=$TSTATUS) $(date '+%F %T') ====================" | tee -a "$TLOG"

# ---- Step 3: seed42 的 best/last 评测 ----
if [ "$TSTATUS" -eq 0 ]; then
    BEST=$(ls -d output/train_newdb_1029v2_3ep_seed${SEED}/best_models/best-epoch=*/score=*.ckpt 2>/dev/null | head -1)
    if [ -n "$BEST" ]; then
        run_eval "$SEED" best "$BEST"
    else
        echo "!! 找不到 seed42 的 best ckpt, 跳过 best 评测" | tee -a "$SUMMARY_MD"
    fi
    run_eval "$SEED" last "output/train_newdb_1029v2_3ep_seed${SEED}/best_models/last.ckpt"
else
    echo "!! 训练失败 (exit=$TSTATUS), 跳过 seed42 评测" | tee -a "$SUMMARY_MD"
fi

echo "==================== 全流程结束 $(date '+%F %T') ====================" | tee -a "$SUMMARY_MD"

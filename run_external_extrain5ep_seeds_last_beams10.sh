#!/bin/bash
# 外部测试集 (4,502 = 4,438 neg + 64 core-pos) 评测:
#   new_db + ex_train 5ep 微调的 seed 0 / 2006 / 42, 各取 last.ckpt,
#   decode_mode = frag / smiles, num_beams = 10 (top-10).
# 单卡 5090 (CUDA id 1; 0=4090, 1=5090). 与 run_external_newdb_seeds_beams10.sh 口径一致.
# 用法: bash run_external_extrain5ep_seeds_last_beams10.sh
#
# 结果:
#   明细 (每 run 一份)  output/external_extrain5ep_seed<seed>_last_<mode>_detail.md
#   汇总                output/external_extrain5ep_seeds_last_beams10_details.md
#   预测 json           output/external_4502_extrain5ep_seed<seed>_last_<mode>_beams10/

set -u
cd /mnt/data2/NMR_Core_Mol_Detection

GPU_ID=1
TEST=./data/nmrmind_cmgnet_1004/external_test.json
PY=/home/lingjiebao/anaconda3/envs/py312/bin/python
SUMMARY_MD=output/external_extrain5ep_seeds_last_beams10_details.md

export CUDA_VISIBLE_DEVICES=$GPU_ID
# 母核判定口径已固定为 指纹库+互变异构指纹 (见 main_cmgnet.py: analyze_molecule),
# 不再需要 CMGNET_FAST_CORE_CHECK 开关。

declare -A CKPTS
CKPTS[0]=output/train_newdb_extrain_5ep_seed0/best_models/last.ckpt
CKPTS[2006]=output/train_newdb_extrain_5ep_seed2006/best_models/last.ckpt
CKPTS[42]=output/train_newdb_extrain_5ep_seed42/best_models/last.ckpt

: > "$SUMMARY_MD"
echo "# 外部测试集 (4502 条: 4438 neg + 64 core-pos) | new_db+ex_train 5ep | last.ckpt | beams=10" >> "$SUMMARY_MD"
echo >> "$SUMMARY_MD"

for seed in 0 2006 42; do
    CKPT=${CKPTS[$seed]}
    if [ ! -e "$CKPT" ]; then
        echo "错误: 找不到 $CKPT" | tee -a "$SUMMARY_MD"
        continue
    fi
    for mode in frag smiles; do
        SAVE=output/external_4502_extrain5ep_seed${seed}_last_${mode}_beams10
        LOG=logs/run_external_extrain5ep_seed${seed}_last_${mode}_beams10.log
        OUT_JSON="$SAVE/test_${mode}_beams10_samples4502.json"

        echo "==================== seed=$seed ckpt=last mode=$mode 开始 $(date '+%F %T') ====================" | tee -a "$LOG"
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
        echo "==================== seed=$seed ckpt=last mode=$mode 结束 (python exit=$STATUS) $(date '+%F %T') ====================" | tee -a "$LOG"

        if [ "$STATUS" -eq 0 ] && [ -e "$OUT_JSON" ]; then
            DETAIL=output/external_extrain5ep_seed${seed}_last_${mode}_detail.md
            TITLE="外部测试集(4502条: 4438 neg + 64 core-pos) | new_db+ex_train 5ep seed=${seed} last | ${mode} | beams=10"
            $PY summarize_core_smiles.py "$DETAIL" "$TITLE" "$OUT_JSON" | tee -a "$LOG"
            cat "$DETAIL" >> "$SUMMARY_MD"
            echo >> "$SUMMARY_MD"
        else
            echo "!! seed=$seed mode=$mode 失败或结果文件缺失: $OUT_JSON" | tee -a "$SUMMARY_MD"
        fi
    done
done

echo "==================== 全部评测结束 $(date '+%F %T') ====================" | tee -a "$SUMMARY_MD"

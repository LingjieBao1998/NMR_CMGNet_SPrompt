#!/bin/bash
# 外部测试集 (4,502 = 4,438 neg + 64 core-pos) 评测:
#   new_db 微调的 seed0 / seed2006, 各取 best (epoch0) 与 last (epoch2),
#   decode_mode = smiles / frag, num_beams = 10.
# 单卡 5090 (CUDA id 1). 与 run_1005_half_single_gpu.sh 的 external 口径一致.
# 用法: bash run_external_newdb_seeds_beams10.sh

set -u
cd /mnt/data2/NMR_Core_Mol_Detection

GPU_ID=1
TEST=./data/nmrmind_cmgnet_1004/external_test.json
PY=/home/lingjiebao/anaconda3/envs/py312/bin/python
SUMMARY_MD=output/external_newdb_seeds_beams10_all_details.md

export CUDA_VISIBLE_DEVICES=$GPU_ID
# 55,130 条测试预测上验证过与原口径 0 差异
export CMGNET_FAST_CORE_CHECK=1

declare -A CKPTS
CKPTS[0_best]=output/train_newdb_1029v2_3ep_seed0/best_models/best-epoch=00-step=033296-val/score=-0.0417.ckpt
CKPTS[0_last]=output/train_newdb_1029v2_3ep_seed0/best_models/last.ckpt
CKPTS[2006_best]=output/train_newdb_1029v2_3ep_seed2006/best_models/best-epoch=00-step=033296-val/score=-0.0422.ckpt
CKPTS[2006_last]=output/train_newdb_1029v2_3ep_seed2006/best_models/last.ckpt

: > "$SUMMARY_MD"

for seed in 0 2006; do
    for kind in best last; do
        CKPT=${CKPTS[${seed}_${kind}]}
        if [ ! -e "$CKPT" ]; then
            echo "错误: 找不到 $CKPT" | tee -a "$SUMMARY_MD"
            continue
        fi
        for mode in smiles frag; do
            SAVE=output/external_4502_newdb_seed${seed}_${kind}_${mode}_beams10
            LOG=run_external_newdb_seed${seed}_${kind}_${mode}_beams10.log
            OUT_JSON="$SAVE/test_${mode}_beams10_samples4502.json"

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
            STATUS=${PIPESTATUS[0]}
            echo "==================== seed=$seed ckpt=$kind mode=$mode 结束 (python exit=$STATUS) $(date '+%F %T') ====================" | tee -a "$LOG"

            if [ "$STATUS" -eq 0 ] && [ -e "$OUT_JSON" ]; then
                DETAIL=output/external_newdb_seed${seed}_${kind}_${mode}_detail.md
                TITLE="外部测试集(1008b, 4502条: 4438 neg + 64 core-pos) | new_db 1029v2 微调 seed=${seed} ${kind} | ${mode} | beams=10"
                $PY summarize_core_smiles.py "$DETAIL" "$TITLE" "$OUT_JSON" | tee -a "$LOG"
                cat "$DETAIL" >> "$SUMMARY_MD"
                echo >> "$SUMMARY_MD"
            else
                echo "!! seed=$seed ckpt=$kind mode=$mode 失败或结果文件缺失: $OUT_JSON" | tee -a "$SUMMARY_MD"
            fi
        done
    done
done

echo "==================== 全部评测结束 $(date '+%F %T') ====================" | tee -a "$SUMMARY_MD"

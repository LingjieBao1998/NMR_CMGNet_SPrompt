#!/bin/bash
# full4 1ep 重训·无NMR噪声版: seed 0 与 seed 2006 双卡并行(各自单卡)
# 与 run_full4_1ep_retrain_s0_s2006.sh 唯一区别: --nmr_aug_scale 3.0 -> 0.0
#   (main_cmgnet.py 中 nmr_aug_scale>0 才掷币增强, scale=0 即完全关闭 NMR 噪声;
#    smiles_aug_proba 0.5 等其余参数原样保留)
# 训练集 = 4 个来源:
#   1) new_db.lmdb (3,196,326 条)
#   2) new_data_1209.filtered.ring.lmdb (77,328 条, PubChem 正例)
#   3) ex_train_add_label_filtered.evalclean.ring.json (24,520 条)
#   4) ex_train_cdk_analogs.json (746 条 CDK 类似物正例)
# 超参: lr 5e-5, 1 epoch, batch48/gradacc2, 基座 1029_v2_best
# 用法(在 tmux 里跑, 勿 Ctrl-C):
#   tmux new-session -d -s full4_nmr0 'bash run_full4_nmr0_1ep_retrain_s0_s2006.sh > logs/run_full4_nmr0_2card.log 2>&1'

set -u
cd /mnt/data2/NMR_Core_Mol_Detection

PY=/home/lingjiebao/anaconda3/envs/py312/bin/python
BASE=models/cmgnet/train_cmgnet_ring_system_3.2_1029_v2_best.ckpt
VALID=./data/nmrmind_cmgnet_1004/validation.json
TEST=./data/nmrmind_cmgnet_1004/test.json
EX_TRAIN=data/extra_training/ex_train_add_label_filtered.evalclean.ring.json
CDK_JSON=data/ex_train_cdk_analogs.json
NEW_DB=./data/NMR_other_data_1020_add_ring_system_1029/new_db.lmdb
LMDB_1209=data/new_data_1209.filtered.ring.lmdb

for f in "$BASE" "$EX_TRAIN" "$CDK_JSON" "$NEW_DB/data.mdb" "$LMDB_1209/data.mdb" "$VALID" "$TEST"; do
    if [ ! -e "$f" ]; then
        echo "错误: 找不到 $f"
        exit 1
    fi
done

train_one () {  # $1=seed  $2=cuda_id  $3=save_path
    local SEED=$1
    local GPU_ID=$2
    local SAVE=$3
    local LOG=$SAVE.log
    mkdir -p "$SAVE"
    echo "==================== train $SAVE (seed=$SEED gpu=$GPU_ID) 开始 $(date '+%F %T') ====================" | tee -a "$LOG"

    CUDA_VISIBLE_DEVICES=$GPU_ID NCCL_P2P_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        $PY -u main_cmgnet.py \
        --save_path "$SAVE" \
        --ckpt "$BASE" \
        --train_file "$EX_TRAIN,$CDK_JSON" \
        --valid_file "$VALID" \
        --test_file "$TEST" \
        --lmdb_path "$NEW_DB,$LMDB_1209" \
        --lr 5e-5 \
        --num_workers 12 \
        --epochs 1 --eval_per_epoch 1 \
        --warmup_ratio 0.02 \
        --label_smoothing 0. \
        --seed "$SEED" \
        --batch_size 48 \
        --gradient_accumulation_steps 2 \
        --do_train \
        --gpus 1 \
        --smiles_aug_proba 0.5 \
        --nmr_aug_proba 0.5 \
        --nmr_aug_scale 0.0 \
        --use_ring_system >> "$LOG" 2>&1
    local STATUS=$?
    echo "EXIT:$STATUS $(date '+%F %T')" | tee -a "$LOG"
    echo "==================== train $SAVE 结束 (exit=$STATUS) $(date '+%F %T') ===================="
    return $STATUS
}

echo "==================== full4_nmr0 2卡 1ep 重训 开始 $(date '+%F %T') ===================="
train_one 0    0 output/train_full4_nmr0_1ep_seed0_retrain &
PID0=$!
train_one 2006 1 output/train_full4_nmr0_1ep_seed2006_retrain &
PID1=$!

wait $PID0; S0=$?
wait $PID1; S1=$?
echo "==================== 全部结束 (seed0=$S0 seed2006=$S1) $(date '+%F %T') ===================="

#!/bin/bash
# 纯文献实验语料 30ep 训练 (0910 基座, 3 seed):
#   数据: /mnt/data1/nmr_data/newdb_experimental_lit_1024/new_db_exp_lit.lmdb (约5.0万, 文献实验库提取, 不含 NMRium)
#         + data/extra_training/ex_train_add_label_filtered.evalclean.ring.json (24,520, 全 nmrshiftdb)
#         合计约 74,504 条/epoch
#   底座: checkpoints/saved_hf_model_0910 (--model_dir 默认值, 不传 --ckpt)
#   30 epochs, eval_per_epoch 1 -> SaveEveryEpoch 每 epoch 落一份 epoch_NN.ckpt (413MB, 每种子约12.5G)
#   save_path 在 /mnt/data1 (data2 仅剩 38G, 放不下 3 种子共 ~37G 的 ckpt)
#   超参口径与 run_1010_cmgnet_seed_single_gpu.sh / seed42_dual_retrain 完全一致 (batch48×accum2, lr1e-4, warmup0.02)
#
# 注意样本暴露量: 30ep × 7.45万 = 224万次样本访问, 还不到旧配方 (330万) 1个 epoch.
#   若要与旧 1ep 对照的暴露量对齐约需 45ep, 与 5ep 对照对齐需 ~220ep. 需要改就调 EPOCHS.
#
# 预计耗时 (5090): steps/epoch = ceil(74,504/48) = 1,553
#   -> 训练 ~2.5 min/epoch + 验证/存档 ≈ 4 min/epoch
#   -> 每种子 30ep ≈ 2h, 3 种子顺序 ≈ 6h
#
# 用法: tmux new-session -d -s exp30ep 'bash run_expdb_30ep_3seeds.sh > logs/exp30ep_driver.log 2>&1'
#   默认 GPU_ID=1 (5090), SEEDS="42 0 2006"; 可改下面两个变量.

set -u
cd /mnt/data2/NMR_Core_Mol_Detection

GPU_ID=1
SEEDS="42 0 2006"
EPOCHS=30

PY=/home/lingjiebao/anaconda3/envs/py312/bin/python
EXP_LMDB=/mnt/data1/nmr_data/newdb_experimental_lit_1024/new_db_exp_lit.lmdb
EX_TRAIN=data/extra_training/ex_train_add_label_filtered.evalclean.ring.json
VALID=./data/nmrmind_cmgnet_1004/validation.json
TEST=./data/nmrmind_cmgnet_1004/test.json
SAVE_ROOT=/mnt/data1/nmr_data/train_exp30ep

export CUDA_VISIBLE_DEVICES=$GPU_ID

# ---- 前置检查 ----
if ! grep -q "seed_everything" main_cmgnet.py; then
    echo "错误: main_cmgnet.py 中未找到 seed_everything, --seed 不会生效, 终止"; exit 1
fi
for f in "$EXP_LMDB/data.mdb" "$EX_TRAIN" "$VALID" "$TEST"; do
    if [ ! -e "$f" ]; then echo "错误: 找不到 $f"; exit 1; fi
done
FREE_DF=$(df --output=avail -BG /mnt/data1 | tail -1 | tr -d 'G')
if [ "$FREE_DF" -lt 45 ]; then
    echo "错误: /mnt/data1 仅剩 ${FREE_DF}G, 3 种子 ckpt 需约 40G, 终止"; exit 1
fi
echo "前置检查通过: gpu=$GPU_ID(CUDA编号) seeds=[$SEEDS] epochs=$EPOCHS /mnt/data1 剩余 ${FREE_DF}G"

for SEED in $SEEDS; do
    SAVE=$SAVE_ROOT/seed${SEED}
    LOG=$SAVE.log
    mkdir -p "$SAVE"
    echo "==================== train $SAVE 开始 $(date '+%F %T') ====================" | tee -a "$LOG"

    NCCL_P2P_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u main_cmgnet.py \
        --save_path "$SAVE" \
        --train_file "$EX_TRAIN" \
        --valid_file "$VALID" \
        --test_file "$TEST" \
        --lmdb_path "$EXP_LMDB" \
        --lr 1e-4 \
        --num_workers 12 \
        --epochs $EPOCHS --eval_per_epoch 1 \
        --warmup_ratio 0.02 \
        --label_smoothing 0. \
        --seed "$SEED" \
        --batch_size 48 \
        --gradient_accumulation_steps 2 \
        --do_train \
        --gpus 1 \
        --smiles_aug_proba 0.5 \
        --nmr_aug_proba 0.5 \
        --nmr_aug_scale 3.0 \
        --use_ring_system >> "$LOG" 2>&1
    STATUS=$?
    echo "EXIT:$STATUS $(date '+%F %T')" | tee -a "$LOG"
    echo "==================== train $SAVE 结束 (exit=$STATUS) $(date '+%F %T') ===================="
done
echo "==================== 全部种子结束 $(date '+%F %T') ===================="

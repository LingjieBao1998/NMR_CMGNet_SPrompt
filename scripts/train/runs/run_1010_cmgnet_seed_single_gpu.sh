#!/bin/bash
# NMRMind 数据训练 CMGNet (ring_system 三模式混合训练) —— 单卡 + 显式随机种子
# 参照 run_1004_cmgnet_nmrmind.sh (双卡版):
#   - 初始权重: checkpoints/saved_hf_model_0910 (--model_dir 默认值, 无需显式传)
#   - 数据: data/nmrmind_cmgnet_1004/train.lmdb (+ validation.json / test.json)
#   - 10 epochs (原 1004 实验实际训到 epoch 9 共 10 轮后终止)
# 与双卡版的差异:
#   - 单卡 (--gpus 1), 无 DDP 多卡同步/NCCL watchdog 问题
#   - 全局 batch 仍为 96 = 单卡 48 × 梯度累积 2 (steps/epoch 同为 22269, LR schedule 与 1004 一致)
#   - --seed 真正生效 (main_cmgnet.py 的 main() 已加 pl.seed_everything(args.seed, workers=True))
#
# 用法: bash run_1010_cmgnet_seed_single_gpu.sh <seed> <gpu_id>
#   gpu_id 按 CUDA 编号: 0=4090(24G), 1=5090(32G)   <-- 与 nvidia-smi 的顺序相反!
# 例 (同一张卡顺序跑两个种子):
#   bash run_1010_cmgnet_seed_single_gpu.sh 42 1
#   bash run_1010_cmgnet_seed_single_gpu.sh 2026 1

SEED=$1
GPU_ID=$2
if [ -z "$SEED" ] || [ -z "$GPU_ID" ]; then
    echo "用法: bash $0 <seed> <gpu_id>   (gpu_id: 0=4090, 1=5090)"
    exit 1
fi

cd /mnt/data2/NMR_Core_Mol_Detection

# 前置检查: --seed 必须真正生效, 防止补丁丢失后白跑一整天
if ! grep -q "seed_everything" main_cmgnet.py; then
    echo "错误: main_cmgnet.py 中未找到 seed_everything, --seed 不会生效, 终止"
    exit 1
fi

BATCH_PER_DEVICE=48   # 三模式每条样本展开 3 行, 48*3=144 行峰值约 18G (4090/5090 均安全)
ACCUM_STEP=2          # 全局 batch = 48 * 2 = 96, 与 1004 实验每次优化一致
EPOCHS=10

SAVE_PATH=output/train_cmgnet_ring_system_3.2_nmrmind_1010_seed${SEED}
LOG_FILE=run_1010_seed${SEED}.log

export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "==================== seed=$SEED gpu=$GPU_ID (CUDA编号) 开始 $(date '+%F %T') ====================" | tee -a $LOG_FILE
mkdir -p $SAVE_PATH
NCCL_P2P_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u main_cmgnet.py \
    --save_path $SAVE_PATH \
    --train_file "" \
    --valid_file ./data/nmrmind_cmgnet_1004/validation.json \
    --test_file ./data/nmrmind_cmgnet_1004/test.json \
    --lr 1e-4 \
    --num_workers 12 \
    --epochs $EPOCHS --eval_per_epoch 1 \
    --warmup_ratio 0.02 \
    --label_smoothing 0. \
    --seed $SEED \
    --batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $ACCUM_STEP \
    --do_train \
    --do_test \
    --gpus 1 \
    --smiles_aug_proba 0.5 \
    --nmr_aug_proba 0.5 \
    --nmr_aug_scale 3.0 \
    --lmdb_path ./data/nmrmind_cmgnet_1004/train.lmdb \
    --use_ring_system 2>&1 | tee -a $LOG_FILE
STATUS=${PIPESTATUS[0]}
echo "==================== seed=$SEED 结束 (python exit=$STATUS) $(date '+%F %T') ====================" | tee -a $LOG_FILE
exit $STATUS

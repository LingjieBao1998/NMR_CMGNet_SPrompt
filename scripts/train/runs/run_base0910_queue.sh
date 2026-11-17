#!/bin/bash
# 排队脚本: 等 oracle nmr+frag 评测(内部两半)结束后, 自动接上 base0910 权重评测
#   阶段1: 外部(4502)@5090 + 内部奇数半(27575)@4090 并行
#   阶段2: 内部偶数半(27575)@5090
# 注: CUDA_VISIBLE_DEVICES=0 实为 4090, =1 为 5090

cd /mnt/data2/NMR_Core_Mol_Detection

echo "等待 oracle 任务结束 $(date '+%F %T')"
while pgrep -f "train_cmgnet_oracle_oddhal[f]" >/dev/null 2>&1 || pgrep -f "train_cmgnet_oracle_evenhal[f]" >/dev/null 2>&1; do
    sleep 60
done
sleep 30
# 等显存回落 (正常退出会释放; 兜底轮询)
for i in $(seq 1 30); do
    U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
    [ "$U" -lt 3000 ] && break
    sleep 20
done
echo "oracle 结束, base0910 评测开始 $(date '+%F %T')"

bash run_1005_base_single_gpu.sh 1 external output/train_base0910_smiles_external > run_base0910_external.log 2>&1 &
P1=$!
bash run_1005_base_single_gpu.sh 0 odd output/train_base0910_smiles_oddhalf > run_base0910_oddhalf.log 2>&1 &
P2=$!

wait $P1
echo "外部完成, 内部偶数半开始@5090 $(date '+%F %T')"
bash run_1005_base_single_gpu.sh 1 even output/train_base0910_smiles_evenhalf > run_base0910_evenhalf.log 2>&1 &
P3=$!

wait $P2
wait $P3
echo "base0910 全部完成 $(date '+%F %T')"

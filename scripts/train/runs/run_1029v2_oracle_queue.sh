#!/bin/bash
# 1029_v2 (web 线上权重) oracle nmr+frag beams=10 全量评测队列:
#   阶段1: 外部(4502)@5090 + 内部奇数半(27575)@4090 并行
#   阶段2: 内部偶数半(27575)@5090
#   结束后自动汇总:
#     output/1029v2_oracle_external_confusion.md
#     output/1029v2_oracle_internal_confusion.md
# 注: CUDA_VISIBLE_DEVICES=0 实为 4090, =1 为 5090
cd /mnt/data2/NMR_Core_Mol_Detection
PY=/home/lingjiebao/anaconda3/envs/py312/bin/python

echo "1029v2 oracle 队列启动 $(date '+%F %T')"

bash run_1029v2_oracle_single_gpu.sh 1 external output/train_1029v2_oracle_external > run_1029v2_oracle_external.log 2>&1 &
P1=$!
bash run_1029v2_oracle_single_gpu.sh 0 odd output/train_1029v2_oracle_oddhalf > run_1029v2_oracle_oddhalf.log 2>&1 &
P2=$!

wait $P1
echo "外部完成, 内部偶数半开始@5090 $(date '+%F %T')"
bash run_1029v2_oracle_single_gpu.sh 1 even output/train_1029v2_oracle_evenhalf > run_1029v2_oracle_evenhalf.log 2>&1 &
P3=$!

wait $P2
wait $P3
echo "1029v2 oracle 全部完成 $(date '+%F %T')"

$PY summarize_confusion.py output/1029v2_oracle_external_confusion.md \
    "外部测试集 1029_v2 (web 线上权重) oracle nmr+frag (beams=10)" \
    output/train_1029v2_oracle_external/test_*.json || echo "外部汇总失败(结果未生成)"
$PY summarize_confusion.py output/1029v2_oracle_internal_confusion.md \
    "内部测试集 1029_v2 (web 线上权重) oracle nmr+frag (beams=10)" \
    output/train_1029v2_oracle_oddhalf/test_*.json output/train_1029v2_oracle_evenhalf/test_*.json || echo "内部汇总失败(结果未生成)"
echo "汇总完成 $(date '+%F %T')"

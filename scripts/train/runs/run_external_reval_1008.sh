#!/bin/bash
# 2026-10-08 外部测试集更新后的重评队列 (单卡 5090, CUDA 序 idx=1)
# 新 external_test.json: 4502 条, 含母核 64 条
# 口径与上轮完全一致: beams=10, batch=48, CMGNET_FAST_CORE_CHECK=1, 同权重同 aug 参数
# 结果写入 *_1008 新目录, 上轮原始 JSON 保留以便新旧对比
#   1) main(nmrmind_1004 best-09) smiles
#   2) main frag
#   3) main nmr+frag (oracle_ring, label=1 给真实环系)
#   4) 1029_v2 (web 线上权重) nmr+frag (oracle_ring)
#   5) base0910 (初始权重) smiles
# 每步独立: 单步失败不中断后续; 跑完自动出各模式混淆矩阵 + 合并文件
export PATH=/home/lingjiebao/anaconda3/envs/py312/bin:$PATH
cd /mnt/data2/NMR_Core_Mol_Detection
PY=/home/lingjiebao/anaconda3/envs/py312/bin/python
G=1   # CUDA 序 1 = 5090 (torch 视角 0=4090, 1=5090)

echo "外部集重评(1008)启动 $(date '+%F %T') gpu=$G (5090)"

# ---- 1) main · smiles ----
echo "---- [1/5] main smiles ----"
bash run_1005_half_single_gpu.sh $G smiles external output/train_cmgnet_smiles_external_1008 \
    > run_1008_main_smiles_external.log 2>&1
$PY summarize_confusion.py output/external_smiles_confusion_1008.md \
    "外部测试集(1008版) 主模型 best-09 smiles (beams=10)" \
    output/train_cmgnet_smiles_external_1008/test_*.json || echo "!! [1/5] main smiles 汇总失败"
date '+%F %T'

# ---- 2) main · frag ----
echo "---- [2/5] main frag ----"
bash run_1005_half_single_gpu.sh $G frag external output/train_cmgnet_frag_external_1008 \
    > run_1008_main_frag_external.log 2>&1
$PY summarize_confusion.py output/external_frag_confusion_1008.md \
    "外部测试集(1008版) 主模型 best-09 frag (beams=10)" \
    output/train_cmgnet_frag_external_1008/test_*.json || echo "!! [2/5] main frag 汇总失败"
date '+%F %T'

# ---- 3) main · nmr+frag oracle ----
echo "---- [3/5] main nmr+frag oracle ----"
bash run_1005_half_single_gpu.sh $G "nmr+frag" external output/train_cmgnet_oracle_external_1008 1 \
    > run_1008_main_oracle_external.log 2>&1
$PY summarize_confusion.py output/oracle_external_confusion_1008.md \
    "外部测试集(1008版) 主模型 best-09 oracle nmr+frag (beams=10)" \
    output/train_cmgnet_oracle_external_1008/test_*.json || echo "!! [3/5] main oracle 汇总失败"
date '+%F %T'

# ---- 4) 1029_v2 · nmr+frag oracle ----
echo "---- [4/5] 1029_v2 oracle ----"
bash run_1029v2_oracle_single_gpu.sh $G external output/train_1029v2_oracle_external_1008 \
    > run_1008_1029v2_oracle_external.log 2>&1
$PY summarize_confusion.py output/1029v2_oracle_external_confusion_1008.md \
    "外部测试集(1008版) 1029_v2(web线上权重) oracle nmr+frag (beams=10)" \
    output/train_1029v2_oracle_external_1008/test_*.json || echo "!! [4/5] 1029v2 汇总失败"
date '+%F %T'

# ---- 5) base0910 · smiles ----
echo "---- [5/5] base0910 smiles ----"
bash run_1005_base_single_gpu.sh $G external output/train_base0910_smiles_external_1008 \
    > run_1008_base0910_smiles_external.log 2>&1
$PY summarize_confusion.py output/base0910_external_confusion_1008.md \
    "外部测试集(1008版) base0910(初始权重) smiles (beams=10)" \
    output/train_base0910_smiles_external_1008/test_*.json || echo "!! [5/5] base0910 汇总失败"
date '+%F %T'

# ---- 合并 ----
cat output/external_smiles_confusion_1008.md \
    output/external_frag_confusion_1008.md \
    output/oracle_external_confusion_1008.md \
    output/1029v2_oracle_external_confusion_1008.md \
    output/base0910_external_confusion_1008.md \
    > output/external_confusion_matrices_1008.md 2>/dev/null

echo "外部集重评(1008)全部完成 $(date '+%F %T')"

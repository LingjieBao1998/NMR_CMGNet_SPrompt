#!/bin/bash
# 2026-10-08 第二版 external_test.json (21:58 更新) 外部集重评队列 — 单卡 5090 (CUDA 序 idx=1)
# 本版测试集: 4502 条 / 64 母核 / 0 重复(手性拆分) / 49 条 0327_csv 正样本峰表更新 / 修正 index=4493 的 1175.9->175.9
# 口径:
#   1) main smiles     全量 4502
#   2) main frag       全量 4502
#   3) main oracle     **只跑阳性** (external_test_pos_only.json, 64 条)
#   4) 1029_v2 oracle  **只跑阳性** (64 条)
#   5) base0910 smiles 全量 4502
# 输出目录统一 _1008b 后缀; 1008 与更早的结果原样保留
export PATH=/home/lingjiebao/anaconda3/envs/py312/bin:$PATH
cd /mnt/data2/NMR_Core_Mol_Detection
PY=/home/lingjiebao/anaconda3/envs/py312/bin/python
G=1   # CUDA 序 1 = 5090 (torch 视角 0=4090, 1=5090)

echo "外部集重评(1008b)启动 $(date '+%F %T') gpu=$G (5090)"

# ---- 1) main · smiles (全量) ----
echo "---- [1/5] main smiles (4502) ----"
bash run_1005_half_single_gpu.sh $G smiles external output/train_cmgnet_smiles_external_1008b \
    > run_1008b_main_smiles_external.log 2>&1
$PY summarize_core_smiles.py output/external_smiles_1008b_detail.md \
    "外部测试集(1008b版) 主模型 best-09 smiles (beams=10, 全量4502)" \
    output/train_cmgnet_smiles_external_1008b/test_*.json || echo "!! [1/5] main smiles 汇总失败"
date '+%F %T'

# ---- 2) main · frag (全量) ----
echo "---- [2/5] main frag (4502) ----"
bash run_1005_half_single_gpu.sh $G frag external output/train_cmgnet_frag_external_1008b \
    > run_1008b_main_frag_external.log 2>&1
$PY summarize_core_smiles.py output/external_frag_1008b_detail.md \
    "外部测试集(1008b版) 主模型 best-09 frag (beams=10, 全量4502)" \
    output/train_cmgnet_frag_external_1008b/test_*.json || echo "!! [2/5] main frag 汇总失败"
date '+%F %T'

# ---- 3) main · nmr+frag oracle (只跑阳性 64 条) ----
echo "---- [3/5] main nmr+frag oracle (posonly 64) ----"
bash run_1005_half_single_gpu.sh $G "nmr+frag" posonly output/train_cmgnet_oracle_external_1008b 1 \
    > run_1008b_main_oracle_posonly.log 2>&1
$PY summarize_core_smiles.py output/oracle_posonly_1008b_detail.md \
    "外部测试集(1008b版) 主模型 best-09 oracle nmr+frag (beams=10, 只评阳性64条)" \
    output/train_cmgnet_oracle_external_1008b/test_*.json || echo "!! [3/5] main oracle 汇总失败"
date '+%F %T'

# ---- 4) 1029_v2 · nmr+frag oracle (只跑阳性 64 条) ----
echo "---- [4/5] 1029_v2 oracle (posonly 64) ----"
bash run_1029v2_oracle_single_gpu.sh $G posonly output/train_1029v2_oracle_external_1008b \
    > run_1008b_1029v2_oracle_posonly.log 2>&1
$PY summarize_core_smiles.py output/1029v2_oracle_posonly_1008b_detail.md \
    "外部测试集(1008b版) 1029_v2(web线上权重) oracle nmr+frag (beams=10, 只评阳性64条)" \
    output/train_1029v2_oracle_external_1008b/test_*.json || echo "!! [4/5] 1029v2 汇总失败"
date '+%F %T'

# ---- 5) base0910 · smiles (全量) ----
echo "---- [5/5] base0910 smiles (4502) ----"
bash run_1005_base_single_gpu.sh $G external output/train_base0910_smiles_external_1008b \
    > run_1008b_base0910_smiles_external.log 2>&1
$PY summarize_core_smiles.py output/base0910_smiles_1008b_detail.md \
    "外部测试集(1008b版) base0910(初始权重) smiles (beams=10, 全量4502)" \
    output/train_base0910_smiles_external_1008b/test_*.json || echo "!! [5/5] base0910 汇总失败"
date '+%F %T'

# ---- 合并 ----
cat output/external_smiles_1008b_detail.md \
    output/external_frag_1008b_detail.md \
    output/oracle_posonly_1008b_detail.md \
    output/1029v2_oracle_posonly_1008b_detail.md \
    output/base0910_smiles_1008b_detail.md \
    > output/external_1008b_all_details.md 2>/dev/null

echo "外部集重评(1008b)全部完成 $(date '+%F %T')"

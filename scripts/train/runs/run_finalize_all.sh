#!/bin/bash
# 全自动收尾汇总 (在 tmux 中运行, 不依赖 VS Code / 对话会话):
#   1) oracle 内部两半跑完 -> output/oracle_internal_confusion.md
#   2) base0910 队列跑完   -> output/base0910_external_confusion.md + output/base0910_internal_confusion.md
cd /mnt/data2/NMR_Core_Mol_Detection
PY=/home/lingjiebao/anaconda3/envs/py312/bin/python

echo "finalize 启动 $(date '+%F %T')"

# ---- 1) 内部 oracle ----
for i in $(seq 1 480); do
    if grep -aq "结束" run_oracle_oddhalf.log 2>/dev/null && grep -aq "结束" run_oracle_evenhalf.log 2>/dev/null; then break; fi
    if ! pgrep -f "train_cmgnet_oracle_oddhal[f]" >/dev/null 2>&1 && ! pgrep -f "train_cmgnet_oracle_evenhal[f]" >/dev/null 2>&1; then
        echo "警告: oracle 进程均已退出(检查日志)"; break
    fi
    sleep 60
done
sleep 20
$PY summarize_confusion.py output/oracle_internal_confusion.md "内部测试集 oracle nmr+frag (beams=10, label=1给真实环系)" \
    output/train_cmgnet_oracle_oddhalf/test_*.json output/train_cmgnet_oracle_evenhalf/test_*.json \
    || echo "oracle 内部汇总失败(可能结果未生成)"
echo "=== oracle 内部汇总完成 $(date '+%F %T') ==="

# ---- 2) base0910 ----
for i in $(seq 1 600); do
    grep -aq "全部完成" run_base0910_queue.log 2>/dev/null && break
    sleep 60
done
sleep 20
$PY summarize_confusion.py output/base0910_external_confusion.md "外部测试集 base0910(初始权重) smiles (beams=10)" \
    output/train_base0910_smiles_external/test_*.json || echo "base 外部汇总失败"
$PY summarize_confusion.py output/base0910_internal_confusion.md "内部测试集 base0910(初始权重) smiles (beams=10)" \
    output/train_base0910_smiles_oddhalf/test_*.json output/train_base0910_smiles_evenhalf/test_*.json || echo "base 内部汇总失败"
echo "=== base0910 汇总完成 $(date '+%F %T') ==="
echo "finalize 全部完成 $(date '+%F %T')"

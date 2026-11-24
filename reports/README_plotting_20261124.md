# Plotting update (2026-11-24)

本次加入今天最终确认的绘图方案（脚本 + 数据 + 图）。

## 文件

- `make_model_top10_f1_bar.py` + `model_top10_f1_bar_data.csv` + `model_top10_f1_bar.png/.pdf`
  - Top-10 各模型 F1 柱状图（Arial、Model 加粗 45°、红色虚线 Best F1=0.962 左侧标注）。
- `make_recall_raw_clean_overlay.py` + `recall_raw_vs_clean_data.csv` + `recall_raw_vs_clean_overlay.png/.pdf`
  - Raw/Clean Recall 重合柱图（Raw 实心，Clean 半透明斜线；Raw 蓝色数值、Clean 红色数值，透明背景白描边；最佳线左侧上方标注）。
- `make_sim5_multimodel_summary.py` + `make_sim5_multimodel_plot.py`
  - 5 条模拟 13C 谱多模型对比（core@k / exact@k），输出 `sim5_multimodel_summary.md/.csv`、`sim5_multimodel_core_exact.png/.pdf`。
- `sim5_nmrsolver_raw.json` + `sim5_nmrsolver_summary.md` + `run_sim5_nmrsolver.py`
  - NMR-Solver（Bohrium NMR-Toolbox）5 条模拟谱提交/轮询结果；raw 含完整响应，cookie 不入库。
- `xgb_sprompt_combined_4502_summary.csv`
  - XGB(τ=0.5)+SPrompt(ex1209 5ep) 在 april18peak 4502 上按 seed 配对的组合指标。
- `demo/sprompt_demo_sample51.png/.svg`, `demo/sim5_sprompt_demo_sample2.png/.svg`
  - SPrompt 优势案例结构对比图。

## 数据依赖

- F1/Recall 图数据已内嵌在对应 `*_data.csv`。
- sim5 多模型图依赖 `data/sim5/fluoroquinolones_nmr.json` 与 `NMR_Core_Mol_Detection/output/sim5_*` dump；在本仓库直接重跑前需保证这些路径可访问，或修改脚本中的路径。
- NMR-Solver 重提交需 Bohrium 会话 cookie（`/tmp/nmrsolver_srv/cookie.txt`），不要入库。

作者：ZCode（Claude），2026-11-24

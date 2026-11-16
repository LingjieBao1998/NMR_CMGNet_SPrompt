#!/usr/bin/env python3
"""
CMGNet-SPrompt 全链路复跑——1029_v2 检查点 × 4502 外部测试集(1008 谱)

目的: 与论文口径(2026-03, F1=0.9545, TP=63/FP=5/FN=1)做同系统对比。
复刻部署端 nmr_core_system/core_detection/ensemble_nmr_predictor.py 的投票逻辑:
  - XGBoost 票:  NMR shifts -> CRESS nmr_encode -> model_xgb_cress_1220.pkl proba >= 0.5
  - Frag 票:     CMGNet frag top-10 内任一 has_core_structure
  - SMILES 票:   CMGNet smals top-10 内任一 has_core_structure
  - 终判:        三票 >= 2 (2/3 规则)
frag/smiles 预测复用 2026-10-28 老栈口径逐样本结果(transformers 4.57.6, beams=10):
  output/external_4502_base1029v2_tf457_{frag,smiles}_beams10/

运行:
  CUDA_VISIBLE_DEVICES=1 nohup ~/venvs/py312_tf457/bin/python \
      rerun_sprompt_fullpipeline_1029v2.py > logs/rerun_sprompt_fullpipeline_1029v2.log 2>&1 &
"""

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

# ---- 路径(与部署端 xgboost_predict.py 默认值一致) ----
XGB_MODEL_PATH = REPO_ROOT / "models/xgboost/model_xgb_cress_1220.pkl"
CRESS_CONFIG_PATH = REPO_ROOT / "models/2_5_w_model/8.json"
CRESS_MODEL_PATH = REPO_ROOT / "models/2_5_w_model/8.pth"
CRESS_TOKENIZER_PATH = REPO_ROOT / "models/tokenizer-smiles-roberta-1e"

TEST_JSON = REPO_ROOT / "data/nmrmind_cmgnet_1004/external_test.json"
FRAG_JSON = REPO_ROOT / "output/external_4502_base1029v2_tf457_frag_beams10/test_frag_beams10_samples4502.json"
SMILES_JSON = REPO_ROOT / "output/external_4502_base1029v2_tf457_smiles_beams10/test_smiles_beams10_samples4502.json"

OUT_DIR = REPO_ROOT / "output/sprompt_fullpipeline_1029v2_20261029"
BATCH_SIZE = 256
XGB_THRESHOLD = 0.5
TOP_K = 10  # 部署端 num_returns/use_top_k 默认

PAPER = dict(tp=63, fp=5, fn=1)  # 论文口径(reviewer 信确认)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def topk_has_core(rec):
    """部署端 use_top_k=True: top-10 内任一 has_core_structure"""
    return any(p.get("has_core_structure", False) for p in rec["predictions"][:TOP_K])


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def report(name, tp, fp, fn, n_pos):
    p, r, f1 = prf(tp, fp, fn)
    log(f"{name:<24} (tp,fp,fn)=({tp},{fp},{fn})  P={p:.4f} R={r:.4f} F1={f1:.4f}")
    return dict(name=name, tp=tp, fp=fp, fn=fn, precision=p, recall=r, f1=f1)


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 1. 数据与已有预测 ----
    test = json.load(open(TEST_JSON))
    frag = json.load(open(FRAG_JSON))
    smiles = json.load(open(SMILES_JSON))
    assert len(test) == len(frag) == len(smiles) == 4502
    for a, b in zip(test, frag):
        assert a["cano_smiles"] == b["reference_smiles"], "frag 顺序错位"
    for a, b in zip(test, smiles):
        assert a["cano_smiles"] == b["reference_smiles"], "smiles 顺序错位"
    labels = [x["label"][0] for x in test]
    n_pos = sum(labels)
    log(f"测试集 {len(test)} 条, 正例 {n_pos}")

    frag_vote = [topk_has_core(r) for r in frag]
    smiles_vote = [topk_has_core(r) for r in smiles]
    log(f"frag 票阳性 {sum(frag_vote)}, smiles 票阳性 {sum(smiles_vote)}")

    # ---- 2. XGBoost 票 (CRESS nmr_encode -> xgb proba) ----
    from nmr_core_system.core_detection.infer import ModelInference
    import joblib
    import torch

    with open(CRESS_CONFIG_PATH) as f:
        config = json.load(f)
    config["model_config"]["roberta_tokenizer_path"] = str(CRESS_TOKENIZER_PATH)
    config["dataset_config"]["smiles_tokenizer"] = str(CRESS_TOKENIZER_PATH)
    tmp_cfg = OUT_DIR / "_cress_config_patched.json"
    json.dump(config, open(tmp_cfg, "w"))

    log("加载 CRESS + XGBoost ...")
    cress = ModelInference(str(tmp_cfg), str(CRESS_MODEL_PATH), device="cuda")
    xgb_model = joblib.load(XGB_MODEL_PATH)
    log("模型加载完成")

    xgb_prob = []
    nmr_list = [x["NMR"] for x in test]
    for i in range(0, len(nmr_list), BATCH_SIZE):
        batch = nmr_list[i:i + BATCH_SIZE]
        with torch.no_grad():
            emb = cress.nmr_encode(batch).cpu().numpy()
        xgb_prob.extend(xgb_model.predict_proba(emb)[:, 1].tolist())
        if (i // BATCH_SIZE) % 4 == 0:
            log(f"  XGB {i + len(batch)}/{len(nmr_list)}")
    xgb_vote = [p >= XGB_THRESHOLD for p in xgb_prob]
    log(f"XGB 票阳性 {sum(xgb_vote)} (阈值 {XGB_THRESHOLD})")

    # ---- 3. 投票与指标 ----
    votes = list(zip(xgb_vote, frag_vote, smiles_vote))
    pos_votes = [sum(v) for v in votes]
    rules = {
        "ensemble_2of3 (2/3规则)": [n >= 2 for n in pos_votes],
        "ensemble_any (敏感)": [n >= 1 for n in pos_votes],
        "ensemble_all (严格)": [n == 3 for n in pos_votes],
    }

    rows = []
    tp = sum(1 for l, v in zip(labels, xgb_vote) if l and v)
    fp = sum(1 for l, v in zip(labels, xgb_vote) if not l and v)
    rows.append(report("xgboost only", tp, fp, n_pos - tp, n_pos))
    tp = sum(1 for l, v in zip(labels, frag_vote) if l and v)
    fp = sum(1 for l, v in zip(labels, frag_vote) if not l and v)
    rows.append(report("cmgnet frag only", tp, fp, n_pos - tp, n_pos))
    tp = sum(1 for l, v in zip(labels, smiles_vote) if l and v)
    fp = sum(1 for l, v in zip(labels, smiles_vote) if not l and v)
    rows.append(report("cmgnet smiles only", tp, fp, n_pos - tp, n_pos))
    for name, pred in rules.items():
        tp = sum(1 for l, v in zip(labels, pred) if l and v)
        fp = sum(1 for l, v in zip(labels, pred) if not l and v)
        rows.append(report(name, tp, fp, n_pos - tp, n_pos))

    # ---- 4. 落盘 ----
    detail = [
        dict(index=i, cano_smiles=test[i]["cano_smiles"], label=labels[i],
             xgb_prob=xgb_prob[i], xgb_vote=xgb_vote[i],
             frag_vote=frag_vote[i], smiles_vote=smiles_vote[i],
             positive_votes=pos_votes[i])
        for i in range(len(test))
    ]
    json.dump(detail, open(OUT_DIR / "per_sample_votes.json", "w"))
    summary = dict(
        date="2026-10-29", checkpoint="base1029v2", xgb="model_xgb_cress_1220.pkl",
        stack="venv py312_tf457 (transformers 4.57.6) + 复用 10-28 老栈 frag/smiles 预测",
        test="data/nmrmind_cmgnet_1004/external_test.json (4502, 正例 64)",
        voting="部署端 ensemble_nmr_predictor.py: top-10 任一 flag + 2/3 规则",
        paper_baseline=PAPER, rows=rows)
    json.dump(summary, open(OUT_DIR / "summary.json", "w"), ensure_ascii=False, indent=2)

    lines = ["# CMGNet-SPrompt 全链路复跑 (1029_v2 × 4502 外部测试)", "",
             f"投票规则 = 部署端口径: top-10 任一 has_core_structure + 2/3 多数; XGB 阈值 {XGB_THRESHOLD}", "",
             "| 配置 | (tp, fp, fn) | Precision | Recall | F1 |", "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['name']} | ({r['tp']}, {r['fp']}, {r['fn']}) | "
                     f"{r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} |")
    lines += ["", f"论文口径对照: (63, 5, 1) P=0.9265 R=0.9844 F1=0.9545", ""]
    (OUT_DIR / "summary.md").write_text("\n".join(lines))
    tmp_cfg.unlink(missing_ok=True)
    log(f"完成, 用时 {time.time() - t0:.0f}s, 输出在 {OUT_DIR}")


if __name__ == "__main__":
    main()

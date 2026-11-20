#!/usr/bin/env python3
"""外部测试集(4,502 = 64 正 / 4,438 负)四模型 top-k 二分类指标复算。

本模块只读**逐样本预测 dump**,不读任何已经表格化的结果,因此表里的每个数
都可以从这里重新算出来。四个模型的数据口径如下:

    XGBoost-CReSS : reports/calib_data/external_apr18peak.npz 的 y / p_raw
                    (判别式模型,无 top-k;"top-1" 即 τ=0.50 的单次判决)
    CMGNet (0910) : results/predictions/cmgnet_0910_smiles_beams10.json
    CMGNet-SPrompt: results/predictions/sprompt_seed0_external4502_beams10.json
                    (释放版权重 sprompt_ex1209_1ep_seed0.ckpt,decode_mode=nmr+frag)
    NMRMind       : results/predictions/nmrmind_external4502_top10.json
                    (官方 200-epoch 权重,april18peak 峰表)

判定口径: top-k 内**任一**候选分子含全母核(3-羧基-4-喹诺酮及其氮杂/萘啶酸
类似物)即把该谱判为阳性。指标定义: P=TP/(TP+FP), R=TP/64,
F1=2PR/(P+R), Specificity=TN/4438, Accuracy=(TP+TN)/4502。

用法:
    PYTHONPATH=. python -m analysis.metrics.topk_external_4502
    python analysis/metrics/topk_external_4502.py --out results/table_topk_external_4502.md

作者:ZCode(Claude),2026-11-20
"""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')

ROOT = Path(__file__).resolve().parents[2]

## ZCode 2026-11-20：母核 SMILES 沿用仓库既有口径（analysis/metrics/
## calculate_cmgnet_metrics.py 与 integrate_models_for_precision.py 同一对
## 模式）。此处用普通 HasSubstructMatch 而不是 rdSubstructLibrary 的互变异构
## 感知版本：两者在 NMRMind 全 4,502 条上逐 k 结果完全相同（45/0、47/1、
## 47/2、48/4，已实测），故取更简单、无状态的那个。
CORE_SMILES = (
    'O=C1C(C(O)=O)=CNC2=CC=CC=C12',   # 4-喹诺酮-3-羧酸（氟喹诺酮母核）
    'O=C1C(C(O)=O)=CNC2=NC=CC=C12',   # 氮杂类似物（萘啶酸母核）
)
CORE_MOLS = [Chem.MolFromSmiles(s) for s in CORE_SMILES]

TOP_KS = (1, 3, 5, 10)
N_POS = 64
N_NEG = 4438


def has_core(smiles):
    """判断单个 SMILES 是否含全母核。

    Args:
        smiles: 候选分子的 SMILES 字符串；NMRMind 原始 dump 里个别记录是
            list,这里只接受 str,其它类型一律判 False。

    Returns:
        bool;SMILES 无法解析时返回 False(与仓库既有判定一致)。
    """
    if not isinstance(smiles, str):
        return False
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    return any(mol.HasSubstructMatch(core) for core in CORE_MOLS)


def metrics_from_counts(tp, fp, fn, tn):
    """由混淆矩阵四项算五个指标。

    Args:
        tp/fp/fn/tn: 整数计数(单位: 条谱)。

    Returns:
        dict,含 precision/recall/f1/specificity/accuracy,全为 0~1 浮点;
        分母为 0 时对应项返回 0.0(与 sklearn zero_division=0 一致)。
    """
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    acc = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) else 0.0
    return {"precision": p, "recall": r, "f1": f1,
            "specificity": spec, "accuracy": acc}


def _confusion(labels, preds):
    """labels/preds 为同长 bool 数组，返回 (tp, fp, fn, tn)。"""
    labels = np.asarray(labels, dtype=bool)
    preds = np.asarray(preds, dtype=bool)
    tp = int(np.sum(labels & preds))
    fp = int(np.sum(~labels & preds))
    fn = int(np.sum(labels & ~preds))
    tn = int(np.sum(~labels & ~preds))
    return tp, fp, fn, tn


def eval_predictions_json(path, pred_key="predictions", top_ks=TOP_KS):
    """评测 CMGNet / CMGNet-SPrompt 的逐样本 dump。

    Args:
        path: JSON 文件路径。结构为 list[{reference_analysis:
            {has_core_structure: bool}, predictions: [{has_core_structure: bool,
            ...}, ...]}, ...]，长度必须为 4,502。
        pred_key: 候选列表字段名(默认 "predictions")。
        top_ks: 要统计的 k 值。

    Returns:
        dict[str, dict]: k -> {"tp","fp","fn","tn","precision","recall",
        "f1","specificity","accuracy"}；同时返回 "n"/"n_pos" 元信息键。
    """
    data = json.load(open(path))
    _check_size(data, path)
    labels = [bool(x["reference_analysis"]["has_core_structure"]) for x in data]
    out = {"n": len(data), "n_pos": int(sum(labels))}
    for k in top_ks:
        preds = [any(p.get("has_core_structure", False)
                     for p in x[pred_key][:k]) for x in data]
        tp, fp, fn, tn = _confusion(labels, preds)
        out[k] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                  **metrics_from_counts(tp, fp, fn, tn)}
    return out


def eval_nmrmind_json(path, top_ks=TOP_KS):
    """评测 NMRMind 官方 dump(结构与上面不同)。

    Args:
        path: JSON 文件路径。结构为 list[{smiles: str|list, index: int,
            result: [SMILES, ...]}, ...]；"smiles" 是输入谱对应的参考结构,
            "result" 是模型给出的 10 个候选。

    Returns:
        dict,布局与 eval_predictions_json 相同。
    """
    data = json.load(open(path))
    _check_size(data, path)

    def ref_smiles(item):
        s = item["smiles"]
        return s[0] if isinstance(s, (list, tuple)) else s

    labels = [has_core(ref_smiles(x)) for x in data]
    out = {"n": len(data), "n_pos": int(sum(labels))}
    for k in top_ks:
        preds = [any(has_core(s) for s in x["result"][:k]) for x in data]
        tp, fp, fn, tn = _confusion(labels, preds)
        out[k] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                  **metrics_from_counts(tp, fp, fn, tn)}
    return out


def eval_xgb_npz(path, threshold=0.5):
    """评测 XGBoost-CReSS(判别式,无 top-k)。

    Args:
        path: reports/calib_data/external_apr18peak.npz,含 y(int) 与
            p_raw(float32, 长度 4,502)。
        threshold: 决策阈值,默认 0.50(与正文 τ 一致)。

    Returns:
        dict,键与 eval_predictions_json 相同,但只有一个 "k" 条目。
    """
    d = np.load(path, allow_pickle=True)
    y, p = d["y"], d["p_raw"]
    _check_size(y, path)
    tp, fp, fn, tn = _confusion(y == 1, p >= threshold)
    return {"n": len(y), "n_pos": int((y == 1).sum()),
            "threshold": threshold,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            **metrics_from_counts(tp, fp, fn, tn)}


def _check_size(seq, path):
    """硬校验: 行数/正例数不对就直接报错,防止口径漂移悄悄进表。"""
    n = len(seq)
    if n != N_POS + N_NEG:
        raise ValueError(f"{path}: 期望 {N_POS + N_NEG} 条,实际 {n} 条")


def render_markdown(rows, xgb):
    """把结果渲染成 Markdown 表 + 出处说明。

    Args:
        rows: [(模型名, 数据文件, {k: metrics})]，按模型分组。
        xgb: eval_xgb_npz 的返回值。

    Returns:
        str,完整 Markdown 文本。
    """
    lines = []
    ap = lines.append
    ap("# 外部测试集 top-k 指标(4,502 = 64 正 / 4,438 负)")
    ap("")
    ap("**全部数值由 `analysis/metrics/topk_external_4502.py` 从逐样本预测文件重算,"
       "未引用任何已有表格。** 判定: top-k 内任一候选含全母核即判阳。")
    ap("")
    ap("| 模型 | Top-k | TP | FP | FN | TN | Precision | Recall | F1 | Specificity | Accuracy |")
    ap("|---|---|---|---|---|---|---|---|---|---|---|")
    for name, _path, res in rows:
        for k in TOP_KS:
            m = res[k]
            ap(f"| {name} | {k} | {m['tp']} | {m['fp']} | {m['fn']} | {m['tn']} "
               f"| {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} "
               f"| {m['specificity']:.4f} | {m['accuracy']:.4f} |")
    m = xgb
    ap(f"| XGBoost-CReSS (τ={m['threshold']:.2f}, 无 top-k) | – | {m['tp']} | {m['fp']} "
       f"| {m['fn']} | {m['tn']} | {m['precision']:.4f} | {m['recall']:.4f} "
       f"| {m['f1']:.4f} | {m['specificity']:.4f} | {m['accuracy']:.4f} |")
    ap("")
    ap("## 数据来源")
    ap("")
    ap("| 模型 | 逐样本文件 |")
    ap("|---|---|")
    for name, path, _res in rows:
        try:
            shown = os.path.relpath(path, ROOT)
        except ValueError:
            shown = path
        ap(f"| {name} | `{shown}` |")
    ap(f"| XGBoost-CReSS | `reports/calib_data/external_apr18peak.npz` (y, p_raw) |")
    ap("")
    ap("说明: CMGNet-SPrompt 行是**生成侧** top-k(模型自己的 10 个候选);"
       "部署管线里的二分类判决由分类器门控,在任何 k 下都是 TP 63 / FP 4 / "
       "FN 1 / TN 4,434(P 0.9403, R 0.9844, F1 0.9618)。")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xgb-npz", default=str(ROOT / "reports/calib_data/external_apr18peak.npz"))
    ap.add_argument("--cmgnet-json",
                    default=str(ROOT / "results/predictions/cmgnet_0910_smiles_beams10.json"))
    ap.add_argument("--sprompt-json",
                    default=str(ROOT / "results/predictions/sprompt_seed0_external4502_beams10.json"))
    ap.add_argument("--nmrmind-json",
                    default=str(ROOT / "results/predictions/nmrmind_external4502_top10.json"))
    ap.add_argument("--out", default=str(ROOT / "results/table_topk_external_4502.md"))
    ap.add_argument("--csv", default=str(ROOT / "results/table_topk_external_4502.csv"))
    args = ap.parse_args()

    rows = [
        ("CMGNet (0910)", args.cmgnet_json, eval_predictions_json(args.cmgnet_json)),
        ("CMGNet-SPrompt (生成侧)", args.sprompt_json, eval_predictions_json(args.sprompt_json)),
        ("NMRMind", args.nmrmind_json, eval_nmrmind_json(args.nmrmind_json)),
    ]
    xgb = eval_xgb_npz(args.xgb_npz)

    for name, _p, res in rows:
        if res["n_pos"] != N_POS:
            raise ValueError(f"{name}: 正例 {res['n_pos']} != {N_POS}")

    md = render_markdown(rows, xgb)
    Path(args.out).write_text(md, encoding="utf-8")

    with open(args.csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "top_k", "tp", "fp", "fn", "tn",
                    "precision", "recall", "f1", "specificity", "accuracy"])
        for name, _p, res in rows:
            for k in TOP_KS:
                m = res[k]
                w.writerow([name, k, m["tp"], m["fp"], m["fn"], m["tn"],
                            f"{m['precision']:.4f}", f"{m['recall']:.4f}",
                            f"{m['f1']:.4f}", f"{m['specificity']:.4f}",
                            f"{m['accuracy']:.4f}"])
        w.writerow(["XGBoost-CReSS", "-", xgb["tp"], xgb["fp"], xgb["fn"], xgb["tn"],
                    f"{xgb['precision']:.4f}", f"{xgb['recall']:.4f}",
                    f"{xgb['f1']:.4f}", f"{xgb['specificity']:.4f}",
                    f"{xgb['accuracy']:.4f}"])

    print(md)
    print(f"written: {args.out}\nwritten: {args.csv}")


if __name__ == "__main__":
    main()

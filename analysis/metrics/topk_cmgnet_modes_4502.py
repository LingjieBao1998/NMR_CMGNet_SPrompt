#!/usr/bin/env python3
"""CMGNet 三种模式 × top-k 指标复算(外部集 4,502 = 64 正 / 4,438 负)。

三种模式(论文 Table 2 / SI Table S1 的分组口径):

    CMGNet-0910      : 0910 基础 checkpoint,decode_mode=smiles,单模型
    CMGNet-SCAFFOLD  : 全量语料重训(1209 extrain 5ep),decode_mode=frag
                       (骨架/片段拼接路径,故称 SCAFFOLD),3 seeds
    CMGNet-SMILES    : 同一批重训权重,decode_mode=smiles(整串直出),3 seeds

输入是逐样本预测 dump(list,长度 4,502,每条含 canonical_reference_smiles
与 predictions[10] 的 rank/smiles)。**本模块不读任何已表格化的结果**,
表里每个数字都由这里从 SMILES 重新判定得到。

判定口径(与 analysis/metrics/topk_external_4502.py 完全一致):
    - 母核 = 4-喹诺酮-3-羧酸 + 其氮杂/萘啶酸类似物,HasSubstructMatch;
    - 样本级命中 = top-k 内任一候选含全母核 → 判阳;
    - Recall 分母固定 64;Specificity 分母固定 4,438;
    - SMILES 准确率 SA = 参考结构与候选 canonical SMILES 完全相等,
      分母固定 64(正例),逐 k 统计(exact@k)。

用法(需 RDKit;仓库根目录执行,PYTHONPATH 必须指向仓库根才能 import
analysis.metrics):
    PYTHONPATH=. python analysis/metrics/topk_cmgnet_modes_4502.py
    PYTHONPATH=. python analysis/metrics/topk_cmgnet_modes_4502.py \
        --corpus-root /mnt/data2/NMR_Core_Mol_Detection/output \
        --out results/table_cmgnet_modes_4502.md

作者:ZCode(Claude),2026-11-21
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger

from analysis.metrics.topk_external_4502 import (  # noqa: F401  (同口径复用)
    N_NEG,
    N_POS,
    TOP_KS,
    has_core,
    metrics_from_counts,
)

RDLogger.DisableLog('rdApp.*')

ROOT = Path(__file__).resolve().parents[2]

## ZCode 2026-11-21：dump 全量预测语料在兄弟仓库，默认路径写死并显式报错，
## 避免换成空目录时静默算出 0。若日后搬进本仓库，改 DEFAULT_CORPUS 即可。
DEFAULT_CORPUS = Path('/mnt/data2/NMR_Core_Mol_Detection/output')


def mode_files(corpus, mode):
    """给出某模式在外部集上的逐样本 dump 路径列表。

    Args:
        corpus: 预测语料根目录(含 apr18_* 子目录)。
        mode: "0910" | "scaffold" | "smiles"。

    Returns:
        list[(seed_label, Path)];目录或文件缺失时抛 FileNotFoundError。
    """
    if mode == "0910":
        spec = [("0910", "apr18_base0910_smiles_beams10/test_smiles_beams10_samples4502.json")]
    elif mode == "scaffold":
        spec = [(f"seed{s}", f"apr18_ex1209_s{s}_frag_beams10/test_frag_beams10_samples4502.json")
                for s in (0, 42, 2006)]
    elif mode == "smiles":
        spec = [(f"seed{s}", f"apr18_ex1209_s{s}_smiles_beams10/test_smiles_beams10_samples4502.json")
                for s in (0, 42, 2006)]
    else:
        raise ValueError(f"unknown mode: {mode}")
    out = []
    for label, rel in spec:
        p = Path(corpus) / rel
        if not p.is_file():
            raise FileNotFoundError(f"{mode}/{label}: 找不到逐样本 dump {p}")
        out.append((label, p))
    return out


def canon(smiles):
    """把候选 SMILES 归一成 RDKit canonical 形式;解析失败返回 None。

    Args:
        smiles: 字符串;dump 里个别字段是 list,这里兼容取第 0 个元素。

    Returns:
        str | None。注意:None 永不等于任何参考串,故不会虚增 SA。
    """
    if isinstance(smiles, (list, tuple)):
        smiles = smiles[0] if smiles else None
    if not isinstance(smiles, str):
        return None
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


def eval_dump(path, top_ks=TOP_KS):
    """评测单个逐样本 dump。

    Args:
        path: dump 路径(4,502 条)。
        top_ks: 要统计的 k。

    Returns:
        dict: {"n": int, "n_pos": int,
               k -> {"tp","fp","fn","tn","precision","recall","f1",
                      "specificity","accuracy","exact","sa"}}。
    """
    data = json.load(open(path))
    if len(data) != N_POS + N_NEG:
        raise ValueError(f"{path}: 期望 {N_POS + N_NEG} 条,实际 {len(data)} 条")

    labels, refs = [], []
    for item in data:
        ref = item["canonical_reference_smiles"]
        labels.append(has_core(ref))
        refs.append(canon(ref))
    if sum(labels) != N_POS:
        raise ValueError(f"{path}: 正例 {sum(labels)} != {N_POS}")

    out = {"n": len(data), "n_pos": int(sum(labels))}
    for k in top_ks:
        tp = fp = exact = 0
        for item, lab, ref in zip(data, labels, refs):
            preds = item["predictions"][:k]
            hit = any(has_core(p.get("smiles")) for p in preds)
            if hit and lab:
                tp += 1
            elif hit and not lab:
                fp += 1
            ## SA 只在 64 个正例上统计(负例没有"参考结构正确"的概念)。
            if lab and ref is not None and any(canon(p.get("smiles")) == ref for p in preds):
                exact += 1
        fn = N_POS - tp
        tn = N_NEG - fp
        out[k] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                  **metrics_from_counts(tp, fp, fn, tn),
                  "exact": exact, "sa": exact / N_POS}
    return out


def aggregate(results):
    """把同模式的多个 seed 结果聚合成 mean / 样本标准差(ddof=1)。

    Args:
        results: list[dict],每个是 eval_dump 的返回值。

    Returns:
        dict: k -> 指标名 -> (mean, sd)。单 seed 时 sd 记为 0.0。
    """
    agg = {}
    for k in TOP_KS:
        keys = ("tp", "fp", "fn", "tn", "precision", "recall", "f1",
                "specificity", "accuracy", "exact", "sa")
        agg[k] = {}
        for key in keys:
            vals = np.array([r[k][key] for r in results], dtype=float)
            sd = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            agg[k][key] = (float(vals.mean()), sd)
    return agg


def fmt(pair, digits=4):
    """mean±sd 渲染;sd 为 0(单 seed 或三 seed 完全一致)时只显示均值。"""
    mean, sd = pair
    return f"{mean:.{digits}f}" if sd == 0 else f"{mean:.{digits}f}±{sd:.{digits}f}"


def fmt_count(pair):
    """计数类字段(TP/FP/FN/TN/exact)的渲染:整数不带小数,否则 2 位小数。"""
    mean, sd = pair
    if sd == 0:
        return f"{mean:.0f}"
    return f"{mean:.2f}±{sd:.2f}"


def run(corpus, out_md, out_csv):
    """跑三模式、打印/落盘 Markdown 与 CSV。"""
    modes = [
        ("CMGNet-0910 (base, SMILES)", "0910"),
        ("CMGNet-SCAFFOLD (frag)", "scaffold"),
        ("CMGNet-SMILES (smiles)", "smiles"),
    ]
    per_mode = []
    for name, mode in modes:
        files = mode_files(corpus, mode)
        results = [eval_dump(p) for _label, p in files]
        per_mode.append((name, mode, files, results, aggregate(results)))

    lines = []
    ap = lines.append
    ap("# CMGNet 三模式 top-k 指标 — 外部集 4,502(64 正 / 4,438 负)")
    ap("")
    ap("阈值/判定: top-k 内任一候选含全母核(4-喹诺酮-3-羧酸及其氮杂类似物)"
       "即判阳;P=TP/(TP+FP), R=TP/64, F1=2PR/(P+R), "
       "Spec=TN/4,438, Acc=(TP+TN)/4,502。SA=参考结构 canonical 完全匹配 / 64(正例)。")
    ap("")
    ap("多 seed 行写成 mean±sd(样本标准差 ddof=1,3 个种子)。")
    ap("")
    ap("| 模型 | Top-k | TP | FP | FN | TN | Precision | Recall | F1 | Specificity | Accuracy | SA (exact@k) |")
    ap("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, _mode, _files, _res, agg in per_mode:
        for k in TOP_KS:
            a = agg[k]
            ap(f"| {name} | {k} | {fmt_count(a['tp'])} | {fmt_count(a['fp'])} "
               f"| {fmt_count(a['fn'])} | {fmt_count(a['tn'])} "
               f"| {fmt(a['precision'])} | {fmt(a['recall'])} | {fmt(a['f1'])} "
               f"| {fmt(a['specificity'])} | {fmt(a['accuracy'])} | {fmt(a['sa'])} |")
    ap("")
    ap("## 逐 seed 明细(可审计)")
    ap("")
    ap("| 模型 | seed | Top-k | TP | FP | FN | TN | Exact |")
    ap("|---|---|---|---|---|---|---|---|")
    for name, _mode, files, results, _agg in per_mode:
        for (label, _p), res in zip(files, results):
            for k in TOP_KS:
                m = res[k]
                ap(f"| {name} | {label} | {k} | {m['tp']} | {m['fp']} | {m['fn']} "
                   f"| {m['tn']} | {m['exact']}/64 |")
    ap("")
    ap("## 数据来源(逐样本 dump,非表格)")
    ap("")
    ap("| 模型 | seed | 文件 |")
    ap("|---|---|---|")
    for name, _mode, files, _res, _agg in per_mode:
        for label, p in files:
            ap(f"| {name} | {label} | `{p}` |")
    ap("")
    ap("复算命令: `PYTHONPATH=. python analysis/metrics/topk_cmgnet_modes_4502.py`")
    md = "\n".join(lines) + "\n"
    Path(out_md).write_text(md, encoding="utf-8")

    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "top_k", "tp", "fp", "fn", "tn", "precision",
                    "recall", "f1", "specificity", "accuracy", "exact", "sa"])
        for name, _mode, _files, _res, agg in per_mode:
            for k in TOP_KS:
                a = agg[k]
                w.writerow([name, k,
                            fmt_count(a["tp"]), fmt_count(a["fp"]),
                            fmt_count(a["fn"]), fmt_count(a["tn"])]
                           + [fmt(a[c]) for c in ("precision", "recall", "f1",
                                                  "specificity", "accuracy")]
                           + [fmt_count(a["exact"]), fmt(a["sa"])])
    return md


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus-root", default=str(DEFAULT_CORPUS))
    ap.add_argument("--out", default=str(ROOT / "results/table_cmgnet_modes_4502.md"))
    ap.add_argument("--csv", default=str(ROOT / "results/table_cmgnet_modes_4502.csv"))
    args = ap.parse_args()
    md = run(args.corpus_root, args.out, args.csv)
    print(md)
    print(f"written: {args.out}\nwritten: {args.csv}")


if __name__ == "__main__":
    main()

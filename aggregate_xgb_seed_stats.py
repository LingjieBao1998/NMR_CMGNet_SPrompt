#!/usr/bin/env python3
"""聚合 xgb_cress_* 各数据配置下 3 个 seed 的评测指标，输出 mean±std 统计表."""
import json
import math
from pathlib import Path

ROOT = Path("/mnt/data2/NMR_Core_Mol_Detection/output")

# 各配置目录；ex_train 的 seed42 落在单独目录，合并进来
CONFIGS = {
    "exp_lit(纯实验语料66正)": ["xgb_cress_exp_lit"],
    "new_dataset(全量+extra7万)": ["xgb_cress_new_dataset"],
    "20261019(全量无extra)": ["xgb_cress_new_dataset_20261019"],
    "20261020_evalfiltered": ["xgb_cress_new_dataset_20261020_evalfiltered"],
    "ex_train(+extra9.4万)": ["xgb_cress_new_dataset_ex_train",
                              "xgb_cress_new_dataset_ex_train_seed42"],
    "no_pubchem(全量无PubChem正)": ["xgb_cress_new_dataset_no_pubchem"],
    "nmrmind(NMRMind训练集)": ["xgb_cress_nmrmind"],
}


def load_merged(dirs):
    """合并多个输出目录的 summary.json（用于 ex_train 拆两处的情况）."""
    merged = {}
    for d in dirs:
        s = json.load(open(ROOT / d / "summary.json"))
        if not merged:
            merged = s
        else:
            merged.setdefault("seeds", {}).update(s.get("seeds", {}))
            # data/config 取第一个目录即可（两目录同配置）
    return merged


def g(m, *path, default=None):
    cur = m
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def first_split(m, base):
    """test split 键名：多数配置为 test@*，nmrmind 为 nmrmind_test@*."""
    for cand in (f"{base}@0.5", f"nmrmind_{base}@0.5"):
        if cand in m:
            return m[cand]
    return {}


def fnum(v, digits=4):
    return "—" if v is None else f"{v:.{digits}f}"


def stats(vals):
    vals = [v for v in vals if v is not None]
    n = len(vals)
    mean = sum(vals) / n
    if n < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)  # 样本方差 ddof=1
    return mean, math.sqrt(var)


def fmt(vals, digits=4):
    m, s = stats(vals)
    return f"{m:.{digits}f}±{s:.{digits}f}"


def main():
    rows = {}  # config -> {seed: {...}}
    for name, dirs in CONFIGS.items():
        s = load_merged(dirs)
        seeds = s["seeds"]
        rows[name] = {
            "train_n": s["data"].get("train", s["data"].get("train_records")),
            "train_pos": s["data"].get("train_pos"),
            "spw": s["config"].get("scale_pos_weight"),
            "seeds_present": sorted(seeds, key=int),
            "per_seed": {},
            "ensemble": s.get("ensemble"),
        }
        for sd, m in seeds.items():
            t05 = first_split(m, "test")
            tvth = m.get("nmrmind_test@val_th", m.get("test@val_th", {}))
            e05 = m.get("external@0.5", {})
            evth = m.get("external@val_th", {})
            rows[name]["per_seed"][sd] = {
                "best_iter": m.get("best_iteration"),
                "val_auprc": m.get("val_auprc"),
                "test_auprc": t05.get("auprc"),
                "test_p@0.5": t05.get("precision"),
                "test_r@0.5": t05.get("recall"),
                "test_f1@0.5": t05.get("f1"),
                "test_p@th": tvth.get("precision"),
                "test_r@th": tvth.get("recall"),
                "test_f1@th": tvth.get("f1"),
                "test_f1@th": tvth.get("f1"),
                "ext_auprc": e05.get("auprc"),
                "ext_p@0.5": e05.get("precision"),
                "ext_r@0.5": e05.get("recall"),
                "ext_f1@0.5": e05.get("f1"),
                "ext_f1@th": evth.get("f1"),
                "ext_p@th": evth.get("precision"),
                "ext_r@th": evth.get("recall"),
                "ext_topk_pr": e05.get("topK", {}).get("precision"),
            }

    lines = []
    ap = lines.append

    ap("# XGBoost 3-seed × 数据配置 聚合统计\n")
    ap("seed = 42 / 0 / 2006；± 为样本标准差 (ddof=1)；@th 指各自验证集 F1 最优阈值。")
    ap("nmrmind 配置的 test split 为 NMRMind 内部 test，其余为 new_db test。\n")

    # ---- 逐配置明细表 ----
    for name, r in rows.items():
        ap(f"## {name}")
        ap(f"- train={r['train_n']:,} (pos {r['train_pos']:,}) | scale_pos_weight≈{r['spw']:.0f} | seeds: {','.join(r['seeds_present'])}")
        ap("")
        ap("| seed | best_iter | val AUPRC | test AUPRC | test F1@0.5 | test F1@th | ext AUPRC | ext P@0.5 | ext R@0.5 | ext F1@0.5 | ext F1@th | ext topK R |")
        ap("|---|---|---|---|---|---|---|---|---|---|---|---|")
        order = [s for s in ["42", "0", "2006"] if s in r["per_seed"]]
        for sd in order:
            m = r["per_seed"][sd]
            ap(f"| {sd} | {m['best_iter']} | {fnum(m['val_auprc'])} | {fnum(m['test_auprc'])} | "
               f"{fnum(m['test_f1@0.5'])} | {fnum(m['test_f1@th'])} | {fnum(m['ext_auprc'])} | "
               f"{fnum(m['ext_p@0.5'])} | {fnum(m['ext_r@0.5'])} | {fnum(m['ext_f1@0.5'])} | "
               f"{fnum(m['ext_f1@th'])} | {fnum(m['ext_topk_pr'])} |")
        # ensemble 行
        ens = r["ensemble"]
        if ens:
            e05 = ens.get("external@0.5", {})
            t05 = first_split(ens, "test")
            evth = ens.get("external@val_th_mean", {})
            tvth = ens.get("test@val_th_mean", ens.get("nmrmind_test@val_th_mean", {}))
            ap(f"| ens | - | - | {fnum(t05.get('auprc'))} | {fnum(t05.get('f1'))} | "
               f"{fnum(tvth.get('f1'))} | {fnum(e05.get('auprc'))} | "
               f"{fnum(e05.get('precision'))} | {fnum(e05.get('recall'))} | "
               f"{fnum(e05.get('f1'))} | {fnum(evth.get('f1'))} | "
               f"{fnum(e05.get('topK', {}).get('recall'))} |")
        # mean±std 行
        ap("| **mean±std** | | " + " | ".join([
            fmt([r["per_seed"][s][k] for s in order]) for k in
            ["val_auprc", "test_auprc", "test_f1@0.5", "test_f1@th",
             "ext_auprc", "ext_p@0.5", "ext_r@0.5", "ext_f1@0.5", "ext_f1@th", "ext_topk_pr"]
        ]) + " |")
        ap("")

    # ---- Precision / Recall / F1 专项 ----
    PRF = ["test_p@0.5", "test_r@0.5", "test_f1@0.5",
           "test_p@th", "test_r@th", "test_f1@th",
           "ext_p@0.5", "ext_r@0.5", "ext_f1@0.5",
           "ext_p@th", "ext_r@th", "ext_f1@th"]
    ap("## Precision / Recall / F1 专项\n")
    ap("@0.5 = 固定阈值 0.5；@th = 各自验证集 F1 最优阈值。\n")
    for name, r in rows.items():
        ap(f"### {name}\n")
        ap("| seed | test P@0.5 | test R@0.5 | test F1@0.5 | test P@th | test R@th | test F1@th "
           "| ext P@0.5 | ext R@0.5 | ext F1@0.5 | ext P@th | ext R@th | ext F1@th |")
        ap("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        order = [s for s in ["42", "0", "2006"] if s in r["per_seed"]]
        for sd in order:
            m = r["per_seed"][sd]
            ap(f"| {sd} | " + " | ".join(fnum(m[k]) for k in PRF) + " |")
        ap("| **mean±std** | " + " | ".join(
            fmt([r["per_seed"][s][k] for s in order]) for k in PRF) + " |")
        ap("")

    # ---- 汇总对比表（核心指标 mean±std）----
    ap("## 跨配置汇总（seed 间 mean±std）\n")
    ap("| 配置 | n_seed | test P@0.5 | test R@0.5 | test F1@0.5 | ext P@0.5 | ext R@0.5 | ext F1@0.5 |")
    ap("|---|---|---|---|---|---|---|---|")
    for name, r in rows.items():
        order = [s for s in ["42", "0", "2006"] if s in r["per_seed"]]
        cells = [
            fmt([r["per_seed"][s]["test_p@0.5"] for s in order]),
            fmt([r["per_seed"][s]["test_r@0.5"] for s in order]),
            fmt([r["per_seed"][s]["test_f1@0.5"] for s in order]),
            fmt([r["per_seed"][s]["ext_p@0.5"] for s in order]),
            fmt([r["per_seed"][s]["ext_r@0.5"] for s in order]),
            fmt([r["per_seed"][s]["ext_f1@0.5"] for s in order]),
        ]
        ap(f"| {name} | {len(order)} | " + " | ".join(cells) + " |")
    ap("")

    out = Path("/mnt/data2/NMR_Core_Mol_Detection/output/xgb_seed_aggregate_stats.md")
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()

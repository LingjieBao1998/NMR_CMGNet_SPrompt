#!/usr/bin/env python3
"""外部集 4,502 上各指标 95% 置信区间(分层 bootstrap + Wilson 对照)。

为什么用 bootstrap 而不是正态近似:
    - Specificity ≈ 0.999、Accuracy ≈ 0.998 这类比例贴近 1,Wald(normal) 区间
      会越界(>1)且明显偏窄;
    - F1 / Precision 的分母(TP+FP)本身是随机量,不是固定 n 的二项比例,
      没有闭式方差,只能重采样。
故统一用**按类分层**的非参数 bootstrap(正例 64、负例 4,438 各自有放回重抽,
保持原样本量),取 2.5%/97.5% 分位数;另外对二项比例给出 Wilson 区间做交叉核对。

多 seed 模型(SCAFFOLD / SMILES / SPrompt)的口径:
    每个 bootstrap 重抽样本上先把**每个 seed 的指标各算一遍**,再对 seed 取平均,
    与表里 "mean±sd over 3 seeds" 的估计量一致;区间因此描述"3 次运行的平均表现"
    的测试集抽样不确定性,seed 之间的散布仍由 ±s.d. 单独报告。

用法(仓库根目录,RDKit + sklearn):
    PYTHONPATH=. python analysis/metrics/ci_bootstrap_4502.py
    PYTHONPATH=. python analysis/metrics/ci_bootstrap_4502.py --n-boot 20000

作者:ZCode(Claude),2026-11-22
"""
import argparse
import json
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from sklearn.metrics import average_precision_score, roc_auc_score

from analysis.metrics.topk_external_4502 import N_NEG, N_POS, TOP_KS, metrics_from_counts

RDLogger.DisableLog('rdApp.*')

ROOT = Path(__file__).resolve().parents[2]
CORPUS = Path('/mnt/data2/NMR_Core_Mol_Detection/output')

## ZCode 2026-11-22：与 topk_cmgnet_modes_4502.py 同一母核模板与判定。
CORE = [Chem.MolFromSmiles(s) for s in
        ('O=C1C(C(O)=O)=CNC2=CC=CC=C12', 'O=C1C(C(O)=O)=CNC2=NC=CC=C12')]


def has_core(smiles):
    """SMILES 是否含喹诺酮母核(解析失败记 False)。"""
    if not isinstance(smiles, str):
        return False
    mol = Chem.MolFromSmiles(smiles)
    return bool(mol) and any(mol.HasSubstructMatch(c) for c in CORE)


def canon(smiles):
    """RDKit canonical SMILES;解析失败返回 None。"""
    if isinstance(smiles, (list, tuple)):
        smiles = smiles[0] if smiles else None
    if not isinstance(smiles, str):
        return None
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


def load_dump(path, top_ks=TOP_KS):
    """读逐样本 dump,返回按类分好的命中原语。

    Args:
        path: 4,502 条的预测 dump。
        top_ks: 需要统计的 k。

    Returns:
        dict: {"pos_hit": {k: bool[64]}, "neg_hit": {k: bool[4438]},
               "pos_exact": {k: bool[64]}}
        按 dump 中正例/负例的原始出现顺序排列(与标签一一对应)。
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

    out = {"pos_hit": {}, "neg_hit": {}, "pos_exact": {}}
    for k in top_ks:
        ph, nh, pe = [], [], []
        for item, lab, ref in zip(data, labels, refs):
            preds = item["predictions"][:k]
            hit = any(has_core(p.get("smiles")) for p in preds)
            if lab:
                ph.append(hit)
                pe.append(ref is not None and
                          any(canon(p.get("smiles")) == ref for p in preds))
            else:
                nh.append(hit)
        out["pos_hit"][k] = np.array(ph, dtype=bool)
        out["neg_hit"][k] = np.array(nh, dtype=bool)
        out["pos_exact"][k] = np.array(pe, dtype=bool)
    return out


def wilson(k, n, z=1.959963984540054):
    """二项比例的 Wilson score 95% 区间。"""
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def boot_metrics(runs, k, rng, n_boot):
    """分层 bootstrap:每次重抽正/负例,逐 seed 算指标再对 seed 平均。

    Args:
        runs: list[dict],每个元素是 load_dump 的返回值(= 一个 seed)。
        k: top-k。
        rng: np.random.Generator。
        n_boot: 重抽次数。

    Returns:
        dict[str, np.ndarray]: 指标名 -> 长度 n_boot 的数组;
        以及 "sa" 只在给了 pos_exact 时返回。
    """
    keys = ("precision", "recall", "f1", "specificity", "accuracy")
    acc = {key: np.empty(n_boot) for key in keys}
    sa = np.empty(n_boot)
    for b in range(n_boot):
        # 等价于有放回抽 64 个正例 / 4,438 个负例:用多项分布取各样本被抽中次数。
        w_pos = rng.multinomial(N_POS, np.full(N_POS, 1.0 / N_POS))
        w_neg = rng.multinomial(N_NEG, np.full(N_NEG, 1.0 / N_NEG))
        vals = {key: 0.0 for key in keys}
        sa_val = 0.0
        for run in runs:
            tp = int(w_pos @ run["pos_hit"][k])
            fp = int(w_neg @ run["neg_hit"][k])
            fn, tn = N_POS - tp, N_NEG - fp
            m = metrics_from_counts(tp, fp, fn, tn)
            for key in keys:
                vals[key] += m[key]
            sa_val += float(w_pos @ run["pos_exact"][k]) / N_POS
        n = len(runs)
        for key in keys:
            acc[key][b] = vals[key] / n
        sa[b] = sa_val / n
    acc["sa"] = sa
    return acc


def point_metrics(runs, k):
    """点估计:逐 seed 算指标后对 seed 平均(与表里的 mean 一致)。"""
    keys = ("precision", "recall", "f1", "specificity", "accuracy", "sa")
    vals = {key: [] for key in keys}
    for run in runs:
        tp = int(run["pos_hit"][k].sum())
        fp = int(run["neg_hit"][k].sum())
        m = metrics_from_counts(tp, fp, N_POS - tp, N_NEG - fp)
        for key in ("precision", "recall", "f1", "specificity", "accuracy"):
            vals[key].append(m[key])
        vals["sa"].append(float(run["pos_exact"][k].mean()))
    return {key: float(np.mean(v)) for key, v in vals.items()}


def ci(boot, key):
    """取 2.5/97.5 分位数。"""
    lo, hi = np.percentile(boot[key], [2.5, 97.5])
    return float(lo), float(hi)


def boot_auc(y, probs, rng, n_boot):
    """AUPRC/AUROC 的 bootstrap(整集重抽,不分层)。"""
    n = len(y)
    auprc = np.empty(n_boot)
    auroc = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if yy.min() == yy.max():
            auprc[b] = auroc[b] = np.nan
            continue
        auprc[b] = average_precision_score(yy == 1, probs[idx])
        auroc[b] = roc_auc_score(yy == 1, probs[idx])
    return auprc, auroc


def run(n_boot, seed, out_md):
    rng = np.random.default_rng(seed)

    modes = [
        ("CMGNet-0910 (k=1..10)", ["apr18_base0910_smiles_beams10/"
                                   "test_smiles_beams10_samples4502.json"]),
        ("CMGNet-SCAFFOLD (k=1..10)", [f"apr18_ex1209_s{s}_frag_beams10/"
                                       "test_frag_beams10_samples4502.json"
                                       for s in (0, 42, 2006)]),
        ("CMGNet-SMILES (k=1..10)", [f"apr18_ex1209_s{s}_smiles_beams10/"
                                     "test_smiles_beams10_samples4502.json"
                                     for s in (0, 42, 2006)]),
        ("CMGNet-SPrompt (k=1..10)", ["apr18_sprompt_ex1209_s0/"
                                      "test_nmr+frag_beams10_samples4502.json",
                                      "apr18_sprompt_ex1209_beams10/"
                                      "test_nmr+frag_beams10_samples4502.json",
                                      "apr18_sprompt_ex1209_s2006/"
                                      "test_nmr+frag_beams10_samples4502.json"]),
    ]

    lines = []
    ap = lines.append
    ap("# 外部集 4,502 指标 95% CI(分层 bootstrap)")
    ap("")
    ap(f"重抽次数 B = {n_boot:,},随机种子 = {seed},按类分层"
       f"(正例 {N_POS}、负例 {N_NEG} 各自有放回),取 2.5/97.5 分位数;"
       "多 seed 模型在每个重抽样本上逐 seed 算指标后取平均。")
    ap("")
    ap("| 模型 | Top-k | Precision [95% CI] | Recall [95% CI] | F1 [95% CI] "
       "| Specificity [95% CI] | Accuracy [95% CI] | SA [95% CI] |")
    ap("|---|---|---|---|---|---|---|---|")

    for name, rels in modes:
        runs = [load_dump(CORPUS / rel) for rel in rels]
        for k in TOP_KS:
            p = point_metrics(runs, k)
            boot = boot_metrics(runs, k, rng, n_boot)
            cells = []
            for key in ("precision", "recall", "f1", "specificity", "accuracy", "sa"):
                lo, hi = ci(boot, key)
                cells.append(f"{p[key]:.4f} [{lo:.4f}, {hi:.4f}]")
            ap(f"| {name} | {k} | " + " | ".join(cells) + " |")

    # Wilson 对照(单 seed 的 0910 + 分类器)
    ap("")
    ap("## Wilson 区间对照(二项比例,单次运行)")
    ap("")
    ap("| 指标 | 命中/总数 | Wilson 95% CI |")
    ap("|---|---|---|")
    base = load_dump(CORPUS / modes[0][1][0])
    for k in (1, 10):
        tp = int(base["pos_hit"][k].sum())
        fp = int(base["neg_hit"][k].sum())
        lo, hi = wilson(tp, N_POS)
        ap(f"| 0910 Recall@k={k} | {tp}/{N_POS} | [{lo:.4f}, {hi:.4f}] |")
        lo, hi = wilson(N_NEG - fp, N_NEG)
        ap(f"| 0910 Specificity@k={k} | {N_NEG - fp}/{N_NEG} | [{lo:.4f}, {hi:.4f}] |")

    # XGBoost
    y = np.load(ROOT / "reports/calib_data/external_apr18peak.npz",
                allow_pickle=True)["y"]
    probs = np.load(ROOT / "reports/calib_data/seed_probs.npz",
                    allow_pickle=True)["external_apr18peak_p"]
    ap("")
    ap("## XGBoost-CReSS(τ=0.50,k 无关)")
    ap("")
    ap("| 指标 | 点估计 | 95% CI |")
    ap("|---|---|---|")
    pr_l, pr_h = wilson(63, 67)
    ap(f"| Precision | 0.9403 | [{pr_l:.4f}, {pr_h:.4f}] (Wilson) |")
    lo, hi = wilson(63, N_POS)
    ap(f"| Recall | 0.9844 | [{lo:.4f}, {hi:.4f}] (Wilson) |")
    lo, hi = wilson(4434, N_NEG)
    ap(f"| Specificity | 0.9991 | [{lo:.4f}, {hi:.4f}] (Wilson) |")
    lo, hi = wilson(4497, N_POS + N_NEG)
    ap(f"| Accuracy | 0.9989 | [{lo:.4f}, {hi:.4f}] (Wilson) |")

    auprc_b, auroc_b = boot_auc(y, probs.mean(axis=0), rng, min(n_boot, 2000))
    ap(f"| AUPRC (seed 平均概率) | {average_precision_score(y == 1, probs.mean(0)):.4f} "
       f"| [{np.nanpercentile(auprc_b, 2.5):.4f}, {np.nanpercentile(auprc_b, 97.5):.4f}] |")
    ap(f"| AUROC (seed 平均概率) | {roc_auc_score(y == 1, probs.mean(0)):.6f} "
       f"| [{np.nanpercentile(auroc_b, 2.5):.6f}, {np.nanpercentile(auroc_b, 97.5):.6f}] |")
    ap("")
    ap("注:AUPRC/AUROC 用整集重抽 2,000 次(不分层);F1 无闭式方差,"
       "故与 P/R/Spec/Acc 一并采用 bootstrap 分位数区间。")
    md = "\n".join(lines) + "\n"
    Path(out_md).write_text(md, encoding="utf-8")
    return md


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261122)
    ap.add_argument("--out", default=str(ROOT / "results/ci_bootstrap_4502.md"))
    args = ap.parse_args()
    md = run(args.n_boot, args.seed, args.out)
    print(md)
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()

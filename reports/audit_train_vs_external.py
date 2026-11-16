"""审计:论文实际训练源 vs 外部测试集 64 个阳性化合物的重合情况.

双口径匹配:
  1. raw  : 原始 SMILES 字符串精确相等(复刻训练脚本黑名单的行为)
  2. canon: RDKit CanonicalSmiles 规范化后相等

训练源(与 train_xgb_cress_new_dataset_seed42.py 一致):
  A. data/new_dataset/train.lmdb          (3,196,326)
  B. data/new_data_1209.filtered.lmdb     (77,328)
  C. ex_train_add_label_filtered.evalclean.json (24,520)

输出: reports/train_vs_external_audit.json
"""
import json
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import lmdb
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
ROOT = Path("/mnt/data2/NMR_Core_Mol_Detection")
OUT = ROOT / "reports" / "train_vs_external_audit.json"

EXT_JSON = ROOT / "data/nmrmind_cmgnet_1004/external_test.json"
TRAIN_LMDB = ROOT / "data/new_dataset/train.lmdb"
EXTRA_LMDB = ROOT / "data/new_data_1209.filtered.lmdb"
EXTRA_JSON = ROOT / "data/extra_training/ex_train_add_label_filtered.evalclean.json"

_ext_raw = []
_ext_lab = []
with open(EXT_JSON) as f:
    ext = json.load(f)
for rec in ext:
    s = rec.get("cano_smiles") or rec.get("canonical_smiles") or rec.get("smiles") or ""
    lab = rec.get("label")
    lab = lab[0] if isinstance(lab, list) else lab
    _ext_raw.append(s)
    _ext_lab.append(lab)

def canon(s):
    if not s:
        return ""
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m) if m else ""

# 只审计阳性(64 个);阴性面太大且审稿人关心的是正例泄漏
raw_pos = {s: i for i, (s, l) in enumerate(zip(_ext_raw, _ext_lab)) if l == 1}
can_pos = {canon(s): i for i, (s, l) in enumerate(zip(_ext_raw, _ext_lab)) if l == 1}
can_pos.pop("", None)
print(f"external positives: {len(raw_pos)} raw, {len(can_pos)} canonical", flush=True)


def scan_smiles_batch(smiles_batch, source):
    """batch = [(idx, smiles, extra...)] -> hit dicts."""
    hits = []
    for item in smiles_batch:
        idx, s = item[0], item[1]
        rest = item[2:]
        m_raw = raw_pos.get(s)
        m_can = can_pos.get(canon(s))
        if m_raw is not None or m_can is not None:
            hits.append({
                "source": source,
                "record_idx": idx,
                "record_smiles": s,
                "match": "raw" if m_raw is not None else "canonical",
                "ext_index": m_raw if m_raw is not None else m_can,
                "extra": rest,
            })
    return hits


def load_and_scan_lmdb(path, source, limit=None):
    env = lmdb.open(str(path), readonly=True, lock=False, subdir=True)
    t0 = time.time()
    all_hits, n = [], 0
    with env.begin() as txn:
        cur = txn.cursor()
        batch = []
        for k, v in cur:
            try:
                rec = pickle.loads(v)
            except Exception:
                continue
            s = rec.get("canonical_smiles") or rec.get("cano_smiles") or ""
            batch.append((k.decode(), s, rec.get("split"), rec.get("source"),
                          len(rec.get("C_nmr") or []), rec.get("label")))
            n += 1
            if len(batch) >= 200_000:
                all_hits.extend(scan_smiles_batch(batch, source))
                batch = []
                print(f"  {source}: {n:,} scanned {time.time()-t0:.0f}s", flush=True)
            if limit and n >= limit:
                break
        if batch:
            all_hits.extend(scan_smiles_batch(batch, source))
    print(f"{source}: done {n:,} records in {time.time()-t0:.0f}s, hits={len(all_hits)}", flush=True)
    return n, all_hits


def load_and_scan_json(path, source):
    t0 = time.time()
    with open(path) as f:
        data = json.load(f)
    batch = []
    for i, rec in enumerate(data):
        s = rec.get("cano_smiles") or rec.get("canonical_smiles") or rec.get("smiles") or ""
        lab = rec.get("label")
        lab = lab[0] if isinstance(lab, list) else lab
        batch.append((str(i), s, rec.get("source"), len(rec.get("C_nmr") or rec.get("NMR") or []), lab))
    hits = scan_smiles_batch(batch, source)
    print(f"{source}: done {len(data):,} records in {time.time()-t0:.0f}s, hits={len(hits)}", flush=True)
    return len(data), hits


def main():
    result = {"external_pos_raw": len(raw_pos), "external_pos_canonical": len(can_pos), "sources": {}}
    for path, source in [(TRAIN_LMDB, "new_dataset/train.lmdb"),
                         (EXTRA_LMDB, "new_data_1209.filtered.lmdb"),
                         (EXTRA_JSON, "ex_train.evalclean.json")]:
        if "lmdb" in str(path):
            n, hits = load_and_scan_lmdb(path, source)
        else:
            if not path.exists():
                # 找一下真实路径
                cands = list((ROOT / "data").glob("*ex_train*evalclean*.json")) + \
                        list((ROOT / "data").glob("*ex_train*.json"))
                print(f"  {path} 不存在, 候选: {cands}", flush=True)
                if not cands:
                    result["sources"][source] = {"error": "file not found"}
                    continue
                path = cands[0]
            n, hits = load_and_scan_json(path, source)
        result["sources"][source] = {
            "path": str(path), "records": n, "hits": hits,
            "n_hits": len(hits),
            "unique_ext_idx": sorted({h["ext_index"] for h in hits}),
            "hits_in_train_split": sorted({h["ext_index"] for h in hits
                                           if h["extra"] and h["extra"][0] == "train"}),
        }
    with open(OUT, "w") as f:
        json.dump(result, f, indent=1, ensure_ascii=False)
    print("saved ->", OUT, flush=True)


if __name__ == "__main__":
    main()

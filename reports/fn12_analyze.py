"""Retrieve the nearest training neighbours of FN #12 in the CReSS embedding
space (cosine similarity only; no fingerprint similarity).

Output: reports/fn12_neighbors200.json
"""
import json, time
import numpy as np
import torch
import lmdb, pickle

import os as _os
B = _os.environ.get('NMR_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
B = B.rstrip('/') + '/'
VT = B + 'viz_tmp/'
NV = 3_298_042
CUT1 = 3_196_194          # new_db valid rows
CUT2 = 3_273_522          # + 1209 valid rows
D = 768
K = 200


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


y = np.load(VT + 'train_labels.npy')
emb = np.memmap(VT + 'train_emb.npy', dtype=np.float16, mode='r', shape=(3_300_000, D))
te = np.load(B + 'reports/calib_data/apr18_cress_emb.npy').astype(np.float32)
q = torch.tensor(te[12] / np.linalg.norm(te[12])).cuda().half()

vals, inds = [], []
CH = 200_000
for k in range(0, NV, CH):
    b = np.asarray(emb[k:min(k + CH, NV)], dtype=np.float32)
    bt = torch.nn.functional.normalize(torch.from_numpy(b).cuda(), dim=1).half()
    v, i = torch.topk(bt @ q, min(K, len(b)))
    vals.append(v.cpu()); inds.append(i.cpu() + k)
V = torch.cat(vals); I = torch.cat(inds)
v, i = torch.topk(V, K)
idx = I[i].numpy(); sims = v.numpy()
log(f"top{K} sims {sims.min():.3f}-{sims.max():.3f}; "
    f"pos {int(y[idx].sum())} neg {int((1 - y[idx]).sum())}")

targets = {int(t) for t in idx}
found = {}


def keep(i, rec, s):
    if i in targets:
        nmr = rec.get('C_nmr') if rec else None
        found[i] = dict(idx=i, label=int(y[i]), smiles=s,
                        nmr=(None if nmr is None else [float(x) for x in nmr]))


# --- new_db ---
env = lmdb.open(B + 'data/new_dataset/train.lmdb', readonly=True, lock=False, subdir=True)
n = 0
with env.begin() as txn:
    for _k, val in txn.cursor():
        o = pickle.loads(val)
        s = o.get('canonical_smiles'); nmr = o.get('C_nmr')
        if not s or nmr is None or len(nmr) == 0:
            continue
        keep(n, o, s); n += 1
log(f"new_db scanned valid {n}")

# --- 1209 ---
env = lmdb.open(B + 'data/new_data_1209.filtered.lmdb', readonly=True, lock=False, subdir=True)
n2 = 0
with env.begin() as txn:
    for _k, val in txn.cursor():
        o = pickle.loads(val)
        s = o.get('cano_smiles', o.get('canonical_smiles')); nmr = o.get('C_nmr')
        if not s or nmr is None or len(nmr) == 0:
            continue
        keep(CUT1 + n2, o, s); n2 += 1
log(f"1209 scanned valid {n2}")

# --- ex_train ---
ex = json.load(open(B + 'data/extra_training/ex_train_add_label_filtered.evalclean.json'))
for j, o in enumerate(ex):
    keep(CUT2 + j, o, o.get('cano_smiles'))

rows = []
for t, s in zip(idx, sims):
    r = found.get(int(t), {})
    rows.append(dict(idx=int(t), cosine=float(s), label=int(y[t]),
                     smiles=r.get('smiles'), nmr=r.get('nmr')))
json.dump(rows, open(B + 'reports/fn12_neighbors200.json', 'w'),
          ensure_ascii=False, indent=1)

log("top 25 by cosine:")
for r in rows[:25]:
    log(f"  idx={r['idx']:>8} cos={r['cosine']:.3f} label={r['label']} "
        f"{str(r['smiles'])[:85]}")

for lab in (1, 0):
    sub = [r for r in rows if r['label'] == lab][:10]
    print(f"\n=== nearest {'POSITIVE' if lab else 'NEGATIVE'} (top 10) ===")
    for r in sub:
        print(f"cos={r['cosine']:.3f} idx={r['idx']} {r['smiles']}")

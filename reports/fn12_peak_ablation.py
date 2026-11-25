"""Peak-level sensitivity analysis for the single FN (#12)."""
import json, os, sys
import numpy as np, torch
import os as _os
B = _os.environ.get('NMR_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
B = B.rstrip('/') + '/'
os.chdir(B); sys.path.insert(0, B)

from nmr_core_system.core_detection.infer import ModelInference
cfg = json.load(open(B + 'models/2_5_w_model/8.json'))
json.dump(cfg, open(B + 'viz_tmp/_cress_cfg_full.json', 'w'))
cress = ModelInference(B + 'viz_tmp/_cress_cfg_full.json', B + 'checkpoints/cress/2_5_w_model/8.pth', device='cpu')

import joblib
m = joblib.load(B + 'models/xgboost/model_xgb_cress_new_dataset_seed42.pkl')
ext = json.load(open(B + 'data/nmrmind_cmgnet_1004/external_test_april18peak.json'))
nmr = [float(x) for x in ext[12]['NMR']]

def enc(spectra):
    with torch.no_grad():
        return cress.nmr_encode([list(map(float, s)) for s in spectra]).cpu().numpy()

def pred(E):
    return m.predict_proba(np.asarray(E, dtype=np.float32))[:, 1]

E0 = enc([nmr])
p0 = pred(E0)[0]
print(f"FN #12 full spectrum ({len(nmr)} peaks): p={p0:.4f}")
print("NMR:", nmr)

# leave-one-peak-out
abl = []
for i in range(len(nmr)):
    s = nmr[:i] + nmr[i + 1:]
    p = pred(enc([s]))[0]
    abl.append((nmr[i], float(p), float(p - p0)))
print("\nLeave-one-peak-out (peak, p, delta):")
for pk, p, d in sorted(abl, key=lambda t: t[1]):
    print(f"  {pk:7.2f}  p={p:.4f}  dp={d:+.4f}")

# analog reference
an = [float(x) for x in [9.91, 37.16, 110.06, 115.0, 116.06, 126.1, 126.5, 127.6,
                         130.9, 132.0, 141.0, 150.6, 155.9, 162.8, 165.5, 175.5]]
Ea = enc([an])
pa = pred(Ea)[0]
cos = float((E0[0] @ Ea[0]) / (np.linalg.norm(E0[0]) * np.linalg.norm(Ea[0])))
print(f"\nclose positive analog 3263827: p={pa:.4f}, cos(#12, analog)={cos:.4f}")
json.dump(dict(full_p=float(p0), nmr=nmr, ablation=abl, analog_p=float(pa), cos=cos),
          open('/tmp/fn12_peak_ablation.json', 'w'), ensure_ascii=False, indent=1)

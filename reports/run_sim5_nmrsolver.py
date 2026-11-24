#!/usr/bin/env python3
"""提交 5 条模拟 13C 谱到 NMR-Solver (Bohrium NMR-Toolbox)，轮询结果并算指标。

输入：/mnt/data2/NMR_Core_Mol_Detection/data/sim5/fluoroquinolones_nmr.json
认证：/tmp/nmrsolver_srv/cookie.txt（不入库）

输出：
  /mnt/data2/NMR_Core_Mol_Detection/reports/sim5_nmrsolver_raw.json
  /mnt/data2/NMR_Core_Mol_Detection/reports/sim5_nmrsolver_summary.md

用法：
  /home/lingjiebao/venvs/py312_tf457/bin/python /tmp/nmrsolver_srv/run_sim5_nmrsolver.py

作者：ZCode（Claude），2026-11-23
"""
import json
import os
import time
import urllib.request

from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')

BASE = 'https://nmr-toolbox-uuid1753076625.appspace.bohrium.com'
COOKIE = open('/tmp/nmrsolver_srv/cookie.txt').read().strip()
SIM5 = '/mnt/data2/NMR_Core_Mol_Detection/data/sim5/fluoroquinolones_nmr.json'
OUT_RAW = '/mnt/data2/NMR_Core_Mol_Detection/reports/sim5_nmrsolver_raw.json'
OUT_MD = '/mnt/data2/NMR_Core_Mol_Detection/reports/sim5_nmrsolver_summary.md'

CORE_MOLS = [Chem.MolFromSmiles(s) for s in (
    'O=C1C(C(O)=O)=CNC2=CC=CC=C12',
    'O=C1C(C(O)=O)=CNC2=NC=CC=C12',
)]


def has_core(s):
    m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
    return bool(m) and any(m.HasSubstructMatch(c) for c in CORE_MOLS)


def canon(s):
    m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
    return Chem.MolToSmiles(m) if m else None


def post(path, payload):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={
            'Content-Type': 'application/json',
            'Accept': '*/*',
            'Cookie': COOKIE,
            'Origin': BASE,
            'Referer': BASE + '/task',
        },
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def submit_one(name, c_shifts):
    payload = {
        'name': name,
        'input_data': {
            'config': {
                'sigma_h': 1, 'sigma_c': 10, 'split_coef': 0.8, 'max_iter': 2,
                'num_search': 1000, 'num_pool': 1000, 'num_filter_pair': 200000,
                'num_filter_mol': 1000, 'num_mutate_mol': 1000, 'topk': 1000,
                'max_cycle_length': 6, 'use_H_split': False, 'use_stereo': False,
                'optional_halogens': ['F', 'Cl', 'Br', 'I'],
                'invalid_patterns': ['[O][O]', '[R]=[R]=[R]', '[r3,r4]=[r3,r4]', '[O][F,Cl,Br,I]'],
                'include_active_hs': 'yes',
            },
            'search': {
                'H_split': [], 'H_shifts': [], 'C_shifts': c_shifts,
                'num_search': 1000, 'topk': 1000,
                'allowed_elements': ['C', 'H', 'N', 'O', 'P', 'Si', 'B', 'F', 'Cl', 'S', 'Br', 'I'],
            },
        },
    }
    return post('/api/submit_task', payload)


def poll(task_id, timeout_s=600, interval=15):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        d = post('/api/get_task_result_by_task_id', {'task_id': task_id})
        if d.get('code') != 0:
            return d
        data = d.get('data') or {}
        status = data.get('status')
        if status == 'Finished':
            return d
        if status in ('Failed', 'Error'):
            return d
        time.sleep(interval)
    return {'code': -1, 'msg': 'timeout', 'data': None}


def main():
    sim5 = json.load(open(SIM5))
    records = []
    for i, rec in enumerate(sim5):
        name = f'sim5_{i}_{rec["key"]}'
        c_shifts = rec['C_nmr']
        print(f'[{i}] submit {name} n_peaks={len(c_shifts)}')
        sub = submit_one(name, c_shifts)
        print(' submit:', sub)
        task_id = sub.get('data') if sub.get('code') == 0 else None
        records.append({
            'index': i, 'name': name, 'task_id': task_id,
            'cano_smiles': rec['cano_smiles'], 'C_nmr': c_shifts,
            'submit_resp': sub,
        })

    # poll
    for r in records:
        tid = r['task_id']
        if not tid:
            r['result'] = None
            continue
        print(f'poll {tid}')
        res = poll(tid)
        r['result'] = res
        status = ((res.get('data') or {}).get('status'))
        print(' ->', status)

    # compute metrics
    summary_rows = []
    for r in records:
        ref = canon(r['cano_smiles'])
        preds = []
        data = (r.get('result') or {}).get('data') or {}
        result = data.get('result') or {}
        plist = result.get('data') or []
        if isinstance(plist, list):
            preds = [p.get('smiles') for p in plist if isinstance(p, dict)]
        r['top10_smiles'] = preds[:10]
        r['n_pred'] = len(preds)
        row = {'index': r['index'], 'ref': r['cano_smiles'], 'n_pred': len(preds)}
        for k in (1, 3, 5, 10):
            top = preds[:k]
            row[f'core@{k}'] = int(any(has_core(s) for s in top))
            row[f'exact@{k}'] = int(any(canon(s) == ref for s in top))
        summary_rows.append(row)

    os.makedirs(os.path.dirname(OUT_RAW), exist_ok=True)
    with open(OUT_RAW, 'w') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    # markdown
    lines = ['# NMR-Solver on 5 simulated 13C spectra', '',
             'Data: `data/sim5/fluoroquinolones_nmr.json` (5 quinolone simulated spectra).', '',
             '| idx | ref (canonical prefix) | n_pred | core@1 | core@10 | exact@1 | exact@10 |',
             '|---:|---|---:|---:|---:|---:|---:|']
    for row in summary_rows:
        ref_short = row['ref'][:45] + ('...' if len(row['ref']) > 45 else '')
        lines.append(f"| {row['index']} | `{ref_short}` | {row['n_pred']} | {row['core@1']} | {row['core@10']} | {row['exact@1']} | {row['exact@10']} |")
    n = len(summary_rows)
    for k in (1, 3, 5, 10):
        c = sum(r[f'core@{k}'] for r in summary_rows) / n
        e = sum(r[f'exact@{k}'] for r in summary_rows) / n
        lines.append(f'- k={k}: core={c:.3f}, exact={e:.3f}')
    with open(OUT_MD, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print('saved', OUT_RAW)
    print('saved', OUT_MD)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""逐样本对比复跑(4090/py312_tf457) vs 10-28 tf457 基线, 并重算全表"""
import json, sys
from collections import OrderedDict

OLD = {
    'frag':   'output/external_4502_base1029v2_tf457_frag_beams10/test_frag_beams10_samples4502.json',
    'smiles': 'output/external_4502_base1029v2_tf457_smiles_beams10/test_smiles_beams10_samples4502.json',
}
NEW = {
    'frag':   'output/repro1029_4090_b32_frag_beams10/test_frag_beams10_samples4502.json',
    'smiles': 'output/repro1029_4090_b32_smiles_beams10/test_smiles_beams10_samples4502.json',
}

def key_preds(item, k=10):
    return tuple((p['rank'], p['smiles'], round(p['score'], 6), p.get('has_core_structure')) for p in item['predictions'][:k])

labels = [1 if x['label'] == [1] else 0 for x in json.load(open('data/nmrmind_cmgnet_1004/external_test.json'))]
xgbv   = [v['xgb_vote'] for v in json.load(open('output/sprompt_fullpipeline_1029v2_20261029/per_sample_votes.json'))]

def cm(pred):
    tp=sum(1 for p,l in zip(pred,labels) if p and l); fp=sum(1 for p,l in zip(pred,labels) if p and not l)
    fn=sum(1 for p,l in zip(pred,labels) if not p and l); tn=sum(1 for p,l in zip(pred,labels) if not p and not l)
    P=tp/(tp+fp) if tp+fp else 0; R=tp/(tp+fn) if tp+fn else 0; F=2*P*R/(P+R) if P+R else 0
    return (tp,fp,fn,tn),P,R,F

hits = {'old':{}, 'new':{}}
for leg in ('frag','smiles'):
    for tag,path in (('old',OLD[leg]),('new',NEW[leg])):
        d=json.load(open(path)); assert len(d)==4502, (path,len(d))
        hits[tag][leg]=[any(p.get('has_core_structure') for p in it['predictions'][:10]) for it in d]

print('='*72)
for leg in ('frag','smiles'):
    same_ref = same_pred = 0
    old=json.load(open(OLD[leg])); new=json.load(open(NEW[leg]))
    for o,n in zip(old,new):
        if o['reference_smiles']==n['reference_smiles']: same_ref+=1
        if key_preds(o)==key_preds(n): same_pred+=1
    (cmo,Po,Ro,Fo)=cm(hits['old'][leg]); (cmn,Pn,Rn,Fn)=cm(hits['new'][leg])
    print(f'[{leg}] 逐样本候选序列完全一致: {same_pred}/4502  (ref一致 {same_ref}/4502)')
    print(f'  10-28基线: ({cmo[0]}, {cmo[1]}, {cmo[2]}, {cmo[3]}) P={Po:.4f} R={Ro:.4f} F1={Fo:.4f}')
    print(f'  本次复跑 : ({cmn[0]}, {cmn[1]}, {cmn[2]}, {cmn[3]}) P={Pn:.4f} R={Rn:.4f} F1={Fn:.4f}')
    diff=[i for i in range(4502) if hits['old'][leg][i]!=hits['new'][leg][i]]
    print(f'  腿投票翻转样本数: {len(diff)}', diff[:20] if diff else '')

print('='*72)
for tag in ('old','new'):
    ens=[sum(t)>=2 for t in zip(hits[tag]['frag'],hits[tag]['smiles'],xgbv)]
    uni=[a or b for a,b in zip(hits[tag]['frag'],hits[tag]['smiles'])]
    (c1,P,R,F)=cm(ens); (c2,_,_,_)=cm(uni)
    print(f'[{tag}] 2/3集成: ({c1[0]}, {c1[1]}, {c1[2]}, {c1[3]}) P={P:.4f} R={R:.4f} F1={F:.4f}   frag∪smiles: {c2}')
print('论文口径: (63, 5, 1, 4433) P=0.9265 R=0.9844 F1=0.9545')

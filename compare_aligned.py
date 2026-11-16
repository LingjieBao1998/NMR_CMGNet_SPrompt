#!/usr/bin/env python3
"""A(现役谱) vs B(4月谱) 同一4500分子集对比; 关键: 8个谱变更FN在B上是否被检出"""
import json
A=json.load(open('data/aligned4500_currentNMR.json')); B=json.load(open('data/aligned4500_aprilNMR.json'))
labels=[1 if r['label']==[1] else 0 for r in A]
FN8={'Cn1cc(C(=O)O)c(=O)c2cc(-c3ccccc3OC(F)(F)F)ccc21','O=C(O)c1cn(C2CC2)c2c(cc(F)c3c4ccccc4[nH]c32)c1=O','O=C(O)c1cn(C2CC2)c2c([N+](=O)[O-])c(-c3ccccc3)c(F)cc2c1=O','Nc1c(-c2cccs2)c(F)cc2c(=O)c(C(=O)O)cn(C3CC3)c12','O=C(O)c1cn(C2CC2)c2c([N+](=O)[O-])c(Cl)c(F)cc2c1=O','Nc1c(-c2ccco2)c(F)cc2c(=O)c(C(=O)O)cn(C3CC3)c12','Cc1ccc(-c2c(F)cc3c(=O)c(C(=O)O)cn(C4CC4)c3c2N)cc1','C[C@H]1COc2c(C3(N)CC3)c(F)cc3c(=O)c(C(=O)O)cn1c23','CCOC(=O)c1cn(C2CC2)c2c(OC)c(F)c(F)cc2c1=O'}
FN_changed={'O=C(O)c1cn(C2CC2)c2c(cc(F)c3c4ccccc4[nH]c32)c1=O','O=C(O)c1cn(C2CC2)c2c([N+](=O)[O-])c(-c3ccccc3)c(F)cc2c1=O','Nc1c(-c2cccs2)c(F)cc2c(=O)c(C(=O)O)cn(C3CC3)c12','O=C(O)c1cn(C2CC2)c2c([N+](=O)[O-])c(Cl)c(F)cc2c1=O','Nc1c(-c2ccco2)c(F)cc2c(=O)c(C(=O)O)cn(C3CC3)c12','Cc1ccc(-c2c(F)cc3c(=O)c(C(=O)O)cn(C4CC4)c3c2N)cc1','C[C@H]1COc2c(C3(N)CC3)c(F)cc3c(=O)c(C(=O)O)cn1c23','CCOC(=O)c1cn(C2CC2)c2c(OC)c(F)c(F)cc2c1=O'}
smi2k={r['cano_smiles']:k for k,r in enumerate(A)}

def cm(pred):
    tp=sum(1 for p,l in zip(pred,labels) if p and l); fp=sum(1 for p,l in zip(pred,labels) if p and not l)
    fn=sum(1 for p,l in zip(pred,labels) if not p and l); tn=sum(1 for p,l in zip(pred,labels) if not p and not l)
    P=tp/(tp+fp) if tp+fp else 0; R=tp/(tp+fn) if tp+fn else 0; F=2*P*R/(P+R) if P+R else 0
    return (tp,fp,fn,tn),P,R,F

hits={}
for tag,d in (('A',A),('B',B)):
    for leg in ('frag','smiles'):
        path=f'output/aligned{tag}_{"cur" if tag=="A" else "apr"}_{leg}_beams10/test_{leg}_beams10_samples4500.json'
        data=json.load(open(path)); assert len(data)==4500
        hits[tag,leg]=[any(p.get('has_core_structure') for p in it['predictions'][:10]) for it in data]

print("="*70)
for leg in ('frag','smiles'):
    (cA,PA,RA,FA)=cm(hits['A',leg]); (cB,PB,RB,FB)=cm(hits['B',leg])
    print(f'[{leg}] A现役谱: ({cA[0]}, {cA[1]}, {cA[2]}, {cA[3]}) P={PA:.4f} R={RA:.4f} F1={FA:.4f}')
    print(f'[{leg}] B四月谱: ({cB[0]}, {cB[1]}, {cB[2]}, {cB[3]}) P={PB:.4f} R={RB:.4f} F1={FB:.4f}')
    tp_gain=[A[k]['cano_smiles'] for k in range(4500) if labels[k] and not hits['A',leg][k] and hits['B',leg][k]]
    tp_loss=[A[k]['cano_smiles'] for k in range(4500) if labels[k] and hits['A',leg][k] and not hits['B',leg][k]]
    print(f'   B检出/A漏检: {len(tp_gain)}   A检出/B漏检: {len(tp_loss)}')
    in8=[s[:55] for s in tp_gain if s in FN_changed]
    print(f'   其中属于8个谱变更FN: {len(in8)}', in8)
    not8=[s for s in tp_gain if s in FN_changed-FN_changed or s in FN8-FN_changed]
    # detail for the 8
    print('   8个谱变更FN逐个(A→B):', [(('DET' if hits['A',leg][smi2k[s]] else 'miss'),('DET' if hits['B',leg][smi2k[s]] else 'miss')) for s in sorted(FN_changed)])
print("="*70)
for tag in ('A','B'):
    ens=[sum(t)>=2 for t in zip(hits[tag,'frag'],hits[tag,'smiles'],[False]*4500)]  # legs-only 2/3 w/o xgb = AND
    uni=[a or b for a,b in zip(hits[tag,'frag'],hits[tag,'smiles'])]
    (c1,_,_,_)=cm(uni)
    print(f'[{tag}] frag∪smiles: ({c1[0]}, {c1[1]}, {c1[2]}, {c1[3]})')

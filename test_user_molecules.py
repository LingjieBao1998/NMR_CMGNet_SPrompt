#!/usr/bin/env python
"""测试用户提供的分子样本"""

from rdkit import Chem
from new_RingSystemFinder import RingSystemFinder

def test_user_molecules():
    """测试用户提供的分子"""
    
    molecules = [
        "C1CCC(CC1)C(=O)O",  # 环己基羧酸
        "c1ccccc1C(=O)O",    # 苯甲酸
        "CC(=O)O.C1CCC(CC1)C(=O)O",  # 乙酸 + 环己基羧酸
        "CN1CCN(c2c(F)c(N)c3c(=O)c(C(=O)O)cn(C4CC4)c3c2F)C(CNS(=O)(=O)C2CC2)C1",
        "O=C(O)c1cn(C2CC2)c2cc(-n3ccnc3)c(F)cc2c1=O",
        "Cn1cc(C(=O)O)c(=O)c2cc(F)c3c(=O)c(C#N)cn(C4CC4)c3c21"
    ]
    
    finder = RingSystemFinder()
    
    for i, smi in enumerate(molecules, 1):
        print(f"测试分子 {i}: {smi}")
        mol = Chem.MolFromSmiles(smi)
        
        if mol is None:
            print("  ❌ 分子解析失败")
            continue
            
        try:
            ring_systems = finder.find_ring_systems(mol, as_mols=True)
            
            print(f"  发现 {len(ring_systems)} 个环系统:")
            for j, ring_mol in enumerate(ring_systems, 1):
                ring_smi = Chem.MolToSmiles(ring_mol)
                atom_count = ring_mol.GetNumAtoms()
                
                # 检查是否包含COOH
                cooh_pattern = Chem.MolFromSmarts("C(=O)[OX2H]")
                has_cohh = ring_mol.HasSubstructMatch(cooh_pattern)
                
                print(f"    系统 {j}: {ring_smi}")
                print(f"      原子数: {atom_count}, 包含COOH: {has_cohh}")
                
                # 如果是COOH，显示具体匹配
                if has_cohh:
                    cooh_matches = ring_mol.GetSubstructMatches(cooh_pattern)
                    print(f"      COOH位置: {cooh_matches}")
            
        except Exception as e:
            print(f"  ❌ 处理失败: {e}")
        
        print()

if __name__ == "__main__":
    test_user_molecules()
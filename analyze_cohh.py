#!/usr/bin/env python
"""详细分析COOH是否被保留"""

from rdkit import Chem
from new_RingSystemFinder import RingSystemFinder

def analyze_cohh_preservation():
    """分析COOH是否被保留"""
    
    test_molecules = [
        ("环己基羧酸", "C1CCC(CC1)C(=O)O"),
        ("苯甲酸", "c1ccccc1C(=O)O"),
        ("单独COOH", "C(=O)O"),
    ]
    
    finder = RingSystemFinder()
    
    for name, smi in test_molecules:
        print(f"\n=== {name} ===")
        print(f"原始SMILES: {smi}")
        
        mol = Chem.MolFromSmiles(smi)
        
        # 检查原始分子的COOH
        cooh_patterns = [
            ("标准COOH", "C(=O)[OX2H]"),
            ("羰基+氧", "C(=O)O"),
            ("酸酐模式", "O=C"),
        ]
        
        print("原始分子分析:")
        for pattern_name, pattern_smarts in cooh_patterns:
            pattern = Chem.MolFromSmarts(pattern_smarts)
            matches = mol.GetSubstructMatches(pattern)
            print(f"  {pattern_name}: {len(matches)} 个匹配")
        
        # 获取环系统
        ring_systems = finder.find_ring_systems(mol, as_mols=True)
        
        print(f"\n环系统分析 (找到 {len(ring_systems)} 个):")
        for i, ring_mol in enumerate(ring_systems, 1):
            ring_smi = Chem.MolToSmiles(ring_mol)
            print(f"  系统 {i}: {ring_smi}")
            
            print("    功能基团检测:")
            for pattern_name, pattern_smarts in cooh_patterns:
                pattern = Chem.MolFromSmarts(pattern_smarts)
                matches = ring_mol.GetSubstructMatches(pattern)
                print(f"      {pattern_name}: {len(matches)} 个匹配")
            
            # 详细原子分析
            print("    原子详情:")
            for atom in ring_mol.GetAtoms():
                symbol = atom.GetSymbol()
                degree = atom.GetDegree()
                charge = atom.GetFormalCharge()
                print(f"      原子 {atom.GetIdx()}: {symbol}, 度={degree}, 电荷={charge}")

if __name__ == "__main__":
    analyze_cohh_preservation()
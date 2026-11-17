#!/usr/bin/env python3
"""
调试预定义母核匹配问题
"""

from rdkit import Chem
from rdkit.Chem import AllChem

def debug_core_matching():
    """调试母核匹配问题"""
    print("=" * 100)
    print("调试预定义母核匹配")
    print("=" * 100)
    
    # 预定义的母核结构
    predefined_cores = [
        'O=c1cc[nH]c2ccccc12',      # 喹诺酮类母核结构1
        'c1ccc2c(=O)cc[nH]c2n1'     # 喹诺酮类母核结构2
    ]
    
    # 测试分子
    test_molecules = [
        {
            "name": "环丙沙星简单表示",
            "smiles": "c1cc2c(cc1)C(=O)N3C(=O)C2=C(C3=O)N"
        },
        {
            "name": "环丙沙星另一表示",
            "smiles": "C1CC(C(N1)C)C(=O)C2=C(C3=CC=CC=C3N2)C(=O)O"
        },
        {
            "name": "包含母核骨架的分子",
            "smiles": "O=C1C2=C(C3=CC=CC=C3N2)C(=O)N1"
        },
        {
            "name": "完全匹配母核1",
            "smiles": "O=c1cc[nH]c2ccccc12"
        },
        {
            "name": "完全匹配母核2", 
            "smiles": "c1ccc2c(=O)cc[nH]c2n1"
        }
    ]
    
    for core_idx, core_smiles in enumerate(predefined_cores, 1):
        print(f"\n{'='*60}")
        print(f"母核 {core_idx}: {core_smiles}")
        print(f"{'='*60}")
        
        core_mol = Chem.MolFromSmiles(core_smiles, sanitize=False)
        if core_mol is None:
            print(f"❌ 母核SMILES无效")
            continue
            
        print(f"✅ 母核有效")
        print(f"   原子数: {core_mol.GetNumAtoms()}")
        print(f"   SMILES标准形式: {Chem.MolToSmiles(core_mol)}")
        
        # 尝试不同的sanitize设置
        core_sanitized = Chem.MolFromSmiles(core_smiles, sanitize=True)
        if core_sanitized:
            print(f"   标准化SMILES: {Chem.MolToSmiles(core_sanitized)}")
        
        for mol_test in test_molecules:
            print(f"\n   测试: {mol_test['name']}")
            print(f"      SMILES: {mol_test['smiles']}")
            
            # 尝试不同的参数组合
            test_variants = [
                ("sanitize=False", Chem.MolFromSmiles(mol_test['smiles'], sanitize=False)),
                ("sanitize=True", Chem.MolFromSmiles(mol_test['smiles'], sanitize=True)),
            ]
            
            found_match = False
            for variant_name, test_mol in test_variants:
                if test_mol is None:
                    continue
                    
                # 子结构匹配
                if test_mol.HasSubstructMatch(core_mol):
                    print(f"      ✅ 匹配成功 ({variant_name})!")
                    matches = test_mol.GetSubstructMatch(core_mol)
                    print(f"         匹配的原子索引: {matches}")
                    found_match = True
                    break
                else:
                    # 尝试反转匹配方向
                    if core_mol.HasSubstructMatch(test_mol):
                        print(f"      ⚠️  反向匹配 ({variant_name})")
                        found_match = True
                        break
            
            if not found_match:
                print(f"      ❌ 未匹配")
                
                # 分析原因
                test_mol = Chem.MolFromSmiles(mol_test['smiles'], sanitize=False)
                if test_mol:
                    print(f"         测试分子原子数: {test_mol.GetNumAtoms()}")
                    print(f"         测试分子SMILES: {Chem.MolToSmiles(test_mol)}")
                    
                    # 尝试MCS匹配
                    try:
                        from rdkit.Chem import rdFMCS
                        mcs = rdFMCS.FindMCS([core_mol, test_mol])
                        print(f"         MCS原子数: {mcs.numAtoms}")
                        if mcs.numAtoms > 0:
                            print(f"         MCS SMILES: {mcs.smartsString}")
                    except Exception as e:
                        print(f"         MCS计算失败: {e}")

if __name__ == "__main__":
    debug_core_matching()
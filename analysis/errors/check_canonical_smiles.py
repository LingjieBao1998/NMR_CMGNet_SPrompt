#!/usr/bin/env python3
"""
检查两个预定义母核的标准化SMILES是否相同
"""

from rdkit import Chem

def check_canonical_smiles():
    """检查母核标准化SMILES"""
    print("=" * 100)
    print("检查预定义母核的标准化SMILES")
    print("=" * 100)
    
    # 原始的两个母核SMILES
    original_cores = [
        'O=c1cc[nH]c2ccccc12',      # 母核结构1
        'c1ccc2c(=O)cc[nH]c2n1'     # 母核结构2
    ]
    
    print(f"\n🎯 原始SMILES:")
    for i, smiles in enumerate(original_cores, 1):
        print(f"   {i}. {smiles}")
    
    print(f"\n🔬 标准化分析:")
    
    canonical_forms = []
    
    for i, smiles in enumerate(original_cores, 1):
        print(f"\n   母核 {i}: {smiles}")
        
        # 尝试不同的sanitize设置
        variants = [
            ("sanitize=False", Chem.MolFromSmiles(smiles, sanitize=False)),
            ("sanitize=True", Chem.MolFromSmiles(smiles, sanitize=True)),
        ]
        
        for variant_name, mol in variants:
            if mol is None:
                print(f"      ❌ {variant_name}: 无效SMILES")
                continue
            
            # 标准化SMILES
            canonical = Chem.MolToSmiles(mol)
            canonical_forms.append(canonical)
            
            print(f"      ✅ {variant_name}:")
            print(f"         标准化: {canonical}")
            print(f"         原子数: {mol.GetNumAtoms()}")
            
            # 分析环系统
            try:
                rings = Chem.GetSymmSSSR(mol)
                print(f"         环数: {len(rings)}")
            except:
                print(f"         环数: 无法计算")
            
            # 分析原子类型
            atom_types = []
            for atom in mol.GetAtoms():
                atom_types.append(f"{atom.GetSymbol()}{atom.GetFormalCharge() if atom.GetFormalCharge() != 0 else ''}")
            print(f"         原子类型: {', '.join(set(atom_types))}")
    
    print(f"\n🔍 比较分析:")
    
    # 检查所有标准化形式是否相同
    unique_canonical = set(canonical_forms)
    
    if len(unique_canonical) == 1:
        print(f"   ✅ 所有标准化SMILES相同:")
        print(f"      共同形式: {list(unique_canonical)[0]}")
    else:
        print(f"   ❌ 标准化SMILES不同:")
        for i, canonical in enumerate(canonical_forms, 1):
            original_smiles = original_cores[(i-1) % len(original_cores)]
            sanitize_method = "sanitize=False" if i % 2 == 1 else "sanitize=True"
            print(f"      {original_smiles} ({sanitize_method}):")
            print(f"         → {canonical}")
    
    # 使用MCS检查相似性
    print(f"\n🧬 最大公共子结构 (MCS) 分析:")
    
    mol1 = Chem.MolFromSmiles(original_cores[0], sanitize=False)
    mol2 = Chem.MolFromSmiles(original_cores[1], sanitize=False)
    
    if mol1 and mol2:
        try:
            from rdkit.Chem import rdFMCS
            mcs = rdFMCS.FindMCS([mol1, mol2])
            
            print(f"   MCS原子数: {mcs.numAtoms}")
            print(f"   MCS键数: {mcs.numBonds}")
            print(f"   MCS SMARTS: {mcs.smartsString}")
            
            # 检查是否完全匹配
            if mcs.numAtoms == mol1.GetNumAtoms() == mol2.GetNumAtoms():
                print(f"   ✅ MCS覆盖所有原子，两个结构完全相同")
            else:
                print(f"   ⚠️  MCS未完全覆盖，结构有差异")
                
                # 分析差异
                print(f"      母核1原子数: {mol1.GetNumAtoms()}")
                print(f"      母核2原子数: {mol2.GetNumAtoms()}")
                print(f"      相同原子数: {mcs.numAtoms}")
                
        except Exception as e:
            print(f"   ❌ MCS计算失败: {e}")
    
    # 测试子结构匹配
    print(f"\n🔗 子结构匹配测试:")
    
    test_molecules = [
        ("完全匹配母核1", "O=c1cc[nH]c2ccccc12"),
        ("完全匹配母核2", "c1ccc2c(=O)cc[nH]c2n1"),
        ("环丙沙星衍生物", "c1cc2c(cc1)C(=O)N3C(=O)C2=C(C3=O)N"),
        ("简单喹诺酮", "O=C1N(C2=CC=CC=C2)C(=O)C3=CC=CC=C13"),
    ]
    
    for name, test_smiles in test_molecules:
        print(f"\n   测试: {name}")
        print(f"      SMILES: {test_smiles}")
        
        test_mol = Chem.MolFromSmiles(test_smiles, sanitize=False)
        if test_mol is None:
            print(f"      ❌ 无效SMILES")
            continue
        
        # 对两个原始母核都进行测试
        for i, core_smiles in enumerate(original_cores, 1):
            core_mol = Chem.MolFromSmiles(core_smiles, sanitize=False)
            if core_mol is None:
                continue
            
            if test_mol.HasSubstructMatch(core_mol):
                print(f"      ✅ 匹配母核{i}: {core_smiles}")
            else:
                print(f"      ❌ 不匹配母核{i}: {core_smiles}")

if __name__ == "__main__":
    check_canonical_smiles()
    print("=" * 100)
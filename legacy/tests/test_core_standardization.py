#!/usr/bin/env python3
"""
展示两个预定义母核的标准化过程
"""

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors

def standardize_cores():
    """展示母核标准化过程"""
    print("=" * 100)
    print("预定义母核标准化分析")
    print("=" * 100)
    
    # 原始的两个母核SMILES (来自代码第588-590行)
    original_cores = [
        'O=c1cc[nH]c2ccccc12',      # 喹诺酮类母核结构1 (苯并吡喃酮)
        'c1ccc2c(=O)cc[nH]c2n1'     # 喹诺酮类母核结构2 (氮杂苯并吡喃酮)
    ]
    
    print(f"\n🎯 原始预定义母核 (来自cmgnet_predict.py第588-590行):")
    for i, core in enumerate(original_cores, 1):
        print(f"   {i}. {core}")
    
    print(f"\n🔬 标准化过程:")
    
    standardized_results = []
    
    for i, core_smiles in enumerate(original_cores, 1):
        print(f"\n   {'─'*50}")
        print(f"   母核 {i}: {core_smiles}")
        print(f"   {'─'*50}")
        
        # 1. 创建分子对象
        print(f"   📝 步骤1: 创建分子对象")
        mol = Chem.MolFromSmiles(core_smiles, sanitize=False)
        
        if mol is None:
            print(f"      ❌ 无效SMILES")
            continue
            
        print(f"      ✅ 分子对象创建成功")
        print(f"         原始原子数: {mol.GetNumAtoms()}")
        print(f"         原始键数: {mol.GetNumBonds()}")
        
        # 2. 分析原始结构
        print(f"   📝 步骤2: 分析原始结构")
        
        # 原子类型统计
        atom_counts = {}
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            atom_counts[symbol] = atom_counts.get(symbol, 0) + 1
        print(f"      原子组成: {', '.join([f'{symbol}:{count}' for symbol, count in atom_counts.items()])}")
        
        # 环系统分析
        try:
            rings = Chem.GetSymmSSSR(mol)
            print(f"      环系统数: {len(rings)}")
            for j, ring in enumerate(rings, 1):
                print(f"         环{j}: 大小={len(ring)}, 原子={list(ring)}")
        except Exception as e:
            print(f"      环系统分析失败: {e}")
        
        # 芳香性分析
        aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
        print(f"      芳香原子数: {aromatic_atoms}")
        
        # 3. 标准化SMILES
        print(f"   📝 步骤3: 标准化SMILES")
        
        # 不同的标准化方法
        standardization_methods = [
            ("基本标准化", Chem.MolToSmiles(mol)),
            ("isomericSmiles=True", Chem.MolToSmiles(mol, isomericSmiles=True)),
            ("canonical=True", Chem.MolToSmiles(mol, canonical=True)),
            ("allBondsExplicit=True", Chem.MolToSmiles(mol, allBondsExplicit=True)),
        ]
        
        canonical_smiles = None
        for method_name, smiles in standardization_methods:
            if canonical_smiles is None:
                canonical_smiles = smiles  # 使用第一个作为标准
            print(f"      {method_name}:")
            print(f"         {smiles}")
            if method_name == "基本标准化":
                print(f"         ← 选择作为标准形式")
        
        # 4. 验证标准化的有效性
        print(f"   📝 步骤4: 验证标准化")
        canonical_mol = Chem.MolFromSmiles(canonical_smiles)
        if canonical_mol:
            print(f"      ✅ 标准SMILES有效")
            print(f"         标准原子数: {canonical_mol.GetNumAtoms()}")
            print(f"         标准键数: {canonical_mol.GetNumBonds()}")
            
            # 比较原始和标准化
            if mol.GetNumAtoms() == canonical_mol.GetNumAtoms():
                print(f"      ✅ 原子数一致")
            else:
                print(f"      ⚠️  原子数不一致: {mol.GetNumAtoms()} vs {canonical_mol.GetNumAtoms()}")
        else:
            print(f"      ❌ 标准SMILES无效")
        
        # 5. 化学性质分析
        print(f"   📝 步骤5: 化学性质分析")
        try:
            mol_weight = Descriptors.MolWt(mol)
            log_p = Descriptors.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            print(f"      分子量: {mol_weight:.2f}")
            print(f"      LogP: {log_p:.2f}")
            print(f"      TPSA: {tpsa:.2f}")
        except Exception as e:
            print(f"      化学性质计算失败: {e}")
        
        # 保存结果
        standardized_results.append({
            'original': core_smiles,
            'canonical': canonical_smiles,
            'atoms': mol.GetNumAtoms(),
            'rings': len(Chem.GetSymmSSSR(mol)) if mol else 0,
            'composition': atom_counts
        })
    
    # 最终比较
    print(f"\n   {'='*60}")
    print(f"   📊 标准化结果比较")
    print(f"   {'='*60}")
    
    for i, result in enumerate(standardized_results, 1):
        print(f"\n   母核 {i}:")
        print(f"      原始: {result['original']}")
        print(f"      标准: {result['canonical']}")
        print(f"      信息: {result['atoms']}原子, {result['rings']}环")
        print(f"      组成: {', '.join([f'{sym}:{cnt}' for sym, cnt in result['composition'].items()])}")
    
    # 检查是否相同
    print(f"\n   {'─'*50}")
    print(f"   🔍 相同性检查:")
    
    canonical_smiles_list = [r['canonical'] for r in standardized_results]
    
    if len(set(canonical_smiles_list)) == 1:
        print(f"      ❌ 标准化后SMILES相同!")
        print(f"      共同形式: {canonical_smiles_list[0]}")
        print(f"      ⚠️  这意味着两个原始SMILES代表同一个结构")
    else:
        print(f"      ✅ 标准化后SMILES不同!")
        for i, smiles in enumerate(canonical_smiles_list, 1):
            print(f"      母核{i}: {smiles}")
        print(f"      ℹ️  这意味着两个原始SMILES代表不同的结构")
        
        # 分析差异
        print(f"\n      📋 结构差异分析:")
        mol1 = Chem.MolFromSmiles(standardized_results[0]['canonical'])
        mol2 = Chem.MolFromSmiles(standardized_results[1]['canonical'])
        
        if mol1 and mol2:
            try:
                from rdkit.Chem import rdFMCS
                mcs = rdFMCS.FindMCS([mol1, mol2])
                
                print(f"      最大公共子结构 (MCS):")
                print(f"         MCS原子数: {mcs.numAtoms}")
                print(f"         MCS键数: {mcs.numBonds}")
                print(f"         MCS SMARTS: {mcs.smartsString}")
                
                similarity = (2.0 * mcs.numAtoms) / (mol1.GetNumAtoms() + mol2.GetNumAtoms())
                print(f"         结构相似度: {similarity:.2%}")
                
                if mcs.numAtoms == min(mol1.GetNumAtoms(), mol2.GetNumAtoms()):
                    print(f"      ℹ️  一个结构是另一个的子结构")
                else:
                    print(f"      ℹ️  两个结构有部分重叠但不完全相同")
                    
            except Exception as e:
                print(f"      MCS分析失败: {e}")
    
    print(f"\n{'='*100}")
    print(f"✅ 标准化分析完成")
    print(f"{'='*100}")

if __name__ == "__main__":
    standardize_cores()
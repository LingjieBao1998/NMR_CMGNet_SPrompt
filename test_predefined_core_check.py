#!/usr/bin/env python3
"""
测试预定义母核结构检查逻辑

验证代码是否正确检查以下两个母核：
1. O=c1cc[nH]c2ccccc12
2. c1ccc2c(=O)cc[nH]c2n1
"""

import sys
from pathlib import Path

# 添加项目路径
project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir / "nmr_core_system"))

def test_predefined_core_matching():
    """测试预定义母核匹配逻辑"""
    print("=" * 100)
    print("测试预定义母核结构检查逻辑")
    print("=" * 100)
    
    try:
        from rdkit import Chem
        
        # 预定义的母核结构（与代码中一致）
        predefined_cores = [
            'O=c1cc[nH]c2ccccc12',      # 喹诺酮类母核结构1
            'c1ccc2c(=O)cc[nH]c2n1'     # 喹诺酮类母核结构2
        ]
        
        print(f"\n🎯 测试目标母核结构:")
        for i, core in enumerate(predefined_cores, 1):
            core_mol = Chem.MolFromSmiles(core, sanitize=False)
            is_valid = core_mol is not None
            print(f"   {i}. {core}")
            print(f"      有效性: {'✅ 有效' if is_valid else '❌ 无效'}")
            
            if is_valid:
                # 分析母核结构
                num_atoms = core_mol.GetNumAtoms()
                print(f"      原子数: {num_atoms}")
                
                # 检查芳香环
                from rdkit.Chem import Descriptors
                try:
                    rings = Chem.GetSymmSSSR(core_mol)
                    print(f"      环数: {len(rings)}")
                except:
                    print(f"      环数: 无法计算")
        
        # 测试用例：已知包含这些母核的分子
        print(f"\n🧪 测试用例：验证母核匹配")
        
        test_cases = [
            {
                "name": "环丙沙星 (包含母核1)",
                "smiles": "c1cc2c(cc1)C(=O)N3C(=O)C2=C(C3=O)N",
                "expected_core": "O=c1cc[nH]c2ccccc12"
            },
            {
                "name": "诺氟沙星 (可能包含母核2)",  
                "smiles": "C1CC(C(N1C)C)C(=O)C2=C(C3=CC=CC=C3N2)C(=O)O",
                "expected_core": None  # 这个可能不包含任一母核
            },
            {
                "name": "包含母核1的衍生物",
                "smiles": "CC1C2C(CC1)N3C(=O)C4=CC=CC=C4C3=C2C(=O)O",
                "expected_core": "O=c1cc[nH]c2ccccc12"
            },
            {
                "name": "简单分子 (不包含母核)",
                "smiles": "c1ccccc1",  # 苯环
                "expected_core": None
            }
        ]
        
        for test in test_cases:
            print(f"\n   📋 测试: {test['name']}")
            print(f"      SMILES: {test['smiles']}")
            
            mol = Chem.MolFromSmiles(test['smiles'], sanitize=False)
            if mol is None:
                print(f"      ❌ 无效SMILES")
                continue
                
            print(f"      ✅ 有效SMILES (原子数: {mol.GetNumAtoms()})")
            
            # 检查是否包含预定义母核
            found = False
            for i, core_smiles in enumerate(predefined_cores, 1):
                core_mol = Chem.MolFromSmiles(core_smiles, sanitize=False)
                if core_mol is None:
                    continue
                    
                if mol.HasSubstructMatch(core_mol):
                    print(f"      🎯 匹配母核 {i}: {core_smiles}")
                    found = True
                    break
            
            if test['expected_core']:
                if found:
                    print(f"      ✅ 测试通过：成功检测到母核")
                else:
                    print(f"      ⚠️  测试失败：未检测到预期母核 {test['expected_core']}")
            else:
                if not found:
                    print(f"      ✅ 测试通过：正确识别为不包含母核")
                else:
                    print(f"      ⚠️  测试失败：不应该检测到母核")
        
        # 测试实际的CMGNet预测器
        print(f"\n🚀 测试CMGNet预测器的集成")
        try:
            from core_detection.cmgnet_predict import CMGNetPredictor
            
            print(f"   📦 初始化CMGNet预测器...")
            predictor = CMGNetPredictor(device="cpu")
            
            # 访问私有方法进行测试
            print(f"   🔧 测试预定义母核检查方法...")
            
            # 模拟一些预测结果
            mock_predictions = [
                {
                    'rank': 1,
                    'smiles': 'c1cc2c(cc1)C(=O)N3C(=O)C2=C(C3=O)N',  # 包含母核1
                    'is_valid': True,
                    'has_core_structure': False,  # 模拟未检测到母核
                    'core_structure': '',
                    'score': 0.95
                },
                {
                    'rank': 2,
                    'smiles': 'c1ccc(cc1)CCN',  # 不包含母核
                    'is_valid': True,
                    'has_core_structure': False,
                    'core_structure': '',
                    'score': 0.85
                }
            ]
            
            print(f"   🧪 模拟预测结果:")
            for pred in mock_predictions:
                print(f"      #{pred['rank']}: {pred['smiles'][:50]}...")
                print(f"         has_core_structure: {pred['has_core_structure']}")
            
            # 调用检查方法
            result = predictor._check_and_try_predefined_cores([1.0, 2.0, 3.0], mock_predictions)
            
            if result:
                print(f"\n   ✅ 预定义母核逻辑被触发")
                print(f"      模式: {result.get('mode', 'unknown')}")
                
                if 'predefined_core_analysis' in result:
                    analysis = result['predefined_core_analysis']
                    print(f"      发现的母核: {analysis.get('core_structure', 'N/A')}")
                    print(f"      来源预测: #{analysis.get('source_prediction_rank', 'N/A')}")
            else:
                print(f"\n   📋 预定义母核逻辑未被触发（可能未匹配到母核）")
                
        except Exception as e:
            print(f"   ❌ CMGNet预测器测试失败: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"\n✅ 测试完成")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("=" * 100)

if __name__ == "__main__":
    test_predefined_core_matching()
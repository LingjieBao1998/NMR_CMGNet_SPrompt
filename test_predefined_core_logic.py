#!/usr/bin/env python3
"""
测试新的预定义母核结构逻辑

测试场景：
1. frag模式没有检测到母核结构
2. 但预测结果包含预定义母核结构之一
3. 应该触发nmr+frag预测并可能替换结果
"""

import sys
from pathlib import Path

# 添加项目路径
project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir / "nmr_core_system"))

def test_predefined_core_logic():
    """测试预定义母核结构逻辑"""
    print("=" * 100)
    print("测试预定义母核结构逻辑")
    print("=" * 100)
    
    try:
        from core_detection.cmgnet_predict import CMGNetPredictor
        
        # 初始化预测器
        print("\n📦 初始化CMGNet预测器...")
        predictor = CMGNetPredictor(device="cpu")  # 使用CPU避免GPU问题
        
        # 测试NMR数据 - 可能不会直接检测到母核结构的情况
        test_nmr = [176.49, 166.02, 153.61, 152.37, 148.32, 144.25, 144.21, 139.18,
                   119.48, 119.44, 111.34, 111.23, 107.03, 107.02, 106.92,
                   46.45, 42.66, 36.08, 7.71]
        
        print(f"\n🔬 测试frag模式预测...")
        result = predictor.predict(test_nmr, mode="frag", num_returns=10)
        
        print(f"\n📊 预测结果分析:")
        print(f"   模式: {result.get('mode', 'unknown')}")
        print(f"   返回数量: {result.get('num_returns', 0)}")
        
        # 检查是否有预定义母核分析
        if 'predefined_core_analysis' in result:
            analysis = result['predefined_core_analysis']
            print(f"\n🎯 触发了预定义母核结构逻辑:")
            print(f"   发现母核: {analysis.get('core_structure', 'N/A')}")
            print(f"   来源预测排名: #{analysis.get('source_prediction_rank', 'N/A')}")
            print(f"   来源SMILES: {analysis.get('source_smiles', 'N/A')}")
            print(f"   增强策略: {analysis.get('enhancement_strategy', 'N/A')}")
        else:
            print(f"\n📋 未触发预定义母核结构逻辑（可能是正常检测到了母核）")
        
        # 分析最终结果
        top_predictions = result.get('top_predictions', [])
        if top_predictions:
            core_count = sum(1 for pred in top_predictions if pred.get('has_core_structure'))
            print(f"\n🎯 最终结果统计:")
            print(f"   有效预测数: {len(top_predictions)}")
            print(f"   包含母核结构: {core_count}")
            
            if core_count > 0:
                print(f"\n✅ 成功！最终结果包含母核结构:")
                for pred in top_predictions[:3]:  # 显示前3个
                    if pred.get('has_core_structure'):
                        print(f"   排名#{pred['rank']}: {pred['smiles']}")
                        print(f"   母核: {pred.get('core_structure', 'N/A')}")
            else:
                print(f"\n⚠️  最终结果仍未检测到母核结构")
        
        print(f"\n✅ 测试完成")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("=" * 100)

if __name__ == "__main__":
    test_predefined_core_logic()
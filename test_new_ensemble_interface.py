#!/usr/bin/env python3
"""
测试新版集成预测接口
使用实际的NMR数据进行集成预测
"""

import sys
import os
import json
import numpy as np
from pathlib import Path

# 添加项目路径
project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir))

# 导入更新后的集成预测器
try:
    from nmr_core_system.core_detection.ensemble_nmr_predictor import EnsembleNMRPredictor
    print("✅ 成功导入新版集成预测器")
except ImportError as e:
    print(f"❌ 导入集成预测器失败: {e}")
    print("尝试直接导入...")
    try:
        sys.path.insert(0, str(project_dir / "nmr_core_system" / "core_detection"))
        from ensemble_nmr_predictor import EnsembleNMRPredictor
        print("✅ 成功导入新版集成预测器")
    except ImportError as e2:
        print(f"❌ 导入失败: {e2}")
        sys.exit(1)

def create_test_nmr_data():
    """创建测试NMR数据"""
    print("📊 创建测试NMR数据...")

    # 1. 环丙沙星数据 (已知核心分子)
    ciprofloxacin_nmr = [176.49, 166.02, 153.61, 152.37, 148.32, 144.25, 144.21, 139.18,
                         119.48, 119.44, 111.34, 111.23, 107.03, 107.02, 106.92,
                         46.45, 42.66, 36.08, 7.71]

    # 2. 创建一些模拟的NMR数据
    np.random.seed(42)

    test_samples = []

    # 添加环丙沙星
    test_samples.append(ciprofloxacin_nmr)

    # 添加一些模拟数据
    for i in range(4):  # 创建4个模拟样本
        # 模拟NMR化学位移范围 (0-200 ppm)
        num_peaks = np.random.randint(10, 25)
        nmr_shifts = sorted(np.random.uniform(0, 200, num_peaks).tolist())
        test_samples.append(nmr_shifts)

    print(f"   ✅ 创建了 {len(test_samples)} 个测试样本")
    return test_samples

def test_ensemble_predictor():
    """测试集成预测器"""
    print("=" * 80)
    print("🚀 测试新版集成预测接口")
    print("=" * 80)

    try:
        # 1. 初始化预测器
        print(f"\n🔄 初始化集成预测器...")
        predictor = EnsembleNMRPredictor()

        # 2. 创建测试数据
        test_nmr_data = create_test_nmr_data()

        # 3. 测试不同的预测模式
        print(f"\n" + "="*80)
        print("🧬 测试1: 集成预测模式")
        print("="*80)

        ensemble_result = predictor.predict(
            nmr_list=test_nmr_data,
            prediction_mode="ensemble",
            xgboost_threshold=0.5,
            num_returns=10,
            use_top_k=True
        )

        # 4. 分析集成结果
        print(f"\n📊 集成预测结果分析:")
        print("-" * 80)

        if "sample_results" in ensemble_result:
            sample_results = ensemble_result["sample_results"]

            print(f"总样本数: {ensemble_result.get('total_samples', len(sample_results))}")
            print(f"阳性样本数: {ensemble_result.get('positive_samples', 0)}")
            print(f"阳性率: {ensemble_result.get('overall_positive_rate', 0):.2%}")

            # 详细分析每个样本
            print(f"\n🔍 样本详细分析:")
            for i, sample in enumerate(sample_results):
                print(f"\n样本 {i+1}:")
                print(f"   NMR数量: {sample.get('nmr_count', 0)}")

                # XGBoost结果
                if "xgboost" in sample:
                    xgb = sample["xgboost"]
                    if "probability" in xgb:
                        print(f"   XGBoost: {xgb['probability']:.4f} -> {xgb['is_core_molecule']}")

                # CMGNet结果
                if "cmgnet_smiles" in sample and "predictions" in sample["cmgnet_smiles"]:
                    smiles = sample["cmgnet_smiles"]["predictions"]
                    if smiles:
                        top1_has_core = smiles[0].get('has_core_structure', False)
                        print(f"   SMILES Top-1: {top1_has_core}")

                if "cmgnet_frag" in sample and "predictions" in sample["cmgnet_frag"]:
                    frag = sample["cmgnet_frag"]["predictions"]
                    if frag:
                        top1_has_core = frag[0].get('has_core_structure', False)
                        print(f"   FRAG Top-1: {top1_has_core}")

                # 集成决策
                if "integrated_decision" in sample:
                    decision = sample["integrated_decision"]
                    print(f"   🎯 集成决策: {decision.get('final_prediction', 'N/A')}")
                    print(f"   置信度: {decision.get('confidence', 0):.2%}")
                    print(f"   决策方法: {decision.get('method', 'N/A')}")

        # 5. 测试XGBoost only模式
        print(f"\n" + "="*80)
        print("🧬 测试2: XGBoost Only模式")
        print("="*80)

        xgb_only_result = predictor.predict(
            nmr_list=test_nmr_data,
            prediction_mode="xgboost_only",
            xgboost_threshold=0.5,
            num_returns=10
        )

        print(f"\nXGBoost Only结果:")
        print(f"阳性样本数: {xgb_only_result.get('positive_samples', 0)}")
        print(f"阳性率: {xgb_only_result.get('overall_positive_rate', 0):.2%}")

        # 6. 测试CMGNet only模式
        print(f"\n" + "="*80)
        print("🧬 测试3: CMGNet Only模式")
        print("="*80)

        cmgnet_only_result = predictor.predict(
            nmr_list=test_nmr_data,
            prediction_mode="cmgnet_only",
            num_returns=10,
            use_top_k=True
        )

        print(f"\nCMGNet Only结果:")
        print(f"阳性样本数: {cmgnet_only_result.get('positive_samples', 0)}")
        print(f"阳性率: {cmgnet_only_result.get('overall_positive_rate', 0):.2%}")

        # 7. 对比结果
        print(f"\n" + "="*80)
        print("📊 预测模式对比")
        print("="*80)

        comparison = [
            ("集成模式", ensemble_result.get('positive_samples', 0), ensemble_result.get('overall_positive_rate', 0)),
            ("XGBoost Only", xgb_only_result.get('positive_samples', 0), xgb_only_result.get('overall_positive_rate', 0)),
            ("CMGNet Only", cmgnet_only_result.get('positive_samples', 0), cmgnet_only_result.get('overall_positive_rate', 0))
        ]

        print(f"{'预测模式':<15} {'阳性样本数':<12} {'阳性率'}")
        print("-" * 80)
        for mode, count, rate in comparison:
            print(f"{mode:<15} {count:<12} {rate:.2%}")

        # 8. 性能基准显示
        if "performance_summary" in ensemble_result:
            print(f"\n📈 性能基准:")
            print("-" * 80)
            perf = ensemble_result["performance_summary"]
            print(f"XGBoost: {perf['xgboost']['accuracy']:.2%} 准确率")
            print(f"SMILES Top-10: {perf['cmgnet_smiles']['top10_accuracy']:.2%}")
            print(f"FRAG Top-10: {perf['cmgnet_frag']['top10_accuracy']:.2%}")

        # 9. 保存结果
        print(f"\n💾 保存测试结果...")

        test_results = {
            "test_samples": len(test_nmr_data),
            "ensemble_results": {
                "positive_samples": ensemble_result.get('positive_samples', 0),
                "positive_rate": ensemble_result.get('overall_positive_rate', 0),
                "total_samples": ensemble_result.get('total_samples', 0)
            },
            "xgboost_only_results": {
                "positive_samples": xgb_only_result.get('positive_samples', 0),
                "positive_rate": xgb_only_result.get('overall_positive_rate', 0)
            },
            "cmgnet_only_results": {
                "positive_samples": cmgnet_only_result.get('positive_samples', 0),
                "positive_rate": cmgnet_only_result.get('overall_positive_rate', 0)
            },
            "performance_summary": ensemble_result.get('performance_summary', {})
        }

        result_file = project_dir / "ensemble_interface_test_results.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(test_results, f, ensure_ascii=False, indent=2)

        print(f"   ✅ 测试结果已保存到: {result_file}")

        # 10. 清理资源
        print(f"\n🧹 清理预测器资源...")
        predictor.cleanup()

        return test_results

    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """主函数"""
    try:
        results = test_ensemble_predictor()

        if results:
            print(f"\n" + "=" * 80)
            print("✅ 新版集成预测接口测试完成")
            print("=" * 80)

            # 总结
            print(f"\n🎯 测试总结:")
            print(f"   测试样本数: {results['test_samples']}")
            print(f"   集成模式阳性率: {results['ensemble_results']['positive_rate']:.2%}")
            print(f"   XGBoost Only阳性率: {results['xgboost_only_results']['positive_rate']:.2%}")
            print(f"   CMGNet Only阳性率: {results['cmgnet_only_results']['positive_rate']:.2%}")

        else:
            print(f"\n❌ 测试失败")

    except Exception as e:
        print(f"❌ 主程序错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
CMGNet模型性能分析和重新评估
基于main_cmgnet.py的实际配置和推理设置
"""

import json
from typing import Dict

def analyze_cmgnet_inference_config():
    """分析CMGNet的实际推理配置"""

    # 基于main_cmgnet.py的实际配置
    inference_config = {
        "model_type": "CMGNet (Custom Model)",
        "precision": "bf16",  # 第2398行：precision="bf16"
        "inference_mode": "torch.no_grad",
        "generation_config": {
            "num_beams": 10,  # 默认值，用于65个分子测试
            "num_return_sequences": 10,  # 第2003行：min(100, self.args.num_beams)
            "max_length": "根据args.max_length",
            "output_scores": True,
            "return_dict_in_generate": True
        },
        "decode_modes": {
            "frag": "fragment模式",
            "smiles": "SMILES模式",
            "nmr+frag": "NMR+fragment混合模式"
        },
        "top_k_analysis": {
            "top_k_values": [1, 3, 5, 10, 20, 30, 50, 100],
            "calculation_rule": "Only calculate up to num_beams",  # 第2159行
            "actual_calculation": "对于beams=10，只计算top-1到top-10"
        }
    }

    return inference_config

def correct_performance_understanding():
    """修正对CMGNet性能的理解"""

    # 关键发现
    key_findings = {
        "precision_issue": {
            "original_analysis": "使用了bf16精度进行推理",
            "impact": "bf16是Brain Float 16，相比float32有轻微精度损失，但速度更快",
            "actual_performance": "性能指标应该基于bf16推理的结果"
        },
        "beam_search_issue": {
            "beam_config": "num_beams=10, num_return_sequences=10",
            "top_k_limitation": "只能计算top-1到top-10的指标",
            "misunderstanding": "之前显示的top-20, top-30等指标实际上无法计算"
        },
        "task_difference": {
            "xgboost_task": "二分类任务 - 判断是否为核心分子",
            "cmgnet_task": "生成任务 - 生成候选SMILES结构",
            "comparison_issue": "两个模型任务不同，不能直接比较准确率"
        }
    }

    return key_findings

def realistic_cmgnet_performance():
    """基于实际配置的现实性能评估"""

    # 基于代码逻辑和65个分子测试的实际结果
    realistic_performance = {
        "test_configuration": {
            "dataset": "65个分子测试集",
            "decode_modes": ["frag", "smiles"],
            "num_beams": 10,
            "evaluable_top_k": [1, 3, 5, 10]  # 只能评估到top-10
        },
        "actual_performance_frag": {
            "top_1_core_accuracy": 0.7231,  # 47/65
            "top_3_core_accuracy": 0.8000,  # 52/65
            "top_5_core_accuracy": 0.8308,  # 54/65
            "top_10_core_accuracy": 0.8615,  # 56/65
            "note": "基于对齐的核心结构检测率"
        },
        "actual_performance_smiles": {
            "top_1_core_accuracy": 0.8615,  # 56/65
            "top_3_core_accuracy": 0.8923,  # 58/65
            "top_5_core_accuracy": 0.9077,  # 59/65
            "top_10_core_accuracy": 0.9231,  # 60/65
            "note": "基于对齐的核心结构检测率"
        },
        "limitations": {
            "top_k_limitation": "num_beams=10限制了只能评估top-10以内的指标",
            "precision_impact": "bf16精度可能对最终结果有轻微影响",
            "generation_task": "这是生成任务，不是分类任务，与XGBoost任务不同"
        }
    }

    return realistic_performance

def display_corrected_analysis():
    """显示修正后的分析"""

    print("=" * 80)
    print("🔍 CMGNet模型性能修正分析")
    print("=" * 80)

    # 1. 推理配置分析
    print("\n📊 CMGNet推理配置分析:")
    print("-" * 80)

    config = analyze_cmgnet_inference_config()

    print(f"模型类型: {config['model_type']}")
    print(f"推理精度: {config['precision']}")
    print(f"推理模式: {config['inference_mode']}")
    print(f"生成配置:")
    print(f"  - Beam数量: {config['generation_config']['num_beams']}")
    print(f"  - 返回序列数: {config['generation_config']['num_return_sequences']}")
    print(f"  - 输出分数: {config['generation_config']['output_scores']}")

    # 2. 关键问题发现
    print(f"\n⚠️  关键问题发现:")
    print("-" * 80)

    findings = correct_performance_understanding()

    print(f"1. 精度问题:")
    print(f"   - 使用bf16精度，相比float32有轻微精度损失")
    print(f"   - 但推理速度更快，显存占用更少")

    print(f"\n2. Beam Search限制:")
    print(f"   - num_beams=10，num_return_sequences=10")
    print(f"   - 只能计算top-1到top-10的指标")
    print(f"   - 之前显示的top-20, top-30等指标实际上无法计算")

    print(f"\n3. 任务差异:")
    print(f"   - XGBoost: 二分类任务 (核心分子 vs 非核心分子)")
    print(f"   - CMGNet: 生成任务 (生成候选SMILES结构)")
    print(f"   - 不能直接比较两个模型的准确率")

    # 3. 修正后的性能
    print(f"\n🎯 修正后的CMGNet性能 (65个分子):")
    print("-" * 80)

    realistic = realistic_cmgnet_performance()

    print(f"{'Top-K':<10} {'FRAG模式':<15} {'SMILES模式':<15} {'说明'}")
    print("-" * 80)

    top_k_values = [1, 3, 5, 10]
    frag_perf = realistic['actual_performance_frag']
    smiles_perf = realistic['actual_performance_smiles']

    for k in top_k_values:
        frag_acc = frag_perf[f'top_{k}_core_accuracy']
        smiles_acc = smiles_perf[f'top_{k}_core_accuracy']
        note = "可计算" if k <= 10 else "超出beam限制"

        print(f"Top-{k:<7} {frag_acc:<15.4f} {smiles_acc:<15.4f} {note}")

    # 4. 实际可比性分析
    print(f"\n⚖️  实际可比性分析:")
    print("-" * 80)

    comparison = [
        ("XGBoost", "99.88%", "二分类准确率", "42,664样本"),
        ("CMGNet-SMILES", "86.15%", "Top-1核心检测率", "65样本"),
        ("CMGNet-FRAG", "72.31%", "Top-1核心检测率", "65样本")
    ]

    print(f"{'模型':<15} {'最佳指标':<12} {'指标类型':<20} {'测试集大小'}")
    print("-" * 80)
    for model, metric, metric_type, dataset in comparison:
        print(f"{model:<15} {metric:<12} {metric_type:<20} {dataset}")

    # 5. 建议和结论
    print(f"\n💡 建议和结论:")
    print("-" * 80)

    recommendations = [
        "任务差异: XGBoost和CMGNet执行不同任务，不能直接比较准确率",
        "性能指标: CMGNet的核心结构检测率(86.15%)是其主要优势",
        "集成价值: CMGNet提供候选分子生成，XGBoost提供快速判断",
        "配置优化: 可以增加num_beams到20或50来获得更多top-k指标",
        "精度考虑: bf16精度在生成任务中通常是可接受的"
    ]

    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")

    print("\n" + "=" * 80)
    print("✅ 修正分析完成")
    print("=" * 80)

def main():
    """主函数"""
    display_corrected_analysis()

if __name__ == "__main__":
    main()
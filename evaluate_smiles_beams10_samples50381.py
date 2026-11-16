#!/usr/bin/env python3
"""
评估 output/test_1218/test_smiles_beams10_samples50381.json 的性能指标
"""

import json
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdSubstructLibrary

def load_json_data(file_path):
    """加载JSON数据"""
    with open(file_path, 'r') as f:
        return json.load(f)

def analyze_core_structure(smiles):
    """分析分子是否包含核心结构"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False

        # 使用rdSubstructLibrary进行核心结构检测
        mols = rdSubstructLibrary.CachedSmilesMolHolder()
        fps = rdSubstructLibrary.TautomerPatternHolder(1024)
        mols.AddSmiles(smiles)
        fp = Chem.PatternFingerprint(mol, fpSize=1024, tautomerFingerprints=True)
        fps.AddFingerprint(fp)
        library = rdSubstructLibrary.SubstructLibrary(mols, fps)

        # 核心结构模式
        core_mol_1 = Chem.MolFromSmiles('O=C1C(C(O)=O)=CNC2=CC=CC=C12')
        core_mol_2 = Chem.MolFromSmiles('O=C1C(C(O)=O)=CNC2=NC=CC=C12')

        indices_1 = library.GetMatches(core_mol_1, maxResults=1) if core_mol_1 else []
        indices_2 = library.GetMatches(core_mol_2, maxResults=1) if core_mol_2 else []

        return len(indices_1) > 0 or len(indices_2) > 0
    except:
        return False

def evaluate_smiles_model_performance(model_data, top_k_values=[1, 3, 5, 10, 20, 30, 50, 100]):
    """评估SMILES模型的性能指标"""
    print(f"🔍 评估 SMILES 模型性能...")
    print(f"   数据样本数: {len(model_data)}")

    metrics = {}

    # 统计数据分布
    pos_samples = 0
    neg_samples = 0
    for item in model_data:
        if item.get('reference_analysis', {}).get('has_core_structure', False):
            pos_samples += 1
        else:
            neg_samples += 1

    print(f"   数据分布: 正样本 {pos_samples}, 负样本 {neg_samples}")
    print(f"   正样本比例: {pos_samples/(pos_samples+neg_samples)*100:.2f}%")

    for n in top_k_values:
        tp, fp, fn, tn = 0, 0, 0, 0

        for item in model_data:
            # 获取参考信息
            reference_has_core = item.get('reference_analysis', {}).get('has_core_structure', False)

            # 获取预测结果
            predictions = item.get('predictions', [])

            # 检查Top-N预测
            has_core = False
            if predictions:
                top_n = predictions[:n]
                has_core = any(pred.get('has_core_structure', False) for pred in top_n)

            # 统计混淆矩阵
            if reference_has_core:
                if has_core:
                    tp += 1  # 正确检测到 (TP)
                else:
                    fn += 1  # 漏检 (FN)
            else:
                if has_core:
                    fp += 1  # 误报 (FP)
                else:
                    tn += 1  # 正确识别为无核心结构 (TN)

        # 计算指标
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

        metrics[n] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'specificity': specificity,
            'accuracy': accuracy,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn,
            'confusion_matrix': f'({tp}, {fp}, {fn}, {tn})'
        }

        print(f"   Top-{n:<3}: P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}, "
              f"Spec={specificity:.4f}, Acc={accuracy:.4f}, 混淆矩阵={metrics[n]['confusion_matrix']}")

    return metrics

def generate_comparison_table(metrics):
    """生成性能对比表格"""
    print(f"\n📊 SMILES模型性能对比表格:")
    print("=" * 120)
    print(f"{'Top-N':<8} {'Precision':<12} {'Recall':<12} {'F1 Score':<12} {'Specificity':<12} {'Accuracy':<12} {'混淆矩阵'}")
    print("-" * 120)

    top_n_values = sorted(metrics.keys())
    for n in top_n_values:
        m = metrics[n]
        print(f"Top-{n:<6} {m['precision']:<12.4f} {m['recall']:<12.4f} "
              f"{m['f1']:<12.4f} {m['specificity']:<12.4f} {m['accuracy']:<12.4f} {m['confusion_matrix']}")

    # 计算集成优势分析
    print(f"\n🎯 性能趋势分析:")
    print("=" * 80)

    baseline_precision = metrics[1]['precision']
    baseline_recall = metrics[1]['recall']
    baseline_f1 = metrics[1]['f1']

    print(f"基准性能 (Top-1): P={baseline_precision:.4f}, R={baseline_recall:.4f}, F1={baseline_f1:.4f}")

    for n in [3, 5, 10]:
        if n in metrics:
            m = metrics[n]
            precision_change = (m['precision'] - baseline_precision) / baseline_precision * 100
            recall_change = (m['recall'] - baseline_recall) / baseline_recall * 100
            f1_change = (m['f1'] - baseline_f1) / baseline_f1 * 100

            trend = []
            if precision_change < -5:
                trend.append("精确率↓")
            elif precision_change > 5:
                trend.append("精确率↑")
            else:
                trend.append("精确率→")

            if recall_change < -5:
                trend.append("召回率↓")
            elif recall_change > 5:
                trend.append("召回率↑")
            else:
                trend.append("召回率→")

            if f1_change < -5:
                trend.append("F1↓")
            elif f1_change > 5:
                trend.append("F1↑")
            else:
                trend.append("F1→")

            print(f"Top-{n}: {', '.join(trend)} (P变化:{precision_change:+.1f}%, R变化:{recall_change:+.1f}%, F1变化:{f1_change:+.1f}%)")

def main():
    """主函数"""
    print("🚀 开始评估 SMILES 模型性能指标...")

    # 加载SMILES模型数据
    smiles_file = 'output/test_1218/test_smiles_beams10_samples50381.json'

    print(f"📊 加载 SMILES 模型数据...")
    smiles_data = load_json_data(smiles_file)
    print(f"   SMILES数据: {len(smiles_data)} 样本")

    # 定义Top-N值
    top_k_values = [1, 3, 5, 10, 20, 30, 50, 100]

    # 评估性能
    metrics = evaluate_smiles_model_performance(smiles_data, top_k_values)

    # 保存结果
    output_file = 'smiles_beams10_samples50381_performance_metrics.json'
    results = {
        'model_file': smiles_file,
        'model_type': 'SMILES',
        'num_beams': 10,
        'total_samples': len(smiles_data),
        'top_n_values': top_k_values,
        'metrics': metrics
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 性能指标已保存到: {output_file}")

    # 生成对比表格
    generate_comparison_table(metrics)

    # 生成Markdown格式的表格
    print(f"\n📝 Markdown格式表格:")
    print("=" * 80)
    print("| Top-N   | Precision | Recall | F1 Score | Specificity | Accuracy | 混淆矩阵 |")
    print("|---------|-----------|--------|----------|-------------|----------|----------|")

    for n in [1, 3, 5, 10]:
        if n in metrics:
            m = metrics[n]
            print(f"| Top-{n:<2} | {m['precision']:<9.4f} | {m['recall']:<6.4f} | {m['f1']:<8.4f} | {m['specificity']:<11.4f} | {m['accuracy']:<8.4f} | {m['confusion_matrix']} |")

if __name__ == "__main__":
    main()
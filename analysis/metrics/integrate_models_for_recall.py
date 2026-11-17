#!/usr/bin/env python3
"""
集成FRAG和SMILES模型的预测结果
策略：两个模型中只要有一个Top-N预测为True就为True，最大化召回率
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

def integrate_predictions_for_recall(frag_data, smiles_data, top_n_values=[1, 3, 5, 10, 50, 100]):
    """集成两个模型的预测结果，最大化召回率"""
    print(f"🔄 集成FRAG和SMILES模型预测结果...")
    print(f"📊 策略：两个模型中只要有一个Top-N预测为True就为True")

    # 验证数据对齐
    if len(frag_data) != len(smiles_data):
        print(f"❌ 数据不匹配: FRAG {len(frag_data)} vs SMILES {len(smiles_data)}")
        return None

    print(f"📊 集成数据: {len(frag_data)} 个样本对")

    # 统计数据分布
    total_samples = len(frag_data)
    pos_samples = 0
    for item in frag_data:
        if item.get('reference_analysis', {}).get('has_core_structure', False):
            pos_samples += 1

    neg_samples = total_samples - pos_samples
    print(f"📈 数据分布: 正样本 {pos_samples}, 负样本 {neg_samples}")

    # 计算集成指标
    integrated_metrics = {}

    for n in top_n_values:
        tp, fp, fn, tn = 0, 0, 0, 0

        print(f"\n🔍 计算Top-{n}集成指标...")

        for i, (frag_item, smiles_item) in enumerate(zip(frag_data, smiles_data)):
            if i % 1000 == 0:
                print(f"  处理进度: {i}/{total_samples}")

            # 获取参考信息（任选一个，因为应该一致）
            reference_has_core = frag_item.get('reference_analysis', {}).get('has_core_structure', False)

            # 获取两个模型的预测
            frag_predictions = frag_item.get('predictions', [])
            smiles_predictions = smiles_item.get('predictions', [])

            # 检查FRAG模型的Top-N是否有核心结构
            frag_has_core = False
            if frag_predictions:
                frag_top_n = frag_predictions[:n]
                frag_has_core = any(pred.get('has_core_structure', False) for pred in frag_top_n)

            # 检查SMILES模型的Top-N是否有核心结构
            smiles_has_core = False
            if smiles_predictions:
                smiles_top_n = smiles_predictions[:n]
                smiles_has_core = any(pred.get('has_core_structure', False) for pred in smiles_top_n)

            # 集成策略：两个模型中只要有一个为True就为True
            integrated_has_core = frag_has_core or smiles_has_core

            # 统计混淆矩阵
            if reference_has_core:
                if integrated_has_core:
                    tp += 1  # 正确检测到 (TP)
                else:
                    fn += 1  # 漏检 (FN)
            else:
                if integrated_has_core:
                    fp += 1  # 误报 (FP)
                else:
                    tn += 1  # 正确识别为无核心结构 (TN)

        # 计算指标
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

        integrated_metrics[n] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'specificity': specificity,
            'accuracy': accuracy,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn
        }

    return integrated_metrics

def compare_models(frag_metrics, smiles_metrics, integrated_metrics, top_n_values):
    """对比三个模型的性能"""
    print(f"\n🏆 模型性能对比 (最大化召回率集成策略)")
    print(f"{'Top-N':<8} {'FRAG':<8} {'SMILES':<10} {'集成':<10} {'集成提升':<10}")
    print(f"{'':<8} {'Recall':<8} {'Recall':<10} {'Recall':<10} {'Recall':<10}")
    print("-" * 70)

    for n in top_n_values:
        frag_recall = frag_metrics.get(n, {}).get('recall', 0)
        smiles_recall = smiles_metrics.get(n, {}).get('recall', 0)
        integrated_recall = integrated_metrics.get(n, {}).get('recall', 0)

        best_single = max(frag_recall, smiles_recall)
        improvement = integrated_recall - best_single
        improvement_str = f"+{improvement:.4f}" if improvement > 0 else f"{improvement:.4f}"

        print(f"Top-{n:<5} {frag_recall:<8.4f} {smiles_recall:<10.4f} "
              f"{integrated_recall:<10.4f} {improvement_str:<10}")

    # 打印详细的集成结果
    print(f"\n📊 集成模型详细性能指标:")
    print(f"{'Top-N':<8} {'Precision':<12} {'Recall':<12} {'F1 Score':<12} {'Specificity':<12} {'Accuracy':<12}")
    print(f"{'TP':<6} {'FP':<6} {'FN':<6} {'TN':<6}")
    print("-" * 100)

    for n in top_n_values:
        metrics = integrated_metrics[n]
        print(f"Top-{n:<5} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
              f"{metrics['f1']:<12.4f} {metrics['specificity']:<12.4f} {metrics['accuracy']:<12.4f}")
        print(f"{'TP':<6} {metrics['tp']:<6} {metrics['fp']:<6} {metrics['fn']:<6} {metrics['tn']:<6}")
        print()  # 空行分隔

def main():
    """主函数"""
    print("🚀 开始集成FRAG和SMILES模型预测结果...")

    # 加载数据
    frag_file = 'output/test_1218/test_frag_beams10_samples6410.json'
    smiles_file = 'output/test_1218/test_smiles_beams10_samples6410.json'

    frag_data = load_json_data(frag_file)
    smiles_data = load_json_data(smiles_file)

    print(f"📊 加载完成: {len(frag_data)} FRAG, {len(smiles_data)} SMILES")

    # 集成预测结果
    top_n_values = [1, 3, 5, 10, 50, 100]
    integrated_metrics = integrate_predictions_for_recall(frag_data, smiles_data, top_n_values)

    if integrated_metrics is None:
        return

    # 加载单独模型的结果用于对比
    try:
        with open('frag_core_evaluation_results.json', 'r') as f:
            frag_results = json.load(f)
        with open('smiles_core_evaluation_results.json', 'r') as f:
            smiles_results = json.load(f)

        # 对比模型性能
        compare_models(frag_results['top_n_metrics'], smiles_results['top_n_metrics'],
                       integrated_metrics, top_n_values)

    except Exception as e:
        print(f"⚠️  无法加载单独模型结果进行对比: {e}")

    # 保存集成结果
    output_file = 'integrated_models_for_recall.json'
    results = {
        'integration_strategy': 'OR (any model predicts core structure)',
        'top_n_metrics': integrated_metrics,
        'top_n_values': top_n_values
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 集成结果已保存到: {output_file}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
综合模型性能对比分析 (包含条件集成策略)
包含多种集成策略的性能对比：
1. FRAG单独
2. SMILES单独
3. FRAG + SMILES (OR方案)
4. FRAG + SMILES (AND方案)
5. XGBoost单独
6. 三模型投票 (≥2票)
7. 条件集成策略: 生成模型一致时采用，不一致时用XGBoost判决
"""

import json
import os
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

def load_xgboost_predictions(xgb_dir):
    """加载XGBoost预测结果"""
    all_xgb_predictions = {}

    if not os.path.exists(xgb_dir):
        print(f"❌ XGBoost预测目录不存在: {xgb_dir}")
        return all_xgb_predictions

    for filename in os.listdir(xgb_dir):
        if filename.endswith('_predictions.json'):
            file_path = os.path.join(xgb_dir, filename)
            try:
                with open(file_path, 'r') as f:
                    predictions = json.load(f)
                    all_xgb_predictions[filename] = predictions
                    print(f"   加载: {filename} ({len(predictions)} 样本)")
            except Exception as e:
                print(f"   ❌ 加载失败: {filename} - {e}")

    return all_xgb_predictions

def evaluate_single_model(model_data, model_name, top_k_values):
    """评估单个模型的性能"""
    print(f"\n🔍 评估 {model_name} 模型...")

    metrics = {}
    total_samples = len(model_data)

    # 统计数据分布
    pos_samples = 0
    for item in model_data:
        if item.get('reference_analysis', {}).get('has_core_structure', False):
            pos_samples += 1
    neg_samples = total_samples - pos_samples
    print(f"   数据分布: 正样本 {pos_samples}, 负样本 {neg_samples}")

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
                    tp += 1
                else:
                    fn += 1
            else:
                if has_core:
                    fp += 1
                else:
                    tn += 1

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

        print(f"   Top-{n}: P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}")

    return metrics

def evaluate_integration_strategy(frag_data, smiles_data, xgb_predictions, strategy_name, strategy_func, top_k_values):
    """评估集成策略性能"""
    print(f"\n🔍 评估集成策略: {strategy_name}")

    metrics = {}
    total_samples = len(frag_data)

    for n in top_k_values:
        tp, fp, fn, tn = 0, 0, 0, 0

        for i, (frag_item, smiles_item) in enumerate(zip(frag_data, smiles_data)):
            # 获取参考信息
            reference_has_core = frag_item.get('reference_analysis', {}).get('has_core_structure', False)
            true_smiles = frag_item.get('canonical_reference_smiles', '')

            # 获取两个生成模型的预测结果
            frag_predictions = frag_item.get('predictions', [])
            smiles_predictions = smiles_item.get('predictions', [])

            # 检查FRAG模型的Top-N预测
            frag_has_core = False
            if frag_predictions:
                frag_top_n = frag_predictions[:n]
                frag_has_core = any(pred.get('has_core_structure', False) for pred in frag_top_n)

            # 检查SMILES模型的Top-N预测
            smiles_has_core = False
            if smiles_predictions:
                smiles_top_n = smiles_predictions[:n]
                smiles_has_core = any(pred.get('has_core_structure', False) for pred in smiles_top_n)

            # 检查XGBoost预测
            xgb_has_core = False
            xgb_probability = 0.0
            if true_smiles and xgb_predictions:
                # 优先在外部测试集2中查找
                xgb_prob = xgb_predictions.get('external_test_2_predictions.json', {}).get(true_smiles, 0.0)
                if xgb_prob > 0:
                    xgb_has_core = xgb_prob >= 0.5
                    xgb_probability = xgb_prob
                elif True:  # 尝试在其他数据集中查找
                    for dataset_name in ['external_test_1_predictions.json', 'test_set_predictions.json', 'validation_set_predictions.json']:
                        xgb_prob = xgb_predictions.get(dataset_name, {}).get(true_smiles, 0.0)
                        if xgb_prob > 0:
                            xgb_has_core = xgb_prob >= 0.5
                            xgb_probability = xgb_prob
                            break

            # 应用集成策略
            integrated_has_core = strategy_func(frag_has_core, smiles_has_core, xgb_has_core, xgb_probability)

            # 统计混淆矩阵
            if reference_has_core:
                if integrated_has_core:
                    tp += 1
                else:
                    fn += 1
            else:
                if integrated_has_core:
                    fp += 1
                else:
                    tn += 1

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

        print(f"   Top-{n}: P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}")

    return metrics

def evaluate_xgboost_only(frag_data, xgb_predictions, top_k_values):
    """评估XGBoost单独性能"""
    print(f"\n🔍 评估 XGBoost 单独模型...")

    metrics = {}
    total_samples = len(frag_data)

    for n in top_k_values:
        tp, fp, fn, tn = 0, 0, 0, 0

        for frag_item in frag_data:
            # 获取参考信息
            reference_has_core = frag_item.get('reference_analysis', {}).get('has_core_structure', False)
            true_smiles = frag_item.get('canonical_reference_smiles', '')

            # 检查XGBoost预测
            xgb_has_core = False
            if true_smiles and xgb_predictions:
                # 优先在外部测试集2中查找
                xgb_prob = xgb_predictions.get('external_test_2_predictions.json', {}).get(true_smiles, 0.0)
                if xgb_prob > 0:
                    xgb_has_core = xgb_prob >= 0.5
                elif True:  # 尝试在其他数据集中查找
                    for dataset_name in ['external_test_1_predictions.json', 'test_set_predictions.json', 'validation_set_predictions.json']:
                        xgb_prob = xgb_predictions.get(dataset_name, {}).get(true_smiles, 0.0)
                        if xgb_prob > 0:
                            xgb_has_core = xgb_prob >= 0.5
                            break

            # 统计混淆矩阵
            if reference_has_core:
                if xgb_has_core:
                    tp += 1
                else:
                    fn += 1
            else:
                if xgb_has_core:
                    fp += 1
                else:
                    tn += 1

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

        print(f"   Top-{n}: P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}")

    return metrics

def main():
    """主函数"""
    print("🚀 开始综合模型性能对比分析 (包含条件集成策略)...")

    # 加载生成模型数据
    frag_file = 'output/test_1218/test_frag_beams10_samples6410.json'
    smiles_file = 'output/test_1218/test_smiles_beams10_samples6410.json'

    print(f"📊 加载生成模型数据...")
    frag_data = load_json_data(frag_file)
    smiles_data = load_json_data(smiles_file)
    print(f"   FRAG数据: {len(frag_data)} 样本")
    print(f"   SMILES数据: {len(smiles_data)} 样本")

    # 加载XGBoost预测结果
    print(f"\n🔍 加载XGBoost预测结果...")
    xgb_predictions_dir = 'xgboost_predictions'
    all_xgb_predictions = load_xgboost_predictions(xgb_predictions_dir)

    # 定义Top-N值
    top_n_values = [1, 3, 5, 10, 50, 100]

    # 1. 评估单个模型
    frag_metrics = evaluate_single_model(frag_data, "FRAG", top_n_values)
    smiles_metrics = evaluate_single_model(smiles_data, "SMILES", top_n_values)

    # 2. 评估XGBoost单独
    xgb_metrics = evaluate_xgboost_only(frag_data, all_xgb_predictions, top_n_values)

    # 3. 定义集成策略
    def conditional_strategy(frag_has_core, smiles_has_core, xgb_has_core, xgb_probability):
        """条件集成策略: 生成模型一致时采用，不一致时用XGBoost判决"""
        if frag_has_core == smiles_has_core:
            # 两个生成模型预测一致，直接采用
            return frag_has_core
        else:
            # 两个生成模型预测不一致，用XGBoost判决
            return xgb_has_core  # 这里已经考虑了0.5阈值

    strategies = {
        "FRAG+SMILES (OR)": lambda frag, smiles, xgb, prob: frag or smiles,
        "FRAG+SMILES (AND)": lambda frag, smiles, xgb, prob: frag and smiles,
        "三模型投票 (≥2票)": lambda frag, smiles, xgb, prob: sum([frag, smiles, xgb]) >= 2,
        "条件集成策略": conditional_strategy
    }

    # 4. 评估集成策略
    integration_metrics = {}
    for strategy_name, strategy_func in strategies.items():
        integration_metrics[strategy_name] = evaluate_integration_strategy(
            frag_data, smiles_data, all_xgb_predictions, strategy_name, strategy_func, top_n_values
        )

    # 5. 汇总所有结果
    all_results = {
        "FRAG单独": frag_metrics,
        "SMILES单独": smiles_metrics,
        "XGBoost单独": xgb_metrics,
        **integration_metrics
    }

    # 保存完整结果
    output_file = 'comprehensive_model_comparison_with_conditional_results.json'
    results = {
        'comparison_strategies': list(all_results.keys()),
        'top_n_values': top_n_values,
        'results': all_results,
        'data_info': {
            'total_samples': len(frag_data),
            'positive_samples': sum(1 for item in frag_data if item.get('reference_analysis', {}).get('has_core_structure', False)),
            'negative_samples': sum(1 for item in frag_data if not item.get('reference_analysis', {}).get('has_core_structure', False))
        }
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 综合对比结果已保存到: {output_file}")

    # 6. 打印对比表格
    print(f"\n📊 综合性能对比表格:")
    print("=" * 160)

    for n in top_n_values:
        print(f"\n{'Top-' + str(n):<8} {'策略':<25} {'Precision':<12} {'Recall':<12} {'F1 Score':<12} {'Specificity':<12} {'Accuracy':<12} {'混淆矩阵'}")
        print("-" * 160)

        for strategy_name, metrics in all_results.items():
            if n in metrics:
                m = metrics[n]
                print(f"{'':<8} {strategy_name:<25} {m['precision']:<12.4f} {m['recall']:<12.4f} "
                      f"{m['f1']:<12.4f} {m['specificity']:<12.4f} {m['accuracy']:<12.4f} {m['confusion_matrix']}")

    # 7. 找出每个指标的最佳策略
    print(f"\n🏆 最佳性能策略:")
    print("=" * 80)

    for n in top_n_values:
        print(f"\nTop-{n} 最佳性能:")

        best_precision = max([(name, metrics[n]['precision']) for name, metrics in all_results.items() if n in metrics], key=lambda x: x[1])
        best_recall = max([(name, metrics[n]['recall']) for name, metrics in all_results.items() if n in metrics], key=lambda x: x[1])
        best_f1 = max([(name, metrics[n]['f1']) for name, metrics in all_results.items() if n in metrics], key=lambda x: x[1])

        print(f"  最高精确度: {best_precision[0]} ({best_precision[1]:.4f})")
        print(f"  最高召回率: {best_recall[0]} ({best_recall[1]:.4f})")
        print(f"  最高F1分数: {best_f1[0]} ({best_f1[1]:.4f})")

    # 8. 条件策略详细分析
    print(f"\n🔍 条件集成策略详细分析:")
    print("=" * 80)

    # 统计条件策略的决策分布
    consistent_true = 0
    consistent_false = 0
    inconsistent_use_xgb = 0
    total_samples = len(frag_data)

    for i, (frag_item, smiles_item) in enumerate(zip(frag_data, smiles_data)):
        true_smiles = frag_item.get('canonical_reference_smiles', '')

        # 获取预测结果
        frag_predictions = frag_item.get('predictions', [])
        smiles_predictions = smiles_item.get('predictions', [])

        # Top-1预测
        frag_has_core = any(pred.get('has_core_structure', False) for pred in frag_predictions[:1])
        smiles_has_core = any(pred.get('has_core_structure', False) for pred in smiles_predictions[:1])

        if frag_has_core == smiles_has_core:
            if frag_has_core:
                consistent_true += 1
            else:
                consistent_false += 1
        else:
            inconsistent_use_xgb += 1

    print(f"Top-1决策分布:")
    print(f"  生成模型一致为True: {consistent_true} ({consistent_true/total_samples*100:.1f}%)")
    print(f"  生成模型一致为False: {consistent_false} ({consistent_false/total_samples*100:.1f}%)")
    print(f"  生成模型不一致(使用XGBoost): {inconsistent_use_xgb} ({inconsistent_use_xgb/total_samples*100:.1f}%)")
    print(f"  总样本数: {total_samples}")

if __name__ == "__main__":
    main()
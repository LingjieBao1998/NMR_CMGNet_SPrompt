#!/usr/bin/env python3
"""
三模型投票集成推理系统
包含FRAG模型 + SMILES模型 + XGBoost模型
集成规则：三个模型中至少有两个预测结果与参考分子相同则为True
"""

import json
import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdSubstructLibrary
import pickle
import xgboost as xgb

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

def voting_three_models(frag_data, smiles_data, xgb_predictions, top_k_values=[1, 3, 5, 10, 50, 100]):
    """三模型投票集成推理"""
    print(f"🔄 开始三模型投票集成推理...")
    print(f"📊 集成策略: 三个模型中至少有两个预测结果与参考分子相同则为True")

    # 验证数据对齐
    if len(frag_data) != len(smiles_data):
        print(f"❌ 生成模型数据不匹配: FRAG {len(frag_data)} vs SMILES {len(smiles_data)}")
        return None

    total_samples = len(frag_data)
    print(f"📊 集成数据: {total_samples} 个样本")

    # 统计数据分布
    pos_samples = 0
    for item in frag_data:
        if item.get('reference_analysis', {}).get('has_core_structure', False):
            pos_samples += 1
    neg_samples = total_samples - pos_samples
    print(f"📈 数据分布: 正样本 {pos_samples}, 负样本 {neg_samples}")

    # 计算集成指标
    integrated_metrics = {}

    for n in top_k_values:
        tp, fp, fn, tn = 0, 0, 0, 0

        print(f"\n🔍 计算Top-{n}三模型投票集成指标...")

        for i, (frag_item, smiles_item) in enumerate(zip(frag_data, smiles_data)):
            if i % 1000 == 0:
                print(f"  处理进度: {i}/{total_samples}")

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

            # 尝试从多个XGBoost预测文件中找到匹配的SMILES
            if true_smiles and xgb_predictions:
                # 优先在外部测试集2中查找
                xgb_prob = xgb_predictions.get('external_test_2_predictions.json', {}).get(true_smiles, 0.0)
                if xgb_prob > 0:
                    xgb_has_core = xgb_prob >= 0.5
                    xgb_probability = xgb_prob
                elif True:  # 尝试在外部测试集1中查找
                    xgb_prob = xgb_predictions.get('external_test_1_predictions.json', {}).get(true_smiles, 0.0)
                    if xgb_prob > 0:
                        xgb_has_core = xgb_prob >= 0.5
                        xgb_probability = xgb_prob
                    elif True:  # 尝试在测试集中查找
                        xgb_prob = xgb_predictions.get('test_set_predictions.json', {}).get(true_smiles, 0.0)
                        if xgb_prob > 0:
                            xgb_has_core = xgb_prob >= 0.5
                            xgb_probability = xgb_prob
                        else:  # 最后尝试在验证集中查找
                            xgb_prob = xgb_predictions.get('validation_set_predictions.json', {}).get(true_smiles, 0.0)
                            xgb_has_core = xgb_prob >= 0.5
                            xgb_probability = xgb_prob

            # 三模型投票策略: 至少有两个预测结果与参考分子相同
            model_predictions = [frag_has_core, smiles_has_core, xgb_has_core]
            true_votes = sum(model_predictions)

            # 至少有两个模型预测为True，则集成预测为True
            integrated_has_core = true_votes >= 2

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
            'tn': tn,
            'confusion_matrix': f'({tp}, {fp}, {fn}, {tn})'
        }

        print(f"  Top-{n} 结果: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
        print(f"  Top-{n} 指标: P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}")

    return integrated_metrics

def main():
    """主函数"""
    print("🚀 开始三模型投票集成推理分析...")

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

    if not all_xgb_predictions:
        print(f"❌ 未找到任何XGBoost预测文件")
        return

    # 三模型投票分析
    top_n_values = [1, 3, 5, 10, 50, 100]
    integrated_metrics = voting_three_models(frag_data, smiles_data, all_xgb_predictions, top_n_values)

    if integrated_metrics is None:
        return

    # 保存集成结果
    output_file = 'three_models_voting_metrics.json'
    results = {
        'integration_strategy': '至少两个模型预测结果与参考分子相同则为True (>=2 votes)',
        'xgb_threshold': 0.5,
        'xgb_files_used': list(all_xgb_predictions.keys()),
        'top_n_metrics': integrated_metrics,
        'top_n_values': top_n_values,
        'data_info': {
            'total_samples': len(frag_data),
            'frag_samples': len(frag_data),
            'smiles_samples': len(smiles_data),
            'positive_samples': sum(1 for item in frag_data if item.get('reference_analysis', {}).get('has_core_structure', False)),
            'negative_samples': sum(1 for item in frag_data if not item.get('reference_analysis', {}).get('has_core_structure', False))
        }
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 三模型投票结果已保存到: {output_file}")

    # 打印详细结果表格
    print(f"\n📊 三模型投票集成性能指标:")
    print("=" * 120)
    print(f"{'Top-N':<8} {'Precision':<12} {'Recall':<12} {'F1 Score':<12} {'Specificity':<12} {'Accuracy':<12} {'混淆矩阵 (TP, FP, FN, TN)'}")
    print("-" * 120)

    for n in top_n_values:
        metrics = integrated_metrics[n]
        print(f"Top-{n:<5} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
              f"{metrics['f1']:<12.4f} {metrics['specificity']:<12.4f} {metrics['accuracy']:<12.4f} "
              f"({metrics['tp']}, {metrics['fp']}, {metrics['fn']}, {metrics['tn']})")

    # 对比分析
    print(f"\n🔍 投票集成策略分析:")
    print(f"   策略: 三个模型中至少有两个预测结果与参考分子相同则为True")
    print(f"   目标: 通过投票机制提高预测的稳定性和准确性")
    print(f"   XGBoost阈值: 0.5")
    print(f"   投票阈值: >= 2票")

    # 计算平均指标
    avg_precision = np.mean([integrated_metrics[n]['precision'] for n in top_n_values])
    avg_recall = np.mean([integrated_metrics[n]['recall'] for n in top_n_values])
    avg_f1 = np.mean([integrated_metrics[n]['f1'] for n in top_n_values])

    print(f"\n📈 平均性能指标:")
    print(f"   平均精确度: {avg_precision:.4f}")
    print(f"   平均召回率: {avg_recall:.4f}")
    print(f"   平均F1分数: {avg_f1:.4f}")

if __name__ == "__main__":
    main()
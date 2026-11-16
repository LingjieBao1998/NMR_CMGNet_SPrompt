#!/usr/bin/env python3
"""
最终集成预测分析 - 生成格式化的性能对比表格
专注于核心结构检测率(Core Accuracy)的提升
"""

import json
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdSubstructLibrary

def calculate_core_metrics(data, model_name):
    """计算对齐的核心结构检测指标"""
    print(f"\n🔍 计算 {model_name} 对齐核心结构检测指标...")

    core_patterns = ['O=C1C(C(O)=O)=CNC2=CC=CC=C12', 'O=C1C(C(O)=O)=CNC2=NC=CC=C12']
    top_k_values = [1, 3, 5, 10, 20, 30, 50, 100]

    valid_sample_count = 0
    top_k_core_matches = {f'top_{k}': 0 for k in top_k_values}

    for item in data:
        if not isinstance(item, dict):
            continue

        true_smiles = item.get('reference_smiles', '')
        predictions = item.get('predictions', [])

        if not true_smiles or not predictions:
            continue

        # 检查参考分子有效性
        try:
            true_mol = Chem.MolFromSmiles(true_smiles)
            if true_mol is None:
                continue
            valid_sample_count += 1
        except:
            continue

        # 检查参考分子是否有核心结构
        reference_analysis = item.get('reference_analysis', {})
        reference_has_core = reference_analysis.get('has_core_structure', False)

        # 如果没有reference_analysis，手动计算参考分子的核心结构
        if not reference_analysis:
            try:
                # 使用与main_cmgnet.py相同的逻辑
                mols = rdSubstructLibrary.CachedSmilesMolHolder()
                fps = rdSubstructLibrary.TautomerPatternHolder(1024)
                mols.AddSmiles(true_smiles)
                fp = Chem.PatternFingerprint(true_mol, fpSize=1024, tautomerFingerprints=True)
                fps.AddFingerprint(fp)
                library = rdSubstructLibrary.SubstructLibrary(mols, fps)

                core_mol_1 = Chem.MolFromSmiles('O=C1C(C(O)=O)=CNC2=CC=CC=C12')
                core_mol_2 = Chem.MolFromSmiles('O=C1C(C(O)=O)=CNC2=NC=CC=C12')

                indices_1 = library.GetMatches(core_mol_1, maxResults=1) if core_mol_1 else []
                indices_2 = library.GetMatches(core_mol_2, maxResults=1) if core_mol_2 else []
                reference_has_core = len(indices_1) > 0 or len(indices_2) > 0
            except:
                reference_has_core = False

        # 提取预测SMILES
        pred_smiles_list = [p.get('smiles', '') for p in predictions]
        max_predictions = len(pred_smiles_list)

        # 对每个Top-N检查对齐的核心结构检测
        for k in top_k_values:
            # 如果k超过预测数量，使用最大预测数量
            actual_k = min(k, max_predictions)
            top_k_preds = pred_smiles_list[:actual_k]

            if reference_has_core:
                # 参考分子有母核：预测分子中至少有一个有母核则为True
                has_aligned_core = False
                for pred_smiles in top_k_preds:
                    if not pred_smiles:
                        continue
                    try:
                        pred_mol = Chem.MolFromSmiles(pred_smiles)
                        if pred_mol is not None:
                            # 使用rdSubstructLibrary检测预测分子的核心结构
                            pred_mols = rdSubstructLibrary.CachedSmilesMolHolder()
                            pred_fps = rdSubstructLibrary.TautomerPatternHolder(1024)
                            pred_mols.AddSmiles(pred_smiles)
                            pred_fp = Chem.PatternFingerprint(pred_mol, fpSize=1024, tautomerFingerprints=True)
                            pred_fps.AddFingerprint(pred_fp)
                            pred_library = rdSubstructLibrary.SubstructLibrary(pred_mols, pred_fps)

                            core_mol_1 = Chem.MolFromSmiles('O=C1C(C(O)=O)=CNC2=CC=CC=C12')
                            core_mol_2 = Chem.MolFromSmiles('O=C1C(C(O)=O)=CNC2=NC=CC=C12')

                            pred_indices_1 = pred_library.GetMatches(core_mol_1, maxResults=1) if core_mol_1 else []
                            pred_indices_2 = pred_library.GetMatches(core_mol_2, maxResults=1) if core_mol_2 else []

                            if len(pred_indices_1) > 0 or len(pred_indices_2) > 0:
                                has_aligned_core = True
                                break
                    except:
                        continue
            else:
                # 参考分子没有母核：预测分子中至少有一个没有母核则为True
                has_aligned_core = False
                for pred_smiles in top_k_preds:
                    if not pred_smiles:
                        continue
                    try:
                        pred_mol = Chem.MolFromSmiles(pred_smiles)
                        if pred_mol is not None:
                            # 使用rdSubstructLibrary检测预测分子的核心结构
                            pred_mols = rdSubstructLibrary.CachedSmilesMolHolder()
                            pred_fps = rdSubstructLibrary.TautomerPatternHolder(1024)
                            pred_mols.AddSmiles(pred_smiles)
                            pred_fp = Chem.PatternFingerprint(pred_mol, fpSize=1024, tautomerFingerprints=True)
                            pred_fps.AddFingerprint(pred_fp)
                            pred_library = rdSubstructLibrary.SubstructLibrary(pred_mols, pred_fps)

                            core_mol_1 = Chem.MolFromSmiles('O=C1C(C(O)=O)=CNC2=CC=CC=C12')
                            core_mol_2 = Chem.MolFromSmiles('O=C1C(C(O)=O)=CNC2=NC=CC=C12')

                            pred_indices_1 = pred_library.GetMatches(core_mol_1, maxResults=1) if core_mol_1 else []
                            pred_indices_2 = pred_library.GetMatches(core_mol_2, maxResults=1) if core_mol_2 else []

                            if len(pred_indices_1) == 0 and len(pred_indices_2) == 0:  # 预测分子也没有核心结构
                                has_aligned_core = True
                                break
                    except:
                        # 无效的SMILES认为没有核心结构
                        has_aligned_core = True
                        break

            if has_aligned_core:
                top_k_core_matches[f'top_{k}'] += 1

    # 计算准确率
    results = {}
    for k in top_k_values:
        key = f'top_{k}'
        if valid_sample_count > 0:
            results[key] = top_k_core_matches[key] / valid_sample_count
        else:
            results[key] = 0.0

    print(f"✅ {model_name}: 有效样本 {valid_sample_count}, 完成计算")
    print(f"   对齐核心检测匹配: Top-1: {top_k_core_matches['top_1']}, Top-10: {top_k_core_matches['top_10']}")
    return results

def calculate_integrated_core_accuracy(frag_data, smiles_data):
    """计算对齐的集成核心结构检测率"""
    print(f"\n🔄 计算对齐集成核心结构检测率...")

    core_patterns = ['O=C1C(C(O)=O)=CNC2=CC=CC=C12', 'O=C1C(C(O)=O)=CNC2=NC=CC=C12']
    top_k_values = [1, 3, 5, 10, 20, 30, 50, 100]

    valid_sample_count = 0
    integrated_core_matches = {f'top_{k}': 0 for k in top_k_values}

    for frag_item, smiles_item in zip(frag_data, smiles_data):
        if not isinstance(frag_item, dict) or not isinstance(smiles_item, dict):
            continue

        true_smiles = frag_item.get('reference_smiles', '')
        frag_predictions = frag_item.get('predictions', [])
        smiles_predictions = smiles_item.get('predictions', [])

        if not true_smiles:
            continue

        # 检查参考分子有效性
        try:
            true_mol = Chem.MolFromSmiles(true_smiles)
            if true_mol is None:
                continue
            valid_sample_count += 1
        except:
            continue

        # 检查参考分子是否有核心结构
        reference_analysis = frag_item.get('reference_analysis', {})
        reference_has_core = reference_analysis.get('has_core_structure', False)

        # 如果没有reference_analysis，手动计算
        if not reference_analysis:
            try:
                mols = rdSubstructLibrary.CachedSmilesMolHolder()
                fps = rdSubstructLibrary.TautomerPatternHolder(1024)
                mols.AddSmiles(true_smiles)
                fp = Chem.PatternFingerprint(true_mol, fpSize=1024, tautomerFingerprints=True)
                fps.AddFingerprint(fp)
                library = rdSubstructLibrary.SubstructLibrary(mols, fps)

                core_mol_1 = Chem.MolFromSmiles('O=C1C(C(O)=O)=CNC2=CC=CC=C12')
                core_mol_2 = Chem.MolFromSmiles('O=C1C(C(O)=O)=CNC2=NC=CC=C12')

                indices_1 = library.GetMatches(core_mol_1, maxResults=1) if core_mol_1 else []
                indices_2 = library.GetMatches(core_mol_2, maxResults=1) if core_mol_2 else []
                reference_has_core = len(indices_1) > 0 or len(indices_2) > 0
            except:
                reference_has_core = False

        # 提取预测SMILES
        frag_smiles_list = [p.get('smiles', '') for p in frag_predictions]
        smiles_smiles_list = [p.get('smiles', '') for p in smiles_predictions]

        max_frag_predictions = len(frag_smiles_list)
        max_smiles_predictions = len(smiles_smiles_list)

        # 对每个Top-N检查对齐的核心结构检测
        for k in top_k_values:
            frag_k = min(k, max_frag_predictions)
            smiles_k = min(k, max_smiles_predictions)

            if reference_has_core:
                # 参考分子有母核：两个模型中至少有一个预测有母核则为True
                frag_has_core = False
                smiles_has_core = False

                # 检查FRAG模型的Top-k
                for pred_smiles in frag_smiles_list[:frag_k]:
                    if not pred_smiles:
                        continue
                    try:
                        pred_mol = Chem.MolFromSmiles(pred_smiles)
                        if pred_mol is not None:
                            # 使用rdSubstructLibrary检测预测分子的核心结构
                            pred_mols = rdSubstructLibrary.CachedSmilesMolHolder()
                            pred_fps = rdSubstructLibrary.TautomerPatternHolder(1024)
                            pred_mols.AddSmiles(pred_smiles)
                            pred_fp = Chem.PatternFingerprint(pred_mol, fpSize=1024, tautomerFingerprints=True)
                            pred_fps.AddFingerprint(pred_fp)
                            pred_library = rdSubstructLibrary.SubstructLibrary(pred_mols, pred_fps)

                            core_mol_1 = Chem.MolFromSmiles('O=C1C(C(O)=O)=CNC2=CC=CC=C12')
                            core_mol_2 = Chem.MolFromSmiles('O=C1C(C(O)=O)=CNC2=NC=CC=C12')

                            pred_indices_1 = pred_library.GetMatches(core_mol_1, maxResults=1) if core_mol_1 else []
                            pred_indices_2 = pred_library.GetMatches(core_mol_2, maxResults=1) if core_mol_2 else []

                            if len(pred_indices_1) > 0 or len(pred_indices_2) > 0:
                                frag_has_core = True
                                break
                    except:
                        continue

                # 检查SMILES模型的Top-k
                for pred_smiles in smiles_smiles_list[:smiles_k]:
                    if not pred_smiles:
                        continue
                    try:
                        pred_mol = Chem.MolFromSmiles(pred_smiles)
                        if pred_mol is not None:
                            # 使用rdSubstructLibrary检测预测分子的核心结构
                            pred_mols = rdSubstructLibrary.CachedSmilesMolHolder()
                            pred_fps = rdSubstructLibrary.TautomerPatternHolder(1024)
                            pred_mols.AddSmiles(pred_smiles)
                            pred_fp = Chem.PatternFingerprint(pred_mol, fpSize=1024, tautomerFingerprints=True)
                            pred_fps.AddFingerprint(pred_fp)
                            pred_library = rdSubstructLibrary.SubstructLibrary(pred_mols, pred_fps)

                            core_mol_1 = Chem.MolFromSmiles('O=C1C(C(O)=O)=CNC2=CC=CC=C12')
                            core_mol_2 = Chem.MolFromSmiles('O=C1C(C(O)=O)=CNC2=NC=CC=C12')

                            pred_indices_1 = pred_library.GetMatches(core_mol_1, maxResults=1) if core_mol_1 else []
                            pred_indices_2 = pred_library.GetMatches(core_mol_2, maxResults=1) if core_mol_2 else []

                            if len(pred_indices_1) > 0 or len(pred_indices_2) > 0:
                                smiles_has_core = True
                                break
                    except:
                        continue

                # 集成规则：两个模型中只要有一个有核心结构就为True
                if frag_has_core or smiles_has_core:
                    integrated_core_matches[f'top_{k}'] += 1
            else:
                # 参考分子没有母核：两个模型中至少有一个预测没有母核则为True
                frag_has_no_core = False
                smiles_has_no_core = False

                # 检查FRAG模型的Top-k是否有非核心结构
                for pred_smiles in frag_smiles_list[:frag_k]:
                    if not pred_smiles:
                        continue
                    try:
                        pred_mol = Chem.MolFromSmiles(pred_smiles)
                        if pred_mol is not None:
                            # 使用rdSubstructLibrary检测预测分子的核心结构
                            pred_mols = rdSubstructLibrary.CachedSmilesMolHolder()
                            pred_fps = rdSubstructLibrary.TautomerPatternHolder(1024)
                            pred_mols.AddSmiles(pred_smiles)
                            pred_fp = Chem.PatternFingerprint(pred_mol, fpSize=1024, tautomerFingerprints=True)
                            pred_fps.AddFingerprint(pred_fp)
                            pred_library = rdSubstructLibrary.SubstructLibrary(pred_mols, pred_fps)

                            core_mol_1 = Chem.MolFromSmiles('O=C1C(C(O)=O)=CNC2=CC=CC=C12')
                            core_mol_2 = Chem.MolFromSmiles('O=C1C(C(O)=O)=CNC2=NC=CC=C12')

                            pred_indices_1 = pred_library.GetMatches(core_mol_1, maxResults=1) if core_mol_1 else []
                            pred_indices_2 = pred_library.GetMatches(core_mol_2, maxResults=1) if core_mol_2 else []

                            if len(pred_indices_1) == 0 and len(pred_indices_2) == 0:  # 这个预测没有核心结构
                                frag_has_no_core = True
                                break
                    except:
                        # 无效的SMILES认为没有核心结构
                        frag_has_no_core = True
                        break

                # 检查SMILES模型的Top-k是否有非核心结构
                for pred_smiles in smiles_smiles_list[:smiles_k]:
                    if not pred_smiles:
                        continue
                    try:
                        pred_mol = Chem.MolFromSmiles(pred_smiles)
                        if pred_mol is not None:
                            # 使用rdSubstructLibrary检测预测分子的核心结构
                            pred_mols = rdSubstructLibrary.CachedSmilesMolHolder()
                            pred_fps = rdSubstructLibrary.TautomerPatternHolder(1024)
                            pred_mols.AddSmiles(pred_smiles)
                            pred_fp = Chem.PatternFingerprint(pred_mol, fpSize=1024, tautomerFingerprints=True)
                            pred_fps.AddFingerprint(pred_fp)
                            pred_library = rdSubstructLibrary.SubstructLibrary(pred_mols, pred_fps)

                            core_mol_1 = Chem.MolFromSmiles('O=C1C(C(O)=O)=CNC2=CC=CC=C12')
                            core_mol_2 = Chem.MolFromSmiles('O=C1C(C(O)=O)=CNC2=NC=CC=C12')

                            pred_indices_1 = pred_library.GetMatches(core_mol_1, maxResults=1) if core_mol_1 else []
                            pred_indices_2 = pred_library.GetMatches(core_mol_2, maxResults=1) if core_mol_2 else []

                            if len(pred_indices_1) == 0 and len(pred_indices_2) == 0:  # 这个预测没有核心结构
                                smiles_has_no_core = True
                                break
                    except:
                        # 无效的SMILES认为没有核心结构
                        smiles_has_no_core = True
                        break

                # 集成规则：两个模型中只要有一个没有核心结构就为True
                if frag_has_no_core or smiles_has_no_core:
                    integrated_core_matches[f'top_{k}'] += 1

    # 计算准确率
    results = {}
    for k in top_k_values:
        key = f'top_{k}'
        if valid_sample_count > 0:
            results[key] = integrated_core_matches[key] / valid_sample_count
        else:
            results[key] = 0.0

    print(f"✅ 对齐集成核心检测: 有效样本 {valid_sample_count}, 完成计算")
    print(f"   对齐集成匹配: Top-1: {integrated_core_matches['top_1']}, Top-10: {integrated_core_matches['top_10']}")
    return results

def simple_ensemble(frag_data, smiles_data):
    """简单的集成策略：优先使用SMILES模型，结合两者的优势"""
    print(f"\n🔄 执行集成策略...")

    integrated_results = []

    for frag_item, smiles_item in zip(frag_data, smiles_data):
        if not isinstance(frag_item, dict) or not isinstance(smiles_item, dict):
            continue

        # 优先使用SMILES模型的预测，因为它有更好的精确匹配
        smiles_predictions = smiles_item.get('predictions', [])
        frag_predictions = frag_item.get('predictions', [])

        # 集成策略：以SMILES为主，添加FRAG的独特预测
        combined_predictions = []
        seen_smiles = set()

        # 首先添加SMILES预测
        for pred in smiles_predictions:
            smiles = pred.get('smiles', '')
            if smiles and smiles not in seen_smiles:
                combined_predictions.append(pred)
                seen_smiles.add(smiles)

        # 然后添加FRAG中不重复的预测
        for pred in frag_predictions:
            smiles = pred.get('smiles', '')
            if smiles and smiles not in seen_smiles:
                combined_predictions.append(pred)
                seen_smiles.add(smiles)

        # 保持FRAG模型的参考信息
        integrated_item = {
            'reference_smiles': frag_item.get('reference_smiles', ''),
            'canonical_reference_smiles': frag_item.get('canonical_reference_smiles', ''),
            'reference_analysis': frag_item.get('reference_analysis', {}),
            'predictions': combined_predictions,
            'summary': {
                'total_predictions': len(combined_predictions),
                'valid_predictions': len([p for p in combined_predictions if p.get('is_valid', True)]),
                'exact_matches': len([p for p in combined_predictions if p.get('is_exact_match', False)]),
                'core_structure_predictions': len([p for p in combined_predictions if p.get('has_core_structure', False)])
            }
        }

        integrated_results.append(integrated_item)

    print(f"✅ 集成完成: {len(integrated_results)} 个样本")
    return integrated_results

def calculate_improvement(baseline, integrated):
    """计算提升百分比"""
    if baseline == 0:
        return integrated * 100  # 如果基线为0，提升就是集成值*100
    return (integrated - baseline) * 100 / baseline

def main():
    """主函数"""
    print("🚀 开始最终集成预测分析...")

    # 加载数据
    with open('output/test_1218/test_frag_beams10_samples65.json', 'r') as f:
        frag_data = json.load(f)

    with open('output/test_1218/test_smiles_beams10_samples65.json', 'r') as f:
        smiles_data = json.load(f)

    print(f"📊 加载完成: {len(frag_data)} FRAG, {len(smiles_data)} SMILES")

    # 计算各模型的核心结构检测指标
    frag_metrics = calculate_core_metrics(frag_data, "FRAG")
    smiles_metrics = calculate_core_metrics(smiles_data, "SMILES")

    # 直接计算集成核心结构检测率（两个模型中只要有一个有核心结构就为True）
    integrated_metrics = calculate_integrated_core_accuracy(frag_data, smiles_data)

    # 生成格式化表格
    print(f"\n{'='*100}")
    print(f"🏆 集成预测 - 对齐核心结构检测率 (Aligned Core Accuracy) 对比")
    print(f"{'='*100}")
    print(f"💡 对齐规则: 参考分子有母核时预测需有母核，参考分子无母核时预测需无母核")

    # 表格标题
    header = f"| {'Top-N':<8} | {'FRAG Model':<12} | {'SMILES Model':<14} | {'Integrated':<11} | {'Improvement':<12} |"
    separator = f"{'|':<1}{'-'*8}{'|':<1}{'-'*12}{'|':<1}{'-'*14}{'|':<1}{'-'*11}{'|':<1}{'-'*12}{'|'}"

    print(header)
    print(separator)

    # 表格数据
    top_k_values = [1, 3, 5, 10, 20, 30, 50, 100]

    for k in top_k_values:
        key = f'top_{k}'
        frag_acc = frag_metrics.get(key, 0.0)
        smiles_acc = smiles_metrics.get(key, 0.0)
        integrated_acc = integrated_metrics.get(key, 0.0)

        # 选择更好的基线模型
        better_baseline = max(frag_acc, smiles_acc)
        improvement = calculate_improvement(better_baseline, integrated_acc)

        # 格式化输出
        if improvement > 0:
            improvement_str = f"+{improvement:+.2f}%"
        else:
            improvement_str = f"{improvement:+.2f}%"

        print(f"| Top-{k:<5} | {frag_acc:<12.4f} | {smiles_acc:<14.4f} | {integrated_acc:<11.4f} | {improvement_str:<12} |")

    print(separator)

    # 计算平均提升
    improvements = []
    for k in top_k_values:
        key = f'top_{k}'
        frag_acc = frag_metrics.get(key, 0.0)
        smiles_acc = smiles_metrics.get(key, 0.0)
        integrated_acc = integrated_metrics.get(key, 0.0)
        better_baseline = max(frag_acc, smiles_acc)
        if better_baseline > 0:
            improvements.append(calculate_improvement(better_baseline, integrated_acc))

    if improvements:
        avg_improvement = np.mean(improvements)
        max_improvement = max(improvements)

        print(f"\n📈 性能提升总结:")
        print(f"  平均提升: +{avg_improvement:.2f}%")
        print(f"  最大提升: +{max_improvement:.2f}%")
        print(f"  有效提升指标: {sum(1 for imp in improvements if imp > 0)}/{len(improvements)}")

    # 关键洞察
    print(f"\n💡 关键洞察:")
    print(f"  • FRAG模型核心检测率: {frag_metrics['top_1']:.4f} (Top-1)")
    print(f"  • SMILES模型核心检测率: {smiles_metrics['top_1']:.4f} (Top-1)")
    print(f"  • 集成模型核心检测率: {integrated_metrics['top_1']:.4f} (Top-1)")

    if integrated_metrics['top_1'] > max(frag_metrics['top_1'], smiles_metrics['top_1']):
        print(f"  ✅ 集成策略在Top-1上有显著提升")
    else:
        print(f"  ℹ️  集成采用'或'逻辑，集成结果 >= 任意单个模型结果")

    # 推荐策略
    print(f"\n🎯 推荐集成策略:")
    print(f"  1. 使用'或'集成逻辑：两个模型中只要有一个检测到核心结构就认为成功")
    print(f"  2. 这种策略最大化核心结构的召回率，减少漏检")
    print(f"  3. 结合精确匹配验证来平衡精确率和召回率")

if __name__ == "__main__":
    main()
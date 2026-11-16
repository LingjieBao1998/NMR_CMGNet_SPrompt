#!/usr/bin/env python3
"""
4子图可视化：包含TMAP图和Wasserstein分析

子图布局：
1. 左上：匈牙利匹配的Wasserstein距离分布
2. 右上：最小值匹配的化学位移差异分布
3. 左下：TMAP可视化（合成 vs 实验数据）
4. 右下：最小值匹配的累积分布函数(CDF)
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from scipy.stats import wasserstein_distance
from collections import defaultdict


def load_json_data(json_path):
    """加载JSON文件"""
    print(f"Loading {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    print(f"  Total samples: {len(data)}")
    return data


def match_molecules(experimental_data, synthetic_data):
    """通过cano_smiles匹配分子对"""
    experimental_dict = {item.get('cano_smiles'): item for item in experimental_data if item.get('cano_smiles')}
    synthetic_dict = {item.get('cano_smiles'): item for item in synthetic_data if item.get('cano_smiles')}

    common_smiles = set(experimental_dict.keys()) & set(synthetic_dict.keys())
    print(f"\n匹配的分子数量: {len(common_smiles)}")

    pairs = []
    for smiles in common_smiles:
        exp_item = experimental_dict[smiles]
        syn_item = synthetic_dict[smiles]
        pairs.append({
            'smiles': smiles,
            'experimental_nmr': exp_item.get('C_nmr', exp_item.get('NMR', [])),
            'synthetic_nmr': syn_item.get('C_nmr', syn_item.get('NMR', []))
        })

    return pairs


def calculate_wasserstein_distance(experimental_nmr, synthetic_nmr):
    """计算两个NMR谱之间的Wasserstein距离"""
    if len(experimental_nmr) == 0 or len(synthetic_nmr) == 0:
        return None

    # 使用try-except处理可能的错误
    try:
        distance = wasserstein_distance(experimental_nmr, synthetic_nmr)
        return distance
    except:
        return None


def hungarian_match_nmr(experimental_nmr, synthetic_nmr):
    """
    匈牙利匹配算法：寻找最优的一对一匹配

    这里简化实现：对所有可能的配对计算距离矩阵，然后找最优匹配
    """
    if len(experimental_nmr) == 0 or len(synthetic_nmr) == 0:
        return [], []

    from scipy.optimize import linear_sum_assignment

    # 构建距离矩阵
    n_exp = len(experimental_nmr)
    n_syn = len(synthetic_nmr)

    # 如果某一边的点数更多，需要补齐
    max_size = max(n_exp, n_syn)
    cost_matrix = np.zeros((max_size, max_size))

    # 填充距离矩阵
    for i, exp_val in enumerate(experimental_nmr):
        for j, syn_val in enumerate(synthetic_nmr):
            cost_matrix[i, j] = abs(exp_val - syn_val)

    # 对于不匹配的点，给一个很大的惩罚值
    penalty = 1000.0
    for i in range(n_exp, max_size):
        for j in range(n_syn, max_size):
            cost_matrix[i, j] = 0  # 两个虚拟点匹配，成本为0
    for i in range(n_exp, max_size):
        for j in range(n_syn):
            cost_matrix[i, j] = penalty  # 实验点匹配虚拟预测点
    for i in range(n_exp):
        for j in range(n_syn, max_size):
            cost_matrix[i, j] = penalty  # 预测点匹配虚拟实验点

    # 使用匈牙利算法
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # 提取匹配对
    matched_exp = []
    matched_syn = []
    total_diff = []

    for i, j in zip(row_ind, col_ind):
        if i < n_exp and j < n_syn:  # 只统计真实匹配
            matched_exp.append(experimental_nmr[i])
            matched_syn.append(synthetic_nmr[j])
            diff = abs(experimental_nmr[i] - synthetic_nmr[j])
            total_diff.append(diff)

    return matched_exp, matched_syn, total_diff


def min_value_match_nmr(experimental_nmr, synthetic_nmr):
    """
    最小值匹配：对于每个实验值，找到最接近的预测值
    （简单、快速，但不是全局最优）
    """
    if len(experimental_nmr) == 0 or len(synthetic_nmr) == 0:
        return [], []

    matched_pairs = []
    total_diff = []

    for exp_val in experimental_nmr:
        # 找到最接近的预测值
        min_diff = float('inf')
        closest_syn = None

        for syn_val in synthetic_nmr:
            diff = abs(exp_val - syn_val)
            if diff < min_diff:
                min_diff = diff
                closest_syn = syn_val

        matched_pairs.append((exp_val, closest_syn))
        total_diff.append(min_diff)

    return matched_pairs, total_diff


def compute_statistics(data_list):
    """计算统计信息"""
    data_array = np.array(data_list)
    return {
        'mean': np.mean(data_array),
        'median': np.median(data_array),
        'std': np.std(data_array),
        'min': np.min(data_array),
        'max': np.max(data_array)
    }


def visualize_4plots_with_tmap(hungarian_distances, min_match_distances,
                               hungarian_diffs, min_match_diffs,
                               results_min=None,
                               tmap_img_path="/homeb/baolingjie/git_download/NMR_Core_Mol_Detection/output/tmap_minhash_sampled_500k/tmap_minhash_synthetic_experimental_v3_clean.png",
                               save_path=None):
    """
    4子图可视化：包含TMAP图和Wasserstein分析

    Args:
        hungarian_distances: 匈牙利匹配的Wasserstein距离列表
        min_match_distances: 最小值匹配的Wasserstein距离列表
        hungarian_diffs: 匈牙利匹配的化学位移差异列表的列表
        min_match_diffs: 最小值匹配的化学位移差异列表的列表
        results_min: 用于NMR示例的结果数据
        tmap_img_path: TMAP图片路径
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 展平所有差异
    hungarian_all_diffs = []
    for diffs in hungarian_diffs:
        hungarian_all_diffs.extend(diffs)
    hungarian_all_diffs = np.abs(np.array(hungarian_all_diffs))

    min_match_all_diffs = []
    for diffs in min_match_diffs:
        min_match_all_diffs.extend(diffs)
    min_match_all_diffs = np.abs(np.array(min_match_all_diffs))

    # === 左上：Wasserstein距离分布（仅匈牙利匹配） ===
    ax = axes[0, 0]
    ax.hist(hungarian_distances, bins=20, color='lightblue', alpha=0.7,
            label='Hungarian Matching', edgecolor='black')
    ax.axvline(x=np.mean(hungarian_distances), color='blue', linestyle='--',
               linewidth=2, label=f'Mean: {np.mean(hungarian_distances):.2f} ppm')
    ax.axvline(x=np.median(hungarian_distances), color='green', linestyle='--',
               linewidth=2, label=f'Median: {np.median(hungarian_distances):.2f} ppm')
    ax.set_xlabel('Wasserstein Distance (ppm)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('(a) Hungarian Matching: Wasserstein Distance', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)  # 增大图例
    ax.grid(True, alpha=0.3)

    # === 右上：最小值匹配的化学位移差异分布 ===
    ax = axes[0, 1]
    ax.hist(min_match_all_diffs, bins=50, color='lightcoral', alpha=0.7,
            edgecolor='black')
    ax.axvline(x=np.mean(min_match_all_diffs), color='red', linestyle='--',
               linewidth=2, label=f'Mean: {np.mean(min_match_all_diffs):.2f} ppm')
    ax.axvline(x=np.median(min_match_all_diffs), color='green', linestyle='--',
               linewidth=2, label=f'Median: {np.median(min_match_all_diffs):.2f} ppm')
    ax.axvline(x=5, color='orange', linestyle=':', linewidth=2, label='5 ppm threshold')
    ax.set_xlabel('Absolute Difference |Exp - Syn| (ppm)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('(b) Min-value Matching: Single Chemistry Shift Differences', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)  # 增大图例
    ax.grid(True, alpha=0.3)

    # === 左下：TMAP可视化（合成 vs 实验数据） ===
    ax = axes[1, 0]

    # 尝试加载TMAP图片
    try:
        if os.path.exists(tmap_img_path):
            tmap_img = imread(tmap_img_path)
            ax.imshow(tmap_img)
            ax.axis('off')  # 关闭坐标轴
            ax.set_title('(c) TMAP: Synthetic vs Experimental', fontsize=14, fontweight='bold')
        else:
            # 如果图片不存在，显示提示文本
            ax.text(0.5, 0.5, f'TMAP image not found:\n{tmap_img_path}',
                   ha='center', va='center', fontsize=10, color='red',
                   transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title('(c) TMAP Visualization', fontsize=14, fontweight='bold')
    except Exception as e:
        # 如果加载失败，显示错误信息
        ax.text(0.5, 0.5, f'Error loading TMAP:\n{str(e)}',
               ha='center', va='center', fontsize=10, color='red',
               transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('(c) TMAP Visualization', fontsize=14, fontweight='bold')

    # === 右下：NMR谱图对比示例 ===
    ax = axes[1, 1]

    # 获取一个代表性的分子进行展示
    exp_nmr_example = None
    syn_nmr_example = None

    if results_min is not None and len(results_min) > 0:
        # 找一个有足够峰的分子
        for result in results_min:
            if len(result['experimental_nmr']) >= 5 and len(result['synthetic_nmr']) >= 5:
                exp_nmr_example = result['experimental_nmr']
                syn_nmr_example = result['synthetic_nmr']
                break

        # 如果没找到合适的，用第一个
        if exp_nmr_example is None:
            exp_nmr_example = results_min[0]['experimental_nmr']
            syn_nmr_example = results_min[0]['synthetic_nmr']

    # 如果仍然没有，使用示例数据
    if exp_nmr_example is None or len(exp_nmr_example) == 0:
        exp_nmr_example = [140.5, 128.3, 128.1, 127.9, 60.2, 40.1, 25.3, 22.1]
        syn_nmr_example = [142.1, 130.5, 129.8, 129.5, 128.9, 61.5, 42.3, 26.8, 23.5]

    # 绘制NMR谱图
    all_shifts = exp_nmr_example + syn_nmr_example
    shift_max = max(all_shifts) + 10
    shift_min = min(all_shifts) - 10

    # 实验数据（坐标轴之上，蓝色）
    for shift in exp_nmr_example:
        ax.plot([shift, shift], [0, 1], 'b-', linewidth=2)
        ax.plot(shift, 1, 'bo', markersize=6)

    # 合成数据（坐标轴之下，红色）
    for shift in syn_nmr_example:
        ax.plot([shift, shift], [0, -1], 'r-', linewidth=2)
        ax.plot(shift, -1, 'rs', markersize=6)

    ax.axhline(y=0, color='k', linestyle='-', linewidth=1)
    ax.set_xlim(shift_max, shift_min)  # NMR习惯：从右到左
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel('Chemical Shift (ppm)', fontsize=12, fontweight='bold')
    ax.set_yticks([])
    ax.set_title('(d) Experiment vs Synthetic NMR', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    # 添加图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='b', marker='o', label='Experimental', linewidth=2, markersize=6),
        Line2D([0], [0], color='r', marker='s', label='Synthetic', linewidth=2, markersize=6)
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)  # 增大图例

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 4子图对比图已保存: {save_path}")

    plt.show()

    # === 打印统计结果 ===
    print("\n" + "="*80)
    print("统计对比结果")
    print("="*80)
    print(f"{'Metric':<25} {'Hungarian':>15} {'Min-value':>15} {'Difference':>15}")
    print("-" * 80)

    print(f"{'Wasserstein Mean':<25} {np.mean(hungarian_distances):>15.3f} "
          f"{np.mean(min_match_distances):>15.3f} {np.mean(min_match_distances) - np.mean(hungarian_distances):>15.3f}")
    print(f"{'Wasserstein Median':<25} {np.median(hungarian_distances):>15.3f} "
          f"{np.median(min_match_distances):>15.3f} {np.median(min_match_distances) - np.median(hungarian_distances):>15.3f}")
    print(f"{'Shift Diff Mean':<25} {np.mean(hungarian_all_diffs):>15.3f} "
          f"{np.mean(min_match_all_diffs):>15.3f} {np.mean(min_match_all_diffs) - np.mean(hungarian_all_diffs):>15.3f}")
    print(f"{'Shift Diff Median':<25} {np.median(hungarian_all_diffs):>15.3f} "
          f"{np.median(min_match_all_diffs):>15.3f} {np.median(min_match_all_diffs) - np.median(hungarian_all_diffs):>15.3f}")
    print(f"{'< 5 ppm (%)':<25} {np.mean(hungarian_all_diffs < 5)*100:>15.1f}% "
          f"{np.mean(min_match_all_diffs < 5)*100:>15.1f}% "
          f"{(np.mean(min_match_all_diffs < 5) - np.mean(hungarian_all_diffs < 5))*100:>15.1f}%")
    print(f"{'< 3 ppm (%)':<25} {np.mean(hungarian_all_diffs < 3)*100:>15.1f}% "
          f"{np.mean(min_match_all_diffs < 3)*100:>15.1f}% "
          f"{(np.mean(min_match_all_diffs < 3) - np.mean(hungarian_all_diffs < 3))*100:>15.1f}%")
    print(f"{'< 1 ppm (%)':<25} {np.mean(hungarian_all_diffs < 1)*100:>15.1f}% "
          f"{np.mean(min_match_all_diffs < 1)*100:>15.1f}% "
          f"{(np.mean(min_match_all_diffs < 1) - np.mean(hungarian_all_diffs < 1))*100:>15.1f}%")

    print("\n" + "="*80)


def load_existing_results():
    """从现有的结果文件加载数据"""
    print("加载现有的Wasserstein分析结果...")

    # 1. 加载匈牙利匹配结果
    hungarian_file = "nmr_wasserstein_distances_with_matching.json"
    min_match_file = "nmr_wasserstein_distances_min_matching.json"

    if not os.path.exists(hungarian_file) or not os.path.exists(min_match_file):
        print(f"未找到结果文件:")
        print(f"  {hungarian_file}: {'✓' if os.path.exists(hungarian_file) else '✗'}")
        print(f"  {min_match_file}: {'✓' if os.path.exists(min_match_file) else '✗'}")
        return None, None, None, None, None

    # 加载匈牙利匹配结果
    with open(hungarian_file, 'r') as f:
        hungarian_data = json.load(f)

    # 加载最小值匹配结果
    with open(min_match_file, 'r') as f:
        min_match_data = json.load(f)

    # 提取数据
    hungarian_distances = []
    hungarian_diffs = []

    for mol in hungarian_data.get('molecules', []):
        if 'distance' in mol:
            hungarian_distances.append(mol['distance'])
        if 'match_result' in mol and 'differences' in mol['match_result']:
            hungarian_diffs.append(mol['match_result']['differences'])

    min_match_distances = []
    min_match_diffs = []
    results_min = []

    for mol in min_match_data.get('molecules', []):
        result_entry = {
            'experimental_nmr': mol.get('experimental_nmr', []),
            'synthetic_nmr': mol.get('synthetic_nmr', [])
        }
        results_min.append(result_entry)

        if 'distance' in mol:
            min_match_distances.append(mol['distance'])
        if 'match_result' in mol and 'differences' in mol['match_result']:
            min_match_diffs.append(mol['match_result']['differences'])

    print(f"✓ 匈牙利匹配: {len(hungarian_distances)} 个分子")
    print(f"✓ 最小值匹配: {len(min_match_distances)} 个分子")
    print(f"✓ NMR示例数据: {len(results_min)} 个分子")

    return hungarian_distances, min_match_distances, hungarian_diffs, min_match_diffs, results_min


def main():
    """主函数"""
    print("="*80)
    print("4子图可视化：TMAP + Wasserstein分析")
    print("="*80)

    # 从现有结果文件加载
    hungarian_distances, min_match_distances, hungarian_diffs, min_match_diffs, results_min = load_existing_results()

    if hungarian_distances is None:
        print("\n请先运行Wasserstein分析生成结果文件")
        return

    # 绘制4子图
    print("\n绘制4子图...")
    visualize_4plots_with_tmap(
        hungarian_distances,
        min_match_distances,
        hungarian_diffs,
        min_match_diffs,
        results_min=results_min,
        tmap_img_path="/homeb/baolingjie/git_download/NMR_Core_Mol_Detection/output/tmap_minhash_sampled_500k/tmap_minhash_synthetic_experimental_v3_clean.png",
        save_path="4plots_with_tmap.png"
    )

    print("\n✅ 完成!")


if __name__ == "__main__":
    main()

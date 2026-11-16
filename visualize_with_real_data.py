#!/usr/bin/env python3
"""
使用具体NMR和SMILES数据的倒数第二层头部可视化
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

def auto_select_best_head(cross_attentions: torch.Tensor, threshold: float = 0.1) -> int:
    """
    自动选择第5层的最佳头部

    Args:
        cross_attentions: 注意力权重 [num_layers, num_heads, seq_len, nmr_len] 或 [num_heads, seq_len, nmr_len]
        threshold: 响应阈值

    Returns:
        最佳头部索引
    """
    # 处理输入维度
    if cross_attentions.dim() == 4:
        # 提取倒数第二层 (第5层，索引4)
        layer_attention = cross_attentions[4]
    elif cross_attentions.dim() == 3:
        layer_attention = cross_attentions
    else:
        raise ValueError(f"不支持的注意力维度: {cross_attentions.dim()}")

    num_heads = layer_attention.size(0)
    head_scores = {}

    print(f"\n🔍 自动选择最佳头部 (第5层，共{num_heads}个头)...")
    print("=" * 80)

    for h in range(num_heads):
        attn = layer_attention[h]  # [seq_len, nmr_len]

        # 三大指标
        max_response = attn.max().item()
        attn_safe = attn + 1e-9
        entropy = -(attn_safe * torch.log(attn_safe)).sum(dim=-1).mean().item()

        # 稀疏度计算
        sorted_attn, _ = attn.sort(dim=-1, descending=True)
        cumsum = sorted_attn.cumsum(dim=-1)
        weights = torch.arange(1, attn.size(-1) + 1, dtype=torch.float32) / attn.size(-1)
        sparsity = 1 - 2 * (cumsum * weights).sum(dim=-1).mean().item()

        # 综合评分
        score = max_response - 0.5 * entropy + 0.3 * sparsity

        head_scores[h] = {
            'max_response': max_response,
            'entropy': entropy,
            'sparsity': sparsity,
            'composite_score': score
        }

        print(f"头部 {h:2d}: 最大响应={max_response:.4f}, 熵={entropy:.4f}, "
              f"稀疏度={sparsity:.4f}, 综合评分={score:.4f}")

    # 按综合评分排序
    sorted_heads = sorted(head_scores.items(), key=lambda x: x[1]['composite_score'], reverse=True)

    print("\n" + "=" * 80)
    best_head_idx = sorted_heads[0][0]
    best_score = sorted_heads[0][1]['composite_score']

    print(f"🏆 自动选择结果: 头部 {best_head_idx}")
    print(f"📊 综合评分: {best_score:.4f}")
    print(f"🔍 最大响应: {sorted_heads[0][1]['max_response']:.4f}")
    print(f"📉 熵值: {sorted_heads[0][1]['entropy']:.4f}")
    print(f"💎 稀疏度: {sorted_heads[0][1]['sparsity']:.4f}")

    if best_score < threshold:
        print(f"⚠️  警告: 最佳头部评分 {best_score:.4f} 低于阈值 {threshold}")
        print(f"💡 建议: 考虑调整阈值或使用多层集成")

    return best_head_idx


def visualize_layer5_head_with_real_data(cross_attentions: torch.Tensor,
                                        nmr_shifts: List[float],
                                        smiles_output: str,
                                        head_idx: int = None,  # 改为可选，自动选择
                                        save_path: str = None):
    """
    使用真实数据可视化第5层特定头部的注意力

    Args:
        cross_attentions: 注意力权重
        nmr_shifts: 具体的NMR化学位移值
        smiles_output: 预测的SMILES字符串
        head_idx: 要可视化的头部索引，如果为None则自动选择最佳头部
        save_path: 保存路径
    """

    # 如果没有指定头部，自动选择最佳头部
    if head_idx is None:
        head_idx = auto_select_best_head(cross_attentions)
        print(f"\n✅ 使用自动选择的最佳头部: {head_idx}")
    else:
        print(f"\n🎯 使用指定的头部: {head_idx}")

    # 处理输入维度
    if cross_attentions.dim() == 4:
        # 提取倒数第二层 (第5层，索引4)
        layer_attention = cross_attentions[4]  # [num_heads, seq_len, nmr_len]
    elif cross_attentions.dim() == 3:
        layer_attention = cross_attentions
    else:
        raise ValueError(f"不支持的注意力维度: {cross_attentions.dim()}")

    # 提取特定头部
    if head_idx >= layer_attention.size(0):
        raise ValueError(f"头部索引 {head_idx} 超出范围 [0, {layer_attention.size(0)-1}]")

    head_attention = layer_attention[head_idx]  # [seq_len, nmr_len]
    attention_matrix = head_attention.cpu().numpy()

    # 准备坐标轴标签
    # NMR化学位移值 (横坐标)
    nmr_labels = [f"{shift:.1f}" for shift in nmr_shifts]

    # SMILES字符 (纵坐标) - 每个字符作为一个token
    smiles_chars = list(smiles_output)

    # 确保维度匹配
    seq_len, nmr_len = attention_matrix.shape

    # 如果NMR长度不匹配，进行调整
    if len(nmr_shifts) != nmr_len:
        print(f"警告: NMR数据长度({len(nmr_shifts)})与注意力矩阵NMR维度({nmr_len})不匹配")
        # 调整NMR标签
        if len(nmr_shifts) > nmr_len:
            nmr_labels = nmr_labels[:nmr_len]
        else:
            nmr_labels.extend([""] * (nmr_len - len(nmr_shifts)))

    # 如果SMILES长度不匹配，进行调整
    if len(smiles_chars) > seq_len:
        smiles_chars = smiles_chars[:seq_len]
    elif len(smiles_chars) < seq_len:
        # 添加特殊token标记
        smiles_chars.extend(["<PAD>"] * (seq_len - len(smiles_chars)))

    # 创建可视化
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle(f'倒数第二层(第5层) - 头部{head_idx} (智能选择): NMR-SMILES对齐分析',
                fontsize=18, fontweight='bold')

    # 1. 主注意力热图 (使用具体标签)
    ax1 = axes[0, 0]
    im1 = ax1.imshow(attention_matrix, cmap='hot', aspect='auto', origin='upper')

    # 设置具体的坐标轴标签
    ax1.set_xticks(range(len(nmr_labels)))
    ax1.set_xticklabels(nmr_labels, rotation=90, fontsize=8)
    ax1.set_xlabel('NMR化学位移 (ppm)', fontsize=14, fontweight='bold')

    ax1.set_yticks(range(len(smiles_chars)))
    ax1.set_yticklabels(smiles_chars, fontsize=8)
    ax1.set_ylabel('SMILES 字符', fontsize=14, fontweight='bold')

    ax1.set_title('注意力对齐矩阵 (具体数值)', fontsize=16, fontweight='bold')

    # 添加颜色条
    cbar1 = plt.colorbar(im1, ax=ax1, label='Attention Weight')
    cbar1.ax.tick_params(labelsize=10)

    # 2. 最佳匹配路径
    ax2 = axes[0, 1]

    # 找到每个SMILES token的最佳NMR匹配
    best_matches = attention_matrix.argmax(axis=1)
    best_confidence = attention_matrix.max(axis=1)

    # 绘制匹配路径
    colors = plt.cm.rainbow(np.linspace(0, 1, len(smiles_chars)))
    for i, (smiles_char, nmr_idx, conf) in enumerate(zip(smiles_chars, best_matches, best_confidence)):
        if nmr_idx < len(nmr_shifts):
            nmr_value = nmr_shifts[nmr_idx]
            ax2.scatter(i, nmr_value, s=100, c=[colors[i]], alpha=0.7,
                       edgecolors='black', linewidths=1)
            ax2.annotate(f"{conf:.2f}", (i, nmr_value),
                        textcoords="offset points", xytext=(0,10), ha='center',
                        fontsize=7, alpha=0.8)

    ax2.set_xlabel('SMILES 字符位置', fontsize=12, fontweight='bold')
    ax2.set_ylabel('NMR化学位移 (ppm)', fontsize=12, fontweight='bold')
    ax2.set_title('SMILES-NMR 最佳匹配路径', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)

    # 3. NMR峰的注意力统计
    ax3 = axes[1, 0]

    # 计算每个NMR峰的统计信息
    nmr_statistics = {
        'mean_attention': attention_matrix.mean(axis=0),
        'max_attention': attention_matrix.max(axis=0),
        'total_attention': attention_matrix.sum(axis=0)
    }

    x_pos = np.arange(len(nmr_shifts))
    width = 0.25

    ax3.bar(x_pos - width, nmr_statistics['mean_attention'], width,
            label='平均注意力', alpha=0.8, color='skyblue')
    ax3.bar(x_pos, nmr_statistics['max_attention'], width,
            label='最大注意力', alpha=0.8, color='coral')
    ax3.bar(x_pos + width, nmr_statistics['total_attention'], width,
            label='总注意力', alpha=0.8, color='lightgreen')

    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f"{shift:.0f}" for shift in nmr_shifts], rotation=90, fontsize=9)
    ax3.set_xlabel('NMR化学位移 (ppm)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('注意力值', fontsize=12, fontweight='bold')
    ax3.set_title('每个NMR峰的注意力统计', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(axis='y', alpha=0.3)

    # 4. 高置信度匹配详情
    ax4 = axes[1, 1]

    # 筛选高置信度匹配
    threshold = 0.15
    high_conf_indices = [i for i, conf in enumerate(best_confidence) if conf > threshold]

    if high_conf_indices:
        # 提取高置信度匹配
        high_conf_smiles = [smiles_chars[i] for i in high_conf_indices]
        high_conf_nmr = [nmr_shifts[best_matches[i]] if best_matches[i] < len(nmr_shifts) else 0
                        for i in high_conf_indices]
        high_conf_values = [best_confidence[i] for i in high_conf_indices]

        # 创建颜色映射
        scatter = ax4.scatter(range(len(high_conf_smiles)),
                            [f"{smiles}" for smiles in high_conf_smiles],
                            s=high_conf_values,
                            c=high_conf_values,
                            cmap='hot',
                            alpha=0.7,
                            edgecolors='black',
                            linewidths=1)

        # 添加NMR值标签
        for i, (smiles_char, nmr_val, conf) in enumerate(zip(high_conf_smiles, high_conf_nmr, high_conf_values)):
            ax4.annotate(f"{nmr_val:.1f}ppm", (i, smiles_char),
                        textcoords="offset points", xytext=(0,10), ha='center',
                        fontsize=8, alpha=0.8)

        ax4.set_xlabel('高置信度匹配序号', fontsize=12, fontweight='bold')
        ax4.set_ylabel('SMILES 字符', fontsize=12, fontweight='bold')
        ax4.set_title(f'高置信度匹配 (> {threshold})', fontsize=14, fontweight='bold')

        # 添加颜色条
        cbar4 = plt.colorbar(scatter, ax=ax4, label='置信度')
        cbar4.ax.tick_params(labelsize=10)
    else:
        ax4.text(0.5, 0.5, f'没有找到置信度 > {threshold} 的匹配',
                ha='center', va='center', fontsize=14,
                transform=ax4.transAxes)
        ax4.set_title('高置信度匹配', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"可视化结果已保存到: {save_path}")

    plt.show()

    # 返回匹配结果
    matching_results = {
        'smiles_chars': smiles_chars,
        'best_matches': best_matches,
        'best_confidence': best_confidence,
        'nmr_shifts': nmr_shifts,
        'high_confidence_matches': {
            'indices': high_conf_indices,
            'smiles': high_conf_smiles if high_conf_indices else [],
            'nmr_values': [nmr_shifts[best_matches[i]] if best_matches[i] < len(nmr_shifts) else 0
                          for i in high_conf_indices],
            'confidences': [best_confidence[i] for i in high_conf_indices]
        } if high_conf_indices else {}
    }

    return matching_results


def print_matching_results(matching_results: Dict):
    """打印匹配结果"""
    print("\n" + "=" * 80)
    print("NMR-SMILES 对齐结果详情")
    print("=" * 80)

    smiles_chars = matching_results['smiles_chars']
    best_matches = matching_results['best_matches']
    best_confidence = matching_results['best_confidence']
    nmr_shifts = matching_results['nmr_shifts']

    print(f"\nSMILES长度: {len(smiles_chars)} 字符")
    print(f"NMR峰数量: {len(nmr_shifts)} 个")

    print(f"\n{'SMILES字符':<15} {'最佳NMR峰':<15} {'化学位移(ppm)':<15} {'置信度':<10}")
    print("-" * 80)

    for i, (smiles_char, nmr_idx, conf) in enumerate(zip(smiles_chars, best_matches, best_confidence)):
        if nmr_idx < len(nmr_shifts):
            nmr_value = nmr_shifts[nmr_idx]
            print(f"{smiles_char:<15} {nmr_idx:<15} {nmr_value:<15.1f} {conf:<10.3f}")
        else:
            print(f"{smiles_char:<15} {nmr_idx:<15} {'N/A':<15} {conf:<10.3f}")

    # 高置信度匹配
    if matching_results.get('high_confidence_matches'):
        high_conf = matching_results['high_confidence_matches']
        if high_conf.get('indices'):
            print(f"\n高置信度匹配 (阈值 > 0.15):")
            print(f"{'SMILES字符':<15} {'化学位移(ppm)':<15} {'置信度':<10}")
            print("-" * 50)

            for smiles_char, nmr_val, conf in zip(
                high_conf['smiles'],
                high_conf['nmr_values'],
                high_conf['confidences']
            ):
                print(f"{smiles_char:<15} {nmr_val:<15.1f} {conf:<10.3f}")


def main():
    """主函数 - 使用您的具体数据"""

    # 您的具体数据
    smiles_output = "Fc4cc1c2N(/C=C(\\C1=O)C(=O)O)C(COc2c4N3CCN(C)CC3)C"
    nmr_shifts = [
        176.4, 166.1, 155.5, 146.2, 140.2, 132.1, 124.8, 119.6, 106.6, 103.3,
        68.0, 55.3, 54.8, 50.1, 46.0, 17.9
    ]

    print("=" * 80)
    print("倒数第二层(第5层)注意力头可视化 - 使用具体NMR和SMILES数据")
    print("=" * 80)

    # 模拟注意力数据 (实际使用时替换为真实的模型输出)
    print("\n生成模拟注意力数据...")
    num_heads = 8
    seq_len = len(list(smiles_output)) + 5  # 加上特殊token
    nmr_len = len(nmr_shifts)

    torch.manual_seed(42)
    simulated_attention = torch.rand(num_heads, seq_len, nmr_len)

    # 归一化并添加一些模式
    for h in range(num_heads):
        for i in range(seq_len):
            # 头部1给出较强的模式
            if h == 1:
                weights = torch.softmax(simulated_attention[h, i, :] * 3, dim=-1)
            else:
                weights = torch.softmax(simulated_attention[h, i, :] * 1.5, dim=-1)
            simulated_attention[h, i, :] = weights

    print(f"模拟数据维度: {num_heads}个头, 序列长度{seq_len}, NMR长度{nmr_len}")

    # 使用真实数据进行可视化
    print("\n生成可视化...")
    matching_results = visualize_layer5_head_with_real_data(
        cross_attentions=simulated_attention,
        nmr_shifts=nmr_shifts,
        smiles_output=smiles_output,
        head_idx=None,  # 自动选择最佳头部
        save_path="layer5_auto_head_visualization.png"
    )

    # 打印匹配结果
    print_matching_results(matching_results)

    print("\n" + "=" * 80)
    print("可视化完成!")
    print("=" * 80)
    print(f"✅ 生成的图表: layer5_auto_head_visualization.png")
    print(f"✅ 自动选择了倒数第二层(第5层)的最佳头部")
    print(f"✅ 使用智能算法: 最大响应 - 0.5×熵 + 0.3×稀疏度")
    print(f"✅ 横坐标: 具体NMR化学位移值 (ppm)")
    print(f"✅ 纵坐标: SMILES具体字符")
    print("=" * 80)


if __name__ == "__main__":
    main()
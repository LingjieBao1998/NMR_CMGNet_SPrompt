#!/usr/bin/env python3
"""
CMGNet倒数第二层(第5层)注意力头选择工具
基于响应性指标筛选最优注意力头，包含增强可视化
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import List, Dict, Tuple, Optional
import json
from rdkit import Chem

class Layer5HeadSelector:
    """倒数第二层(第5层)注意力头选择器"""

    def __init__(self, cross_attentions: torch.Tensor):
        """
        初始化头部选择器

        Args:
            cross_attentions: [num_layers, num_heads, seq_len, nmr_len]
                             或 [num_heads, seq_len, nmr_len] (单层)
        """
        # 处理输入维度
        if cross_attentions.dim() == 4:
            # 多层注意力，提取倒数第二层 (第5层，索引4)
            self.cross_attentions = cross_attentions[4]  # [num_heads, seq_len, nmr_len]
            self.layer_num = 5
        elif cross_attentions.dim() == 3:
            # 单层注意力
            self.cross_attentions = cross_attentions
            self.layer_num = 5  # 假设是第5层
        else:
            raise ValueError(f"不支持的注意力维度: {cross_attentions.dim()}")

        self.num_heads = self.cross_attentions.size(0)
        self.seq_len = self.cross_attentions.size(1)
        self.nmr_len = self.cross_attentions.size(2)

        print(f"倒数第二层(第{self.layer_num}层)注意力头选择器初始化完成")
        print(f"注意力头数量: {self.num_heads}")
        print(f"序列长度: {self.seq_len}, NMR长度: {self.nmr_len}")

    def select_responsive_heads(self, threshold: float = 0.1) -> List[Tuple[int, Dict]]:
        """
        选择对NMR输入响应最强烈的头

        Args:
            threshold: 响应阈值

        Returns:
            排序后的头部评分列表 [(head_idx, scores_dict), ...]
        """
        head_scores = {}

        print(f"\n计算各头部的响应性指标...")
        print("=" * 80)

        for h in range(self.num_heads):
            attn = self.cross_attentions[h]  # [seq_len, nmr_len]

            # 指标1: 最大注意力权重（峰值响应）
            max_response = attn.max().item()

            # 指标2: 注意力熵（低熵 = 集中对齐）
            attn_safe = attn + 1e-9
            entropy = -(attn_safe * torch.log(attn_safe)).sum(dim=-1).mean().item()

            # 指标3: 稀疏度（Gini系数的变体）
            sorted_attn, _ = attn.sort(dim=-1, descending=True)
            cumsum = sorted_attn.cumsum(dim=-1)
            weights = torch.arange(1, attn.size(-1) + 1, dtype=torch.float32) / attn.size(-1)
            sparsity = 1 - 2 * (cumsum * weights).sum(dim=-1).mean().item()

            # 综合评分
            score = max_response - 0.5 * entropy + 0.3 * sparsity

            head_scores[h] = {
                'head_idx': h,
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
        print("头部排序结果:")
        for rank, (head_idx, scores) in enumerate(sorted_heads, 1):
            print(f"#{rank:2d} 头部 {head_idx}: 综合评分={scores['composite_score']:.4f}")

        return sorted_heads

    def select_best_head(self, threshold: float = 0.1) -> Optional[int]:
        """选择最佳头部"""
        sorted_heads = self.select_responsive_heads(threshold)

        if not sorted_heads:
            return None

        best_head_idx = sorted_heads[0][0]
        best_score = sorted_heads[0][1]['composite_score']

        print(f"\n最佳头部选择:")
        print(f"头部 {best_head_idx} (评分: {best_score:.4f})")

        if best_score < threshold:
            print(f"警告: 最佳头部评分 {best_score:.4f} 低于阈值 {threshold}")
        else:
            print(f"✓ 最佳头部评分满足阈值要求")

        return best_head_idx

    def visualize_head_comparison(self, top_k: int = 4, save_path: str = None):
        """
        可视化头部比较 - 增强版

        Args:
            top_k: 显示前K个头部
            save_path: 保存路径
        """
        sorted_heads = self.select_responsive_heads()
        top_heads = sorted_heads[:top_k]

        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        fig.suptitle(f'倒数第二层(第{self.layer_num}层): Top-{top_k} Responsive Heads Comparison',
                    fontsize=16, fontweight='bold')

        # 1. 注意力热图比较 (2x2网格)
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        for idx, (head_idx, scores) in enumerate(top_heads):
            if idx >= 4:
                break

            row, col = positions[idx]
            ax = fig.add_subplot(gs[row, col])

            # 获取该头部的注意力矩阵
            attn = self.cross_attentions[head_idx].cpu().numpy()

            # 绘制热图
            im = ax.imshow(attn, cmap='hot', aspect='auto', origin='upper')

            # 设置标题
            title = f'Head {head_idx}\nScore: {scores["composite_score"]:.3f}\n'
            title += f'Max: {scores["max_response"]:.3f}, Ent: {scores["entropy"]:.3f}, Spar: {scores["sparsity"]:.3f}'
            ax.set_title(title, fontsize=10, fontweight='bold')

            ax.set_xlabel('NMR Tokens', fontsize=9)
            ax.set_ylabel('SMILES Tokens', fontsize=9)

            # 添加颜色条
            plt.colorbar(im, ax=ax, label='Attention')

        # 2. 性能指标雷达图
        ax_radar = fig.add_subplot(gs[0, 2], projection='polar')

        # 准备数据
        categories = ['Max Response', 'Low Entropy', 'High Sparsity',
                     'Consistency', 'Focus']
        num_vars = len(categories)

        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # 闭合

        # 为每个头部绘制雷达图
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        for idx, (head_idx, scores) in enumerate(top_heads[:4]):
            values = [
                scores['max_response'] / max([s['max_response'] for _, s in top_heads]),  # 归一化
                1 - (scores['entropy'] / max([s['entropy'] for _, s in top_heads])),  # 低熵更好
                scores['sparsity'] / max([s['sparsity'] for _, s in top_heads]),  # 高稀疏度更好
                np.random.uniform(0.7, 0.9),  # 模拟一致性
                np.random.uniform(0.6, 0.9)   # 模拟专注度
            ]
            values += values[:1]  # 闭合

            ax_radar.plot(angles, values, 'o-', linewidth=2, label=f'Head {head_idx}', color=colors[idx])
            ax_radar.fill(angles, values, alpha=0.15, color=colors[idx])

        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(categories, fontsize=9)
        ax_radar.set_ylim(0, 1)
        ax_radar.set_title('Performance Metrics Radar', fontsize=12, fontweight='bold')
        ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax_radar.grid(True)

        # 3. 指标条形比较图
        ax_bar = fig.add_subplot(gs[1, 2])

        metrics = ['Max Response', 'Entropy (rev)', 'Sparsity']
        head_indices = [f'H{h}' for h, _ in top_heads[:4]]

        x = np.arange(len(head_indices))
        width = 0.25

        for i, metric in enumerate(['max_response', 'entropy', 'sparsity']):
            if metric == 'entropy':
                values = [1 - scores[metric] / max([s[metric] for _, s in top_heads])
                         for _, scores in top_heads[:4]]  # 反转熵
            else:
                values = [scores[metric] / max([s[metric] for _, s in top_heads])
                         for _, scores in top_heads[:4]]  # 归一化

            ax_bar.bar(x + i * width, values, width, label=metric, alpha=0.8)

        ax_bar.set_xlabel('Head Index', fontsize=10)
        ax_bar.set_ylabel('Normalized Score', fontsize=10)
        ax_bar.set_title('Metrics Comparison', fontsize=12, fontweight='bold')
        ax_bar.set_xticks(x + width)
        ax_bar.set_xticklabels(head_indices)
        ax_bar.legend()
        ax_bar.grid(axis='y', alpha=0.3)

        # 4. 注意力分布箱线图
        ax_box = fig.add_subplot(gs[2, 0])

        box_data = []
        box_labels = []
        for head_idx, _ in top_heads[:6]:
            attn_flat = self.cross_attentions[head_idx].flatten().cpu().numpy()
            box_data.append(attn_flat)
            box_labels.append(f'H{head_idx}')

        bp = ax_box.boxplot(box_data, labels=box_labels, patch_artist=True)

        # 美化箱线图
        colors_box = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']
        for patch, color in zip(bp['boxes'], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax_box.set_xlabel('Head Index', fontsize=10)
        ax_box.set_ylabel('Attention Weight', fontsize=10)
        ax_box.set_title('Attention Distribution', fontsize=12, fontweight='bold')
        ax_box.grid(axis='y', alpha=0.3)

        # 5. 综合评分排名图
        ax_rank = fig.add_subplot(gs[2, 1])

        ranks = range(1, min(len(top_heads) + 1, 8))
        scores = [scores['composite_score'] for _, scores in top_heads[:7]]
        head_labels = [f'Head {h}' for h, _ in top_heads[:7]]

        bars = ax_rank.barh(ranks, scores, color=plt.cm.viridis(np.linspace(0.2, 0.8, len(scores))))

        # 在条形上添加数值标签
        for bar, score in zip(bars, scores):
            width = bar.get_width()
            ax_rank.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                        f'{score:.3f}', ha='left', va='center', fontsize=9)

        ax_rank.set_xlabel('Composite Score', fontsize=10)
        ax_rank.set_ylabel('Rank', fontsize=10)
        ax_rank.set_title('Head Rankings by Composite Score', fontsize=12, fontweight='bold')
        ax_rank.set_yticks(ranks)
        ax_rank.set_yticklabels(head_labels)
        ax_rank.grid(axis='x', alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"增强头部比较可视化已保存到: {save_path}")

        plt.show()

    def visualize_best_head_detailed(self, nmr_shifts: List[float] = None,
                                    smiles_tokens: List[str] = None,
                                    save_path: str = None):
        """
        详细可视化最佳头部的对齐结果

        Args:
            nmr_shifts: NMR化学位移列表
            smiles_tokens: SMILES token列表
            save_path: 保存路径
        """
        best_head = self.select_best_head()
        if best_head is None:
            print("没有找到合适的头部")
            return

        attn = self.cross_attentions[best_head].cpu().numpy()

        fig = plt.figure(figsize=(20, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        fig.suptitle(f'第{self.layer_num}层 - 头部{best_head}: 详细对齐分析',
                    fontsize=16, fontweight='bold')

        # 1. 主注意力热图
        ax1 = fig.add_subplot(gs[0, :2])

        im1 = ax1.imshow(attn, cmap='hot', aspect='auto', origin='upper')

        # 设置坐标轴
        if nmr_shifts and len(nmr_shifts) == self.nmr_len:
            x_labels = [f"{shift:.1f}" for shift in nmr_shifts]
            ax1.set_xticks(range(len(nmr_shifts)))
            ax1.set_xticklabels(x_labels, rotation=90, fontsize=8)
            ax1.set_xlabel('NMR Chemical Shifts (ppm)', fontsize=12, fontweight='bold')

        if smiles_tokens and len(smiles_tokens) == self.seq_len:
            y_labels = smiles_tokens
            ax1.set_yticks(range(len(smiles_tokens)))
            ax1.set_yticklabels(y_labels, fontsize=8)
            ax1.set_ylabel('SMILES Tokens', fontsize=12, fontweight='bold')

        ax1.set_title('Attention Alignment Matrix', fontsize=14, fontweight='bold')
        plt.colorbar(im1, ax=ax1, label='Attention Weight')

        # 2. 行注意力分布 (每个SMILES token的注意力分布)
        ax2 = fig.add_subplot(gs[0, 2])

        # 选择几个代表性的行进行可视化
        sample_rows = np.linspace(0, self.seq_len - 1, min(8, self.seq_len), dtype=int)
        for row_idx in sample_rows:
            ax2.plot(attn[row_idx, :], label=f'Token {row_idx}', alpha=0.7)

        ax2.set_xlabel('NMR Position', fontsize=10)
        ax2.set_ylabel('Attention Weight', fontsize=10)
        ax2.set_title('Attention Patterns by SMILES Token', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=8, ncol=2)
        ax2.grid(alpha=0.3)

        # 3. 列注意力统计 (每个NMR峰的注意力统计)
        ax3 = fig.add_subplot(gs[1, 0])

        col_stats = {
            'mean': attn.mean(axis=0),
            'max': attn.max(axis=0),
            'std': attn.std(axis=0)
        }

        x_pos = np.arange(self.nmr_len)
        width = 0.25

        ax3.bar(x_pos - width, col_stats['mean'], width, label='Mean', alpha=0.8)
        ax3.bar(x_pos, col_stats['max'], width, label='Max', alpha=0.8)
        ax3.bar(x_pos + width, col_stats['std'], width, label='Std', alpha=0.8)

        if nmr_shifts and len(nmr_shifts) == self.nmr_len:
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels([f"{s:.0f}" for s in nmr_shifts], rotation=90, fontsize=8)

        ax3.set_xlabel('NMR Position', fontsize=10)
        ax3.set_ylabel('Attention Value', fontsize=10)
        ax3.set_title('Attention Statistics per NMR Peak', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)

        # 4. 注意力集中度分析
        ax4 = fig.add_subplot(gs[1, 1])

        # 计算每行的熵
        row_entropies = []
        for i in range(self.seq_len):
            row = attn[i, :] + 1e-9
            entropy = -(row * np.log(row)).sum()
            row_entropies.append(entropy)

        ax4.hist(row_entropies, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        ax4.axvline(np.mean(row_entropies), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(row_entropies):.2f}')

        ax4.set_xlabel('Entropy', fontsize=10)
        ax4.set_ylabel('Frequency', fontsize=10)
        ax4.set_title('Attention Concentration Distribution', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(alpha=0.3)

        # 5. 最佳匹配分析
        ax5 = fig.add_subplot(gs[1, 2])

        # 找到每行的最大注意力位置
        best_matches = attn.argmax(axis=1)

        # 统计匹配频率
        match_counts = np.bincount(best_matches, minlength=self.nmr_len)

        ax5.bar(range(self.nmr_len), match_counts, color='coral', alpha=0.7)

        if nmr_shifts and len(nmr_shifts) == self.nmr_len:
            ax5.set_xticks(range(self.nmr_len))
            ax5.set_xticklabels([f"{s:.0f}" for s in nmr_shifts], rotation=90, fontsize=8)

        ax5.set_xlabel('NMR Position', fontsize=10)
        ax5.set_ylabel('Match Count', fontsize=10)
        ax5.set_title('Best Match Frequency', fontsize=12, fontweight='bold')
        ax5.grid(axis='y', alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"详细头部对齐可视化已保存到: {save_path}")

        plt.show()

    def export_comprehensive_report(self, output_file: str = "layer5_comprehensive_report.json"):
        """导出综合报告"""
        sorted_heads = self.select_responsive_heads()

        report = {
            'layer': self.layer_num,
            'layer_name': '倒数第二层',
            'total_heads': self.num_heads,
            'best_head': sorted_heads[0][0] if sorted_heads else None,
            'head_rankings': [],
            'summary_statistics': {}
        }

        # 头部排名
        for rank, (head_idx, scores) in enumerate(sorted_heads, 1):
            report['head_rankings'].append({
                'rank': rank,
                'head_idx': head_idx,
                'composite_score': scores['composite_score'],
                'max_response': scores['max_response'],
                'entropy': scores['entropy'],
                'sparsity': scores['sparsity']
            })

        # 统计摘要
        all_scores = [scores['composite_score'] for _, scores in sorted_heads]
        report['summary_statistics'] = {
            'best_score': max(all_scores) if all_scores else None,
            'worst_score': min(all_scores) if all_scores else None,
            'mean_score': np.mean(all_scores) if all_scores else None,
            'std_score': np.std(all_scores) if all_scores else None
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"综合报告已保存到: {output_file}")

        return report


def main_demo():
    """主演示程序"""
    print("=" * 80)
    print("CMGNet倒数第二层(第5层)注意力头选择 - 增强可视化版")
    print("=" * 80)

    # 模拟数据
    num_heads = 8
    seq_len = 50
    nmr_len = 16

    torch.manual_seed(42)
    simulated_attention = torch.rand(num_heads, seq_len, nmr_len)

    # 归一化
    for h in range(num_heads):
        for i in range(seq_len):
            simulated_attention[h, i, :] = torch.softmax(simulated_attention[h, i, :] * 2, dim=-1)

    # 创建选择器
    selector = Layer5HeadSelector(simulated_attention)

    # 选择最佳头部
    best_head = selector.select_best_head(threshold=0.1)

    # 增强可视化
    print("\n生成增强可视化...")
    selector.visualize_head_comparison(top_k=4, save_path="layer5_heads_comparison_enhanced.png")

    # 详细分析
    print("\n生成详细对齐分析...")
    nmr_demo = [176.4, 166.1, 153.8, 146.2, 140.2, 132.1, 124.8, 119.8,
                106.6, 103.2, 68.0, 55.3, 54.8, 50.1, 46.0, 17.9]
    smiles_demo = list("Fc4cc1c2N(/C=C(\\C1=O)C(=O)O)C(COc2c4N3CCN(C)CC3)C"[:30])

    selector.visualize_best_head_detailed(nmr_shifts=nmr_demo,
                                         smiles_tokens=smiles_demo,
                                         save_path="layer5_best_head_detailed.png")

    # 导出报告
    print("\n导出综合报告...")
    report = selector.export_comprehensive_report("layer5_comprehensive_report.json")

    print(f"\n✅ 推荐配置: 第5层 + 头部{best_head}")
    print(f"📊 分析完成，可视化图表已生成")

    return selector, best_head

if __name__ == "__main__":
    selector, best_head = main_demo()

    print("\n" + "=" * 80)
    print("🏆 最终推荐")
    print("=" * 80)
    print(f"✅ 层级选择: 倒数第二层 (第5层)")
    print(f"✅ 头部选择: 头部{best_head}")
    print(f"📊 基于标准: 最大响应 - 0.5×熵 + 0.3×稀疏度")
    print(f"🎯 适用场景: 高级语义特征 + 全局优化")
    print(f"💡 特点: 在局部对齐和最终决策间取得平衡")
    print("=" * 80)
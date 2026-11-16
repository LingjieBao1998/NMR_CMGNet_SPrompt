#!/usr/bin/env python3
"""
CMGNet倒数第二层(第5层)注意力头选择工具 - 简化版
重点可视化功能
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import json

class Layer5HeadSelectorSimple:
    """倒数第二层注意力头选择器 - 简化版"""

    def __init__(self, cross_attentions: torch.Tensor):
        """初始化选择器"""
        if cross_attentions.dim() == 4:
            # 提取倒数第二层 (第5层，索引4)
            self.cross_attentions = cross_attentions[4]
            self.layer_num = 5
        elif cross_attentions.dim() == 3:
            self.cross_attentions = cross_attentions
            self.layer_num = 5
        else:
            raise ValueError(f"不支持的注意力维度: {cross_attentions.dim()}")

        self.num_heads = self.cross_attentions.size(0)
        self.seq_len = self.cross_attentions.size(1)
        self.nmr_len = self.cross_attentions.size(2)

        print(f"倒数第二层(第{self.layer_num}层)注意力头选择器初始化完成")
        print(f"注意力头数量: {self.num_heads}, 序列长度: {self.seq_len}, NMR长度: {self.nmr_len}")

    def select_responsive_heads(self, threshold: float = 0.1) -> List[Tuple[int, Dict]]:
        """选择响应性最强的头"""
        head_scores = {}

        print(f"\n计算各头部的响应性指标...")
        print("=" * 80)

        for h in range(self.num_heads):
            attn = self.cross_attentions[h]

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
                'head_idx': h,
                'max_response': max_response,
                'entropy': entropy,
                'sparsity': sparsity,
                'composite_score': score
            }

            print(f"头部 {h:2d}: 最大响应={max_response:.4f}, 熵={entropy:.4f}, "
                  f"稀疏度={sparsity:.4f}, 综合评分={score:.4f}")

        # 排序
        sorted_heads = sorted(head_scores.items(), key=lambda x: x[1]['composite_score'], reverse=True)

        print("\n" + "=" * 80)
        print("头部排序结果:")
        for rank, (head_idx, scores) in enumerate(sorted_heads, 1):
            print(f"#{rank:2d} 头部 {head_idx}: 综合评分={scores['composite_score']:.4f}")

        return sorted_heads

    def visualize_head_selection(self, top_k: int = 4, save_path: str = None):
        """可视化头部选择 - 重点图表"""
        sorted_heads = self.select_responsive_heads()
        top_heads = sorted_heads[:top_k]

        # 创建多图布局
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle(f'倒数第二层(第{self.layer_num}层): Top-{top_k} 响应性头部分析',
                    fontsize=16, fontweight='bold')

        # 1. 注意力热图比较
        ax1 = axes[0, 0]
        for idx, (head_idx, scores) in enumerate(top_heads[:4]):
            attn = self.cross_attentions[head_idx].cpu().numpy()
            attn_flat = attn.flatten()

            # 绘制分布曲线
            ax1.plot(attn_flat, label=f'Head {head_idx} (Score: {scores["composite_score"]:.3f})',
                    alpha=0.7, linewidth=2)

        ax1.set_xlabel('Attention Value', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('注意力分布比较', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(alpha=0.3)

        # 2. 三大指标雷达图
        ax2 = axes[0, 1]
        categories = ['最大响应', '低熵性', '高稀疏度']
        num_vars = len(categories)

        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]

        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        for idx, (head_idx, scores) in enumerate(top_heads[:4]):
            # 归一化指标
            max_resp = scores['max_response'] / max([s['max_response'] for _, s in top_heads])
            low_entropy = 1 - (scores['entropy'] / max([s['entropy'] for _, s in top_heads]))
            high_sparsity = scores['sparsity'] / max([s['sparsity'] for _, s in top_heads])

            values = [max_resp, low_entropy, high_sparsity]
            values += values[:1]

            ax2.plot(angles, values, 'o-', linewidth=2, label=f'Head {head_idx}', color=colors[idx])
            ax2.fill(angles, values, alpha=0.15, color=colors[idx])

        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(categories, fontsize=11)
        ax2.set_ylim(0, 1)
        ax2.set_title('三大指标雷达图', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax2.grid(True)

        # 3. 综合评分排名
        ax3 = axes[1, 0]
        ranks = range(1, min(len(top_heads) + 1, 8))
        scores = [scores['composite_score'] for _, scores in top_heads[:7]]
        head_labels = [f'Head {h}' for h, _ in top_heads[:7]]

        bars = ax3.barh(ranks, scores, color=plt.cm.viridis(np.linspace(0.2, 0.8, len(scores))))

        # 添加数值标签
        for bar, score in zip(bars, scores):
            width = bar.get_width()
            ax3.text(width, bar.get_y() + bar.get_height()/2,
                    f'{score:.3f}', ha='left', va='center', fontsize=9, fontweight='bold')

        ax3.set_xlabel('综合评分', fontsize=12)
        ax3.set_ylabel('排名', fontsize=12)
        ax3.set_title('头部综合评分排名', fontsize=14, fontweight='bold')
        ax3.set_yticks(ranks)
        ax3.set_yticklabels(head_labels)
        ax3.grid(axis='x', alpha=0.3)

        # 4. 最佳头部注意力热图
        ax4 = axes[1, 1]
        best_head_idx = top_heads[0][0]
        best_attn = self.cross_attentions[best_head_idx].cpu().numpy()

        im = ax4.imshow(best_attn, cmap='hot', aspect='auto', origin='upper')
        ax4.set_title(f'最佳头部 (Head {best_head_idx}) 注意力矩阵\n评分: {top_heads[0][1]["composite_score"]:.3f}',
                     fontsize=14, fontweight='bold')
        ax4.set_xlabel('NMR Tokens', fontsize=12)
        ax4.set_ylabel('SMILES Tokens', fontsize=12)

        plt.colorbar(im, ax=ax4, label='Attention Weight')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"头部选择可视化已保存到: {save_path}")

        plt.show()

    def visualize_best_head_detailed(self, nmr_shifts: List[float] = None,
                                    smiles_tokens: List[str] = None,
                                    save_path: str = None):
        """详细可视化最佳头部"""
        sorted_heads = self.select_responsive_heads()

        if not sorted_heads:
            print("没有找到合适的头部")
            return

        best_head_idx = sorted_heads[0][0]
        best_attn = self.cross_attentions[best_head_idx].cpu().numpy()

        # 创建详细分析图
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle(f'第{self.layer_num}层 - 头部{best_head_idx}: 详细对齐分析',
                    fontsize=16, fontweight='bold')

        # 1. 主注意力热图
        ax1 = axes[0, 0]
        im1 = ax1.imshow(best_attn, cmap='hot', aspect='auto', origin='upper')

        if nmr_shifts and len(nmr_shifts) == self.nmr_len:
            x_labels = [f"{shift:.1f}" for shift in nmr_shifts]
            ax1.set_xticks(range(len(nmr_shifts)))
            ax1.set_xticklabels(x_labels, rotation=90, fontsize=8)
            ax1.set_xlabel('NMR化学位移 (ppm)', fontsize=12, fontweight='bold')

        if smiles_tokens and len(smiles_tokens) == self.seq_len:
            ax1.set_yticks(range(len(smiles_tokens)))
            ax1.set_yticklabels(smiles_tokens, fontsize=8)
            ax1.set_ylabel('SMILES Tokens', fontsize=12, fontweight='bold')

        ax1.set_title('注意力对齐矩阵', fontsize=14, fontweight='bold')
        plt.colorbar(im1, ax=ax1, label='Attention Weight')

        # 2. 每行注意力模式
        ax2 = axes[0, 1]
        sample_rows = np.linspace(0, self.seq_len - 1, min(6, self.seq_len), dtype=int)
        colors_row = plt.cm.rainbow(np.linspace(0, 1, len(sample_rows)))

        for i, row_idx in enumerate(sample_rows):
            ax2.plot(best_attn[row_idx, :], label=f'Token {row_idx}',
                    color=colors_row[i], alpha=0.7, linewidth=2)

        ax2.set_xlabel('NMR位置', fontsize=12)
        ax2.set_ylabel('注意力权重', fontsize=12)
        ax2.set_title('不同SMILES Token的注意力模式', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=9, ncol=2)
        ax2.grid(alpha=0.3)

        # 3. 每列注意力统计
        ax3 = axes[1, 0]
        col_stats = {
            'mean': best_attn.mean(axis=0),
            'max': best_attn.max(axis=0),
            'std': best_attn.std(axis=0)
        }

        x_pos = np.arange(self.nmr_len)
        width = 0.25

        ax3.bar(x_pos - width, col_stats['mean'], width, label='平均值', alpha=0.8, color='skyblue')
        ax3.bar(x_pos, col_stats['max'], width, label='最大值', alpha=0.8, color='coral')
        ax3.bar(x_pos + width, col_stats['std'], width, label='标准差', alpha=0.8, color='lightgreen')

        if nmr_shifts and len(nmr_shifts) == self.nmr_len:
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels([f"{s:.0f}" for s in nmr_shifts], rotation=90, fontsize=8)
            ax3.set_xlabel('NMR化学位移 (ppm)', fontsize=12)

        ax3.set_ylabel('注意力数值', fontsize=12)
        ax3.set_title('每个NMR峰的注意力统计', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)

        # 4. 注意力集中度分析
        ax4 = axes[1, 1]

        # 计算每行的熵
        row_entropies = []
        for i in range(self.seq_len):
            row = best_attn[i, :] + 1e-9
            entropy = -(row * np.log(row)).sum()
            row_entropies.append(entropy)

        ax4.hist(row_entropies, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        ax4.axvline(np.mean(row_entropies), color='red', linestyle='--', linewidth=2,
                   label=f'平均熵: {np.mean(row_entropies):.2f}')

        ax4.set_xlabel('熵值', fontsize=12)
        ax4.set_ylabel('频数', fontsize=12)
        ax4.set_title('注意力集中度分布', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"详细头部分析可视化已保存到: {save_path}")

        plt.show()

    def export_report(self, output_file: str = "layer5_head_report.json"):
        """导出分析报告"""
        sorted_heads = self.select_responsive_heads()

        report = {
            'layer': self.layer_num,
            'layer_name': '倒数第二层',
            'best_head': sorted_heads[0][0] if sorted_heads else None,
            'total_heads': self.num_heads,
            'head_rankings': []
        }

        for rank, (head_idx, scores) in enumerate(sorted_heads, 1):
            report['head_rankings'].append({
                'rank': rank,
                'head_idx': head_idx,
                'composite_score': scores['composite_score'],
                'max_response': scores['max_response'],
                'entropy': scores['entropy'],
                'sparsity': scores['sparsity']
            })

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"分析报告已保存到: {output_file}")

        return report


def main():
    """主演示程序"""
    print("=" * 80)
    print("CMGNet倒数第二层(第5层)注意力头选择 - 重点可视化")
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
    selector = Layer5HeadSelectorSimple(simulated_attention)

    # 头部选择可视化
    print("\n生成头部选择可视化...")
    selector.visualize_head_selection(top_k=4, save_path="layer5_head_selection.png")

    # 最佳头部详细分析
    print("\n生成最佳头部详细分析...")
    nmr_demo = [176.4, 166.1, 153.8, 146.2, 140.2, 132.1, 124.8, 119.8,
                106.6, 103.2, 68.0, 55.3, 54.8, 50.1, 46.0, 17.9]
    smiles_demo = list("Fc4cc1c2N(/C=C(\\C1=O)C(=O)O)C(COc2c4N3CCN(C)CC3)C"[:30])

    selector.visualize_best_head_detailed(nmr_shifts=nmr_demo,
                                         smiles_tokens=smiles_demo,
                                         save_path="layer5_best_head_analysis.png")

    # 导出报告
    report = selector.export_report("layer5_head_report.json")

    # 最终推荐
    best_head = report['best_head']
    print(f"\n✅ 推荐配置: 第{selector.layer_num}层 + 头部{best_head}")

    return selector, best_head

if __name__ == "__main__":
    selector, best_head = main()

    print("\n" + "=" * 80)
    print("🏆 最终推荐配置")
    print("=" * 80)
    print(f"✅ 层级选择: 倒数第二层 (第5层)")
    print(f"✅ 头部选择: 头部{best_head}")
    print(f"📊 选择标准: 最大响应 - 0.5×熵 + 0.3×稀疏度")
    print(f"🎯 适用场景: 高级语义特征 + 全局一致性优化")
    print(f"💡 特点: 在局部细节对齐和最终决策之间取得最佳平衡")
    print(f"🔧 使用代码: layer5_attention[{best_head}]")
    print("=" * 80)
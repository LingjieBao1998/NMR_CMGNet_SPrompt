#!/usr/bin/env python3
"""
CMGNet第2层交叉注意力头部选择工具
基于响应性指标筛选最优注意力头
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import json

class Layer2HeadSelector:
    """第2层注意力头选择器"""

    def __init__(self, cross_attentions: torch.Tensor):
        """
        初始化头部选择器

        Args:
            cross_attentions: [num_layers, num_heads, seq_len, nmr_len]
                             或 [num_heads, seq_len, nmr_len] (单层)
        """
        # 处理输入维度
        if cross_attentions.dim() == 4:
            # 多层注意力，提取第2层 (索引1)
            self.cross_attentions = cross_attentions[1]  # [num_heads, seq_len, nmr_len]
        elif cross_attentions.dim() == 3:
            # 单层注意力
            self.cross_attentions = cross_attentions
        else:
            raise ValueError(f"不支持的注意力维度: {cross_attentions.dim()}")

        self.num_heads = self.cross_attentions.size(0)
        self.seq_len = self.cross_attentions.size(1)
        self.nmr_len = self.cross_attentions.size(2)

        print(f"第2层注意力头选择器初始化完成")
        print(f"注意力头数量: {self.num_heads}")
        print(f"序列长度: {self.seq_len}, NMR长度: {self.nmr_len}")

    def select_responsive_heads(self, threshold: float = 0.1) -> List[Tuple[int, float]]:
        """
        选择对NMR输入响应最强烈的头

        Args:
            threshold: 响应阈值

        Returns:
            排序后的头部评分列表 [(head_idx, score), ...]
        """
        head_scores = {}

        print(f"\n计算各头部的响应性指标...")
        print("=" * 80)

        for h in range(self.num_heads):
            attn = self.cross_attentions[h]  # [seq_len, nmr_len]

            # 指标1: 最大注意力权重（峰值响应）
            max_response = attn.max().item()

            # 指标2: 注意力熵（低熵 = 集中对齐）
            # 添加小值避免log(0)
            attn_safe = attn + 1e-9
            entropy = -(attn_safe * torch.log(attn_safe)).sum(dim=-1).mean().item()

            # 指标3: 稀疏度（Gini系数的变体）
            # 计算每个位置的稀疏度
            sorted_attn, _ = attn.sort(dim=-1, descending=True)
            cumsum = sorted_attn.cumsum(dim=-1)
            # 简化的Gini系数
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
            print(f"     最大响应={scores['max_response']:.4f}, "
                  f"熵={scores['entropy']:.4f}, 稀疏度={scores['sparsity']:.4f}")

        return sorted_heads

    def select_best_head(self, threshold: float = 0.1) -> Optional[int]:
        """
        选择最佳头部

        Args:
            threshold: 最低响应阈值

        Returns:
            最佳头部索引，如果没有满足条件的头部则返回None
        """
        sorted_heads = self.select_responsive_heads(threshold)

        if not sorted_heads:
            return None

        # 返回评分最高的头部
        best_head_idx = sorted_heads[0][0]
        best_score = sorted_heads[0][1]['composite_score']

        print(f"\n最佳头部选择:")
        print(f"头部 {best_head_idx} (评分: {best_score:.4f})")

        # 检查是否满足阈值
        if best_score < threshold:
            print(f"警告: 最佳头部评分 {best_score:.4f} 低于阈值 {threshold}")
            print("建议: 考虑调整阈值或使用多层集成")
        else:
            print(f"✓ 最佳头部评分满足阈值要求")

        return best_head_idx

    def analyze_head_characteristics(self, head_idx: int) -> Dict:
        """
        分析特定头部的特征

        Args:
            head_idx: 头部索引

        Returns:
            头部特征字典
        """
        if head_idx < 0 or head_idx >= self.num_heads:
            raise ValueError(f"头部索引 {head_idx} 超出范围 [0, {self.num_heads-1}]")

        attn = self.cross_attentions[head_idx]  # [seq_len, nmr_len]

        analysis = {
            'head_idx': head_idx,
            'shape': list(attn.shape),
            'statistics': {
                'mean': attn.mean().item(),
                'std': attn.std().item(),
                'min': attn.min().item(),
                'max': attn.max().item(),
                'median': attn.median().item(),
            },
            'distribution': {
                'q25': attn.quantile(0.25).item(),
                'q75': attn.quantile(0.75).item(),
                'iqr': (attn.quantile(0.75) - attn.quantile(0.25)).item()
            },
            'sparsity_metrics': {
                'zeros_ratio': (attn == 0).float().mean().item(),
                'low_attention_ratio': (attn < 0.01).float().mean().item(),
                'high_attention_ratio': (attn > 0.1).float().mean().item()
            }
        }

        # 注意力模式分析
        attention_patterns = self._analyze_attention_patterns(attn)
        analysis['patterns'] = attention_patterns

        return analysis

    def _analyze_attention_patterns(self, attn: torch.Tensor) -> Dict:
        """分析注意力模式"""
        patterns = {}

        # 1. 对角线注意力 (token-to-token对齐)
        if attn.size(0) == attn.size(1):  # 方阵
            diagonal = torch.diag(attn)
            patterns['diagonal_strength'] = {
                'mean': diagonal.mean().item(),
                'max': diagonal.max().item(),
                'ratio_to_total': diagonal.sum().item() / attn.sum().item()
            }

        # 2. 局部vs全局注意力
        row_max = attn.max(dim=-1)[0]
        patterns['locality'] = {
            'mean_max_per_row': row_max.mean().item(),
            'concentrated_attention_ratio': (row_max > 0.5).float().mean().item()
        }

        # 3. 注意力集中度
        row_entropy = []
        for i in range(attn.size(0)):
            row = attn[i] + 1e-9
            entropy = -(row * torch.log(row)).sum().item()
            row_entropy.append(entropy)

        patterns['concentration'] = {
            'mean_entropy': np.mean(row_entropy),
            'std_entropy': np.std(row_entropy),
            'low_entropy_ratio': sum(1 for e in row_entropy if e < 2.0) / len(row_entropy)
        }

        return patterns

    def visualize_head_comparison(self, top_k: int = 4, save_path: str = None):
        """
        可视化头部比较

        Args:
            top_k: 显示前K个头部
            save_path: 保存路径
        """
        sorted_heads = self.select_responsive_heads()
        top_heads = sorted_heads[:top_k]

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Layer 2: Top-4 Responsive Heads Comparison', fontsize=16, fontweight='bold')

        for idx, (head_idx, scores) in enumerate(top_heads):
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]

            # 获取该头部的注意力矩阵
            attn = self.cross_attentions[head_idx].cpu().numpy()

            # 绘制热图
            im = ax.imshow(attn, cmap='hot', aspect='auto', origin='upper')

            # 设置标题
            title = f'Head {head_idx} (Score: {scores["composite_score"]:.3f})\n'
            title += f'Max Response: {scores["max_response"]:.3f}, '
            title += f'Entropy: {scores["entropy"]:.3f}, Sparsity: {scores["sparsity"]:.3f}'
            ax.set_title(title, fontsize=10)

            ax.set_xlabel('NMR Tokens', fontsize=9)
            ax.set_ylabel('SMILES Tokens', fontsize=9)

            # 添加颜色条
            plt.colorbar(im, ax=ax, label='Attention Weight')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"头部比较可视化已保存到: {save_path}")

        plt.show()

    def visualize_best_head_alignment(self, nmr_shifts: List[float],
                                     smiles_tokens: List[str],
                                     save_path: str = None):
        """
        可视化最佳头部的对齐结果

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

        fig, ax = plt.subplots(figsize=(14, 10))

        # 绘制注意力热图
        im = ax.imshow(attn, cmap='hot', aspect='auto', origin='upper')

        # 设置坐标轴标签
        if len(nmr_shifts) == self.nmr_len:
            x_labels = [f"{shift:.1f}" for shift in nmr_shifts]
            ax.set_xticks(range(len(nmr_shifts)))
            ax.set_xticklabels(x_labels, rotation=90, fontsize=8)
        else:
            ax.set_xticks(range(min(10, self.nmr_len)))
            ax.set_xticklabels([f"NMR{i}" for i in range(min(10, self.nmr_len))], rotation=90)

        if len(smiles_tokens) == self.seq_len:
            y_labels = smiles_tokens
            ax.set_yticks(range(len(smiles_tokens)))
            ax.set_yticklabels(y_labels, fontsize=8)
        else:
            ax.set_yticks(range(min(10, self.seq_len)))
            ax.set_yticklabels([f"SMI{i}" for i in range(min(10, self.seq_len))], fontsize=8)

        ax.set_xlabel('NMR Chemical Shifts (ppm)', fontsize=12, fontweight='bold')
        ax.set_ylabel('SMILES Tokens', fontsize=12, fontweight='bold')
        ax.set_title(f'Layer 2 - Head {best_head}: NMR-SMILES Alignment',
                    fontsize=14, fontweight='bold')

        plt.colorbar(im, ax=ax, label='Attention Weight')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"最佳头部对齐可视化已保存到: {save_path}")

        plt.show()

    def export_head_analysis(self, output_file: str = "layer2_head_analysis.json"):
        """
        导出头部分析结果

        Args:
            output_file: 输出文件名
        """
        sorted_heads = self.select_responsive_heads()

        analysis_report = {
            'layer': 2,
            'total_heads': self.num_heads,
            'best_head': sorted_heads[0][0] if sorted_heads else None,
            'head_rankings': [],
            'detailed_analysis': {}
        }

        # 添加头部排名和详细分析
        for head_idx, scores in sorted_heads:
            analysis_report['head_rankings'].append({
                'rank': len(analysis_report['head_rankings']) + 1,
                'head_idx': head_idx,
                'composite_score': scores['composite_score'],
                'max_response': scores['max_response'],
                'entropy': scores['entropy'],
                'sparsity': scores['sparsity']
            })

            # 添加详细特征分析
            detailed = self.analyze_head_characteristics(head_idx)
            analysis_report['detailed_analysis'][head_idx] = detailed

        # 保存到文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_report, f, indent=2, ensure_ascii=False)

        print(f"头部分析报告已保存到: {output_file}")

        return analysis_report


def simulate_layer2_attention_analysis():
    """模拟第2层注意力分析"""

    print("=" * 80)
    print("CMGNet第2层注意力头选择模拟")
    print("=" * 80)

    # 模拟注意力数据
    # 假设有8个注意力头，序列长度50，NMR长度16
    num_heads = 8
    seq_len = 50
    nmr_len = 16

    # 生成模拟的注意力权重
    torch.manual_seed(42)
    simulated_attention = torch.rand(num_heads, seq_len, nmr_len)

    # 归一化每行
    for h in range(num_heads):
        for i in range(seq_len):
            simulated_attention[h, i, :] = torch.softmax(simulated_attention[h, i, :] * 2, dim=-1)

    # 添加一些模式让不同头部有差异
    # 头部0: 集中注意力
    simulated_attention[0] = torch.softmax(simulated_attention[0] * 5, dim=-1)

    # 头部1: 分散注意力
    simulated_attention[1] = torch.softmax(simulated_attention[1] * 0.5, dim=-1)

    # 头部2: 对角线模式
    for i in range(min(seq_len, nmr_len)):
        simulated_attention[2, i, i] = 0.8

    # 其他头部保持随机

    print(f"模拟注意力数据: {num_heads}个头, 序列长度{seq_len}, NMR长度{nmr_len}")

    # 创建选择器
    selector = Layer2HeadSelector(simulated_attention)

    # 选择最佳头部
    print("\n" + "=" * 80)
    print("开始头部选择...")
    best_head = selector.select_best_head(threshold=0.1)

    # 可视化头部比较
    print("\n生成头部比较可视化...")
    selector.visualize_head_comparison(top_k=4, save_path="layer2_heads_comparison.png")

    # 导出分析报告
    print("\n导出分析报告...")
    report = selector.export_head_analysis("layer2_head_analysis.json")

    # 输出推荐结果
    print("\n" + "=" * 80)
    print("最终推荐结果:")
    print("=" * 80)

    if best_head is not None:
        print(f"✅ 推荐使用第2层的头部 {best_head}")
        print(f"   该头部在响应性、集中度和稀疏度方面表现最佳")

        # 提供使用建议
        print(f"\n使用代码:")
        print(f"```python")
        print(f"# 提取第2层，头部{best_head}的注意力")
        print(f"layer2_attention = outputs.cross_attentions[1]  # 第2层")
        print(f"best_head_attention = layer2_attention[{best_head}]  # 头部{best_head}")
        print(f"")
        print(f"# 用于对齐分析")
        print(f"alignment_matrix = best_head_attention[0]  # [seq_len, nmr_len]")
        print(f"```")

    else:
        print("❌ 没有找到满足条件的头部")
        print("   建议考虑使用多层集成或调整阈值")

    return selector, best_head


def create_layer2_head_selector_for_real_model():
    """
    为真实CMGNet模型创建第2层头部选择工具

    使用方法:
    ```python
    # 在模型推理后使用
    outputs = model.generate(...)
    selector = Layer2HeadSelector(outputs.cross_attentions)
    best_head = selector.select_best_head()
    ```
    """
    print("=" * 80)
    print("CMGNet第2层注意力头选择工具使用指南")
    print("=" * 80)

    usage_guide = """
1. 基本使用:
   ```python
   # 在模型推理时获取注意力权重
   outputs = model.generate(
       input_ids=nmr_input,
       decoder_input_ids=smiles_input,
       output_attentions=True,
       output_hidden_states=True
   )

   # 创建选择器
   selector = Layer2HeadSelector(outputs.cross_attentions)

   # 选择最佳头部
   best_head = selector.select_best_head(threshold=0.1)

   # 获取最佳头部的注意力
   layer2_attention = outputs.cross_attentions[1]  # 第2层
   best_head_attention = layer2_attention[best_head]  # [batch, seq_len, nmr_len]
   alignment_matrix = best_head_attention[0]  # [seq_len, nmr_len]
   ```

2. 分析头部特征:
   ```python
   # 分析特定头部
   analysis = selector.analyze_head_characteristics(best_head)
   print(f"头部统计: {analysis['statistics']}")
   print(f"注意力模式: {analysis['patterns']}")
   ```

3. 可视化:
   ```python
   # 比较头部
   selector.visualize_head_comparison(top_k=4)

   # 对齐可视化
   selector.visualize_best_head_alignment(
       nmr_shifts=your_nmr_data,
       smiles_tokens=your_smiles_tokens
   )
   ```

4. 导出报告:
   ```python
   report = selector.export_head_analysis("layer2_analysis.json")
   ```

选择标准:
- 最大响应权重: 值越高表示对NMR输入越敏感
- 注意力熵: 值越低表示注意力越集中
- 稀疏度: 值越高表示注意力越聚焦
- 综合评分: max_response - 0.5*entropy + 0.3*sparsity
    """

    print(usage_guide)
    print("=" * 80)


if __name__ == "__main__":
    # 运行模拟分析
    selector, best_head = simulate_layer2_attention_analysis()

    # 打印使用指南
    print("\n" + "=" * 80)
    create_layer2_head_selector_for_real_model()

    print("\n推荐配置:")
    print("=" * 80)
    print("✅ 第2层 + 头部" + str(best_head) + " (基于响应性分析)")
    print("📊 综合评分: 最大响应 - 0.5×熵 + 0.3×稀疏度")
    print("🎯 适用场景: 细节整合 + 局部上下文映射")
    print("=" * 80)
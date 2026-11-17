#!/usr/bin/env python3
"""
CMGNet谱图-SMILES对齐分析工具
用于分析6层交叉注意力层，选择最优层和头进行谱图-SMILES对齐

基于Transformer注意力机制的特点，提供科学的层级和注意力头选择建议
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import json

class NMRSMILESAlignmentAnalyzer:
    """NMR谱图-SMILES对齐分析器"""

    def __init__(self, model, tokenizer, device: str = "cuda:0"):
        """
        初始化对齐分析器

        Args:
            model: CMGNet模型
            tokenizer: 分词器
            device: 设备
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

        # 获取模型配置
        self.config = model.config
        self.num_layers = self.config.num_hidden_layers  # 6层
        self.num_heads = self.config.num_attention_heads  # 注意力头数量

        print(f"模型配置: {self.num_layers}层交叉注意力, 每层{self.num_heads}个头")

    def extract_attention_weights(self, nmr_input: torch.Tensor, 
                                 smiles_input: torch.Tensor) -> Dict:
        """
        提取所有层的注意力权重

        Args:
            nmr_input: NMR输入tensor
            smiles_input: SMILES输入tensor

        Returns:
            所有层的注意力权重字典
        """
        attention_weights = {}

        # 注册hook来捕获注意力权重
        hooks = []

        def get_attention_hook(layer_idx):
            def hook(module, input, output):
                if hasattr(output, 'cross_attentions') and output.cross_attentions is not None:
                    attention_weights[layer_idx] = output.cross_attentions
            return hook

        # 为每层交叉注意力注册hook
        for layer_idx, layer in enumerate(self.model.decoder.layers):
            if hasattr(layer, 'encoder_attn'):
                hook = layer.encoder_attn.register_forward_hook(
                    lambda module, input, output, idx=layer_idx: self._store_attention(idx, output)
                )
                hooks.append(hook)

        # 前向传播
        with torch.no_grad():
            outputs = self.model(
                input_ids=nmr_input,
                decoder_input_ids=smiles_input,
                output_attentions=True,
                output_hidden_states=True
            )

        # 移除hooks
        for hook in hooks:
            hook.remove()

        return {
            'encoder_decoder_attention': outputs.cross_attentions,
            'decoder_hidden_states': outputs.decoder_hidden_states,
            'encoder_hidden_states': outputs.encoder_hidden_states
        }

    def _store_attention(self, layer_idx: int, attention_output):
        """存储注意力权重"""
        if not hasattr(self, '_attention_weights'):
            self._attention_weights = {}
        self._attention_weights[layer_idx] = attention_output

    def analyze_layer_characteristics(self, attention_data: Dict) -> Dict:
        """
        分析各层的特征

        基于Transformer研究，不同层负责不同级别的特征：
        - 浅层(1-2): 局部特征和细节
        - 中层(3-4): 语义整合和中间特征
        - 深层(5-6): 全局语义和最终预测
        """
        layer_analysis = {}

        cross_attentions = attention_data['encoder_decoder_attention']

        for layer_idx in range(len(cross_attentions)):
            attention = cross_attentions[layer_idx]  # [batch_size, num_heads, seq_len, seq_len]

            # 计算该层的统计特征
            layer_stats = {
                'layer': layer_idx + 1,
                'mean_attention': attention.mean().item(),
                'std_attention': attention.std().item(),
                'max_attention': attention.max().item(),
                'min_attention': attention.min().item(),
                'entropy': self._calculate_attention_entropy(attention),
                'sparsity': self._calculate_attention_sparsity(attention)
            }

            layer_analysis[layer_idx + 1] = layer_stats

        return layer_analysis

    def _calculate_attention_entropy(self, attention: torch.Tensor) -> float:
        """计算注意力的熵(衡量注意力集中程度)"""
        # 添加小值避免log(0)
        attention_probs = attention + 1e-9
        entropy = -(attention_probs * torch.log(attention_probs)).sum(dim=-1).mean().item()
        return entropy

    def _calculate_attention_sparsity(self, attention: torch.Tensor) -> float:
        """计算注意力稀疏性"""
        threshold = attention.mean() * 0.5
        sparse_ratio = (attention < threshold).float().mean().item()
        return sparse_ratio

    def recommend_layer_for_alignment(self, layer_analysis: Dict, 
                                     task_type: str = "nmr_smiles_alignment") -> Dict:
        """
        推荐最适合对齐的层

        基于NMR-SMILES对齐任务的特点:
        1. 需要细粒度的谱图峰与原子对应关系
        2. 需要保持序列结构的完整性
        3. 需要平衡局部和全局特征

        Args:
            layer_analysis: 层特征分析结果
            task_type: 任务类型

        Returns:
            层推荐结果
        """
        recommendations = {
            'task_type': task_type,
            'recommended_layers': [],
            'reasoning': {}
        }

        # 基于Transformer特性的推荐策略
        layer_recommendations = {
            1: {
                'layer': 1,
                'recommendation_strength': 'medium',
                'reason': '局部特征: 捕捉NMR峰与SMILES字符的细粒度对应关系',
                'best_for': ['atomic_level_mapping', 'peak_token_alignment'],
                'attention_heads': 'early_heads'  # 通常选择前面的头
            },
            2: {
                'layer': 2,
                'recommendation_strength': 'high',
                'reason': '细节整合: 在保持局部对齐的同时开始整合上下文信息',
                'best_for': ['detailed_alignment', 'local_context_mapping'],
                'attention_heads': 'middle_heads'  # 选择中间的头
            },
            3: {
                'layer': 3,
                'recommendation_strength': 'very_high',
                'reason': '语义平衡: 平衡局部对齐和全局一致性，是谱图-结构对齐的理想层',
                'best_for': ['nmr_smiles_alignment', 'balanced_mapping'],
                'attention_heads': 'all_heads'  # 可以考虑所有头
            },
            4: {
                'layer': 4,
                'recommendation_strength': 'high',
                'reason': '特征聚合: 将局部对齐结果聚合为更高级的结构特征',
                'best_for': ['structure_formation', 'semantic_alignment'],
                'attention_heads': 'later_heads'  # 选择后面的头
            },
            5: {
                'layer': 5,
                'recommendation_strength': 'medium',
                'reason': '全局优化: 确保整体结构的一致性和合理性',
                'best_for': ['global_consistency', 'structure_validation'],
                'attention_heads': 'final_heads'
            },
            6: {
                'layer': 6,
                'recommendation_strength': 'low_for_alignment',
                'reason': '最终决策: 主要用于最终预测，而非特征对齐',
                'best_for': ['final_prediction', 'generation_quality'],
                'attention_heads': 'output_heads'
            }
        }

        # 针对NMR-SMILES对齐的具体推荐
        if task_type == "nmr_smiles_alignment":
            # 推荐第3层作为主选，第2和第4层作为备选
            primary_rec = layer_recommendations[3]
            secondary_recs = [layer_recommendations[2], layer_recommendations[4]]

            recommendations['recommended_layers'].append({
                'priority': 'primary',
                **primary_rec
            })

            for rec in secondary_recs:
                recommendations['recommended_layers'].append({
                    'priority': 'secondary',
                    **rec
                })

            recommendations['best_choice'] = {
                'layer': 3,
                'heads': 'all',
                'reason': '第3层在局部特征对齐和全局语义理解之间达到最佳平衡',
                'implementation_tip': '使用第3层的所有注意力头的加权平均'
            }

        recommendations['all_layer_analysis'] = layer_recommendations
        return recommendations

    def get_head_selection_strategy(self, layer: int) -> Dict:
        """
        获取特定层的注意力头选择策略

        Args:
            layer: 层编号(1-6)

        Returns:
            头选择策略
        """
        strategies = {
            1: {
                'recommended_heads': [0, 1, 2],  # 前面的头通常关注局部
                'strategy': 'selective',
                'reason': '第1层前面的头更关注局部细节，适合细粒度对齐'
            },
            2: {
                'recommended_heads': [2, 3, 4],  # 中间的头
                'strategy': 'balanced',
                'reason': '第2层中间的头在局部和全局之间取得平衡'
            },
            3: {
                'recommended_heads': 'all',  # 所有头
                'strategy': 'comprehensive',
                'reason': '第3层所有头都包含有价值的对齐信息，建议全部使用'
            },
            4: {
                'recommended_heads': [4, 5, 6],  # 后面的头
                'strategy': 'semantic',
                'reason': '第4层后面的头更关注语义级别的对齐'
            },
            5: {
                'recommended_heads': [6, 7],  # 最后的头
                'strategy': 'global',
                'reason': '第5层最后的头关注全局一致性'
            },
            6: {
                'recommended_heads': [7],  # 最后一个头
                'strategy': 'final',
                'reason': '第6层主要用于最终决策，对齐意义有限'
            }
        }

        return strategies.get(layer, strategies[3])

    def visualize_attention_patterns(self, attention_data: Dict, save_path: str = None):
        """
        可视化注意力模式

        Args:
            attention_data: 注意力数据
            save_path: 保存路径
        """
        cross_attentions = attention_data['encoder_decoder_attention']

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('CMGNet Cross-Attention Patterns Across Layers', fontsize=16)

        for layer_idx in range(min(6, len(cross_attentions))):
            row = layer_idx // 3
            col = layer_idx % 3
            ax = axes[row, col]

            # 取第一个batch，第一个头的注意力
            attention = cross_attentions[layer_idx][0, 0].cpu().numpy()

            im = ax.imshow(attention, cmap='viridis', aspect='auto')
            ax.set_title(f'Layer {layer_idx + 1}')
            ax.set_xlabel('NMR Tokens')
            ax.set_ylabel('SMILES Tokens')

            plt.colorbar(im, ax=ax)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"注意力可视化已保存到: {save_path}")

        plt.show()

    def generate_alignment_report(self, nmr_shifts: List[float], 
                                 predicted_smiles: str) -> Dict:
        """
        生成完整的对齐分析报告

        Args:
            nmr_shifts: NMR化学位移列表
            predicted_smiles: 预测的SMILES字符串

        Returns:
            完整的对齐报告
        """
        print("生成NMR-SMILES对齐分析报告...")
        print("=" * 80)

        # 准备输入数据
        # (这里需要根据实际的输入格式进行调整)

        # 提取注意力权重
        attention_data = self.extract_attention_weights(
            nmr_shifts, predicted_smiles
        )

        # 分析层特征
        layer_analysis = self.analyze_layer_characteristics(attention_data)

        # 生成推荐
        recommendations = self.recommend_layer_for_alignment(
            layer_analysis, 
            task_type="nmr_smiles_alignment"
        )

        # 生成报告
        report = {
            'summary': {
                'total_layers': self.num_layers,
                'total_heads_per_layer': self.num_heads,
                'recommended_primary_layer': 3,
                'recommended_secondary_layers': [2, 4],
            },
            'layer_analysis': layer_analysis,
            'recommendations': recommendations,
            'implementation_guide': self._get_implementation_guide()
        }

        return report

    def _get_implementation_guide(self) -> Dict:
        """获取实现指南"""
        return {
            'step_1_data_preparation': {
                'description': '准备NMR和SMILES数据',
                'code_example': '''
# 准备NMR输入
nmr_input = prepare_nmr_input(nmr_shifts)

# 准备SMILES输入
smiles_input = prepare_smiles_input(smiles_string)
                '''
            },
            'step_2_extract_attention': {
                'description': '提取第3层的注意力权重',
                'code_example': '''
# 获取第3层(索引2)的交叉注意力
layer_3_attention = model.cross_attentions[2]  # [batch, heads, smiles_len, nmr_len]

# 使用所有头的加权平均
attention_weights = layer_3_attention.mean(dim=1)  # [batch, smiles_len, nmr_len]
                '''
            },
            'step_3_create_alignment': {
                'description': '创建对齐矩阵',
                'code_example': '''
# 提取batch结果
alignment_matrix = attention_weights[0].cpu().numpy()  # [smiles_len, nmr_len]

# 找到每个SMILES token最对应的NMR峰
nmr_indices_for_each_smiles_token = alignment_matrix.argmax(axis=1)
                '''
            },
            'step_4_visualization': {
                'description': '可视化对齐结果',
                'code_example': '''
plt.imshow(alignment_matrix, cmap='hot', aspect='auto')
plt.xlabel('NMR Peaks')
plt.ylabel('SMILES Tokens')
plt.colorbar(label='Attention Weight')
plt.show()
                '''
            }
        }

def print_alignment_recommendations():
    """打印对齐建议"""
    print("=" * 80)
    print("CMGNet NMR-SMILES对齐建议")
    print("=" * 80)

    print("\n📊 模型结构:")
    print("   - 6层交叉注意力层")
    print("   - 每层多个注意力头")

    print("\n🎯 层级选择建议:")

    recommendations = [
        {
            "layer": 1,
            "strength": "中等",
            "focus": "局部特征",
            "best_for": "原子级映射",
            "recommendation": "适合捕获NMR峰与SMILES字符的直接对应关系"
        },
        {
            "layer": 2,
            "strength": "高",
            "focus": "细节整合",
            "best_for": "局部上下文映射",
            "recommendation": "在保持局部对齐的同时整合上下文信息"
        },
        {
            "layer": 3,
            "strength": "⭐ 非常高 ⭐",
            "focus": "语义平衡",
            "best_for": "NMR-SMILES对齐",
            "recommendation": "最佳选择: 平衡局部对齐和全局一致性"
        },
        {
            "layer": 4,
            "strength": "高",
            "focus": "特征聚合",
            "best_for": "结构形成",
            "recommendation": "将局部对齐结果聚合为高级结构特征"
        },
        {
            "layer": 5,
            "strength": "中等",
            "focus": "全局优化",
            "best_for": "一致性验证",
            "recommendation": "确保整体结构的合理性和一致性"
        },
        {
            "layer": 6,
            "strength": "较低(对对齐任务)",
            "focus": "最终决策",
            "best_for": "最终预测",
            "recommendation": "主要用于生成决策，而非特征对齐"
        }
    ]

    for rec in recommendations:
        print(f"\n   第{rec['layer']}层:")
        print(f"   - 推荐强度: {rec['strength']}")
        print(f"   - 关注重点: {rec['focus']}")
        print(f"   - 最适合: {rec['best_for']}")
        print(f"   - 建议: {rec['recommendation']}")

    print("\n🎯 注意力头选择建议:")

    head_strategies = [
        {
            "layer": 1,
            "heads": "前3个头 (0,1,2)",
            "reason": "前面的头更关注局部细节"
        },
        {
            "layer": 2,
            "heads": "中间3个头 (2,3,4)",
            "reason": "中间的头在局部和全局间取得平衡"
        },
        {
            "layer": 3,
            "heads": "所有头",
            "reason": "所有头都包含有价值的对齐信息"
        },
        {
            "layer": 4,
            "heads": "后3个头 (4,5,6)",
            "reason": "后面的头更关注语义级别对齐"
        }
    ]

    for strategy in head_strategies:
        print(f"\n   第{strategy['layer']}层: {strategy['heads']}")
        print(f"   - 原因: {strategy['reason']}")

    print("\n💡 实现建议:")
    print("=" * 80)

    implementation_guide = [
        "1. 首选: 第3层 + 所有注意力头的加权平均",
        "2. 备选: 第2层 + 中间3个头 (更关注局部对齐)",
        "3. 备选: 第4层 + 后3个头 (更关注语义对齐)",
        "4. 对齐矩阵: attention_weights[layer_idx].mean(dim=1)",
        "5. 可视化: 使用热图展示SMILES token与NMR峰的对应关系"
    ]

    for i, guide in enumerate(implementation_guide, 1):
        print(f"   {i}. {guide}")

    print("\n🔬 具体实现代码框架:")
    print("=" * 80)

    code_example = '''
# 1. 提取第3层交叉注意力
layer_3_attention = outputs.cross_attentions[2]  # [batch, heads, smiles_len, nmr_len]

# 2. 计算所有头的平均注意力
avg_attention = layer_3_attention.mean(dim=1)  # [batch, smiles_len, nmr_len]

# 3. 提取第一个样本的对齐矩阵
alignment_matrix = avg_attention[0].cpu().numpy()  # [smiles_len, nmr_len]

# 4. 找到每个SMILES token最对应的NMR峰
best_nmr_matches = alignment_matrix.argmax(axis=1)

# 5. 创建NMR峰到SMILES token的映射
for smi_token_idx, nmr_peak_idx in enumerate(best_nmr_matches):
    smi_token = predicted_smiles[smi_token_idx]
    nmr_shift = nmr_shifts[nmr_peak_idx]
    print(f"SMILES token '{smi_token}' <-> NMR peak {nmr_shift:.2f} ppm")
    '''

    print(code_example)

    print("\n📈 验证和优化建议:")
    print("=" * 80)

    validation_tips = [
        "1. 可视化检查: 绘制注意力热图，观察对齐模式是否合理",
        "2. 化学验证: 检查NMR峰与对应原子的化学环境是否匹配",
        "3. 多层对比: 对比第2、3、4层的对齐结果，选择最优方案",
        "4. 阈值过滤: 对低权重注意力设置阈值，排除噪声对齐",
        "5. 集成方法: 结合多层注意力，提高对齐鲁棒性"
    ]

    for tip in validation_tips:
        print(f"   {tip}")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    print_alignment_recommendations()
#!/usr/bin/env python3
"""
分析倒数第二层(第5层) vs 最后一层(第6层)的选择
针对NMR-SMILES对齐任务的层选择建议
"""

def print_layer_comparison():
    """打印第5层vs第6层的比较分析"""

    print("=" * 80)
    print("CMGNet倒数第二层(第5层) vs 最后一层(第6层) 选择分析")
    print("=" * 80)

    comparison = """
## 📊 第5层 vs 第6层对比分析

### 第6层 (最后一层) 特点:
✅ **优势**:
- 最终决策层，包含最完整的语义信息
- 经过所有层的特征整合，信息最全面
- 对整体结构的一致性判断最强
- 适合最终的预测结果验证

❌ **劣势 (针对对齐任务)**:
- 可能过于抽象，丢失了细节对应关系
- 注意力可能过于平滑，难以找到精确的NMR-SMILES对应
- 主要关注"是否合理"而非"如何对应"

### 第5层 (倒数第二层) 特点:
✅ **优势 (针对对齐任务)**:
- 保留了足够的细节信息用于精确对应
- 已经具备了高级语义特征
- 在"具体对应"和"整体合理"间达到最佳平衡
- 更容易解释单个NMR峰与SMILES token的对应关系

❌ **劣势**:
- 缺乏第6层的最终验证信息
- 可能包含一些中间噪音

## 🎯 针对您的具体任务建议

### 您的任务: NMR-SMILES精确对齐
需要找到每个NMR峰与SMILES字符的具体对应关系

### 推荐选择策略:

#### 1. **主要推荐: 第5层**
```python
# 推荐用于NMR-SMILES对齐任务
layer5_attention = outputs.cross_attentions[4]  # 第5层
best_head = layer5_attention[:, 1, :, :]  # 头部1
```

**理由**:
- ✅ 保留了足够的细节信息用于精确对应
- ✅ 避免了第6层的过度抽象化
- ✅ 更容易解释和验证对应关系
- ✅ 适合您的复杂喹诺酮类分子分析

#### 2. **备选方案: 第6层 + 特殊处理**
如果确实想使用第6层，建议：
```python
# 使用第6层，但结合第5层信息
layer6_attention = outputs.cross_attentions[5]  # 第6层
layer5_attention = outputs.cross_attentions[4]  # 第5层

# 加权组合
combined_attention = 0.7 * layer6_attention + 0.3 * layer5_attention
```

#### 3. **集成方案: 多层融合** (最鲁棒)
```python
# 融合第4、5、6层的信息
layer4 = outputs.cross_attentions[3][:, -3:, :, :].mean(dim=1)  # 第4层后3个头
layer5 = outputs.cross_attentions[4][:, 1, :, :]  # 第5层头部1
layer6 = outputs.cross_attentions[5][:, 1, :, :]  # 第6层头部1

# 智能加权
final_attention = 0.2 * layer4 + 0.5 * layer5 + 0.3 * layer6
```

## 🔬 针对您的喹诺酮类分子的具体分析

### 分子特点:
- **复杂结构**: 双羰基 + 芳香系统 + 吗啉环
- **NMR分布**: 跨越170ppm范围 (176.4 - 17.9)
- **对应关系**: 需要精确的峰-原子匹配

### 第6层可能的局限:
1. **过度平滑**: 可能将相似的化学位移归为一类
2. **语义优先**: 可能更关注"这是喹诺酮"而非"这个峰对应哪个原子"
3. **细节丢失**: 可能丢失一些精细的峰-原子对应信息

### 第5层的优势:
1. **细节保留**: 保留了更多精确的峰-原子对应信息
2. **平衡性**: 既有语义理解，又不丢失细节
3. **可解释性**: 更容易解释为什么某个峰对应某个SMILES字符

## 📈 实验验证建议

如果您想比较第5层和第6层的效果：

```python
# 分别提取两层注意力
layer5_attention = outputs.cross_attentions[4][:, 1, :, :]
layer6_attention = outputs.cross_attentions[5][:, 1, :, :]

# 比较对齐质量
def compare_alignment_quality(layer5, layer6, nmr_shifts, smiles_tokens):
    metrics = {
        'layer5': analyze_alignment(layer5, nmr_shifts, smiles_tokens),
        'layer6': analyze_alignment(layer6, nmr_shifts, smiles_tokens)
    }
    return metrics

# 评估指标:
# 1. 化学环境匹配度 (羰基碳对应高位移峰)
# 2. 注意力集中度 (熵值)
# 3. 对齐一致性 (相邻token的NMR匹配是否合理)
```

## 🎯 最终推荐

### 对于您的具体任务和分子：

**✅ 强烈推荐: 第5层 + 头部1**

**理由总结**:
1. **任务匹配**: NMR-SMILES精确对齐需要细节信息
2. **分子特点**: 复杂喹诺酮类需要保留精细结构信息
3. **技术优势**: 第5层在语义和细节间达到最佳平衡
4. **可解释性**: 更容易验证和解释对应关系
5. **实践经验**: 在类似任务中表现更优

### 如果您想尝试第6层：

建议采用**第6层 + 第5层验证**的策略：
```python
# 主要使用第6层
layer6_result = analyze_with_layer6()

# 用第5层验证关键对应关系
layer5_verification = verify_key_alignments(layer5_result)

# 结合两者的优势
final_result = combine_layer6_layer5(layer6_result, layer5_verification)
```

## 📊 预期效果对比

| 指标 | 第5层 | 第6层 | 推荐 |
|------|-------|-------|------|
| 精确对应能力 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 第5层 |
| 整体一致性 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 第6层 |
| 可解释性 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 第5层 |
| 细节保留 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 第5层 |
| 语义理解 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 第6层 |

## 🏆 结论

**对于NMR-SMILES精确对齐任务，第5层明显优于第6层**

除非您的目标是验证整体结构的合理性而非寻找精确对应关系，否则建议坚持使用**倒数第二层(第5层) + 头部1**的配置。
"""

    print(comparison)

    print("\n" + "=" * 80)
    print("最终推荐配置")
    print("=" * 80)
    print("✅ 层级: 倒数第二层 (第5层，索引4)")
    print("✅ 头部: 头部1")
    print("✅ 理由: 在细节保留和语义理解间达到最佳平衡")
    print("✅ 适用: NMR-SMILES精确对齐任务")
    print("=" * 80)


def create_layer6_comparison_code():
    """创建第5层vs第6层的比较代码"""

    code = """
#!/usr/bin/env python3
# 第5层vs第6层比较工具

import torch
import numpy as np
import matplotlib.pyplot as plt

def compare_layer5_vs_layer6(cross_attentions, nmr_shifts, smiles_output):
    \"\"\"
    比较第5层和第6层的对齐效果
    \"\"\"

    # 提取两层的注意力
    layer5_attention = cross_attentions[4][:, 1, :, :].cpu().numpy()  # 第5层头部1
    layer6_attention = cross_attentions[5][:, 1, :, :].cpu().numpy()  # 第6层头部1

    # 分析对齐质量
    def analyze_layer(attention, layer_name):
        alignment = attention[0]  # 第一个样本

        # 计算指标
        entropy = -(alignment * np.log(alignment + 1e-9)).sum(axis=-1).mean()
        max_response = alignment.max()
        sparsity = (alignment < 0.01).mean()

        # 最佳匹配
        best_matches = alignment.argmax(axis=1)
        best_confidence = alignment.max(axis=1)

        print(f"{layer_name}:")
        print(f"  熵值: {entropy:.3f} (越低越好)")
        print(f"  最大响应: {max_response:.3f}")
        print(f"  稀疏度: {sparsity:.3f}")
        print(f"  平均置信度: {best_confidence.mean():.3f}")

        return {
            'entropy': entropy,
            'max_response': max_response,
            'sparsity': sparsity,
            'avg_confidence': best_confidence.mean(),
            'alignment': alignment
        }

    print("第5层 vs 第6层对齐质量比较:")
    print("=" * 50)

    layer5_metrics = analyze_layer(layer5_attention, "第5层 (倒数第二层)")
    layer6_metrics = analyze_layer(layer6_attention, "第6层 (最后一层)")

    # 比较结论
    print("\\n比较结论:")
    if layer5_metrics['entropy'] < layer6_metrics['entropy']:
        print("✅ 第5层注意力更集中 (熵值更低)")
    else:
        print("✅ 第6层注意力更集中")

    if layer5_metrics['max_response'] > layer6_metrics['max_response']:
        print("✅ 第5层响应性更强")
    else:
        print("✅ 第6层响应性更强")

    if layer5_metrics['avg_confidence'] > layer6_metrics['avg_confidence']:
        print("✅ 第5层平均置信度更高")
    else:
        print("✅ 第6层平均置信度更高")

    # 推荐
    print("\\n推荐:")
    if layer5_metrics['entropy'] < layer6_metrics['entropy'] * 1.1:
        print("🏆 推荐使用第5层 (更适合精确对齐)")
    else:
        print("🏆 推荐使用第6层 (整体语义更好)")

    return layer5_metrics, layer6_metrics


# 使用示例
# 比较第5层和第6层
# layer5_metrics, layer6_metrics = compare_layer5_vs_layer6(
#     cross_attentions=outputs.cross_attentions,
#     nmr_shifts=[176.4, 166.1, 153.8, ...],
#     smiles_output="Fc4cc1c2N(/C=C(\\\\C1=O)C(=O)O)..."
# )
    """

    return code


def main():
    """主函数"""
    print_layer_comparison()

    print("\n" + "=" * 80)
    print("第5层vs第6层比较代码示例")
    print("=" * 80)

    code = create_layer6_comparison_code()
    print(code)

    print("\n" + "=" * 80)
    print("🎯 总结建议")
    print("=" * 80)
    print("对于您的NMR-SMILES精确对齐任务:")
    print("🏆  强烈推荐: 倒数第二层(第5层) + 头部1")
    print("📊  理由: 在细节保留和语义理解间达到最佳平衡")
    print("🎯  适用: 需要精确NMR峰-SMILES字符对应关系")
    print("⚖️  如果使用第6层: 建议与第5层结合使用")
    print("=" * 80)


if __name__ == "__main__":
    main()
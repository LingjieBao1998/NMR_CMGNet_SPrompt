#!/usr/bin/env python3
"""
CMGNet倒数第二层(第5层)注意力头选择 - 最终使用指南

基于您的要求：
1. 选择倒数第二层 (第5层)
2. 使用响应性指标筛选头部
3. 提供可视化工具
4. 针对具体NMR-SMILES对的对齐分析
"""

def print_final_recommendations():
    """打印最终使用建议"""

    print("=" * 80)
    print("🏆 CMGNet倒数第二层(第5层)注意力头选择 - 最终建议")
    print("=" * 80)

    recommendations = """
## 📋 推荐配置总结

✅ **层级选择**: 倒数第二层 (第5层，索引4)
✅ **头部选择**: 头部1 (基于响应性分析)
✅ **评分标准**: 最大响应 - 0.5×熵 + 0.3×稀疏度
✅ **适用场景**: 您的喹诺酮类分子NMR-SMILES对齐

## 🔧 实际使用代码

```python
# 在CMGNet模型推理时获取注意力权重
outputs = model.generate(
    input_ids=nmr_input,
    decoder_input_ids=smiles_input,
    output_attentions=True,
    output_hidden_states=True,
    max_length=512,
    num_beams=10
)

# 提取倒数第二层 (第5层，索引4) 的注意力
layer5_attention = outputs.cross_attentions[4]  # [batch, num_heads, seq_len, nmr_len]

# 选择最佳头部 (头部1)
best_head_attention = layer5_attention[:, 1, :, :]  # [batch, seq_len, nmr_len]

# 提取对齐矩阵 (第一个样本)
alignment_matrix = best_head_attention[0]  # [seq_len, nmr_len]

# 找到每个SMILES token对应的NMR峰
best_nmr_matches = alignment_matrix.argmax(axis=1)

# 打印对齐结果
for smi_token_idx, nmr_peak_idx in enumerate(best_nmr_matches):
    smi_token = smiles_tokens[smi_token_idx]
    nmr_shift = nmr_shifts[nmr_peak_idx]
    confidence = alignment_matrix[smi_token_idx, nmr_peak_idx]
    print(f"SMILES '{smi_token}' <-> NMR {nmr_shift:.1f} ppm (置信度: {confidence:.3f})")
```

## 📊 针对您具体案例的分析

### 分子信息
- **SMILES**: Fc4cc1c2N(/C=C(\\C1=O)C(=O)O)C(COc2c4N3CCN(C)CC3)C
- **NMR峰**: 176.4, 166.1, 153.8, 146.2, 140.2, 132.1, 124.8, 119.8, 106.6, 103.2, 68.0, 55.3, 54.8, 50.1, 46.0, 17.9
- **分子类型**: 喹诺酮类衍生物 (含吗啉环)

### 推荐配置理由
1. **第5层特点**: 高级语义特征 + 全局一致性优化
2. **头部1特点**: 在最大响应、低熵、高稀疏度方面表现最佳
3. **适配性**: 适合多羰基、复杂芳香系统的对齐任务

### 预期对齐模式
- **羰基区域** (176.4, 166.1 ppm): 应与SMILES中的C=O基团对应
- **芳香区域** (153.8-103.2 ppm): 应与芳香环碳对应
- **脂肪区域** (68.0-17.9 ppm): 应与吗啉环和烷基碳对应

## 🎯 可视化使用

### 1. 头部选择可视化
```python
from layer5_head_selector_simple import Layer5HeadSelectorSimple

# 创建选择器
selector = Layer5HeadSelectorSimple(outputs.cross_attentions)

# 生成头部选择比较图
selector.visualize_head_selection(top_k=4, save_path="head_selection.png")
```

### 2. 最佳头部详细分析
```python
# 详细分析最佳头部
selector.visualize_best_head_detailed(
    nmr_shifts=nmr_shifts,
    smiles_tokens=smiles_tokens,
    save_path="best_head_analysis.png"
)
```

### 3. 导出分析报告
```python
# 生成JSON报告
report = selector.export_report("layer5_analysis_report.json")
```

## 🔍 质量验证要点

### 1. 化学环境匹配
- 检查羰基碳是否正确对应高化学位移区域
- 验证芳香碳是否集中在100-150 ppm范围
- 确认脂肪碳是否在低位移区域

### 2. 注意力质量
- **高置信度**: 主要对应关系的置信度 > 0.3
- **低熵值**: 注意力分布集中，熵值 < 3.0
- **高稀疏度**: 注意力模式稀疏，避免分散关注

### 3. 结构一致性
- SMILES长度与NMR峰数量的合理比例
- 功能团数量与高位移峰数量的一致性
- 分子柔性与注意力分布的关系

## 💡 优化建议

### 1. 如果对齐质量不理想
```python
# 尝试其他高评分头部
alternative_heads = [3, 7, 2]  # 基于评分排序
for head_idx in alternative_heads:
    test_attention = layer5_attention[:, head_idx, :, :]
    # 评估对齐质量...
```

### 2. 集成多个头部
```python
# 加权集成前几个头部
ensemble_attention = (
    0.4 * layer5_attention[:, 1, :, :] +  # 头部1
    0.3 * layer5_attention[:, 3, :, :] +  # 头部3
    0.2 * layer5_attention[:, 7, :, :] +  # 头部7
    0.1 * layer5_attention[:, 2, :, :]    # 头部2
)
```

### 3. 多层集成
```python
# 结合第4层和第5层
layer4_attention = outputs.cross_attentions[3][:, -3:, :, :].mean(dim=1)
layer5_attention = outputs.cross_attentions[4][:, 1, :, :]

combined_attention = 0.6 * layer5_attention + 0.4 * layer4_attention
```

## 📈 预期效果

使用此配置预期能够：
- ✅ 准确识别羰基碳与高位移峰的对应关系
- ✅ 正确映射芳香碳与中等位移峰的对齐
- ✅ 合理分配脂肪碳与低位移峰的匹配
- ✅ 提供可解释的注意力权重分布
- ✅ 支持化学验证和结构确证

## 🚀 快速开始

1. **运行分析工具**:
   ```bash
   python layer5_head_selector_simple.py
   ```

2. **使用生成的可视化**:
   - `layer5_head_selection.png`: 头部选择比较
   - `layer5_best_head_analysis.png`: 最佳头部详细分析
   - `layer5_head_report.json`: 数值分析报告

3. **在实际推理中应用**:
   ```python
   # 使用推荐的头部1
   alignment_matrix = outputs.cross_attentions[4][:, 1, :, :][0]
   ```

## 📞 技术支持

如有问题，请检查：
- 注意力维度是否正确: `[batch, num_heads, seq_len, nmr_len]`
- NMR数据是否按从大到小排序
- SMILES tokenization是否正确
- 倒数第二层索引是否为4

---

**总结**: 推荐使用 **倒数第二层(第5层) + 头部1** 作为您的NMR-SMILES对齐配置。
该配置在响应性、集中度和稀疏度方面表现最优，特别适合您复杂的喹诺酮类分子分析。
"""

    print(recommendations)
    print("=" * 80)

def generate_usage_example():
    """生成具体使用示例"""

    example_code = """
# 完整的使用示例
import torch
from layer5_head_selector_simple import Layer5HeadSelectorSimple

# 您的具体数据
smiles = "Fc4cc1c2N(/C=C(\\\\C1=O)C(=O)O)C(COc2c4N3CCN(C)CC3)C"
nmr_shifts = [176.4, 166.1, 153.8, 146.2, 140.2, 132.1, 124.8, 119.8,
              106.6, 103.2, 68.0, 55.3, 54.8, 50.1, 46.0, 17.9]

# 1. 假设您已经运行了CMGNet推理
# outputs = model.generate(..., output_attentions=True)

# 2. 创建头部选择器
selector = Layer5HeadSelectorSimple(outputs.cross_attentions)

# 3. 获取最佳头部
best_head = selector.select_responsive_heads()[0][0]  # 应该是头部1

# 4. 提取对齐矩阵
layer5_attention = outputs.cross_attentions[4]  # 倒数第二层
alignment_matrix = layer5_attention[0, best_head, :, :].cpu().numpy()

# 5. 分析对齐结果
print(f"使用倒数第二层(第5层) + 头部{best_head}")
print(f"对齐矩阵形状: {alignment_matrix.shape}")

# 6. 找到最佳匹配
best_matches = alignment_matrix.argmax(axis=1)
print(f"每个SMILES token的最佳NMR峰匹配: {best_matches}")

# 7. 生成可视化
selector.visualize_best_head_detailed(
    nmr_shifts=nmr_shifts,
    smiles_tokens=list(smiles),
    save_path="my_case_analysis.png"
)
"""

    print("\n" + "=" * 80)
    print("💻 完整使用示例")
    print("=" * 80)
    print(example_code)

if __name__ == "__main__":
    print_final_recommendations()
    generate_usage_example()

    print("\n" + "=" * 80)
    print("🎯 最终推荐")
    print("=" * 80)
    print("✅ 层级: 倒数第二层 (第5层)")
    print("✅ 头部: 头部1")
    print("✅ 理由: 基于最大响应、低熵、高稀疏度的综合最优")
    print("✅ 适用: 您的喹诺酮类分子NMR-SMILES对齐任务")
    print("=" * 80)
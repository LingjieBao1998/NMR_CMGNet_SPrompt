#!/usr/bin/env python3
"""
解释为什么NMR-SMILES对齐中会出现PAD标记
以及如何处理PAD标记
"""

def explain_pad_tokens():
    """详细解释PAD标记的出现原因和处理方法"""

    explanation = """
## 🔍 为什么会出现PAD标记？

### 1. **序列长度不匹配**
在深度学习模型中，输入序列需要统一长度，这就产生了PAD：

```
您的SMILES: "Fc4cc1c2N(/C=C(\\C1=O)C(=O)O)C(COc2c4N3CCN(C)CC3)C"
长度: 49个字符

模型内部处理长度: 54 (需要padding到固定长度)
NMR峰数量: 16个

差异: 54 - 49 = 5个PAD标记
```

### 2. **模型处理要求**
```python
# 模型内部的处理流程
original_smiles = "Fc4cc1c2N(/C=C(\\C1=O)C(=O)O)C(COc2c4N3CCN(C)CC3)C"  # 49字符

# Tokenization (字符级)
tokens = list(original_smiles)  # ['F', 'c', '4', 'c', 'c', '1', ...]

# 添加特殊标记
tokens = ['<BOS>'] + tokens + ['<EOS>']  # 开始和结束标记

# Padding到固定长度 (比如64)
while len(tokens) < 64:
    tokens.append('<PAD>')

# 最终序列长度: 64
# 其中包含: 1个<BOS> + 49个字符 + 1个<EOS> + 13个<PAD>
```

### 3. **批量处理需求**
```
如果同时处理多个样本:
样本1: "ABC" (3字符)
样本2: "Fc4cc1c2N(/C=C(\\C1=O)C(=O)O)C(COc2c4N3CCN(C)CC3)C" (49字符)

需要统一长度: max(3, 49) = 49
样本1需要补46个PAD标记
```

## 🛠️ 如何处理PAD标记？

### 方法1: **过滤PAD标记** (推荐)
```python
def remove_pad_tokens(smiles_tokens, alignment_matrix):
    \"\"\"
    移除PAD标记及其对应的注意力
    \"\"\"
    # 找出非PAD的索引
    valid_indices = [i for i, token in enumerate(smiles_tokens) if token != '<PAD>']

    # 过滤tokens和alignment
    filtered_tokens = [smiles_tokens[i] for i in valid_indices]
    filtered_alignment = alignment_matrix[valid_indices, :]

    return filtered_tokens, filtered_alignment

# 使用
smiles_output = "Fc4cc1c2N(/C=C(\\C1=O)C(=O)O)C(COc2c4N3CCN(C)CC3)C"
smiles_chars = list(smiles_output)

# 如果有padding
smiles_chars.append('<PAD>')
smiles_chars.append('<PAD>')

# 过滤
clean_chars, clean_alignment = remove_pad_tokens(smiles_chars, alignment_matrix)
```

### 方法2: **忽略PAD的注意力权重**
```python
def analyze_ignore_pad(alignment_matrix, smiles_tokens):
    \"\"\"
    分析时忽略PAD标记
    \"\"\"
    valid_mask = [token != '<PAD>' for token in smiles_tokens]

    # 只考虑有效的token
    valid_alignment = alignment_matrix[valid_mask, :]

    # 分析有效对齐
    for i, (token, attn_row) in enumerate(zip(
        [smiles_tokens[i] for i, valid in enumerate(valid_mask) if valid],
        valid_alignment
    )):
        best_nmr_idx = attn_row.argmax()
        confidence = attn_row[best_nmr_idx]
        print(f"Token '{token}' -> NMR[{best_nmr_idx}] (置信度: {confidence:.3f})")
```

### 方法3: **修正序列长度生成** (最佳)
```python
def generate_attention_without_pad(model, nmr_input, smiles_output):
    \"\"\"
    生成刚好匹配SMILES长度的注意力，避免PAD
    \"\"\"
    # 1. 计算需要的精确长度
    smiles_len = len(list(smiles_output)) + 2  # +2 for <BOS>, <EOS>

    # 2. 设置max_length为精确长度
    outputs = model.generate(
        input_ids=nmr_input,
        max_length=smiles_len,  # 精确长度
        output_attentions=True,
        # ... 其他参数
    )

    # 3. 提取注意力 (无PAD)
    cross_attentions = outputs.cross_attentions
    layer5_attention = cross_attentions[4][:, 1, :, :smiles_len, :]

    return layer5_attention
```

## 📊 修改可视化代码处理PAD

```python
def visualize_with_real_data_no_pad(cross_attentions, nmr_shifts, smiles_output):
    \"\"\"
    修改后的可视化，自动处理PAD标记
    \"\"\"

    # 提取注意力
    layer_attention = cross_attentions[4]  # 第5层
    head_attention = layer_attention[1]   # 头部1
    attention_matrix = head_attention.cpu().numpy()

    # 准备SMILES tokens
    smiles_chars = list(smiles_output)

    # 检查是否有长度不匹配
    seq_len, nmr_len = attention_matrix.shape

    if len(smiles_chars) != seq_len:
        print(f"⚠️  检测到序列长度不匹配:")
        print(f"   SMILES字符: {len(smiles_chars)}")
        print(f"   注意力矩阵: {seq_len}")

        # 如果SMILES更短，说明有PAD
        if len(smiles_chars) < seq_len:
            print(f"   包含 {seq_len - len(smiles_chars)} 个PAD标记")

            # 截取有效长度
            attention_matrix = attention_matrix[:len(smiles_chars), :]
            print(f"   ✅ 已截取到有效长度: {len(smiles_chars)}")

        # 如果SMILES更长，进行padding
        else:
            print(f"   需要补充 {(len(smiles_chars) - seq_len)} 个位置")
            # 这种情况较少见，通常模型会自动处理

    # 继续可视化...
    # 使用截取后的attention_matrix
```

## 🎯 具体到您的案例

### 您的输出中有PAD的原因:
```
原始SMILES: "Fc4cc1c2N(/C=C(\\C1=O)C(=O)O)C(COc2c4N3CCN(C)CC3)C" (49字符)
模型处理长度: 54字符
差异: 5个PAD标记

可能的结构:
49个SMILES字符 + 1个<EOS> + 4个<PAD> = 54字符
或者:
1个<BOS> + 49个SMILES字符 + 1个<EOS> + 3个<PAD> = 54字符
```

### 修正建议:
```python
# 在生成时设置精确长度
outputs = model.generate(
    input_ids=nmr_input,
    max_length=len(list(smiles_output)) + 2,  # 精确长度
    output_attentions=True
)

# 或者在分析时过滤
valid_indices = [i for i, char in enumerate(smiles_chars) if char != '<PAD>']
clean_alignment = alignment_matrix[valid_indices, :]
clean_smiles = [smiles_chars[i] for i in valid_indices]
```

## 💡 最佳实践

1. **生成阶段**: 设置合适的max_length避免过多PAD
2. **分析阶段**: 过滤掉PAD标记和对应的注意力
3. **可视化阶段**: 只显示有效的SMILES字符

这样可以确保您的NMR-SMILES对齐分析更加准确和清晰！
"""

    print(explanation)

    print("\n" + "=" * 80)
    print("实用代码示例")
    print("=" * 80)

    practical_code = """
#!/usr/bin/env python3
# 处理PAD标记的实用函数

import torch
import numpy as np

def clean_alignment_data(smiles_output, alignment_matrix, attention_threshold=0.01):
    \"\"\"
    清理对齐数据，移除PAD和低注意力权重的影响

    Args:
        smiles_output: 原始SMILES字符串
        alignment_matrix: 注意力矩阵 [seq_len, nmr_len]
        attention_threshold: 注意力阈值，低于此值的认为是对齐噪音

    Returns:
        清理后的tokens, alignment, 和统计信息
    \"\"\"

    # 1. 准备tokens
    smiles_tokens = list(smiles_output)

    # 2. 检测并处理PAD
    seq_len = alignment_matrix.shape[0]

    if len(smiles_tokens) < seq_len:
        # 检测到PAD标记
        print(f"检测到 {seq_len - len(smiles_tokens)} 个PAD标记")

        # 截取有效长度
        alignment_matrix = alignment_matrix[:len(smiles_tokens), :]

    elif len(smiles_tokens) > seq_len:
        # SMILES更长，需要考虑特殊标记
        # 假设有<BOS>和<EOS>
        expected_special = 2  # <BOS> + <EOS>
        if len(smiles_tokens) + expected_special == seq_len:
            print("检测到包含特殊标记(<BOS>, <EOS>)")
            # 插入特殊标记
            smiles_tokens = ['<BOS>'] + smiles_tokens + ['<EOS>']
        else:
            print(f"警告: 长度不匹配 (SMILES:{len(smiles_tokens)}, Matrix:{seq_len})")
            # 截取或填充
            smiles_tokens = smiles_tokens[:seq_len]

    # 3. 过滤低注意力权重
    # 将低于阈值的注意力设为0
    clean_alignment = alignment_matrix.copy()
    clean_alignment[clean_alignment < attention_threshold] = 0

    # 重新归一化每行
    row_sums = clean_alignment.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # 避免除零
    clean_alignment = clean_alignment / row_sums

    # 4. 统计信息
    stats = {
        'original_length': seq_len,
        'clean_length': len(smiles_tokens),
        'pad_count': seq_len - len(smiles_tokens),
        'avg_attention': clean_alignment.mean(),
        'max_attention': clean_alignment.max(),
        'alignment_quality': 'good' if clean_alignment.mean() > 0.05 else 'poor'
    }

    return smiles_tokens, clean_alignment, stats


# 使用示例
smiles = "Fc4cc1c2N(/C=C(\\\\C1=O)C(=O)O)C(COc2c4N3CCN(C)CC3)C"
alignment = np.random.rand(54, 16)  # 假设包含54个位置(含PAD)

clean_tokens, clean_alignment, stats = clean_alignment_data(smiles, alignment)

print(f"清理结果:")
print(f"  有效tokens: {len(clean_tokens)}")
print(f"  PAD数量: {stats['pad_count']}")
print(f"  平均注意力: {stats['avg_attention']:.3f}")
print(f"  对齐质量: {stats['alignment_quality']}")
    """

    print(practical_code)

    print("\n" + "=" * 80)
    print("🎯 针对您的情况的具体建议")
    print("=" * 80)

    specific_advice = """
您的案例:
- SMILES: "Fc4cc1c2N(/C=C(\\C1=O)C(=O)O)C(COc2c4N3CCN(C)CC3)C" (49字符)
- 模型输出: 54个位置
- PAD数量: 5个

处理方案:
1. 在分析时截取前49个字符的注意力
2. 忽略最后5个PAD位置的注意力权重
3. 重新归一化注意力分布

修改后的代码:
```python
# 您的原始数据
smiles_output = "Fc4cc1c2N(/C=C(\\\\C1=O)C(=O)O)C(COc2c4N3CCN(C)CC3)C"
alignment_matrix = outputs.cross_attentions[4][0, 1, :, :].cpu().numpy()  # [54, 16]

# 方案1: 直接截取有效长度
valid_alignment = alignment_matrix[:49, :]  # [49, 16]
valid_smiles = list(smiles_output)  # 49个字符

# 方案2: 智能检测有效长度
def find_valid_length(smiles, alignment):
    smiles_len = len(list(smiles))
    align_len = alignment.shape[0]

    if smiles_len < align_len:
        return smiles_len  # 截取SMILES长度
    else:
        return align_len  # 使用完整alignment

valid_len = find_valid_length(smiles_output, alignment_matrix)
clean_alignment = alignment_matrix[:valid_len, :]
clean_smiles = list(smiles_output)[:valid_len]

# 继续您的分析...
for i, (char, attn_row) in enumerate(zip(clean_smiles, clean_alignment)):
    best_nmr = attn_row.argmax()
    confidence = attn_row[best_nmr]
    print(f"字符 '{char}' -> NMR峰 {best_nmr} (置信度: {confidence:.3f})")
```

这样可以确保您的分析只关注有效的SMILES字符，不被PAD标记干扰！
"""

    print(specific_advice)


def main():
    """主函数"""
    explain_pad_tokens()

if __name__ == "__main__":
    main()
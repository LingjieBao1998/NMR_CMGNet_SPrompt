#!/usr/bin/env python3
"""
测试批量处理修复
"""
import sys
sys.path.append('/home/lingjiebao/project/NMR_Core_Mol_Detection/nmr_core_system')

from core_detection.ensemble_nmr_predictor import EnsembleNMRPredictor
import pandas as pd

# 读取测试CSV文件
csv_file = '/home/lingjiebao/project/NMR_Core_Mol_Detection/test_batch.csv'
df = pd.read_csv(csv_file)

# 解析NMR数据
nmr_data = []
for idx, row in df.iterrows():
    nmr_text = str(row['NMR']).strip()
    if ',' in nmr_text:
        values = [v.strip() for v in nmr_text.split(',') if v.strip()]
    else:
        values = [v.strip() for v in nmr_text.split() if v.strip()]
    
    float_values = []
    for val in values:
        try:
            float_val = float(val)
            float_values.append(float_val)
        except ValueError:
            continue
    
    if float_values:
        nmr_data.append(float_values)

print(f"🔬 测试数据: {len(nmr_data)} 个样本")
print(f"   样本1: {nmr_data[0]}")
print(f"   样本2: {nmr_data[1]}")
print(f"   样本3: {nmr_data[2]}")

# 创建预测器
predictor = EnsembleNMRPredictor()

# 测试批量处理
print(f"\n🧪 开始批量处理测试...")
result = predictor.predict(
    nmr_list=nmr_data,
    celery_task_id="test_batch_123",
    xgboost_threshold=0.5,
    num_returns=5,
    prediction_mode="xgboost_only"  # 使用xgboost_only加快测试速度
)

print(f"\n📊 结果分析:")
print(f"   批量模式: {result.get('batch_mode', False)}")
print(f"   总样本数: {result.get('total_samples', 0)}")
print(f"   阳性样本: {result.get('positive_samples', 0)}")
print(f"   样本结果数量: {len(result.get('sample_results', []))}")

# 清理
predictor.cleanup()

print(f"\n✅ 测试完成！")
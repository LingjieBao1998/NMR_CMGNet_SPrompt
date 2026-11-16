#!/usr/bin/env python3
"""
XGBoost核心分子检测推理代码

功能:
    输入NMR化学位移列表，输出XGBoost预测结果

使用方法:
    python xgboost_predict.py
"""

import sys
import os
import random
from pathlib import Path

# 设置随机种子以确保结果可重现（参照main_cmgnet.py）
RANDOM_SEED = 42

def set_random_seed(seed: int = RANDOM_SEED):
    """
    设置所有相关库的随机种子，确保结果可重现

    Args:
        seed: 随机种子值（默认42，与main_cmgnet.py一致）
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # 确保CUDA操作的可重现性（参照main_cmgnet.py的deterministic设置）
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

# 在导入其他模块之前设置随机种子
set_random_seed(RANDOM_SEED)

# 获取当前文件所在目录
current_file_path = Path(__file__).resolve()
# 获取项目根目录 - 修复为正确的app目录
# core_detection在/app/nmr_core_system/，所以需要往上两级到/app/
project_dir = current_file_path.parent.parent.parent  # /app/
# 将项目根目录添加到sys.path
sys.path.insert(0, str(project_dir))

import torch
import numpy as np
import joblib
from typing import List, Dict
# 修改为从同级目录导入infer
from .infer import ModelInference


class XGBoostPredictor:
    """XGBoost预测器"""

    def __init__(self,
                 xgb_model_path: str = None,
                 cress_config_path: str = None,
                 cress_model_path: str = None,
                 device: str = "cuda:0"):
        """
        初始化XGBoost预测器

        Args:
            xgb_model_path: XGBoost模型路径
            cress_config_path: CRESS配置文件路径
            cress_model_path: CRESS模型路径
            device: 设备 (cuda:0 或 cpu)
        """
        # 使用project_dir作为基础路径
        base_path = project_dir

        # 默认路径
        if xgb_model_path is None:
            xgb_model_path = str(base_path / "models/xgboost/model_xgb_cress_1220.pkl")
        if cress_config_path is None:
            cress_config_path = str(base_path / "models/2_5_w_model/8.json")
        if cress_model_path is None:
            cress_model_path = str(base_path / "models/2_5_w_model/8.pth")

        print("="*100)
        print("初始化XGBoost预测器")
        print("="*100)
        print(f"项目根目录: {base_path}")

        # 1. 加载CRESS模型 (用于生成embedding)
        print(f"\n📦 加载CRESS模型...")
        print(f"   Config: {cress_config_path}")
        print(f"   Model: {cress_model_path}")
        
        # 读取配置文件并修改tokenizer路径为绝对路径
        import json
        with open(cress_config_path, 'r') as f:
            config = json.load(f)
        
        # 修改tokenizer路径为绝对路径
        tokenizer_path = str(base_path / "models/tokenizer-smiles-roberta-1e")
        config["model_config"]["roberta_tokenizer_path"] = tokenizer_path
        config["dataset_config"]["smiles_tokenizer"] = tokenizer_path
        
        # 创建临时配置文件
        import tempfile
        temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(config, temp_config)
        temp_config.close()
        
        print(f"   修改后tokenizer路径: {tokenizer_path}")
        
        self.cress_model = ModelInference(
            config_path=temp_config.name,
            pretrain_model_path=cress_model_path,
            device=device
        )
        
        # 删除临时配置文件
        import os
        os.unlink(temp_config.name)
        print(f"   ✅ CRESS模型已加载")

        # 2. 加载XGBoost模型
        print(f"\n📦 加载XGBoost模型...")
        print(f"   Path: {xgb_model_path}")
        self.xgb_model = joblib.load(xgb_model_path)
        print(f"   ✅ XGBoost模型已加载")

        self.device = device
        print(f"\n✅ 预测器初始化完成")
        print("="*100)

    def generate_embedding(self, nmr_shifts: List[float]) -> torch.Tensor:
        """
        从NMR化学位移生成CRESS embedding

        Args:
            nmr_shifts: NMR化学位移列表

        Returns:
            CRESS embedding (768维向量)
        """
        # 转换为batch格式
        nmr_batch = [nmr_shifts]

        # 生成embedding
        with torch.no_grad():
            embedding = self.cress_model.nmr_encode(nmr_batch)

        return embedding

    def predict(self, nmr_shifts: List[float], threshold: float = 0.5) -> Dict:
        """
        使用XGBoost预测

        Args:
            nmr_shifts: NMR化学位移列表
            threshold: 判断阈值 (默认0.5)

        Returns:
            预测结果字典:
                - probability: 预测概率
                - prediction: 预测标签 (1=核心分子, 0=非核心分子)
                - threshold: 使用的阈值
        """
        # 1. 生成CRESS embedding
        embedding = self.generate_embedding(nmr_shifts)

        # 2. XGBoost预测
        embedding_np = embedding.cpu().numpy()
        prob = float(self.xgb_model.predict_proba(embedding_np)[:, 1][0])
        pred = 1 if prob >= threshold else 0

        return {
            'probability': prob,
            'prediction': pred,
            'threshold': threshold
        }

    def predict_batch(self, nmr_list: List[List[float]], threshold: float = 0.5, batch_size: int = 32) -> List[Dict]:
        """
        批量预测 - 使用batch处理提高效率

        Args:
            nmr_list: NMR化学位移列表的列表
            threshold: 判断阈值
            batch_size: 批量大小 (默认32)

        Returns:
            预测结果列表
        """
        results = []

        # 分批处理
        for i in range(0, len(nmr_list), batch_size):
            batch_nmr = nmr_list[i:i + batch_size]

            # 生成batch embeddings
            with torch.no_grad():
                batch_embeddings = self.cress_model.nmr_encode(batch_nmr)

            # 转换为numpy
            batch_embeddings_np = batch_embeddings.cpu().numpy()

            # XGBoost批量预测
            batch_probs = self.xgb_model.predict_proba(batch_embeddings_np)[:, 1]
            batch_preds = (batch_probs >= threshold).astype(int)

            # 组装结果
            for prob, pred in zip(batch_probs, batch_preds):
                results.append({
                    'probability': float(prob),
                    'prediction': int(pred),
                    'threshold': threshold
                })

        return results

    def print_result(self, nmr_shifts: List[float], result: Dict):
        """打印预测结果"""
        print("\n" + "="*100)
        print("预测结果")
        print("="*100)
        print(f"\n输入信息:")
        print(f"  NMR化学位移数量: {len(nmr_shifts)}")
        print(f"  化学位移范围: {min(nmr_shifts):.2f} - {max(nmr_shifts):.2f} ppm")

        print(f"\n预测信息:")
        print(f"  概率: {result['probability']:.4f} ({result['probability']:.2%})")
        print(f"  阈值: {result['threshold']}")
        print(f"  预测: {'✅ 核心分子' if result['prediction'] == 1 else '❌ 非核心分子'}")

        if result['prediction'] == 1:
            print(f"\n✅ 该NMR化学位移对应的分子被预测为核心分子！")
        else:
            print(f"\n❌ 该NMR化学位移对应的分子被预测为非核心分子。")

        print("="*100)


def main():
    """主函数 - 演示用法"""
    print("="*100)
    print("XGBoost核心分子检测 - 推理演示")
    print("="*100)

    # 1. 初始化预测器
    predictor = XGBoostPredictor()

    # 2. 测试案例 - 环丙沙星 (Ciprofloxacin)
    cipro_nmr = [
        176.49, 166.02, 153.61, 152.37, 148.32, 144.25, 144.21, 139.18,
        119.48, 119.44, 111.34, 111.23, 107.03, 107.02, 106.92,
        46.45, 42.66, 36.08, 7.71
    ]

    print(f"\n测试案例: 环丙沙星 (Ciprofloxacin)")

    # 3. 预测
    result = predictor.predict(cipro_nmr, threshold=0.5)

    # 4. 打印结果
    predictor.print_result(cipro_nmr, result)

    # 5. 测试不同阈值
    print("\n" + "="*100)
    print("不同阈值测试")
    print("="*100)

    thresholds = [0.3, 0.5, 0.7, 0.9]
    for thresh in thresholds:
        result = predictor.predict(cipro_nmr, threshold=thresh)
        print(f"\n阈值 {thresh}:")
        print(f"  概率: {result['probability']:.4f}")
        print(f"  预测: {'✅ 核心' if result['prediction'] == 1 else '❌ 非核心'}")

    # 6. 批量预测演示
    print("\n" + "="*100)
    print("批量预测演示 (batch_size=32)")
    print("="*100)

    # 创建测试数据 (100个样本)
    print("\n创建100个测试样本...")
    test_nmr_list = [cipro_nmr] * 100

    # 批量预测
    print("执行批量预测...")
    import time
    start_time = time.time()
    batch_results = predictor.predict_batch(test_nmr_list, threshold=0.5, batch_size=32)
    elapsed_time = time.time() - start_time

    print(f"\n批量预测完成:")
    print(f"  样本数量: {len(test_nmr_list)}")
    print(f"  Batch大小: 32")
    print(f"  总耗时: {elapsed_time:.2f}秒")
    print(f"  平均速度: {elapsed_time/len(test_nmr_list):.4f}秒/样本")

    # 统计结果
    core_count = sum(1 for r in batch_results if r['prediction'] == 1)
    print(f"\n预测结果统计:")
    print(f"  核心分子: {core_count}/{len(batch_results)}")
    print(f"  非核心分子: {len(batch_results) - core_count}/{len(batch_results)}")
    print(f"  平均概率: {sum(r['probability'] for r in batch_results)/len(batch_results):.4f}")

    print("\n" + "="*100)


def predict_from_user_input():
    """从用户输入获取NMR并预测"""
    print("="*100)
    print("XGBoost核心分子检测 - 用户输入模式")
    print("="*100)

    # 初始化预测器
    print("\n🔄 初始化预测器...")
    predictor = XGBoostPredictor()

    print("\n请输入NMR化学位移（用逗号或空格分隔）:")
    print("示例: 176.49, 166.02, 153.61, 152.37")

    user_input = input("\n> ")

    # 解析输入
    try:
        nmr_shifts = [float(x.strip()) for x in user_input.replace(',', ' ').split() if x.strip()]
        print(f"\n解析到 {len(nmr_shifts)} 个NMR化学位移")
    except ValueError as e:
        print(f"\n❌ 输入格式错误: {e}")
        return

    # 获取阈值
    print("\n请输入XGBoost阈值 (默认0.5):")
    threshold_input = input("> ")
    threshold = float(threshold_input) if threshold_input.strip() else 0.5

    # 执行预测
    result = predictor.predict(nmr_shifts, threshold=threshold)

    # 打印结果
    predictor.print_result(nmr_shifts, result)


if __name__ == "__main__":
    # 运行演示
    main()

    # 或者使用用户输入模式 (取消下面的注释)
    # predict_from_user_input()

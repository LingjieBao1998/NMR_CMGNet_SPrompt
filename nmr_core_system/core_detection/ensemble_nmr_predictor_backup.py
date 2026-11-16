#!/usr/bin/env python3
"""
集成NMR预测器 - 整合XGBoost和CMGNet

功能:
    输入NMR化学位移列表，输出集成预测结果

接口参数:
    - save_path: 保存路径（默认为None，如果为None，则在project_dir/result/下面，新建一个celery_task_id/result.json文件）
    - celery_task_id: Celery任务ID（用于生成保存路径，默认为None时使用UUID）
    - xgboost_threshold: XGBoost的阈值（默认为0.5）
    - num_returns: Top-K结果（默认为10）
    - prediction_mode: 预测模式（"ensemble"或"xgboost_only"）

集成逻辑:
    - 投票机制：2/3即为True
    - XGBoost: 概率 >= threshold → True
    - CMGNet-Frag: Top1包含母核 → True
    - CMGNet-SMILES: Top1包含母核 → True
"""

import sys
import os
import json
import uuid
import random
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np

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
    np.random.seed(seed)

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

# 导入XGBoost和CMGNet预测器
from .xgboost_predict import XGBoostPredictor
from .cmgnet_predict import CMGNetPredictor


class EnsembleNMRPredictor:
    """集成NMR预测器"""

    def __init__(self):
        """
        初始化集成预测器
        """

        # 获取项目目录 (NMR_Core_Mol_Detection)
        current_dir = Path(__file__).parent
        self.project_dir = current_dir.parent.parent

        # 初始化预测器
        print("🔄 初始化集成预测器...")
        try:
            self.xgboost_predictor = XGBoostPredictor()
            print("   ✅ XGBoost预测器已加载")
        except Exception as e:
            print(f"   ❌ XGBoost预测器加载失败: {e}")
            self.xgboost_predictor = None

        try:
            self.cmgnet_predictor = CMGNetPredictor()
            print("   ✅ CMGNet预测器已加载")
        except Exception as e:
            print(f"   ❌ CMGNet预测器加载失败: {e}")
            self.cmgnet_predictor = None

        print("✅ 集成预测器初始化完成")

    def predict(self,
               nmr_list,
               save_path: Optional[str] = None,
               celery_task_id: Optional[str] = None,
               xgboost_threshold: float = 0.5,
               num_returns: int = 10,
               prediction_mode: str = "ensemble") -> Dict:
        """
        对NMR样本进行集成预测

        Args:
            nmr_list: NMR化学位移列表（支持一维列表[[...]]或二维列表[...]）
            save_path: 保存路径（可选，默认为None时自动生成celery_task_id路径）
            celery_task_id: Celery任务ID（用于生成保存路径，默认为None时使用UUID）
            xgboost_threshold: XGBoost阈值（默认为0.5）
            num_returns: Top-K结果数量（默认为10）
            prediction_mode: 预测模式（"ensemble"或"xgboost_only"）

        Returns:
            预测结果字典
        """
        # 处理输入格式：支持一维列表[[...]]或二维列表[...]
        if isinstance(nmr_list, list) and len(nmr_list) > 0:
            # 检查第一个元素是否为列表（批量处理模式）
            first_element = nmr_list[0]
            if isinstance(first_element, list):
                # 二维列表格式：[[...], [...], ...]，批量处理
                nmr_samples = nmr_list
                print(f"   📊 批量处理模式：检测到 {len(nmr_samples)} 个样本")
            else:
                # 检查是否为数字列表（单样本处理模式）
                if all(isinstance(x, (int, float)) for x in nmr_list):
                    # 一维列表格式：[...]，视为单个样本
                    nmr_samples = [nmr_list]
                    print(f"   📊 单样本处理模式：检测到 1 个样本")
                else:
                    # 混合格式，尝试作为批量处理
                    nmr_samples = nmr_list
                    print(f"   📊 混合批量处理模式：检测到 {len(nmr_samples)} 个样本")
        else:
            return {"error": "无效的NMR数据格式"}

        print(f"\n🔬 开始集成预测...")
        print(f"   样本数量: {len(nmr_samples)}")
        print(f"   预测模式: {prediction_mode}")
        print(f"   XGBoost阈值: {xgboost_threshold}")
        print(f"   Top-K: {num_returns}")

        # 初始化批量结果存储
        batch_results = []
        ensemble_decisions = []

        # 处理每个样本
        for sample_idx, nmr_shifts in enumerate(nmr_samples):
            print(f"\n🔄 处理样本 {sample_idx + 1}/{len(nmr_samples)}...")
            print(f"   NMR数量: {len(nmr_shifts)}")

            sample_result = {
                "sample_id": sample_idx + 1,
                "nmr_shifts": nmr_shifts,
                "nmr_count": len(nmr_shifts),
                "xgboost_threshold": xgboost_threshold,
                "topk": num_returns,
                "prediction_mode": prediction_mode
            }

            # 1. XGBoost预测
            if self.xgboost_predictor:
                print(f"\n📊 XGBoost预测中...")
                try:
                    xgb_result = self.xgboost_predictor.predict(list(nmr_shifts))
                    xgb_probability = xgb_result['probability']
                    xgb_is_core = xgb_probability >= xgboost_threshold

                    sample_result["xgboost"] = {
                        "probability": float(xgb_probability),
                        "is_core_molecule": bool(xgb_is_core),
                        "prediction": "core_molecule" if xgb_is_core else "non_core_molecule"
                    }

                    print(f"   ✅ XGBoost: {xgb_probability:.4f} -> {xgb_is_core}")
                except Exception as e:
                    print(f"   ❌ XGBoost预测失败: {e}")
                    sample_result["xgboost"] = {"error": str(e)}
            else:
                sample_result["xgboost"] = {"error": "XGBoost预测器未初始化"}

            # 2. CMGNet预测（仅在ensemble模式下）
            if prediction_mode == "ensemble" and self.cmgnet_predictor:
                print(f"\n🧬 CMGNet预测中...")

                # Frag模式
                print(f"   🔄 Frag模式...")
                try:
                    frag_result = self.cmgnet_predictor.predict(
                        list(nmr_shifts),
                        mode="frag",
                        num_returns=num_returns
                    )

                    if frag_result.get("top_predictions"):
                        sample_result["cmgnet_frag"] = {
                            "predictions": frag_result["top_predictions"][:num_returns]
                        }

                        print(f"   ✅ Frag: {frag_result['top_predictions'][0].get('has_core_structure', False) if frag_result['top_predictions'] else 'N/A'}")
                    else:
                        sample_result["cmgnet_frag"] = {"error": "无有效预测"}
                        print(f"   ❌ Frag: 无有效预测")

                except Exception as e:
                    print(f"   ❌ Frag预测失败: {e}")
                    sample_result["cmgnet_frag"] = {"error": str(e)}

                # SMILES模式
                print(f"   🔄 SMILES模式...")
                try:
                    smiles_result = self.cmgnet_predictor.predict(
                        list(nmr_shifts),
                        mode="smiles",
                        num_returns=num_returns
                    )

                    if smiles_result.get("top_predictions"):
                        sample_result["cmgnet_smiles"] = {
                            "predictions": smiles_result["top_predictions"][:num_returns]
                        }

                        print(f"   ✅ SMILES: {smiles_result['top_predictions'][0].get('has_core_structure', False) if smiles_result['top_predictions'] else 'N/A'}")
                    else:
                        sample_result["cmgnet_smiles"] = {"error": "无有效预测"}
                        print(f"   ❌ SMILES: 无有效预测")

                except Exception as e:
                    print(f"   ❌ SMILES预测失败: {e}")
                    sample_result["cmgnet_smiles"] = {"error": str(e)}

            # 3. 集成决策（投票机制：2/3即为True）
            if prediction_mode == "ensemble":
                print(f"\n🗳️  投票决策...")

                votes = []
                voting_details = {}

                # XGBoost投票
                if "xgboost" in sample_result and "is_core_molecule" in sample_result["xgboost"]:
                    xgb_vote = sample_result["xgboost"]["is_core_molecule"]
                    votes.append(xgb_vote)
                    voting_details["xgboost"] = bool(xgb_vote)
                    print(f"   XGBoost: {xgb_vote}")
                else:
                    voting_details["xgboost"] = None

                # CMGNet-Frag投票
                if "cmgnet_frag" in sample_result and "predictions" in sample_result["cmgnet_frag"] and sample_result["cmgnet_frag"]["predictions"]:
                    frag_vote = sample_result["cmgnet_frag"]["predictions"][0].get("has_core_structure", False)
                    votes.append(frag_vote)
                    voting_details["cmgnet_frag"] = bool(frag_vote)
                    print(f"   CMGNet-Frag: {frag_vote}")
                else:
                    voting_details["cmgnet_frag"] = None

                # CMGNet-SMILES投票
                if "cmgnet_smiles" in sample_result and "predictions" in sample_result["cmgnet_smiles"] and sample_result["cmgnet_smiles"]["predictions"]:
                    smiles_vote = sample_result["cmgnet_smiles"]["predictions"][0].get("has_core_structure", False)
                    votes.append(smiles_vote)
                    voting_details["cmgnet_smiles"] = bool(smiles_vote)
                    print(f"   CMGNet-SMILES: {smiles_vote}")
                else:
                    voting_details["cmgnet_smiles"] = None

                # 投票结果
                total_votes = len(votes)
                positive_votes = sum(votes)

                # 2/3即为True的规则
                is_core_molecule = positive_votes >= (2 * total_votes / 3)

                sample_result["integrated_decision"] = {
                    "voting_details": voting_details,
                    "total_votes": total_votes,
                    "positive_votes": positive_votes,
                    "is_core_molecule": bool(is_core_molecule),
                    "final_prediction": "core_molecule" if is_core_molecule else "non_core_molecule",
                    "confidence": positive_votes / total_votes if total_votes > 0 else 0
                }

                print(f"\n🎯 最终决策: {is_core_molecule} ({positive_votes}/{total_votes})")
                print(f"   置信度: {sample_result['integrated_decision']['confidence']:.2%}")

                # 保存这个样本的集成决策用于后续统计
                ensemble_decisions.append(is_core_molecule)

            else:
                # XGBoost only模式
                if "xgboost" in sample_result and "is_core_molecule" in sample_result["xgboost"]:
                    sample_result["integrated_decision"] = {
                        "is_core_molecule": sample_result["xgboost"]["is_core_molecule"],
                        "final_prediction": sample_result["xgboost"]["prediction"],
                        "confidence": sample_result["xgboost"]["probability"]
                    }
                    print(f"\n🎯 最终决策: {sample_result['xgboost']['is_core_molecule']}")
                    ensemble_decisions.append(sample_result["xgboost"]["is_core_molecule"])

            batch_results.append(sample_result)

        # 构建批量处理的总结果
        total_positive = sum(ensemble_decisions)
        total_samples = len(ensemble_decisions)

        result = {
            "batch_mode": True,
            "total_samples": total_samples,
            "positive_samples": total_positive,
            "sample_results": batch_results,
            "overall_positive_rate": total_positive / total_samples if total_samples > 0 else 0
        }

        print(f"\n🎉 批量处理完成！")
        print(f"   总样本数: {total_samples}")
        print(f"   阳性样本: {total_positive}")
        print(f"   阳性率: {result['overall_positive_rate']:.2%}")

        # 保存结果到文件
        self._save_result(result, save_path, celery_task_id)

        return result

    def cleanup(self):
        """
        清理预测器资源，释放GPU显存
        """
        import torch
        import gc
        
        print("🧹 开始清理预测器资源...")
        
        try:
            # 清理CMGNet预测器（占用GPU显存的主要部分）
            if hasattr(self, 'cmgnet_predictor') and self.cmgnet_predictor is not None:
                if hasattr(self.cmgnet_predictor, 'model'):
                    del self.cmgnet_predictor.model
                if hasattr(self.cmgnet_predictor, 'tokenizer'):
                    del self.cmgnet_predictor.tokenizer
                del self.cmgnet_predictor
                self.cmgnet_predictor = None
                print("   ✅ CMGNet预测器已清理")
        except Exception as e:
            print(f"   ⚠️  清理CMGNet预测器时出错: {e}")
        
        try:
            # 清理XGBoost预测器
            if hasattr(self, 'xgboost_predictor') and self.xgboost_predictor is not None:
                del self.xgboost_predictor
                self.xgboost_predictor = None
                print("   ✅ XGBoost预测器已清理")
        except Exception as e:
            print(f"   ⚠️  清理XGBoost预测器时出错: {e}")
        
        # 清理CUDA缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("   ✅ CUDA缓存已清理")
        
        # 强制垃圾回收
        gc.collect()
        print("   ✅ 垃圾回收完成")
        print("🧹 预测器资源清理完成")

    def _save_result(self, result: Dict, save_path: Optional[str] = None, celery_task_id: Optional[str] = None):
        """保存预测结果到文件"""
        if save_path is None:
            # 使用celery_task_id作为目录名，如果没有则使用UUID
            result_id = celery_task_id if celery_task_id else str(uuid.uuid4())
            result_dir = self.project_dir / "result" / result_id
            result_dir.mkdir(parents=True, exist_ok=True)
            save_path = result_dir / "result.json"

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"\n💾 结果已保存到: {save_path}")


def main():
    """示例用法"""
    print("="*60)
    print("集成NMR预测器 - 示例")
    print("="*60)

    # 创建集成预测器
    predictor = EnsembleNMRPredictor()

    # 测试NMR数据（环丙沙星）
    nmr_list = [[176.49, 166.02, 153.61, 152.37, 148.32, 144.25, 144.21, 139.18,
                119.48, 119.44, 111.34, 111.23, 107.03, 107.02, 106.92,
                46.45, 42.66, 36.08, 7.71]]

    # 使用新的predict接口
    result = predictor.predict(nmr_list,
                             xgboost_threshold=0.5,
                             num_returns=10,
                             prediction_mode="ensemble")  # 或 "xgboost_only"

    print(f"\n🎯 预测结果:")
    if "integrated_decision" in result:
        decision = result["integrated_decision"]
        print(f"   最终预测: {decision['final_prediction']}")
        print(f"   是否为核心分子: {decision['is_core_molecule']}")
        print(f"   置信度: {decision['confidence']:.2%}")


if __name__ == "__main__":
    main()
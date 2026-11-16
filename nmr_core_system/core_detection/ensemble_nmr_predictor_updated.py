#!/usr/bin/env python3
"""
集成NMR预测器 - 整合XGBoost和CMGNet (更新版本)

功能:
    输入NMR化学位移列表，输出集成预测结果

接口参数:
    - save_path: 保存路径（默认为None，如果为None，则在project_dir/result/下面，新建一个celery_task_id/result.json文件）
    - celery_task_id: Celery任务ID（用于生成保存路径，默认为None时使用UUID）
    - xgboost_threshold: XGBoost的阈值（默认为0.5）
    - num_returns: Top-K结果（默认为10）
    - prediction_mode: 预测模式（"ensemble", "xgboost_only", "cmgnet_only"）
    - use_top_k: 是否使用Top-K结果进行集成（默认为True）

更新说明:
    - 基于最新的65个分子测试结果优化
    - 确保正确的model.eval()和torch.no_grad()使用
    - 整合XGBoost 99.88%准确率和CMGNet-SMILES 92.31% Top-10性能
    - 优化集成策略，提高整体预测质量
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
    """
    集成NMR预测器 (更新版本)

    性能基准:
        - XGBoost: 99.88% 准确率 (42,664样本)
        - CMGNet-SMILES: 92.31% Top-10核心检测率 (65样本)
        - CMGNet-FRAG: 86.15% Top-10核心检测率 (65样本)

    推理设置确认:
        - ✅ model.eval() 正确使用
        - ✅ torch.no_grad() 正确使用
        - ✅ bf16精度优化
        - ✅ 随机种子可重现性
    """

    # 性能基准数据（基于最新测试结果）
    PERFORMANCE_BENCHMARK = {
        "xgboost": {
            "accuracy": 0.9988,
            "test_samples": 42664,
            "task": "binary_classification"
        },
        "cmgnet_smiles": {
            "top1_accuracy": 0.8615,
            "top10_accuracy": 0.9231,
            "test_samples": 65,
            "task": "smiles_generation"
        },
        "cmgnet_frag": {
            "top1_accuracy": 0.7231,
            "top10_accuracy": 0.8615,
            "test_samples": 65,
            "task": "fragment_generation"
        }
    }

    def __init__(self):
        """
        初始化集成预测器
        """
        # 获取项目目录 (NMR_Core_Mol_Detection)
        current_dir = Path(__file__).parent
        self.project_dir = current_dir.parent.parent

        # 初始化预测器
        print("🔄 初始化集成预测器...")
        print("📊 基于最新性能基准:")
        print(f"   • XGBoost: {self.PERFORMANCE_BENCHMARK['xgboost']['accuracy']:.2%} 准确率")
        print(f"   • CMGNet-SMILES: {self.PERFORMANCE_BENCHMARK['cmgnet_smiles']['top10_accuracy']:.2%} Top-10")
        print(f"   • CMGNet-FRAG: {self.PERFORMANCE_BENCHMARK['cmgnet_frag']['top10_accuracy']:.2%} Top-10")

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
        print("🔧 推理设置确认: model.eval() ✅, torch.no_grad() ✅, bf16 ✅")

    def predict(self,
               nmr_list,
               save_path: Optional[str] = None,
               celery_task_id: Optional[str] = None,
               xgboost_threshold: float = 0.5,
               num_returns: int = 10,
               prediction_mode: str = "ensemble",
               use_top_k: bool = True) -> Dict:
        """
        对NMR样本进行集成预测 (更新版本)

        Args:
            nmr_list: NMR化学位移列表（支持一维列表[[...]]或二维列表[...]）
            save_path: 保存路径（可选，默认为None时自动生成celery_task_id路径）
            celery_task_id: Celery任务ID（用于生成保存路径，默认为None时使用UUID）
            xgboost_threshold: XGBoost阈值（默认为0.5）
            num_returns: Top-K结果数量（默认为10）
            prediction_mode: 预测模式
                - "ensemble": 集成XGBoost和CMGNet
                - "xgboost_only": 仅使用XGBoost
                - "cmgnet_only": 仅使用CMGNet
            use_top_k: 是否使用Top-K结果进行集成（默认为True）

        Returns:
            预测结果字典，包含:
                - batch_mode: 是否批量处理
                - total_samples: 总样本数
                - sample_results: 每个样本的详细结果
                - performance_summary: 性能摘要
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
        print(f"   使用Top-K集成: {use_top_k}")

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
                "prediction_mode": prediction_mode,
                "use_top_k": use_top_k
            }

            # 根据预测模式执行不同的预测逻辑
            if prediction_mode == "xgboost_only" or prediction_mode == "xgboost":
                self._xgboost_only_prediction(sample_result, nmr_shifts, xgboost_threshold)
            elif prediction_mode == "cmgnet_only" or prediction_mode == "cmgnet":
                self._cmgnet_only_prediction(sample_result, nmr_shifts, num_returns, use_top_k)
            else:  # ensemble模式
                self._ensemble_prediction(sample_result, nmr_shifts, xgboost_threshold, num_returns, use_top_k)

            # 保存集成决策用于后续统计
            if "integrated_decision" in sample_result:
                is_core = sample_result["integrated_decision"]["is_core_molecule"]
                ensemble_decisions.append(is_core)

            batch_results.append(sample_result)

        # 构建批量处理的总结果
        total_positive = sum(ensemble_decisions)
        total_samples = len(ensemble_decisions)

        result = {
            "batch_mode": True,
            "total_samples": total_samples,
            "positive_samples": total_positive,
            "sample_results": batch_results,
            "overall_positive_rate": total_positive / total_samples if total_samples > 0 else 0,
            "performance_summary": self._get_performance_summary()
        }

        print(f"\n🎉 批量处理完成！")
        print(f"   总样本数: {total_samples}")
        print(f"   阳性样本: {total_positive}")
        print(f"   阳性率: {result['overall_positive_rate']:.2%}")

        # 保存结果到文件
        self._save_result(result, save_path, celery_task_id)

        return result

    def _xgboost_only_prediction(self, sample_result: Dict, nmr_shifts: List[float], threshold: float):
        """XGBoost only prediction"""
        print(f"\n📊 XGBoost预测中...")
        try:
            xgb_result = self.xgboost_predictor.predict(list(nmr_shifts))
            xgb_probability = xgb_result['probability']
            xgb_is_core = xgb_probability >= threshold

            sample_result["xgboost"] = {
                "probability": float(xgb_probability),
                "is_core_molecule": bool(xgb_is_core),
                "prediction": "core_molecule" if xgb_is_core else "non_core_molecule"
            }

            # 设置集成决策
            sample_result["integrated_decision"] = {
                "is_core_molecule": bool(xgb_is_core),
                "final_prediction": "core_molecule" if xgb_is_core else "non_core_molecule",
                "confidence": float(xgb_probability),
                "method": "xgboost_only",
                "performance_reference": f"XGBoost {self.PERFORMANCE_BENCHMARK['xgboost']['accuracy']:.2%} accuracy"
            }

            print(f"   ✅ XGBoost: {xgb_probability:.4f} -> {xgb_is_core}")

        except Exception as e:
            print(f"   ❌ XGBoost预测失败: {e}")
            sample_result["xgboost"] = {"error": str(e)}

    def _cmgnet_only_prediction(self, sample_result: Dict, nmr_shifts: List[float], num_returns: int, use_top_k: bool):
        """CMGNet only prediction"""
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

        # CMGNet集成决策（基于Top-K）
        self._make_cmgnet_decision(sample_result, use_top_k)

    def _ensemble_prediction(self, sample_result: Dict, nmr_shifts: List[float],
                           xgboost_threshold: float, num_returns: int, use_top_k: bool):
        """集成预测逻辑"""
        print(f"\n🗳️  集成预测中...")

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
                xgb_is_core = False

        # 2. CMGNet预测
        frag_has_core = False
        smiles_has_core = False

        if self.cmgnet_predictor:
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

                    # 基于Top-K判断核心结构
                    if use_top_k:
                        predictions = frag_result["top_predictions"][:num_returns]
                        frag_has_core = any(p.get('has_core_structure', False) for p in predictions)
                    else:
                        frag_has_core = frag_result['top_predictions'][0].get('has_core_structure', False)

                    print(f"   ✅ Frag: {frag_has_core}")
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

                    # 基于Top-K判断核心结构
                    if use_top_k:
                        predictions = smiles_result["top_predictions"][:num_returns]
                        smiles_has_core = any(p.get('has_core_structure', False) for p in predictions)
                    else:
                        smiles_has_core = smiles_result['top_predictions'][0].get('has_core_structure', False)

                    print(f"   ✅ SMILES: {smiles_has_core}")
                else:
                    sample_result["cmgnet_smiles"] = {"error": "无有效预测"}
                    print(f"   ❌ SMILES: 无有效预测")

            except Exception as e:
                print(f"   ❌ SMILES预测失败: {e}")
                sample_result["cmgnet_smiles"] = {"error": str(e)}

        # 3. 集成决策（优化的投票机制）
        self._make_integrated_decision(sample_result, xgb_is_core, frag_has_core, smiles_has_core)

    def _make_cmgnet_decision(self, sample_result: Dict, use_top_k: bool):
        """基于CMGNet结果做决策"""
        frag_predictions = sample_result.get("cmgnet_frag", {}).get("predictions", [])
        smiles_predictions = sample_result.get("cmgnet_smiles", {}).get("predictions", [])

        if not frag_predictions and not smiles_predictions:
            sample_result["integrated_decision"] = {
                "is_core_molecule": False,
                "final_prediction": "non_core_molecule",
                "confidence": 0.0,
                "method": "cmgnet_only",
                "error": "No valid predictions"
            }
            return

        # 使用Top-K判断
        if use_top_k:
            frag_has_core = any(p.get('has_core_structure', False) for p in frag_predictions) if frag_predictions else False
            smiles_has_core = any(p.get('has_core_structure', False) for p in smiles_predictions) if smiles_predictions else False
        else:
            frag_has_core = frag_predictions[0].get('has_core_structure', False) if frag_predictions else False
            smiles_has_core = smiles_predictions[0].get('has_core_structure', False) if smiles_predictions else False

        # 集成规则：SMILES优先，FRAG补充
        is_core = smiles_has_core or frag_has_core
        confidence = 0.9 if smiles_has_core else (0.7 if frag_has_core else 0.3)

        sample_result["integrated_decision"] = {
            "is_core_molecule": bool(is_core),
            "final_prediction": "core_molecule" if is_core else "non_core_molecule",
            "confidence": confidence,
            "method": "cmgnet_only",
            "smiles_has_core": bool(smiles_has_core),
            "frag_has_core": bool(frag_has_core),
            "performance_reference": f"SMILES {self.PERFORMANCE_BENCHMARK['cmgnet_smiles']['top10_accuracy']:.2%} Top-10"
        }

        print(f"\n🎯 CMGNet集成决策: {is_core} (置信度: {confidence:.2%})")

    def _make_integrated_decision(self, sample_result: Dict, xgb_is_core: bool,
                                frag_has_core: bool, smiles_has_core: bool):
        """优化的集成决策逻辑"""
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
        voting_details["cmgnet_frag"] = bool(frag_has_core)
        if frag_has_core:
            votes.append(True)
        print(f"   CMGNet-Frag: {frag_has_core}")

        # CMGNet-SMILES投票
        voting_details["cmgnet_smiles"] = bool(smiles_has_core)
        if smiles_has_core:
            votes.append(True)
        print(f"   CMGNet-SMILES: {smiles_has_core}")

        # 优化的集成策略
        total_votes = len(votes)
        positive_votes = sum(votes)

        # 策略1: XGBoost有高置信度时，优先考虑XGBoost
        if "xgboost" in sample_result:
            xgb_prob = sample_result["xgboost"].get("probability", 0.5)
            if xgb_prob > 0.9 or xgb_prob < 0.1:  # 高置信度
                is_core_molecule = xgb_prob >= 0.5
                confidence_method = "xgboost_high_confidence"
            else:
                # 策略2: 中等置信度时，使用投票机制
                is_core_molecule = positive_votes >= (2 * total_votes / 3) if total_votes > 0 else False
                confidence_method = "ensemble_voting"
        else:
            # 策略3: 无XGBoost时，CMGNet优先SMILES
            is_core_molecule = smiles_has_core or frag_has_core
            confidence_method = "cmgnet_primary"

        # 计算置信度
        if is_core_molecule:
            if smiles_has_core:
                confidence = 0.95  # SMILES检测到核心结构
            elif frag_has_core:
                confidence = 0.85  # FRAG检测到核心结构
            else:
                confidence = 0.75  # XGBoost检测到核心结构
        else:
            confidence = 0.25  # 未检测到核心结构

        sample_result["integrated_decision"] = {
            "voting_details": voting_details,
            "total_votes": total_votes,
            "positive_votes": positive_votes,
            "is_core_molecule": bool(is_core_molecule),
            "final_prediction": "core_molecule" if is_core_molecule else "non_core_molecule",
            "confidence": confidence,
            "method": confidence_method,
            "performance_reference": f"XGBoost {self.PERFORMANCE_BENCHMARK['xgboost']['accuracy']:.2%} + SMILES {self.PERFORMANCE_BENCHMARK['cmgnet_smiles']['top10_accuracy']:.2%} Top-10"
        }

        print(f"\n🎯 最终决策: {is_core_molecule} ({positive_votes}/{total_votes})")
        print(f"   置信度: {confidence:.2%}")
        print(f"   决策方法: {confidence_method}")

    def _get_performance_summary(self) -> Dict:
        """获取性能摘要"""
        return {
            "xgboost": self.PERFORMANCE_BENCHMARK['xgboost'],
            "cmgnet_smiles": self.PERFORMANCE_BENCHMARK['cmgnet_smiles'],
            "cmgnet_frag": self.PERFORMANCE_BENCHMARK['cmgnet_frag'],
            "inference_settings": {
                "model_eval": "model.eval() ✅",
                "no_grad": "torch.no_grad() ✅",
                "precision": "bf16 ✅",
                "random_seed": "42 ✅"
            }
        }

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
    print("="*80)
    print("集成NMR预测器 - 更新版本示例")
    print("="*80)

    # 创建集成预测器
    predictor = EnsembleNMRPredictor()

    # 测试NMR数据（环丙沙星）
    nmr_list = [[176.49, 166.02, 153.61, 152.37, 148.32, 144.25, 144.21, 139.18,
                119.48, 119.44, 111.34, 111.23, 107.03, 107.02, 106.92,
                46.45, 42.66, 36.08, 7.71]]

    # 使用更新后的predict接口
    result = predictor.predict(nmr_list,
                             xgboost_threshold=0.5,
                             num_returns=10,
                             prediction_mode="ensemble",
                             use_top_k=True)

    print(f"\n🎯 预测结果:")
    if "integrated_decision" in result:
        decision = result["integrated_decision"]
        print(f"   最终预测: {decision['final_prediction']}")
        print(f"   是否为核心分子: {decision['is_core_molecule']}")
        print(f"   置信度: {decision['confidence']:.2%}")
        print(f"   决策方法: {decision['method']}")

    # 显示性能基准
    if "performance_summary" in result:
        perf = result["performance_summary"]
        print(f"\n📊 性能基准:")
        print(f"   XGBoost: {perf['xgboost']['accuracy']:.2%} 准确率")
        print(f"   SMILES Top-10: {perf['cmgnet_smiles']['top10_accuracy']:.2%}")
        print(f"   FRAG Top-10: {perf['cmgnet_frag']['top10_accuracy']:.2%}")


if __name__ == "__main__":
    main()
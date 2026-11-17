#!/usr/bin/env python3
"""
CMGNet简单推理脚本
基于Django项目中的CMGNet模型推理代码简化而来

功能:
    从NMR化学位移数据预测分子结构

使用方法:
    python cmgnet_simple_inference.py
"""

import os
import sys
import random
from pathlib import Path
from argparse import Namespace

# 设置随机种子
RANDOM_SEED = 42

def set_random_seed(seed: int = RANDOM_SEED):
    """设置随机种子"""
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
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

set_random_seed(RANDOM_SEED)

import torch
import numpy as np
from typing import List, Dict
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from rdkit import Chem
from rdkit.Chem import rdSubstructLibrary
import warnings
from rdkit import RDLogger

# 屏蔽警告
RDLogger.DisableLog('rdApp.warning')
warnings.filterwarnings("ignore", category=DeprecationWarning)

class NmrDataset:
    """NMR数据集处理类"""
    def __init__(self):
        self.min_value = -50
        self.max_value = 350
        self.scale = 10
        self.units = (self.max_value - self.min_value) * self.scale
        self.start = 189
        self.end = 190
        self.number_start = 191

    def fill_item(self, nmr, want_token=True):
        """处理NMR数据"""
        nmr = list(set(nmr))
        nmr = [round((value - self.min_value) * self.scale) for value in nmr]
        nmr = [min(max(0, value), self.units-1) for value in nmr]
        nmr.sort()
        if want_token:
            nmr = [self.start] + [value + self.number_start for value in nmr] + [self.end]
            item = {"input_ids": nmr,
                    "attention_mask": [1 for _ in range(len(nmr))]}
            return item
        else:
            item = np.zeros(self.units)
            item[nmr] = 1
            item = torch.from_numpy(item).to(torch.float32)
            item_ = {"input_ids": [self.start, self.number_start, self.end],
                     "attention_mask": [1, 1, 1]}
            return item, item_

def query_core(ori_smiles):
    """检测分子是否包含母核结构"""
    try:
        mol = Chem.MolFromSmiles(ori_smiles)
        mols = rdSubstructLibrary.CachedSmilesMolHolder()
        fps = rdSubstructLibrary.TautomerPatternHolder(1024)
        mols.AddSmiles(ori_smiles)
        fp = Chem.PatternFingerprint(mol, fpSize=1024, tautomerFingerprints=True)
        fps.AddFingerprint(fp)
        library = rdSubstructLibrary.SubstructLibrary(mols, fps)
        indices_1 = library.GetMatches(Chem.MolFromSmiles('O=C1C(C(O)=O)=CNC2=CC=CC=C12'), maxResults=50000)
        indices_2 = library.GetMatches(Chem.MolFromSmiles('O=C1C(C(O)=O)=CNC2=NC=CC=C12'), maxResults=50000)

        if len(indices_1) > 0:
            core_mol = Chem.MolFromSmiles('O=C(O)c1c[nH]c2ccccc2c1=O')
            canonical_core = Chem.MolToSmiles(core_mol)
            return canonical_core, 1
        elif len(indices_2) > 0:
            core_mol = Chem.MolFromSmiles('O=C(O)c1c[nH]c2ncccc2c1=O')
            canonical_core = Chem.MolToSmiles(core_mol)
            return canonical_core, 1
        else:
            return '', 0
    except Exception as e:
        return '', 0

class CMGNetPredictor:
    """CMGNet预测器"""

    def __init__(self, model_path: str = None, device: str = "cuda:0"):
        """
        初始化CMGNet预测器

        Args:
            model_path: 模型路径
            device: 设备 (cuda:0 或 cpu)
        """
        self.device = device
        
        # 设置默认路径
        project_dir = Path(__file__).parent
        if model_path is None:
            model_path = str(project_dir / "checkpoints/bart_base_0910")

        print("初始化CMGNet预测器")
        print(f"模型路径: {model_path}")
        print(f"设备: {device}")

        # 创建参数
        self.args = Namespace(
            model_dir=model_path,
            num_beams=10,
            max_length=512,
        )

        # 加载分词器和模型
        print("加载分词器...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        print("分词器已加载")

        print("加载模型...")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        print("模型已加载")

        # 初始化NMR数据处理类
        self.nmr_dataset = NmrDataset()

        print("CMGNet预测器初始化完成")

    def predict(self, nmr_shifts: List[float], num_returns: int = 10, mode: str = "smiles") -> Dict:
        """
        预测分子结构

        Args:
            nmr_shifts: NMR化学位移列表
            num_returns: 返回结果数量
            mode: 预测模式 ("frag", "smiles")

        Returns:
            预测结果字典
        """
        try:
            # 处理NMR数据
            nmr_result = self.nmr_dataset.fill_item(nmr_shifts, want_token=True)

            # 构造输入
            inputs = {
                "input_ids": torch.tensor(nmr_result["input_ids"]).unsqueeze(0).to(self.device),
                "attention_mask": torch.tensor(nmr_result["attention_mask"]).unsqueeze(0).to(self.device)
            }

            # 根据模式设置token IDs
            if mode == "smiles":
                bos_token_id = 187
                eos_token_id = 188
                decoder_start_token_id = 187
            else:  # "frag" 模式
                bos_token_id = 183
                eos_token_id = 184
                decoder_start_token_id = 183

            # 生成预测
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=self.args.max_length,
                    num_beams=self.args.num_beams,
                    num_return_sequences=num_returns,
                    bos_token_id=bos_token_id,
                    pad_token_id=1,
                    eos_token_id=eos_token_id,
                    decoder_start_token_id=decoder_start_token_id,
                    output_scores=True,
                    return_dict_in_generate=True
                )

            # 解码结果
            generated_sequences = outputs.sequences
            generated_scores = outputs.scores

            # 清理预测结果
            pred_sequences = []
            for seq in generated_sequences:
                decoded = self.tokenizer.decode(seq, skip_special_tokens=True)
                cleaned = decoded.replace("<fragment>", "").replace("</fragment>", "").replace("<SMILES>", "").replace("</SMILES>", "").replace("<pad>", "").replace("</s>", "").strip()
                pred_sequences.append(cleaned)

            # 处理分数
            if isinstance(generated_scores, tuple):
                final_scores = generated_scores[-1].cpu().numpy() if generated_scores else np.zeros(len(generated_sequences))
            else:
                final_scores = generated_scores.cpu().numpy()

            if len(final_scores.shape) > 1 and final_scores.shape[1] > 0:
                final_scores = final_scores[0, :len(generated_sequences)]

            # 生成预测结果列表
            top_predictions = []
            for i, smiles in enumerate(pred_sequences):
                is_valid = self.validate_smiles(smiles)
                core_structure, has_core = query_core(smiles)

                top_predictions.append({
                    'rank': i + 1,
                    'smiles': smiles,
                    'is_valid': is_valid,
                    'score': float(final_scores[i]),
                    'has_core_structure': bool(has_core),
                    'core_structure': core_structure
                })

            return {
                'top_predictions': top_predictions,
                'num_returns': len(top_predictions),
                'nmr_count': len(nmr_shifts),
                'mode': mode
            }

        except Exception as e:
            return {
                'top_predictions': [],
                'num_returns': 0,
                'error': str(e),
                'nmr_count': len(nmr_shifts)
            }

    def validate_smiles(self, smiles: str) -> bool:
        """验证SMILES字符串是否有效"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False

    def print_result(self, nmr_shifts: List[float], result: Dict):
        """打印预测结果"""
        print("\n" + "="*80)
        print("CMGNet预测结果")
        print("="*80)
        print(f"\n输入信息:")
        print(f"  NMR化学位移数量: {len(nmr_shifts)}")
        print(f"  化学位移范围: {min(nmr_shifts):.2f} - {max(nmr_shifts):.2f} ppm")

        if 'error' in result:
            print(f"\n预测失败: {result['error']}")
        elif not result.get('top_predictions'):
            print(f"\n没有生成预测结果")
        else:
            print(f"\nTop-{len(result['top_predictions'])} 预测结果:")

            for pred in result['top_predictions']:
                validity_mark = "✓" if pred['is_valid'] else "✗"
                core_mark = "★" if pred['has_core_structure'] else "✗"

                print(f"\n  [{pred['rank']:2d}] {validity_mark} 分数: {pred['score']:.4f}")
                print(f"       {core_mark} 母核结构: {pred['has_core_structure']}")
                if pred['has_core_structure']:
                    print(f"       母核SMILES: {pred['core_structure']}")
                print(f"       SMILES: {pred['smiles']}")

        print("="*80)

def main():
    """主函数"""
    print("CMGNet简单推理演示")

    try:
        # 初始化预测器
        predictor = CMGNetPredictor()

        # 测试案例 - 环丙沙星
        cipro_nmr = [
            176.49, 166.02, 153.61, 152.37, 148.32, 144.25, 144.21, 139.18,
            119.48, 119.44, 111.34, 111.23, 107.03, 107.02, 106.92,
            46.45, 42.66, 36.08, 7.71
        ]

        print(f"\n测试案例: 环丙沙星")
        print(f"NMR峰数量: {len(cipro_nmr)}")

        # Frag模式预测
        print(f"\n模式: Frag (母核结构预测)")
        result_frag = predictor.predict(cipro_nmr, mode="frag", num_returns=10)
        predictor.print_result(cipro_nmr, result_frag)

        # SMILES模式预测
        print(f"\n模式: SMILES (完整结构预测)")
        result_smiles = predictor.predict(cipro_nmr, mode="smiles", num_returns=10)
        predictor.print_result(cipro_nmr, result_smiles)

    except Exception as e:
        print(f"\n程序执行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
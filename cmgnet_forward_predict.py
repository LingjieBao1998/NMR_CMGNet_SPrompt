#!/usr/bin/env python3
"""
CMGNet前向预测脚本
基于Django项目中的CMGNet模型，提供分子结构前向预测功能

功能:
    从NMR化学位移数据前向预测分子结构
    支持批量预测和不同预测模式
    提供详细的预测结果分析

使用方法:
    python cmgnet_forward_predict.py --nmr 176.49 166.02 153.61 --mode smiles --top 5
"""

import os
import sys
import random
import argparse
import json
from pathlib import Path
from argparse import Namespace
from typing import List, Dict, Union

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
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from rdkit import Chem
from rdkit.Chem import rdSubstructLibrary, Descriptors, rdMolDescriptors
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

def get_core_atom_indices(smiles, core_smiles):
    """获取母核结构在分子中的原子序号"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        core = Chem.MolFromSmiles(core_smiles, sanitize=False)

        if mol is None or core is None:
            return []

        try:
            matches = mol.GetSubstructMatches(core)
        except Exception as e:
            matches = []

        if len(matches) > 0:
            match_idx_list = list(matches[0])
            return match_idx_list
        else:
            try:
                from rdkit.Chem import rdFMCS
                mcs = rdFMCS.FindMCS([core, mol])
                if mcs.numAtoms == core.GetNumAtoms():
                    pattern = Chem.MolFromSmarts(mcs.smartsString)
                    matches_in_mol = mol.GetSubstructMatches(pattern)
                    if len(matches_in_mol) > 0:
                        match_idx_list = list(matches_in_mol[0])
                        return match_idx_list
                    else:
                        return []
                else:
                    return []
            except Exception as mcs_error:
                return []

    except Exception as e:
        return []

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
            return canonical_core, 1, get_core_atom_indices(ori_smiles, 'O=C(O)c1c[nH]c2ccccc2c1=O')
        elif len(indices_2) > 0:
            core_mol = Chem.MolFromSmiles('O=C(O)c1c[nH]c2ncccc2c1=O')
            canonical_core = Chem.MolToSmiles(core_mol)
            return canonical_core, 1, get_core_atom_indices(ori_smiles, 'O=C(O)c1c[nH]c2ncccc2c1=O')
        else:
            return '', 0, []
    except Exception as e:
        return '', 0, []

def calculate_molecular_properties(smiles: str) -> Dict:
    """计算分子性质"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}

        return {
            'molecular_weight': Descriptors.MolWt(mol),
            'logp': Descriptors.MolLogP(mol),
            'num_atoms': mol.GetNumAtoms(),
            'num_bonds': mol.GetNumBonds(),
            'num_rings': rdMolDescriptors.CalcNumRings(mol),
            'num_heteroatoms': rdMolDescriptors.CalcNumHeteroatoms(mol),
            'tpsa': Descriptors.TPSA(mol),
        }
    except:
        return {}

class CMGNetForwardPredictor:
    """CMGNet前向预测器"""

    def __init__(self, model_path: str = None, device: str = "cuda:0"):
        """
        初始化CMGNet前向预测器

        Args:
            model_path: 模型路径
            device: 设备 (cuda:0 或 cpu)
        """
        self.device = device

        # 设置默认路径
        project_dir = Path(__file__).parent
        if model_path is None:
            model_path = str(project_dir / "checkpoints/saved_hf_model_0910")

        print("=" * 80)
        print("初始化CMGNet前向预测器")
        print("=" * 80)
        print(f"模型路径: {model_path}")
        print(f"设备: {device}")

        # 创建参数
        self.args = Namespace(
            model_dir=model_path,
            num_beams=10,
            max_length=512,
        )

        # 加载分词器和模型
        print("加载分词器和模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        print("加载完成")

        # 初始化NMR数据处理类
        self.nmr_dataset = NmrDataset()

        print("=" * 80)

    def forward_predict(self, nmr_shifts: List[float], num_returns: int = 10, 
                       mode: str = "smiles", calculate_properties: bool = True) -> Dict:
        """
        前向预测分子结构

        Args:
            nmr_shifts: NMR化学位移列表
            num_returns: 返回结果数量
            mode: 预测模式 ("frag", "smiles")
            calculate_properties: 是否计算分子性质

        Returns:
            预测结果字典
        """
        try:
            print(f"\n开始前向预测...")
            print(f"输入NMR峰数量: {len(nmr_shifts)}")
            print(f"预测模式: {mode}")
            print(f"返回Top-{num_returns}结果")

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

            # 前向生成预测
            print("执行模型前向推理...")
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

            # 归一化分数
            scores_exp = np.exp(final_scores - np.max(final_scores))
            normalized_scores = scores_exp / scores_exp.sum()

            # 生成预测结果列表
            top_predictions = []
            for i, smiles in enumerate(pred_sequences):
                is_valid = self.validate_smiles(smiles)
                core_structure, has_core, core_atom_indices = query_core(smiles)

                prediction = {
                    'rank': i + 1,
                    'smiles': smiles,
                    'is_valid': is_valid,
                    'raw_score': float(final_scores[i]),
                    'normalized_score': float(normalized_scores[i]),
                    'has_core_structure': bool(has_core),
                    'core_structure': core_structure,
                    'core_atom_indices': core_atom_indices
                }

                # 计算分子性质
                if calculate_properties and is_valid:
                    properties = calculate_molecular_properties(smiles)
                    prediction.update(properties)

                top_predictions.append(prediction)

            print(f"前向预测完成，生成{len(top_predictions)}个结果")

            return {
                'top_predictions': top_predictions,
                'num_returns': len(top_predictions),
                'nmr_count': len(nmr_shifts),
                'mode': mode,
                'input_info': {
                    'nmr_range': [min(nmr_shifts), max(nmr_shifts)],
                    'nmr_count': len(nmr_shifts)
                }
            }

        except Exception as e:
            print(f"前向预测失败: {e}")
            import traceback
            traceback.print_exc()
            return {
                'top_predictions': [],
                'num_returns': 0,
                'error': str(e),
                'nmr_count': len(nmr_shifts)
            }

    def batch_forward_predict(self, nmr_list: List[List[float]], num_returns: int = 10,
                             mode: str = "smiles") -> List[Dict]:
        """
        批量前向预测

        Args:
            nmr_list: NMR化学位移列表的列表
            num_returns: 每个样本返回的结果数量
            mode: 预测模式

        Returns:
            预测结果列表
        """
        print(f"\n开始批量前向预测，共{len(nmr_list)}个样本")
        results = []

        for i, nmr_shifts in enumerate(nmr_list):
            print(f"\n处理样本 {i+1}/{len(nmr_list)}")
            result = self.forward_predict(nmr_shifts, num_returns, mode)
            results.append(result)

        print(f"\n批量前向预测完成")
        return results

    def validate_smiles(self, smiles: str) -> bool:
        """验证SMILES字符串是否有效"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False

    def print_forward_result(self, result: Dict, detailed: bool = True):
        """打印前向预测结果"""
        print("\n" + "=" * 80)
        print("CMGNet前向预测结果")
        print("=" * 80)

        if 'input_info' in result:
            info = result['input_info']
            print(f"\n输入信息:")
            print(f"  NMR化学位移数量: {info['nmr_count']}")
            print(f"  化学位移范围: {info['nmr_range'][0]:.2f} - {info['nmr_range'][1]:.2f} ppm")

        if 'error' in result:
            print(f"\n预测失败: {result['error']}")
        elif not result.get('top_predictions'):
            print(f"\n没有生成预测结果")
        else:
            print(f"\n预测模式: {result['mode']}")
            print(f"Top-{len(result['top_predictions'])} 预测结果:")

            valid_count = 0
            core_count = 0

            for pred in result['top_predictions']:
                validity_mark = "✓" if pred['is_valid'] else "✗"
                core_mark = "★" if pred['has_core_structure'] else "✗"

                print(f"\n  [#{pred['rank']:2d}] {validity_mark} 分数: {pred['normalized_score']:.4f} (原始: {pred['raw_score']:.2f})")
                print(f"       {core_mark} 母核结构: {pred['has_core_structure']}")

                if pred['has_core_structure']:
                    print(f"       母核SMILES: {pred['core_structure']}")
                    if pred.get('core_atom_indices'):
                        print(f"       母核原子序号: {pred['core_atom_indices']}")
                    core_count += 1

                print(f"       SMILES: {pred['smiles']}")

                # 详细信息
                if detailed and pred['is_valid']:
                    if 'molecular_weight' in pred:
                        print(f"       分子量: {pred['molecular_weight']:.2f}")
                    if 'logp' in pred:
                        print(f"       LogP: {pred['logp']:.2f}")
                    if 'num_atoms' in pred:
                        print(f"       原子数: {pred['num_atoms']}, 环数: {pred['num_rings']}")

                if pred['is_valid']:
                    valid_count += 1

            print(f"\n统计摘要:")
            print(f"  有效SMILES: {valid_count}/{len(result['top_predictions'])}")
            print(f"  包含母核结构: {core_count}/{len(result['top_predictions'])}")

        print("=" * 80)

    def save_results(self, result: Dict, filename: str):
        """保存预测结果到JSON文件"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"结果已保存到: {filename}")
        except Exception as e:
            print(f"保存结果失败: {e}")

def parse_nmr_input(nmr_str: str) -> List[float]:
    """解析NMR输入字符串"""
    try:
        # 尝试解析为JSON数组
        if nmr_str.startswith('['):
            return json.loads(nmr_str)
        # 尝试解析为逗号分隔的数值
        else:
            return [float(x.strip()) for x in nmr_str.split(',')]
    except:
        raise ValueError(f"无法解析NMR输入: {nmr_str}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='CMGNet前向预测')
    parser.add_argument('--nmr', type=str, help='NMR化学位移数据 (逗号分隔或JSON数组)')
    parser.add_argument('--nmr_file', type=str, help='包含NMR数据的文件 (每行一个样本)')
    parser.add_argument('--mode', type=str, default='smiles', choices=['frag', 'smiles'], help='预测模式')
    parser.add_argument('--top', type=int, default=10, help='返回Top-K结果')
    parser.add_argument('--device', type=str, default='cuda:0', help='设备')
    parser.add_argument('--output', type=str, help='输出文件名')
    parser.add_argument('--model', type=str, help='模型路径')
    parser.add_argument('--no_properties', action='store_true', help='不计算分子性质')

    args = parser.parse_args()

    print("CMGNet前向预测系统")
    print("=" * 80)

    try:
        # 初始化预测器
        predictor = CMGNetForwardPredictor(model_path=args.model, device=args.device)

        # 处理输入
        if args.nmr:
            # 单样本预测
            nmr_shifts = parse_nmr_input(args.nmr)
            print(f"输入NMR数据: {len(nmr_shifts)} 个化学位移")

            result = predictor.forward_predict(
                nmr_shifts,
                num_returns=args.top,
                mode=args.mode,
                calculate_properties=not args.no_properties
            )

            predictor.print_forward_result(result)

            if args.output:
                predictor.save_results(result, args.output)

        elif args.nmr_file:
            # 批量预测
            print(f"从文件读取NMR数据: {args.nmr_file}")
            nmr_list = []
            with open(args.nmr_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        nmr_shifts = parse_nmr_input(line)
                        nmr_list.append(nmr_shifts)

            print(f"共读取 {len(nmr_list)} 个样本")

            results = predictor.batch_forward_predict(
                nmr_list,
                num_returns=args.top,
                mode=args.mode
            )

            # 保存批量结果
            if args.output:
                batch_result = {
                    'total_samples': len(results),
                    'mode': args.mode,
                    'results': results
                }
                predictor.save_results(batch_result, args.output)
            else:
                # 打印每个结果
                for i, result in enumerate(results):
                    print(f"\n样本 {i+1} 的预测结果:")
                    predictor.print_forward_result(result, detailed=False)

        else:
            # 演示模式
            print("使用演示数据...")

            # 演示案例 - 环丙沙星
            demo_nmr = [
                176.49, 166.02, 153.61, 152.37, 148.32, 144.25, 144.21, 139.18,
                119.48, 119.44, 111.34, 111.23, 107.03, 107.02, 106.92,
                46.45, 42.66, 36.08, 7.71
            ]

            print(f"\n演示案例: 环丙沙星 (Ciprofloxacin)")
            print(f"NMR峰数量: {len(demo_nmr)}")

            # Frag模式
            print(f"\n" + "=" * 80)
            print("模式1: Frag (母核结构预测)")
            print("=" * 80)
            result_frag = predictor.forward_predict(demo_nmr, mode="frag", num_returns=10)
            predictor.print_forward_result(result_frag)

            # SMILES模式
            print(f"\n" + "=" * 80)
            print("模式2: SMILES (完整结构预测)")
            print("=" * 80)
            result_smiles = predictor.forward_predict(demo_nmr, mode="smiles", num_returns=10)
            predictor.print_forward_result(result_smiles)

    except Exception as e:
        print(f"\n程序执行失败: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
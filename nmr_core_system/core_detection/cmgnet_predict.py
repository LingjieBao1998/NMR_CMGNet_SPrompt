#!/usr/bin/env python3
"""
CMGNet核心分子检测推理代码

功能:
    输入NMR化学位移列表，输出CMGNet预测结果

使用方法:
    python cmgnet_predict.py
"""

import sys
import os
import random
from pathlib import Path
from argparse import Namespace

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

# Flash Attention配置（先尝试使用，失败后禁用）
import os
# 默认不禁用Flash Attention，让系统自动尝试

import torch
import numpy as np
import json
from typing import List, Dict, Optional, Union

# 添加项目路径
current_file_path = Path(__file__).resolve()
project_dir = current_file_path.parent.parent.parent
sys.path.insert(0, str(project_dir))

# 导入必要的模块
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import pytorch_lightning as pl
from torch import nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, rdSubstructLibrary
import warnings
from rdkit import RDLogger

# 屏蔽警告
RDLogger.DisableLog('rdApp.warning')
warnings.filterwarnings("ignore", category=DeprecationWarning)

class NmrDataset:
    """NMR数据集类 - 来自main_cmgnet.py"""
    def __init__(self, **kwargs):
        self.min_value = kwargs.pop('min_value') if 'min_value' in kwargs.keys() else -50
        self.max_value = kwargs.pop('max_value') if 'max_value' in kwargs.keys() else 350
        self.scale = kwargs.pop('scale') if 'scale' in kwargs.keys() else 10
        self.augment_prob = kwargs.pop('augment_prob') if 'augment_prob' in kwargs.keys() else 0.5
        # augment_range unit ±ppm
        self.augment_range = kwargs.pop('augment_range') if 'augment_range' in kwargs.keys() else 1.5
        self.units = (self.max_value - self.min_value) * self.scale
        self.start = kwargs.pop('start') if 'start' in kwargs.keys() else 189
        self.end = kwargs.pop('end') if 'end' in kwargs.keys() else 190
        self.number_start = kwargs.pop('number_start') if 'number_start' in kwargs.keys() else 191

    def get_value(self, value, augment_prob):
        if random.random() < augment_prob:
            return value
        else:
            return 0

    def fill_item(self, nmr, want_token=True):
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

    def augment_item(self, nmr, want_token=True):
        nmr = list(set(nmr))
        offset = np.random.normal(
            loc=0.0, scale=self.augment_range/2, size=len(nmr))
        offset = [min(max(-self.augment_range, i), self.augment_range)
                  for i in offset]
        offset = [self.get_value(i, self.augment_prob) for i in offset]
        new_nmr = [nmr_item + offset_item for nmr_item,
                   offset_item in zip(nmr, offset)]
        return self.fill_item(new_nmr, want_token=want_token)

def get_core_atom_indices(smiles, core_smiles):
    """
    获取母核结构在分子中的原子序号

    参考main_cmgnet.py第915-958行的实现

    Args:
        smiles: 分子SMILES
        core_smiles: 母核结构SMILES

    Returns:
        母核原子序号列表，如果匹配失败返回空列表
    """
    try:
        import random
        from rdkit.Chem import rdFMCS

        mol = Chem.MolFromSmiles(smiles)
        core = Chem.MolFromSmiles(core_smiles, sanitize=False)

        if mol is None or core is None:
            return []

        # 查找子结构匹配
        try:
            matches = mol.GetSubstructMatches(core)
        except Exception as e:
            print("len(matches)==0")
            matches = []

        if len(matches) > 0:
            # 直接匹配成功，返回第一个匹配的原子序号
            match_idx_list = list(matches[0])
            return match_idx_list
        else:
            # 退化到使用MCS (基于main_cmgnet.py第938-958行)
            try:
                mcs = rdFMCS.FindMCS([core, mol])
                if mcs.numAtoms == core.GetNumAtoms():
                    # MCS找到完整的母核结构
                    pattern = Chem.MolFromSmarts(mcs.smartsString)
                    matches_in_mol = mol.GetSubstructMatches(pattern)

                    if len(matches_in_mol) > 0:
                        # 使用MCS匹配的原子序号
                        match_idx_list = list(matches_in_mol[0])
                        return match_idx_list
                    else:
                        return []
                else:
                    return []
            except Exception as mcs_error:
                print(f"MCS处理失败: {mcs_error}")
                return []

    except Exception as e:
        print(f"获取母核原子序号失败: {e}")
        return []
    
def query_core(ori_smiles):
    """检测分子是否包含母核结构"""
    try:
        mol = Chem.MolFromSmiles(ori_smiles)
        # 用 CachedSmilesMolHolder(会 sanitize), 不能用 CachedTrustedSmilesMolHolder:
        # 后者 AddSmiles 不 sanitize, 存的 Kekule 结构与下面 sanitize 后的指纹对不上,
        # 会让母核的非芳香写法被漏判成 False (2026-10-22 修复)。
        mols = rdSubstructLibrary.CachedSmilesMolHolder()
        fps = rdSubstructLibrary.TautomerPatternHolder(1024)
        mols.AddSmiles(ori_smiles)
        fp = Chem.PatternFingerprint(mol,fpSize=1024,tautomerFingerprints=True)
        fps.AddFingerprint(fp)
        library = rdSubstructLibrary.SubstructLibrary(mols,fps)
        indices_1 = library.GetMatches(Chem.MolFromSmiles('O=C1C(C(O)=O)=CNC2=CC=CC=C12'),maxResults=50000)
        indices_2 = library.GetMatches(Chem.MolFromSmiles('O=C1C(C(O)=O)=CNC2=NC=CC=C12'),maxResults=50000)

        if len(indices_1)>0:
            # 返回标准化的母核SMILES
            core_mol = Chem.MolFromSmiles('O=C(O)c1c[nH]c2ccccc2c1=O')
            canonical_core = Chem.MolToSmiles(core_mol)
            return canonical_core, 1, get_core_atom_indices(ori_smiles, 'O=C(O)c1c[nH]c2ccccc2c1=O')
        elif len(indices_2)>0:
            # 返回标准化的母核SMILES
            core_mol = Chem.MolFromSmiles('O=C(O)c1c[nH]c2ncccc2c1=O')
            canonical_core = Chem.MolToSmiles(core_mol)
            return canonical_core, 1, get_core_atom_indices(ori_smiles, 'O=C(O)c1c[nH]c2ncccc2c1=O')
        else:
            return '', 0, []
    except Exception as e:
        print(f"查询核心结构失败: {e}")
        return '', 0, []

class CMGNetPredictor:
    """CMGNet预测器"""

    def __init__(self,
                 model_path: str = None,
                 tokenizer_path: str = None,
                 ckpt_path: str = None,
                 device: str = "cuda:0"):
        """
        初始化CMGNet预测器

        Args:
            model_path: 模型目录路径
            tokenizer_path: 分词器路径  
            ckpt_path: 检查点路径
            device: 设备 (cuda:0 或 cpu)
        """
        # 设置默认参数
        self.device = device
        if model_path is None:
            model_path = str(project_dir / "checkpoints/bart_base_0910")
        if tokenizer_path is None:
            tokenizer_path = str(project_dir / "checkpoints/bart_base_0910")
        if ckpt_path is None:
            ckpt_path = str(project_dir / "checkpoints/train_cmgnet_ring_system_3.2_1029_v2_best.ckpt")

        print("="*100)
        print("初始化CMGNet预测器")
        print("="*100)
        print(f"模型路径: {model_path}")
        print(f"检查点路径: {ckpt_path}")
        print(f"设备: {device}")

        # 创建默认参数
        self.args = Namespace(
            model_dir=str(project_dir / "checkpoints/bart_base_0910"),
            use_extra_encoder=False,
            use_ring_system=True,
            force_aligned=True,
            num_beams=10,
            decode_mode="frag",
            max_length=512,
            smiles_aug_proba=0.5,
            nmr_aug_proba=0.5,
            nmr_aug_scale=3.0,
            batch_size=1,
            num_workers=0  # 推理时使用0避免多进程问题
        )

        # 加载分词器和模型
        print(f"\n📦 加载分词器...")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        print(f"   ✅ 分词器已加载")

        print(f"\n📦 加载模型...")
        # 第一次尝试：使用Flash Attention（如果可用）
        try:
            if os.path.exists(ckpt_path):
                # 尝试从检查点加载
                self.model = self.load_model_from_checkpoint(ckpt_path)
                print(f"   ✅ 从检查点加载模型成功 (Flash Attention可用)")
            else:
                # 从预训练模型加载
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
                print(f"   ✅ 从预训练模型加载成功 (Flash Attention可用)")
        except Exception as e:
            error_msg = str(e)
            # 检查是否为Flash Attention相关错误
            if 'flash_attn' in error_msg.lower() or 'flash attention' in error_msg.lower():
                print(f"   ⚠️  Flash Attention加载失败: {e}")
                print(f"   🔄 禁用Flash Attention，使用标准注意力机制...")
                try:
                    # 禁用Flash Attention，使用标准注意力
                    os.environ['USE_FLASH_ATTENTION'] = '0'
                    os.environ['ENABLE_FLASH_ATTENTION'] = '0'
                    
                    # 重新尝试加载模型
                    if os.path.exists(ckpt_path):
                        self.model = self.load_model_from_checkpoint(ckpt_path)
                    else:
                        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
                    print(f"   ✅ 使用标准注意力机制加载成功")
                except Exception as retry_e:
                    print(f"   ❌ 标准注意力机制加载也失败: {retry_e}")
                    raise
            else:
                print(f"   ❌ 加载模型失败: {e}")
                raise

        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 初始化NMR数据处理类
        self.nmr_dataset = NmrDataset(
            min_value=-50,
            max_value=350,
            scale=10,
            augment_prob=0.0,  # 推理时不进行数据增强
            augment_range=3.0,
            start=189,
            end=190,
            number_start=191
        )
        
        print(f"\n✅ CMGNet预测器初始化完成")
        print("="*100)

    def load_model_from_checkpoint(self, ckpt_path: str):
        """从检查点加载模型"""
        # 这里需要实现自定义的检查点加载逻辑
        # 由于原始代码使用了PyTorch Lightning，我们需要适配
        try:
            # 尝试直接加载模型权重 - 使用weights_only=False解决兼容性问题
            checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            
            # 检查是否是PyTorch Lightning检查点
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                # 移除'model.'前缀（如果有）
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('model.'):
                        new_state_dict[k[6:]] = v
                    else:
                        new_state_dict[k] = v
                
                model = AutoModelForSeq2SeqLM.from_pretrained(self.args.model_dir)
                model.load_state_dict(new_state_dict, strict=False)
                print(f"   ✅ 成功从PyTorch Lightning检查点加载模型")
                return model
            else:
                # 直接是模型权重
                model = AutoModelForSeq2SeqLM.from_pretrained(self.args.model_dir)
                model.load_state_dict(checkpoint, strict=False)
                print(f"   ✅ 成功从直接权重加载模型")
                return model
                
        except Exception as e:
            print(f"   ⚠️  从检查点加载失败，尝试使用预训练模型: {e}")
            return AutoModelForSeq2SeqLM.from_pretrained(self.args.model_dir)

    def _predict_nmr_frag_combined(self, nmr_shifts: List[float], num_returns: int = 10,
                                 skip_smart_selection: bool = True, core_structure: str = None) -> Dict:
        """
        nmr+frag模式的真正实现：将frag预测结果拼接到NMR输入后面

        参考main_cmgnet_fragsmiles.py第1085行：
        input_ids_list.append(temp["nmr_result"]["input_ids"] + temp["ring_system_result"]["input_ids"])

        Args:
            nmr_shifts: NMR化学位移列表
            num_returns: 返回结果数量
            skip_smart_selection: 是否跳过智能选择
            core_structure: 指定的母核结构SMILES (可选)
        """
        try:
            # 第一步：获取frag预测（跳过智能选择避免递归）
            if core_structure:
                # 如果指定了母核结构，直接使用
                print(f"   🎯 使用指定母核: {core_structure}")
                top_frag_smiles = core_structure
            else:
                # 否则获取frag预测
                frag_result = self._predict_single(nmr_shifts, num_returns=1, mode="frag", skip_smart_selection=True)

                if not frag_result.get('top_predictions') or not frag_result['top_predictions']:
                    return {
                        'top_predictions': [],
                        'num_returns': 0,
                        'error': 'frag预测失败',
                        'nmr_count': len(nmr_shifts),
                        'mode': 'nmr+frag'
                    }

                # 选择frag的Top-1预测
                top_frag_smiles = frag_result['top_predictions'][0]['smiles']
                print(f"   🔗 Frag预测: {top_frag_smiles}")
            
            # 第二步：将frag SMILES转换为token IDs (参考main_cmgnet.py的smiles2token函数)
            frag_tokens_dict = self.tokenizer([top_frag_smiles], max_length=self.args.max_length)
            frag_tokens = [187] + frag_tokens_dict["input_ids"][0][1:-1] + [188]  # 添加BOS和EOS标记
            
            # 第三步：处理NMR数据
            nmr_result = self.nmr_dataset.fill_item(nmr_shifts, want_token=True)
            
            # 第四步：拼接NMR和frag的token IDs (参考main_cmgnet.py第1085行)
            combined_input_ids = nmr_result["input_ids"] + frag_tokens
            combined_attention_mask = nmr_result["attention_mask"] + [1] * len(frag_tokens)
            
            # 第五步：构造拼接后的输入
            inputs = {
                "input_ids": torch.tensor(combined_input_ids).unsqueeze(0).to(self.device),
                "attention_mask": torch.tensor(combined_attention_mask).unsqueeze(0).to(self.device)
            }
            
            # 第六步：使用smiles模式的token IDs进行生成
            bos_token_id = 187
            eos_token_id = 188
            decoder_start_token_id = 187
            
            # 第七步：生成完整SMILES
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
                
            # 解码预测结果
            generated_sequences = outputs.sequences
            generated_scores = outputs.scores
            
            # 清理预测结果
            pred_sequences = []
            for seq in generated_sequences:
                decoded = self.tokenizer.decode(seq, skip_special_tokens=True)
                cleaned = decoded.replace("<fragment>","").replace("</fragment>","").replace("<SMILES>", "").replace("</SMILES>", "").replace("<pad>", "").replace("</s>", "").strip()
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
                core_structure, has_core, core_atom_indices = query_core(smiles)

                top_predictions.append({
                    'rank': i + 1,
                    'smiles': smiles,
                    'is_valid': is_valid,
                    'score': float(final_scores[i]),
                    'has_core_structure': bool(has_core),
                    'core_structure': core_structure,
                    'core_atom_indices': core_atom_indices
                })
            
            return {
                'top_predictions': top_predictions,
                'num_returns': len(top_predictions),
                'nmr_count': len(nmr_shifts),
                'mode': 'nmr+frag',
                'frag_input': top_frag_smiles  # 记录使用的frag输入
            }
            
        except Exception as e:
            return {
                'top_predictions': [],
                'num_returns': 0,
                'error': str(e),
                'nmr_count': len(nmr_shifts),
                'mode': 'nmr+frag'
            }

    def _apply_smart_core_selection(self, nmr_shifts: List[float], top_predictions: List[Dict]) -> Optional[Dict]:
        """
        frag模式的智能母核选择策略
        
        策略:
        1. 统计Top10中出现频率最高的合法母核结构
        2. 如果最高频率>1，选择该母核进行nmr+frag预测
        3. 如果所有母核频率都为1，选择概率最高的有母核的预测
        4. 如果没有检测到母核结构，尝试使用预定义的母核结构进行nmr+frag预测
        
        Args:
            nmr_shifts: 原始NMR数据
            top_predictions: Top-10预测结果
            
        Returns:
            增强的预测结果，如果不需要增强则返回None
        """
        # 1. 筛选出包含母核结构的合法分子
        valid_core_predictions = []
        for pred in top_predictions:
            if pred['is_valid'] and pred['has_core_structure'] and pred['core_structure']:
                valid_core_predictions.append(pred)
        
        # 如果没有发现任何包含母核的合法分子，检查是否有预定义母核结构
        if not valid_core_predictions:
            print("   🔍 未检测到母核结构，检查预定义母核...")
            return self._check_and_try_predefined_cores(nmr_shifts, top_predictions)
        
        # 2. 统计母核结构出现频率（使用标准化SMILES）
        core_frequency = {}
        for pred in valid_core_predictions:
            core_structure = pred['core_structure']
            # 标准化母核SMILES，确保相同结构的不同表示被识别为同一个
            try:
                core_mol = Chem.MolFromSmiles(core_structure)
                if core_mol:
                    canonical_core = Chem.MolToSmiles(core_mol)
                    # 更新预测中的母核为标准化形式
                    pred['core_structure'] = canonical_core
                    core_structure = canonical_core
            except:
                pass  # 如果标准化失败，使用原始SMILES
            
            if core_structure not in core_frequency:
                core_frequency[core_structure] = []
            core_frequency[core_structure].append(pred)
        
        # 3. 找出最高频率的母核结构
        max_frequency = 0
        most_frequent_core = None
        most_frequent_predictions = []
        
        for core_structure, predictions in core_frequency.items():
            frequency = len(predictions)
            if frequency > max_frequency:
                max_frequency = frequency
                most_frequent_core = core_structure
                most_frequent_predictions = predictions
        
        # 4. 选择策略
        selected_prediction = None
        
        if max_frequency > 1:
            # 策略1: 选择出现频率最高的母核中分数最高的预测
            most_frequent_predictions.sort(key=lambda x: x['score'], reverse=True)
            selected_prediction = most_frequent_predictions[0]
            print(f"   🧠 智能选择: 频率最高的母核 (出现{max_frequency}次)")
        else:
            # 策略2: 所有母核频率都为1，选择分数最高的有母核预测
            valid_core_predictions.sort(key=lambda x: x['score'], reverse=True)
            selected_prediction = valid_core_predictions[0]
            print(f"   🧠 智能选择: 分数最高的母核 (频率为1)")
        
        if selected_prediction is None:
            return None
        
        print(f"   🎯 选择母核: {selected_prediction['core_structure']}")
        print(f"   📊 原始排名: {selected_prediction['rank']}, 分数: {selected_prediction['score']:.4f}")

        # 5. 使用nmr+frag模式重新预测，传入选定的母核结构
        print(f"   🚀 触发nmr+frag精确生成...")
        try:
            enhanced_result = self._predict_nmr_frag_combined(
                nmr_shifts,
                num_returns=10,
                skip_smart_selection=True,
                core_structure=selected_prediction['core_structure']
            )
            
            # 添加元数据说明这是增强的结果
            if 'top_predictions' in enhanced_result:
                enhanced_result['frag_analysis'] = {
                    'selected_core_structure': selected_prediction['core_structure'],
                    'original_rank': selected_prediction['rank'],
                    'original_score': selected_prediction['score'],
                    'core_frequency': max_frequency,
                    'selection_strategy': 'high_frequency' if max_frequency > 1 else 'high_score',
                    'triggered_mode': 'nmr+frag'
                }
                
                # 更新模式标识
                enhanced_result['mode'] = 'frag->nmr+frag'
                enhanced_result['original_mode'] = 'frag'
            
            return enhanced_result
            
        except Exception as e:
            print(f"   ❌ nmr+frag生成失败: {e}")
            return None

    def _check_and_try_predefined_cores(self, nmr_shifts: List[float], top_predictions: List[Dict]) -> Optional[Dict]:
        """
        检查预测结果中是否包含预定义母核结构，如果有则尝试nmr+frag预测
        
        逻辑:
        1. 检查原始预测中是否有分子包含预定义母核结构
        2. 如果找到了，用该母核进行nmr+frag预测
        3. 如果新预测包含母核结构，替换原有结果；否则保留原有结果
        
        Args:
            nmr_shifts: NMR化学位移列表
            top_predictions: 原始frag预测结果
            
        Returns:
            增强的预测结果（决定是否替换），否则返回None表示保留原结果
        """
        # 预定义的母核结构 (使用SMARTS模式进行更灵活的子结构匹配)
        predefined_cores_raw = [
            'O=c1cc[nH]c2ccccc12',      # 喹诺酮类母核结构1 (苯并吡喃酮)
            'c1ccc2c(=O)cc[nH]c2n1'     # 喹诺酮类母核结构2 (氮杂苯并吡喃酮)
        ]
        
        # 标准化预定义母核结构，去除重复
        predefined_cores = []
        seen_canonical = set()
        
        for core_smiles in predefined_cores_raw:
            try:
                # 标准化SMILES
                core_mol = Chem.MolFromSmiles(core_smiles, sanitize=False)
                if core_mol is None:
                    print(f"      ⚠️  跳过无效母核: {core_smiles}")
                    continue
                
                canonical_core = Chem.MolToSmiles(core_mol)
                
                # 检查是否已经存在相同的标准结构
                if canonical_core in seen_canonical:
                    print(f"      ℹ️  跳过重复母核: {core_smiles} → {canonical_core}")
                    continue
                
                # 添加到列表
                predefined_cores.append(core_smiles)
                seen_canonical.add(canonical_core)
                
                # 显示母核信息
                num_atoms = core_mol.GetNumAtoms()
                try:
                    rings = Chem.GetSymmSSSR(core_mol)
                    num_rings = len(rings)
                except:
                    num_rings = 0
                
                print(f"      ✅ 添加母核: {core_smiles}")
                print(f"         标准形式: {canonical_core}")
                print(f"         原子数: {num_atoms}, 环数: {num_rings}")
                
            except Exception as e:
                print(f"      ❌ 处理母核失败: {core_smiles} - {e}")
                continue
        
        if not predefined_cores:
            print(f"      ❌ 没有有效的预定义母核结构")
            return None
        
        print(f"      📋 共使用 {len(predefined_cores)} 个不同的预定义母核结构")
        print(f"   🔍 检查预测结果中是否包含预定义母核结构...")
        
        # 检查原始预测中是否有分子包含预定义母核结构
        found_core = None
        found_prediction = None
        
        for pred in top_predictions:
            if not pred.get('is_valid'):
                continue
                
            smiles = pred.get('smiles', '')
            if not smiles:
                continue
                
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                    
                # 检查是否包含任一预定义母核结构
                for core_smiles in predefined_cores:
                    core_mol = Chem.MolFromSmiles(core_smiles, sanitize=False)
                    if core_mol is None:
                        continue
                        
                    # 检查子结构匹配
                    if mol.HasSubstructMatch(core_mol):
                        found_core = core_smiles
                        found_prediction = pred
                        print(f"   ✅ 发现预定义母核结构: {core_smiles}")
                        print(f"   📋 来源: 预测排名 #{pred['rank']}, SMILES: {smiles}")
                        break
                if found_core:
                    break
                    
            except Exception as e:
                print(f"   ⚠️  检查预测时出错: {e}")
                continue
        
        # 如果没有找到预定义母核结构，返回None保留原结果
        if not found_core:
            print(f"   ❌ 未在预测结果中发现预定义母核结构")
            print(f"   📋 保留原始frag预测结果")
            return None
        
        # 找到了预定义母核结构，进行nmr+frag预测
        print(f"   🚀 使用发现的母核结构进行nmr+frag预测...")
        
        try:
            enhanced_result = self._predict_nmr_frag_combined(
                nmr_shifts,
                num_returns=10,
                skip_smart_selection=True,
                core_structure=found_core
            )
            
            # 检查新的预测结果是否包含母核结构
            if enhanced_result and enhanced_result.get('top_predictions'):
                has_valid_core = any(
                    pred.get('has_core_structure') and pred.get('core_structure')
                    for pred in enhanced_result['top_predictions']
                )
                
                if has_valid_core:
                    print(f"   ✅ nmr+frag预测成功包含母核结构，替换原始结果")
                    
                    # 添加元数据
                    enhanced_result['predefined_core_analysis'] = {
                        'found_predefined_core': True,
                        'core_structure': found_core,
                        'source_prediction_rank': found_prediction['rank'],
                        'source_smiles': found_prediction['smiles'],
                        'enhancement_strategy': 'predefined_core_nmr_frag'
                    }
                    
                    # 更新模式标识
                    enhanced_result['mode'] = 'frag->predefined_core->nmr+frag'
                    enhanced_result['original_mode'] = 'frag'
                    
                    return enhanced_result
                else:
                    print(f"   ⚠️  nmr+frag预测未包含母核结构，保留原始结果")
                    return None
            else:
                print(f"   ❌ nmr+frag预测失败，保留原始结果")
                return None
                
        except Exception as e:
            print(f"   ❌ nmr+frag预测异常: {e}，保留原始结果")
            return None


    def predict(self, nmr_input, num_returns: int = 10, mode: str = "smiles", batch_size: int = 8):
        """
        统一预测接口 - 自动识别单样本或批量预测

        Args:
            nmr_input: NMR化学位移数据，支持两种格式:
                     - 单样本: List[float] 如 [176.49, 166.02, 153.61, ...]
                     - 批量: List[List[float]] 如 [[176.49, 166.02], [175.4, 164.8], ...]
            num_returns: 返回前k个预测结果，默认10
            mode: 生成模式 ("frag", "smiles", "nmr+frag")
            batch_size: 批量大小（仅批量预测时有效），默认8

        Returns:
            单样本: Dict 预测结果字典
            批量: List[Dict] 预测结果列表
        """
        # 自动识别输入类型
        if isinstance(nmr_input, list) and len(nmr_input) > 0:
            if isinstance(nmr_input[0], list):
                # 批量预测
                return self._predict_batch(nmr_input, batch_size, mode, num_returns)
            elif isinstance(nmr_input[0], (int, float)):
                # 单样本预测
                return self._predict_single(nmr_input, num_returns, mode)
        
        # 无法识别的输入类型
        raise ValueError("无法识别的输入格式，请提供 List[float] 或 List[List[float]]")

    def _predict_single(self, nmr_shifts: List[float], num_returns: int = 10, mode: str = "smiles", skip_smart_selection: bool = False) -> Dict:
        """单样本预测内部实现"""
        try:
            # 特殊处理nmr+frag模式：需要先获取frag预测，然后拼接
            if mode == "nmr+frag":
                return self._predict_nmr_frag_combined(nmr_shifts, num_returns, skip_smart_selection=skip_smart_selection)
            
            # 使用NmrDataset处理NMR数据 - 按照main_cmgnet.py的方式
            nmr_result = self.nmr_dataset.fill_item(nmr_shifts, want_token=True)
            
            # 构造输入
            inputs = {
                "input_ids": torch.tensor(nmr_result["input_ids"]).unsqueeze(0).to(self.device),
                "attention_mask": torch.tensor(nmr_result["attention_mask"]).unsqueeze(0).to(self.device)
            }
            
            # 根据模式设置不同的token IDs - 来自main_cmgnet.py
            if mode in ["smiles", "nmr+frag"]:
                bos_token_id = 187
                eos_token_id = 188
                decoder_start_token_id = 187
            else:  # "frag" 模式
                bos_token_id = 183
                eos_token_id = 184
                decoder_start_token_id = 183
            
            # 生成多个预测结果
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
                
            # 解码所有预测结果 - 按照main_cmgnet.py的方式处理
            generated_sequences = outputs.sequences
            generated_scores = outputs.scores
            
            # 清理预测结果
            pred_sequences = []
            for seq in generated_sequences:
                decoded = self.tokenizer.decode(seq, skip_special_tokens=True)
                # 清理特殊标签
                cleaned = decoded.replace("<fragment>","").replace("</fragment>","").replace("<SMILES>", "").replace("</SMILES>", "").replace("<pad>", "").replace("</s>", "").strip()
                pred_sequences.append(cleaned)
            
            # 处理分数 - 直接指数化转换为概率
            if isinstance(generated_scores, tuple):
                # 取最后一步的分数
                final_scores = generated_scores[-1].cpu().numpy() if generated_scores else np.zeros(len(generated_sequences))
            else:
                final_scores = generated_scores.cpu().numpy()

            # 简单分数处理
            if len(final_scores.shape) > 1 and final_scores.shape[1] > 0:
                final_scores = final_scores[0, :len(generated_sequences)]
            
            # 在单次预测内做softmax归一化，概率总和为1
            # 生成预测结果列表
            top_predictions = []
            for i, smiles in enumerate(pred_sequences):
                # 验证SMILES
                is_valid = self.validate_smiles(smiles)

                # 检测母核结构
                core_structure, has_core, core_atom_indices = query_core(smiles)

                top_predictions.append({
                    'rank': i + 1,
                    'smiles': smiles,
                    'is_valid': is_valid,
                    'score': float(final_scores[i]),
                    'has_core_structure': bool(has_core),
                    'core_structure': core_structure,
                    'core_atom_indices': core_atom_indices
                })
            
            # frag模式的智能母核选择策略（当未跳过时）
            if mode == "frag" and not skip_smart_selection:
                enhanced_result = self._apply_smart_core_selection(nmr_shifts, top_predictions)
                # 检查是否需要替换原始结果
                if enhanced_result and enhanced_result.get('top_predictions'):
                    # 验证增强结果是否包含母核结构
                    has_valid_core = any(
                        pred.get('has_core_structure') and pred.get('core_structure')
                        for pred in enhanced_result['top_predictions']
                    )
                    if has_valid_core:
                        print(f"   🎯 使用增强结果替换原始预测")
                        return enhanced_result
                    else:
                        print(f"   ⚠️  增强结果中没有包含母核结构，保留原有预测结果")
                else:
                    print(f"   ⚠️  增强过程失败或无效，保留原有预测结果")
            
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

    def _predict_batch(self, nmr_list: List[List[float]], batch_size: int = 8, mode: str = "smiles", num_returns: int = 10) -> List[Dict]:
        """
        批量预测 - 按照main_cmgnet.py的方式实现

        Args:
            nmr_list: NMR化学位移列表的列表
            batch_size: 批量大小，最大8，默认8
            mode: 生成模式 ("frag", "smiles", "nmr+frag")
            num_returns: 每个样本返回的结果数量

        Returns:
            预测结果列表
        """
        # 限制batch_size最大为8
        batch_size = min(batch_size, 8)
        
        all_results = []
        
        # 分批处理
        for i in range(0, len(nmr_list), batch_size):
            batch_nmr = nmr_list[i:i + batch_size]
            current_batch_size = len(batch_nmr)
            print(f"\n📦 处理批次 {i//batch_size + 1}/{(len(nmr_list) + batch_size - 1)//batch_size} (样本数: {current_batch_size})")
            
            try:
                # 批量处理NMR数据 - 按照main_cmgnet.py的方式
                batch_input_ids = []
                batch_attention_mask = []
                
                for nmr_shifts in batch_nmr:
                    nmr_result = self.nmr_dataset.fill_item(nmr_shifts, want_token=True)
                    batch_input_ids.append(nmr_result["input_ids"])
                    batch_attention_mask.append(nmr_result["attention_mask"])
                
                # 填充到相同长度
                max_len = max(len(ids) for ids in batch_input_ids)
                padded_input_ids = []
                padded_attention_mask = []
                
                for ids, mask in zip(batch_input_ids, batch_attention_mask):
                    # 填充到最大长度
                    padded_ids = ids + [1] * (max_len - len(ids))  # 1是pad_token_id
                    padded_mask = mask + [0] * (max_len - len(mask))
                    padded_input_ids.append(padded_ids)
                    padded_attention_mask.append(padded_mask)
                
                # 构造batch输入
                inputs = {
                    "input_ids": torch.tensor(padded_input_ids).to(self.device),
                    "attention_mask": torch.tensor(padded_attention_mask).to(self.device)
                }
                
                # 根据模式设置不同的token IDs - 来自main_cmgnet.py
                if mode in ["smiles", "nmr+frag"]:
                    bos_token_id = 187
                    eos_token_id = 188
                    decoder_start_token_id = 187
                else:  # "frag" 模式
                    bos_token_id = 183
                    eos_token_id = 184
                    decoder_start_token_id = 183
                
                # 批量生成 - 按照main_cmgnet.py的方式
                # 注意：num_return_sequences不能超过num_beams
                total_return_sequences = min(num_returns * current_batch_size, self.args.num_beams)
                # 确保每个样本至少有1个结果
                if total_return_sequences < current_batch_size:
                    total_return_sequences = current_batch_size
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_length=self.args.max_length,
                        num_beams=self.args.num_beams,
                        num_return_sequences=total_return_sequences,  # 总共生成的序列数
                        bos_token_id=bos_token_id,
                        pad_token_id=1,
                        eos_token_id=eos_token_id,
                        decoder_start_token_id=decoder_start_token_id,
                        output_scores=True,
                        return_dict_in_generate=True
                    )
                
                # 处理批量结果 - 按照main_cmgnet.py的方式
                generated_sequences = outputs.sequences
                generated_scores = outputs.scores
                
                # 解码所有预测结果
                pred_sequences = []
                for seq in generated_sequences:
                    decoded = self.tokenizer.decode(seq, skip_special_tokens=True)
                    cleaned = decoded.replace("<fragment>","").replace("</fragment>","").replace("<SMILES>", "").replace("</SMILES>", "").replace("<pad>", "").replace("</s>", "").strip()
                    pred_sequences.append(cleaned)
                
                # 处理分数 - 按照main_cmgnet.py的方式
                if isinstance(generated_scores, tuple):
                    final_scores = generated_scores[-1].cpu().numpy() if generated_scores else np.zeros(len(generated_sequences))
                else:
                    final_scores = generated_scores.cpu().numpy()

                # Simple score processing
                if len(final_scores.shape) > 1 and final_scores.shape[1] > 0:
                    final_scores = final_scores[0, :len(generated_sequences)]
                
                # 按照main_cmgnet.py的方式分组 - 为每个输入样本分配结果
                sequences_per_input = len(pred_sequences) // current_batch_size if current_batch_size > 0 else len(pred_sequences)
                
                for batch_idx in range(current_batch_size):
                    start_idx = batch_idx * sequences_per_input
                    end_idx = start_idx + sequences_per_input
                    
                    pred_smiles = pred_sequences[start_idx:end_idx]
                    pred_scores = final_scores[start_idx:end_idx].tolist()
                    
                    # 构建预测结果
                    sample_predictions = []
                    for rank, smiles in enumerate(pred_smiles):
                        is_valid = self.validate_smiles(smiles)

                        # 检测母核结构
                        core_structure, has_core, core_atom_indices = query_core(smiles)

                        sample_predictions.append({
                            'rank': rank + 1,
                            'smiles': smiles,
                            'is_valid': is_valid,
                            'score': float(pred_scores[rank]),
                            'has_core_structure': bool(has_core),
                            'core_structure': core_structure,
                            'core_atom_indices': core_atom_indices
                        })
                    
                    all_results.append({
                        'top_predictions': sample_predictions,
                        'num_returns': len(sample_predictions),
                        'nmr_count': len(batch_nmr[batch_idx]),
                        'mode': mode
                    })
                    
            except Exception as e:
                print(f"❌ 批次处理失败: {e}")
                # 如果batch处理失败，fallback到单个处理
                for nmr_shifts in batch_nmr:
                    try:
                        result = self.predict(nmr_shifts, num_returns=num_returns, mode=mode)
                        all_results.append(result)
                    except Exception as single_error:
                        print(f"❌ 单个样本处理失败: {single_error}")
                        all_results.append({
                            'top_predictions': [],
                            'num_returns': 0,
                            'error': str(single_error),
                            'nmr_count': len(nmr_shifts),
                            'mode': mode
                        })
        return all_results

    # 向后兼容的包装方法
    def predict_batch(self, nmr_list: List[List[float]], batch_size: int = 8, mode: str = "smiles", num_returns: int = 10) -> List[Dict]:
        """
        向后兼容的批量预测接口
        
        注意: 建议直接使用 predict() 方法，它会自动识别输入类型
        """
        return self._predict_batch(nmr_list, batch_size, mode, num_returns)

    def validate_smiles(self, smiles: str) -> bool:
        """
        验证SMILES字符串是否有效

        Args:
            smiles: SMILES字符串

        Returns:
            是否有效
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False

    def print_result(self, nmr_shifts: List[float], result: Dict):
        """打印预测结果"""
        print("\n" + "="*100)
        print("CMGNet预测结果 (Top-{})".format(result.get('num_returns', 10)))
        print("="*100)
        print(f"\n输入信息:")
        print(f"  NMR化学位移数量: {len(nmr_shifts)}")
        print(f"  化学位移范围: {min(nmr_shifts):.2f} - {max(nmr_shifts):.2f} ppm")

        if 'error' in result:
            print(f"\n❌ 预测失败: {result['error']}")
        elif not result.get('top_predictions'):
            print(f"\n❌ 没有生成预测结果")
        else:
            print(f"\n🎯 Top-{len(result['top_predictions'])} 预测结果:")
            
            valid_count = 0
            core_count = 0
            for pred in result['top_predictions']:
                validity_mark = "✅" if pred['is_valid'] else "❌"
                core_mark = "🎯" if pred['has_core_structure'] else "❌"
                
                print(f"\n  [{pred['rank']:2d}] {validity_mark} 分数: {pred['score']:.4f}")
                print(f"       {core_mark} 母核结构: {pred['has_core_structure']}")
                if pred['has_core_structure']:
                    print(f"       母核SMILES: {pred['core_structure']}")
                    core_count += 1
                print(f"       SMILES: {pred['smiles']}")
                
                if pred['is_valid']:
                    valid_count += 1

            print(f"\n📊 统计: {valid_count}/{len(result['top_predictions'])} 个有效SMILES")
            if core_count > 0:
                print(f"🎯 {core_count}/{len(result['top_predictions'])} 个预测包含母核结构")
            if valid_count > 0:
                print(f"✅ 成功生成了 {valid_count} 个有效的分子结构！")
            else:
                print(f"⚠️  所有预测结果都生成了无效的SMILES字符串")

        print("="*100)


def main():
    """主函数 - 演示用法"""
    print("="*100)
    print("CMGNet核心分子检测 - 推理演示")
    print("="*100)

    try:
        # 1. 初始化预测器
        predictor = CMGNetPredictor()
        
        # 2. 测试案例 - 环丙沙星 (Ciprofloxacin)
        cipro_nmr = [
            176.49, 166.02, 153.61, 152.37, 148.32, 144.25, 144.21, 139.18,
            119.48, 119.44, 111.34, 111.23, 107.03, 107.02, 106.92,
            46.45, 42.66, 36.08, 7.71
        ]

        print(f"\n测试案例: 环丙沙星 (Ciprofloxacin)")

        # 3. 演示不同模式的预测
        print(f"\n" + "="*100)
        print("演示不同预测模式")
        print("="*100)
        
        # 3.1 Frag模式 (包含智能选择)
        print(f"\n🔬 模式1: Frag (包含智能母核选择)")
        result_frag = predictor.predict(cipro_nmr, mode="frag", num_returns=10)
        predictor.print_result(cipro_nmr, result_frag)
        
        # 3.2 SMILES模式
        # print(f"\n🔬 模式2: SMILES (直接生成完整结构)")
        # result_smiles = predictor.predict(cipro_nmr, mode="smiles", num_returns=10)
        # predictor.print_result(cipro_nmr, result_smiles)
        
        # 3.3 NMIR+Frag模式
        # print(f"\n🔬 模式3: NMIR+Frag (结合模式)")
        # result_nmr_frag = predictor.predict(cipro_nmr, mode="nmr+frag", num_returns=10)
        # predictor.print_result(cipro_nmr, result_nmr_frag)

    except Exception as e:
        print(f"\n❌ 程序执行失败: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*100)


if __name__ == "__main__":
    main()
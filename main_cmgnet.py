import os
import math
import json
import random
import argparse
import numpy as np
from pathlib import Path

import time

import torch.distributed as dist
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, LightningDataModule
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.strategies.ddp import DDPStrategy
from torch import nn
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import get_scheduler
import json
from functools import reduce
from operator import mul
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

import warnings
from rdkit import RDLogger
from rdkit.Chem import rdMolDescriptors
# Import RDKit for molecule analysis
from rdkit import Chem
from rdkit.Chem import rdSubstructLibrary

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def _register_numpy2_pickle_alias():
    """numpy 2.x 写的 pickle 在 numpy 1.x 环境的反序列化兼容。

    1020 那批 lmdb（如 data/NMR_other_data_1020_add_ring_system_1029/new_db.lmdb）
    是用 numpy 2.x 序列化的，记录里引用 numpy._core.numeric._frombuffer；
    而本训练环境 py312 是 numpy 1.26，pickle.loads 会直接抛
    ModuleNotFoundError: No module named 'numpy._core.numeric'。
    这里把 numpy._core.* 别名到 numpy.core.*（函数体一致），无需升级 numpy。
    版本考古同类问题见 main() 里旧 ckpt 的 BartTokenizerFast 旧路径注册。
    """
    if int(np.__version__.split(".")[0]) >= 2:
        return  # numpy 2 本身就是 _core 布局, 无需处理
    import importlib
    import numpy.core
    sys.modules.setdefault("numpy._core", numpy.core)
    for _sub in ("numeric", "multiarray", "umath", "numerictypes", "fromnumeric",
                 "shape_base", "_methods", "_exceptions", "_multiarray_umath",
                 "overrides", "records", "memmap", "arrayprint", "einsumfunc"):
        try:
            sys.modules.setdefault(
                "numpy._core." + _sub, importlib.import_module("numpy.core." + _sub))
        except Exception:
            pass


_register_numpy2_pickle_alias()

from new_RingSystemFinder import RingSystemFinder

# import useful_rdkit_utils as uru
from rdkit.Chem import rdmolops

import lmdb
import pickle
import copy

from rdkit.Chem import rdSubstructLibrary
import torch
import torch.nn as nn
from transformers import BartModel, BartConfig
from transformers.models.bart.modeling_bart import BartScaledWordEmbedding, BartEncoder, BartDecoder
from transformers.cache_utils import Cache, DynamicCache, EncoderDecoderCache
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers import AutoTokenizer, BartForConditionalGeneration
from typing import Optional, Tuple, Dict, Any, List
try:
    from flash_attn import flash_attn_func  # 必需安装flash_attn库
    from flash_attn.modules.mha import CrossAttention as FlashAttentionCrossAttention
    FLASH_ATTN_AVAILABLE = True
except Exception as e:
    print("flash_attn 不可用:", e)
    FLASH_ATTN_AVAILABLE = False
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from rdkit.Chem.rdFMCS import FindMCS


# 屏蔽 RDKit 的警告（包括 DEPRECATION WARNING）
RDLogger.DisableLog('rdApp.warning')  # 屏蔽 RDKit 自己的 Logger 警告
warnings.filterwarnings("ignore", category=DeprecationWarning)  # 屏蔽 Python 的 DeprecationWarning

def count_parameters(model):
    params = model.parameters()
    num_params = sum(reduce(mul, p.size(), 1) for p in params)
    return num_params

def print_model_size(model):
    num_params = count_parameters(model)
    num_params_m = num_params / 1e6
    print(f"Model size: {num_params_m:.2f}M parameters")

def get_random_smiles(old_mol):
    old_mol = copy.deepcopy(old_mol)
    len_mol = old_mol.GetNumAtoms()
    index_list = list(range(len_mol))
    if len(index_list)<=1:
        return Chem.MolToSmiles(old_mol)
    random.shuffle(index_list)
    li = []
    for index in index_list:
        try:
            new_smiles = Chem.MolToSmiles(old_mol, rootedAtAtom=index)
            li.append(new_smiles)
        except:
            pass
        if len(li) == 1:
            break
    if len(li) >= 1:
        return li[0]
    else:
        return None

def align_mol_with_order(mol, match_idx_list):
    # 构建新的原子顺序：环系统原子在前，其他原子在后
    all_atoms = list(range(mol.GetNumAtoms()))
    other_atoms = [i for i in all_atoms if i not in match_idx_list]
    new_order = match_idx_list + other_atoms
    
    # 重新排序原子并生成SMILES
    mol_renumbered = Chem.RenumberAtoms(mol, new_order)
    aligned_smiles = Chem.MolToSmiles(mol_renumbered, canonical=False)
    return aligned_smiles


def get_atom_count(mol):
    """计算分子中的原子数量"""
    if mol is None:
        return 0
    return mol.GetNumAtoms()

# 母核判定的固定口径 (单一实现, 不再有 env 开关 / 快速路径分支):
# 指纹库 (rdSubstructLibrary) + 互变异构指纹 (tautomerFingerprints=True)。
# 评测结果必须由这里唯一决定, 任何"优化"都要先过全量等价性验证再合入。
#
# 历史坑 (2026-10-22 修复): 原实现用 CachedTrustedSmilesMolHolder, 该 holder
# 的 AddSmiles 不做 sanitize, 存进去的是原始 Kekule 结构, 而指纹却由已 sanitize
# 的 mol 算出, 二者对不上 → 对母核的非芳香写法 (如 O=C1C(C(O)=O)=CNC2=CC=CC=C12
# 本身) 会漏判成 False。改用 CachedSmilesMolHolder (会 sanitize) 后与
# HasSubstructMatch 在全部 297,132 条真实候选上 0 差异。
#
# 查询分子是常量, 在模块级只构建一次。
_CORE_QUERY_SMILES = ('O=C1C(C(O)=O)=CNC2=CC=CC=C12', 'O=C1C(C(O)=O)=CNC2=NC=CC=C12')
CORE_QUERY_MOLS = [m for m in map(Chem.MolFromSmiles, _CORE_QUERY_SMILES) if m is not None]
CORE_FP_SIZE = 1024


def analyze_molecule(smiles):
    """分析单个分子 (is_valid / is_exact_match / has_core_structure)。

    has_core_structure 由指纹库 + 互变异构指纹判定, 固定口径见 CORE_QUERY_MOLS。
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {
                "is_valid": False,
                "is_exact_match": False,
                "has_core_structure": False
            }

        # 检查母核结构: 指纹库(互变异构) 路径
        has_core = False
        try:
            mols = rdSubstructLibrary.CachedSmilesMolHolder()
            fps = rdSubstructLibrary.TautomerPatternHolder(CORE_FP_SIZE)
            mols.AddSmiles(smiles)
            fp = Chem.PatternFingerprint(mol, fpSize=CORE_FP_SIZE, tautomerFingerprints=True)
            fps.AddFingerprint(fp)
            library = rdSubstructLibrary.SubstructLibrary(mols, fps)

            has_core = any(
                len(library.GetMatches(q, maxResults=1)) > 0
                for q in CORE_QUERY_MOLS
            )
        except:
            has_core = False

        return {
            "is_valid": True,
            "has_core_structure": has_core
        }
    except:
        return {
            "is_valid": False,
            "is_exact_match": False,
            "has_core_structure": False
        }

# def find_largest_smiles(ring_systems):
#     """
#     找到原子数量最多的SMILES字符串
#     如果多个分子具有相同的最大原子数，返回所有符合条件的SMILES列表
#     """
#     max_atoms = 0
#     largest_smiles = []  # 改为列表存储所有符合条件的SMILES
    
#     for smiles in ring_systems:
#         mol = Chem.MolFromSmiles(smiles, sanitize=False)
#         if mol is None:
#             continue  # 跳过无效的SMILES
            
#         num_atoms = get_atom_count(mol)
        
#         if num_atoms > max_atoms:
#             # 发现更大的原子数，重置列表
#             max_atoms = num_atoms
#             largest_smiles = [smiles]  # 重新开始列表
            
#         elif num_atoms == max_atoms:
#             # 原子数相同，添加到列表
#             largest_smiles.append(smiles)
    
#     return largest_smiles

def get_nonCH_atom_count(mol):
    """计算非碳非氢原子数量"""
    if mol is None:
        return 0
        
    count = 0
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol != "C" and symbol != "H":
            count += 1
    return count

def has_aromatic_ring(mol):
    """判断分子是否含有芳香环"""
    try:
        # 获取环信息
        sssr = Chem.GetSymmSSSR(mol)
        for ring in sssr:
            # 检查环是否全部为芳香键
            is_aromatic = True
            for i in range(len(ring)):
                bond = mol.GetBondBetweenAtoms(ring[i], ring[(i+1) % len(ring)])
                if bond is None or not bond.GetIsAromatic():
                    is_aromatic = False
                    break
            if is_aromatic:
                return True
        return False
    except:
        return False

def get_aromatic_ring_count(mol):
    """获取芳香环的数量"""
    try:
        aromatic_rings = 0
        sssr = Chem.GetSymmSSSR(mol)
        for ring in sssr:
            is_aromatic = True
            for i in range(len(ring)):
                bond = mol.GetBondBetweenAtoms(ring[i], ring[(i+1) % len(ring)])
                if bond is None or not bond.GetIsAromatic():
                    is_aromatic = False
                    break
            if is_aromatic:
                aromatic_rings += 1
        return aromatic_rings
    except:
        return 0

def find_largest_smiles(ring_systems):
    """
    找到SMILES字符串
    优先考虑有芳香性的环，如果多个环都有芳香性，则按照(原子数量 + 非CH原子个数)作为优先级选择最大的
    """
    candidates = []  # 存储所有有效分子信息
    
    # 第一步：收集所有有效分子的信息
    for smiles in ring_systems:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            continue  # 跳过无效的SMILES
            
        num_atoms = get_atom_count(mol)
        is_aromatic = has_aromatic_ring(mol)
        nonCH_count = get_nonCH_atom_count(mol)
        aromatic_count = get_aromatic_ring_count(mol) if is_aromatic else 0
        
        candidates.append({
            "smiles": smiles,
            "atom_count": num_atoms,
            "is_aromatic": is_aromatic,
            "nonCH_count": nonCH_count,
            "aromatic_count": aromatic_count
        })
    
    # 如果没有候选分子
    if not candidates:
        return ""
    
    # 如果只有一个候选分子
    if len(candidates) == 1:
        return candidates[0]["smiles"]
    
    # 第二步：优先考虑有芳香性的分子
    aromatic_candidates = [c for c in candidates if c["is_aromatic"]]
    non_aromatic_candidates = [c for c in candidates if not c["is_aromatic"]]
    
    # 如果有芳香性分子，只在芳香性分子中选择
    if aromatic_candidates:
        candidates_to_select = aromatic_candidates
        # print(f"在 {len(aromatic_candidates)} 个芳香性分子中选择最大的")
    else:
        candidates_to_select = non_aromatic_candidates
        # print(f"在 {len(non_aromatic_candidates)} 个非芳香性分子中选择最大的")
    
    # 第三步：按照原子数量降序排序，如果原子数相同则按非CH原子个数降序排序
    candidates_to_select.sort(key=lambda x: (x["atom_count"], x["nonCH_count"]), reverse=True)
    
    # 选择最大的候选分子
    selected = candidates_to_select[0]
    
    # 打印所有候选分子信息用于调试
    # for i, candidate in enumerate(candidates_to_select):
    #     print(f"{i+1}. 分子: {candidate['smiles']}, "
    #           f"原子数: {candidate['atom_count']}, "
    #           f"非CH原子: {candidate['nonCH_count']}, "
    #           f"芳香环数: {candidate['aromatic_count']}")
    
    # print(f"最终选择: {selected['smiles']}")
    return selected["smiles"]


def count_carbon_atoms(mol):
    return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)

def remove_specific_terminal_carbons(mol):
    # 创建分子的可编辑版本
    rwmol = Chem.RWMol(mol)
    
    # 获取分子的拓扑信息
    ring_info = mol.GetRingInfo()
    ring_atoms = set()
    for ring in ring_info.AtomRings():
        ring_atoms.update(ring)
    
    # 找出需要删除的原子
    atoms_to_remove = []
    for atom in rwmol.GetAtoms():
        # 检查是否是末端碳原子
        if atom.GetDegree() == 1 and atom.GetAtomicNum() == 6:
            # 获取连接的原子和键
            neighbor = atom.GetNeighbors()[0]
            bond = atom.GetBonds()[0]
            
            # 检查条件：邻居在环上且键不是单键
            if neighbor.GetIdx() in ring_atoms and bond.GetBondType() != Chem.BondType.SINGLE:
                atoms_to_remove.append(atom.GetIdx())
    
    # 按从大到小的顺序删除原子，避免索引问题
    for atom_idx in sorted(atoms_to_remove, reverse=True):
        rwmol.RemoveAtom(atom_idx)
    
    # 返回新的分子
    return rwmol.GetMol()

def get_ring_system(mol, proba=0.0):
    ring_system_finder = RingSystemFinder()
    ring_system_list_with_Rgroups = ring_system_finder.find_ring_systems(mol, as_mols=True)

    if random.random()>proba:
        ring_system = [Chem.MolToSmiles(remove_specific_terminal_carbons(_)) for _ in ring_system_list_with_Rgroups]
    else:
        try:
            ring_system = [get_random_smiles(remove_specific_terminal_carbons(_)) for _ in ring_system_list_with_Rgroups]
        except Exception as e:
            print(e)
            ring_system = [Chem.MolToSmiles(remove_specific_terminal_carbons(_)) for _ in ring_system_list_with_Rgroups]

    return ring_system



class NmrDataset(Dataset):
    def __init__(self, **kwargs):
        super(NmrDataset, self).__init__()
        self.scale = kwargs.pop(
            'scale') if 'scale' in kwargs.keys() else 10
        self.min_value = kwargs.pop(
            'min_value') if 'min_value' in kwargs.keys() else -50
        self.max_value = kwargs.pop(
            'max_value') if 'max_value' in kwargs.keys() else 350
        self.augment_prob = kwargs.pop(
            'augment_prob') if 'augment_prob' in kwargs.keys() else 0.5
        # augment_range unit ±ppm
        self.augment_range = kwargs.pop(
            'augment_range') if 'augment_range' in kwargs.keys() else 1.5
        self.units = (self.max_value - self.min_value) * self.scale
        self.start = kwargs.pop(
            'start') if 'start' in kwargs.keys() else 189
        self.end = kwargs.pop(
            'end') if 'end' in kwargs.keys() else 190
        self.number_start = kwargs.pop(
            'number_start') if 'number_start' in kwargs.keys() else 191

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
            nmr = [189] + [value + self.number_start for value in nmr] + [190]
            item = {"input_ids": nmr,
                    "attention_mask": [1 for _ in range(len(nmr))]}
            return item
        else:
            item = np.zeros(self.units)
            item[nmr] = 1
            item = torch.from_numpy(item).to(torch.float32)
            item_ = {"input_ids": [189, 191, 190],
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

def get_new_smiles(old_mol):
    len_mol = old_mol.GetNumAtoms()
    li = []
    for index in range(len_mol):
        try:
            new_smiles = Chem.MolToSmiles(old_mol, rootedAtAtom=index)
            li.append(new_smiles)
        except:
            pass
    if len(li) >= 1:
        return random.choice(li)
    else:
        return None

def get_args(notebook=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--debug', action='store_true')

    ## model_dir 
    parser.add_argument('--model_dir', default="./checkpoints/saved_hf_model_0910",type=str)

    # Training
    parser.add_argument('--use_lora', action='store_true')
    parser.add_argument('--use_molecule_formular', action='store_true')
    parser.add_argument('--use_ring_system', action='store_true')
    parser.add_argument('--force_aligned', action='store_true')
    parser.add_argument('--use_clear_smiles', action='store_true') # 证明了效果确实不行
    parser.add_argument('--use_extra_encoder', action='store_true') 
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--max_grad_norm', type=float, default=5.)
    parser.add_argument('--scheduler', type=str, choices=['cosine', 'constant'], default='cosine')
    parser.add_argument('--warmup_ratio', type=float, default=0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)

    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--load_encoder_only', action='store_true')
    parser.add_argument('--train_steps_per_epoch', type=int, default=-1)
    parser.add_argument('--eval_per_epoch', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='output/')
    parser.add_argument('--save_mode', type=str, default='best', choices=['best', 'all', 'last'])
    parser.add_argument('--load_ckpt', type=str, default='best')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--num_train_example', type=int, default=None)
    parser.add_argument('--label_smoothing', type=float, default=0.0)

    ## dataset
    parser.add_argument('--train_file', type=str, default="")
    parser.add_argument('--valid_file', type=str, default="")
    parser.add_argument('--test_file', type=str, default="")
    # Inference

    parser.add_argument('--ckpt', type=str, default=None)
    
    ## data_augment
    parser.add_argument('--smiles_aug_proba', type=float, default=0.0)
    parser.add_argument('--nmr_aug_proba', type=float, default=0.0)
    parser.add_argument('--nmr_aug_scale', type=float, default=1.5)

    parser.add_argument('--lmdb_path', type=str, default="")
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--remove_multi', action='store_true')

    parser.add_argument('--decode_mode', type=str, default="smiles")
    # nmr+frag 的 oracle 选项: 仅对 label==1(含母核)的样本附加"真实数据中的环系"作为片段输入,
    # 其余样本不给片段、退化为纯 NMR 输入(等价 smiles 模式); oracle 评测的参考统一用完整分子,
    # 避免参考=环系导致大量分子共享同一参考而在收尾去重时被合并。默认关闭(保持原行为)。
    parser.add_argument('--oracle_ring', action='store_true')
    # SPrompt(CMGNet-SPrompt): XGB 票为 True 的样本, 往 NMR 输入后拼接固定母核结构提示后做
    # nmr+frag 条件生成; 其余样本不给片段、退化为纯 NMR 输入(等价 smiles 模式)。
    # sprompt_votes 为与 test_file 等长的 0/1 列表(json, 按样本位置对齐); 评测参考统一用完整分子。
    parser.add_argument('--sprompt_votes', type=str, default="")
    parser.add_argument('--sprompt_core', type=str, default="O=C(O)c1c[nH]c2ccccc2c1=O")

    args = parser.parse_args([]) if notebook else parser.parse_args()
    args.lmdb_path = args.lmdb_path.split(",")
    args.train_file = args.train_file.split(",")
    args.valid_file = args.valid_file.split(",")
    args.test_file = args.test_file.split(",")

    return args

def smiles2token(tokenizer, smile_string, phase="val", block_size=512, proba=0.5):
    smile_string_tmp = smile_string
    if phase == "train":
        if random.random()<proba:
            old_mol = Chem.MolFromSmiles(smile_string)
            if old_mol is not None:
                try:
                    smile_string_tmp = Chem.MolToSmiles(
                        old_mol, isomericSmiles=False, doRandom=True, canonical=False)
                except:
                    smile_string_tmp2 = get_new_smiles(old_mol)
                    if smile_string_tmp2 is not None:
                        smile_string_tmp = smile_string_tmp2

    tmp = tokenizer([smile_string_tmp], max_length=block_size)
    tmp["input_ids"] = [[187] + i[1:-1] + [188]
                        for i in tmp["input_ids"]]
    tmp["input_ids"] = tmp["input_ids"][0]
    tmp["attention_mask"] = tmp["attention_mask"][0]
    return tmp

def remove_electron(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles  # 无法解析的 SMILES
        
        ## 清除去自由基
        for atom in mol.GetAtoms():
            if atom.GetNumRadicalElectrons() > 0:
                # print(f"发现自由基原子 {atom.GetSymbol()}，有 {atom.GetNumRadicalElectrons()} 个自由电子")
                # 加一个氢原子（简化处理，真实情况可能需要更复杂的逻辑）
                # atom.SetNumExplicitHs(atom.GetTotalNumHs() + 1)
                if True:
                    atom.SetNumRadicalElectrons(0)
        return Chem.MolToSmiles(mol)
    except Exception as e:
        return smiles

class CustomDataset(Dataset):
    def __init__(self, args, tokenizer, file_path="", phase="val"):
        """
        Args:
            data_list (list or array): 样本数据，比如特征列表 [样本1, 样本2, ...]
            labels (list or tensor, optional): 对应的标签，如 [0, 1, 0, ...]。可以为 None（如测试集）
        """
        self.args = args
        self.tokenizer = tokenizer
        self.phase = phase
        self.data = []
        if isinstance(file_path, str) and os.path.exists(file_path):
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext == '.json':
                with open(file_path, "r", encoding='utf-8') as f:
                    self.data.extend(json.load(f))
            elif file_ext in ('.pkl', '.pickle'):  # 修正：使用file_ext
                with open(file_path, "rb") as f:  # 修正：使用file_path
                    self.data.extend(pickle.load(f))
                    
        elif isinstance(file_path, list):
            for _ in file_path:
                if os.path.exists(_):
                    file_ext = os.path.splitext(_)[1].lower()
                    if file_ext == '.json':
                        with open(_, "r", encoding='utf-8') as f:
                            self.data.extend(json.load(f))
                    elif file_ext in ('.pkl', '.pickle'):  # 修正：使用file_ext
                        with open(_, "rb") as f:  # 修正：使用_
                            self.data.extend(pickle.load(f))
        
        if self.args.debug:
            self.data = self.data[:1000]

        # SPrompt: 加载 XGB 门控票(与 test_file 样本位置一一对应的 0/1 列表)
        self.sprompt_votes = None
        if getattr(args, "sprompt_votes", ""):
            with open(args.sprompt_votes, "r") as f:
                self.sprompt_votes = json.load(f)
            assert len(self.sprompt_votes) >= len(self.data), \
                f"sprompt_votes({len(self.sprompt_votes)}) 短于测试集({len(self.data)})"

        self.ori_NmrDataset = NmrDataset(augment_range=self.args.nmr_aug_scale)

        # self.data = [self.data[i] for i in range(len(self.data)) if self.data[i]["source"]!= "NMR BANK"]
        self.env_list = []
        self.key_list = []
        self.env_length_idx = []
        if self.phase == "train":
            if isinstance(args.lmdb_path, str):
                if os.path.exists(args.lmdb_path):
                    env = lmdb.open(
                            args.lmdb_path,
                            readonly=True,
                            map_size= 1000 * 1024**3,
                            lock=False,
                            readahead=False,
                            meminit=False,
                            max_readers=1024,
                        )
                    data_split = np.load(args.split_file_path)
                    if phase == "train":
                        data_indices = data_split['train_index']  # 注意键名是 'train_index'，不是 'train'
                    elif phase == "valid":
                        data_indices = data_split['val_index']
                    else:
                        data_indices = data_split['test_index']
                    
                    if self.args.debug and phase != "train":
                        data_indices = data_indices[:1000]
                    
                    self.env_list.append(env)
                    self.key_list.extend(data_indices)
                    self.env_length_idx.append(len(self.key_list))
            else:
                for temp_i, _ in enumerate(range(len(args.lmdb_path))):
                    if os.path.exists(args.lmdb_path[_]):
                        lmdb_path = args.lmdb_path[_]
                        env = lmdb.open(
                                lmdb_path,
                                map_size= 1000 * 1024**3,
                                readonly=True,
                                lock=False,
                                readahead=False,
                                meminit=False,
                                max_readers=1024,
                            )
                        length = env.stat()['entries']
                        self.env_list.append(env)
                        key_list = [f'{count}'.encode('ascii') for count in range(length)]
                        # if temp_i>0:
                        #     key_list = [f'{count}'.encode('ascii') for count in range(length)]
                        # else:
                        #     print("using 5000000 data as test data")
                        #     key_list = [f'{count}'.encode('ascii') for count in range(length) if count>5000000]
                        self.key_list.extend(key_list)
                        self.env_length_idx.append(len(self.key_list))
        
        if len(self.env_list)>0:
            self.env_list = [None]*len(self.env_list)

    def __len__(self):
        return len(self.data) + len(self.key_list)  # 数据集大小
    
    def __load_env(self, env_idx):
        for temp_i, _ in enumerate(range(len(self.args.lmdb_path))):
            if temp_i == env_idx:
                lmdb_path = self.args.lmdb_path[_]
                env = lmdb.open(
                        lmdb_path,
                        map_size= 1000 * 1024**3,
                        readonly=True,
                        lock=False,
                        readahead=False,
                        meminit=False,
                        max_readers=1024,
                    )
        self.env_list[env_idx] = env
    
    def get_fromula_result(self, smiles):
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            formula = rdMolDescriptors.CalcMolFormula(mol)
            tmp = self.tokenizer([formula])
            tmp["input_ids"] = [[181] + i[1:-1] + [182]
                                for i in tmp["input_ids"]]
            tmp["input_ids"] = tmp["input_ids"][0]
            tmp["attention_mask"] = tmp["attention_mask"][0]
            
            return tmp
        except Exception as e:
            print(e)
            return {}

    def __getitem__(self, idx):
        try:
            if idx < len(self.data):
                sample = self.data[idx]  # 获取第 idx 个样本
                if "C_nmr" in sample:
                    ori_nmr = sample["C_nmr"]
                else:
                    ori_nmr = sample["NMR"]
                if "canonical_smiles" in sample:
                    ori_smiles = sample["canonical_smiles"]
                else:
                    ori_smiles = sample["cano_smiles"]
            else:
                for env_idx in range(len(self.env_length_idx)):
                    if self.env_length_idx[env_idx] + len(self.data)>idx:
                        break
                if self.env_list[env_idx] is None:
                    self.__load_env(env_idx)
                env =  self.env_list[env_idx]
                key = self.key_list[idx-len(self.data)]
                with env.begin() as txn:
                    sample = pickle.loads(txn.get(key))
                
                if "canonical_smiles" in sample:
                    ori_smiles = sample["canonical_smiles"]
                else:
                    ori_smiles = sample["cano_smiles"]
                
                if "C_nmr" in sample:
                    ori_nmr = sample["C_nmr"]
                else:
                    ori_nmr = sample["NMR"]
            
            if len(ori_nmr)==0:
                print("no nmr in data")
                return self.__getitem__(idx+1)
            
            if self.args.remove_multi:
                source = sample.get("source", "")
                print("source",source)
                if ("Multi" in source) or ("multi" in source):
                    return self.__getitem__(random.randint(self.__len__()))
            
            ori_smiles = remove_electron(ori_smiles)
            
            mol = None
            try:
                mol = Chem.MolFromSmiles(ori_smiles)
            except Exception as e:
                print(e)
            if mol is None:
                return self.__getitem__(idx+1)
            number_carbon = count_carbon_atoms(mol)

            # if len(ori_nmr)>number_carbon:
            #     print("too long ori_nmr")
            #     import ipdb
            #     ipdb.set_trace()
            #     return self.__getitem__(idx+1)

            if self.args.use_ring_system:
                if "ring_system" not in sample:
                    ring_system = get_ring_system(mol)
                else:
                    if self.phase == "train" and random.random()<self.args.smiles_aug_proba:
                        try:
                            ring_system = get_ring_system(mol, proba=1.0)
                        except Exception as e:
                            ring_system = sample["ring_system"]
                    else:
                        ring_system = sample["ring_system"]
                
                if len(ring_system)>0:
                    if random.random()<self.args.smiles_aug_proba and self.phase == "train":
                        for _ in range(len(ring_system)):
                            try:
                                ring_system[_] = get_random_smiles(Chem.MolFromSmiles(ring_system[_]))
                            except Exception as e:
                                pass
                        
                        random.shuffle(ring_system)
                    
                    random_ring_system = find_largest_smiles(ring_system)
                
                else:
                    random_ring_system = ""
                # SPrompt: 固定母核提示覆盖样本环系(仅门控样本会被拼入输入, 见 collate_fn)
                if getattr(self.args, "sprompt_core", "") and self.sprompt_votes is not None:
                    random_ring_system = self.args.sprompt_core
                # if "label" in sample:
                #     label = sample["label"]
                # else:
                #     mols = rdSubstructLibrary.CachedTrustedSmilesMolHolder()
                #     fps = rdSubstructLibrary.TautomerPatternHolder(1024)
                #     mols.AddSmiles(ori_smiles)
                #     fp = Chem.PatternFingerprint(mol,fpSize=1024,tautomerFingerprints=True)
                #     fps.AddFingerprint(fp)
                #     library = rdSubstructLibrary.SubstructLibrary(mols,fps)
                #     indices_1 = library.GetMatches(Chem.MolFromSmiles('O=C1C(C(O)=O)=CNC2=CC=CC=C12'),maxResults=50000)
                #     indices_2 = library.GetMatches(Chem.MolFromSmiles('O=C1C(C(O)=O)=CNC2=NC=CC=C12'),maxResults=50000)

                #     if len(indices_1)>0 or len(indices_2)>0:
                #         label = np.array([1])
                #     else:
                #         label = np.array([0])
                
                # label = int(label)
                
                # if label == 0:
                #     if "ring_system" not in sample:
                #         ring_system = get_ring_system(mol)
                #     else:
                #         if self.phase == "train" and random.random()<self.args.smiles_aug_proba:
                #             try:
                #                 ring_system = get_ring_system(mol, proba=1.0)
                #             except Exception as e:
                #                 ring_system = sample["ring_system"]
                #         else:
                #             ring_system = sample["ring_system"]
                    
                #     if len(ring_system)>0:
                #         if random.random()<self.args.smiles_aug_proba and self.phase == "train":
                #             for _ in range(len(ring_system)):
                #                 try:
                #                     ring_system[_] = get_random_smiles(Chem.MolFromSmiles(ring_system[_]))
                #                 except Exception as e:
                #                     pass
                            
                #             random.shuffle(ring_system)
                        
                #         random_ring_system = find_largest_smiles(ring_system)
                    
                #     else:
                #         random_ring_system = ""
                # else:
                #     if "extra_ring_system" not in sample:
                #         if "label" in sample:
                #             mols = rdSubstructLibrary.CachedTrustedSmilesMolHolder()
                #             fps = rdSubstructLibrary.TautomerPatternHolder(1024)
                #             mols.AddSmiles(ori_smiles)
                #             fp = Chem.PatternFingerprint(mol,fpSize=1024,tautomerFingerprints=True)
                #             fps.AddFingerprint(fp)
                #             library = rdSubstructLibrary.SubstructLibrary(mols,fps)
                #             indices_1 = library.GetMatches(Chem.MolFromSmiles('O=C1C(C(O)=O)=CNC2=CC=CC=C12'),maxResults=50000)
                #             indices_2 = library.GetMatches(Chem.MolFromSmiles('O=C1C(C(O)=O)=CNC2=NC=CC=C12'),maxResults=50000)

                #         if len(indices_1) > 0 and len(indices_2) >0 :
                #             if self.phase == "train" and random.random()<self.args.smiles_aug_proba:
                #                 random_ring_system = get_random_smiles(Chem.MolFromSmiles(random.choice(['O=C1C(C(O)=O)=CNC2=CC=CC=C12', 'O=C1C(C(O)=O)=CNC2=NC=CC=C12'])))
                #             else:
                #                 random_ring_system = Chem.MolToSmiles(Chem.MolFromSmiles(random.choice(['O=C1C(C(O)=O)=CNC2=CC=CC=C12', 'O=C1C(C(O)=O)=CNC2=NC=CC=C12'])))

                #         elif len(indices_1) > 0:
                #             if self.phase == "train" and random.random()<self.args.smiles_aug_proba:
                #                 random_ring_system = get_random_smiles(Chem.MolFromSmiles(random.choice(['O=C1C(C(O)=O)=CNC2=CC=CC=C12'])))
                #             else:
                #                 random_ring_system = Chem.MolToSmiles(Chem.MolFromSmiles(random.choice(['O=C1C(C(O)=O)=CNC2=CC=CC=C12'])))

                #         elif len(indices_2) > 0:
                #             if self.phase == "train" and random.random()<self.args.smiles_aug_proba:
                #                 random_ring_system = get_random_smiles(Chem.MolFromSmiles(random.choice(['O=C1C(C(O)=O)=CNC2=NC=CC=C12'])))
                #             else:
                #                 random_ring_system = Chem.MolToSmiles(Chem.MolFromSmiles(random.choice(['O=C1C(C(O)=O)=CNC2=NC=CC=C12'])))

                #     else:
                #         if self.phase == "train" and random.random()<self.args.smiles_aug_proba:
                #             random_ring_system = get_random_smiles(Chem.MolFromSmiles(random.choice(sample["extra_ring_system"])))
                #         else:
                #             random_ring_system = Chem.MolToSmiles(Chem.MolFromSmiles(random.choice(sample["extra_ring_system"])))

            temp_dict = {}
            if "source" in sample:
                temp_dict["source"] = sample["source"]
            else:
                temp_dict["source"] = ""
            # label: 母核标注 (list 如 [1]/[0]); oracle_ring 依据它决定是否给环系片段
            temp_dict["label"] = sample.get("label", [0])
            temp_dict["aligned_flag"] = True
            # SPrompt: 该样本是否注入母核提示(按数据集内位置取 XGB 票; None 表示未启用)
            temp_dict["sprompt_gate"] = bool(self.sprompt_votes[idx]) if self.sprompt_votes is not None else None
            if self.args.use_ring_system is False:
                if self.args.use_molecule_formular is False:
                    if self.args.use_clear_smiles is False:
                        if self.args.smiles_aug_proba == 0:
                            smiles_result = smiles2token(self.tokenizer, ori_smiles, phase=self.phase, block_size=self.args.max_length, proba=self.args.smiles_aug_proba)
                        else:
                            if random.random()<self.args.smiles_aug_proba and self.phase == "train":
                                if "smiles_list" in sample:
                                    temp_smiles_list = sample["smiles_list"]
                                    ori_smiles = random.choice(sample["smiles_list"])
                                else:
                                    mol = Chem.MolFromSmiles(ori_smiles)
                                    ori_smiles = get_random_smiles(mol)

                            smiles_result = smiles2token(self.tokenizer, ori_smiles, phase=self.phase, block_size=self.args.max_length, proba=0)
                    else:
                        smiles_result = smiles2token(self.tokenizer, random.choice(sample["ClearSMILES"][:20]), phase=self.phase, block_size=self.args.max_length, proba=0)

                    temp_dict["decoder_input_ids"] = smiles_result["input_ids"]
                    temp_dict["decoder_attention_mask"] = smiles_result["attention_mask"]
                    if self.args.nmr_aug_scale>0 and self.phase == "train" and random.random()>self.args.nmr_aug_proba:
                        nmr_result = self.ori_NmrDataset.augment_item(ori_nmr)
                    else:
                        nmr_result = self.ori_NmrDataset.fill_item(ori_nmr)
                    
                    temp_dict["input_ids"] = nmr_result["input_ids"]
                    temp_dict["attention_mask"] = nmr_result["attention_mask"]
                    temp_dict["smiles"] = ori_smiles
                else:
                    if self.args.use_clear_smiles is False:
                        if self.args.smiles_aug_proba == 0:
                            smiles_result = smiles2token(self.tokenizer, ori_smiles, phase=self.phase, block_size=self.args.max_length, proba=self.args.smiles_aug_proba)
                        else:
                            if random.random()<self.args.smiles_aug_proba and self.phase == "train":
                                if "smiles_list" in sample:
                                    temp_smiles_list = sample["smiles_list"]
                                    ori_smiles = random.choice(sample["smiles_list"])
                                else:
                                    mol = Chem.MolFromSmiles(ori_smiles)
                                    ori_smiles = get_random_smiles(mol)
                                
                            smiles_result = smiles2token(self.tokenizer, ori_smiles, phase=self.phase, block_size=self.args.max_length, proba=0)
                        temp_dict["smiles_result"] = smiles_result
                    else:
                        smiles_result = smiles2token(self.tokenizer, random.choice(sample["ClearSMILES"][:20]), phase=self.phase, block_size=self.args.max_length, proba=0)

                    temp_dict["smiles_result"] = smiles_result
                    if self.args.nmr_aug_scale>0 and self.phase == "train" and random.random()>self.args.nmr_aug_proba:
                        nmr_result = self.ori_NmrDataset.augment_item(ori_nmr)
                    else:
                        nmr_result = self.ori_NmrDataset.fill_item(ori_nmr)
                    
                    temp_dict["nmr_result"] = nmr_result
                    temp_dict["fromula_result"] = self.get_fromula_result(ori_smiles)
                    temp_dict["smiles"] = ori_smiles
            else:
                if self.args.force_aligned is False or random_ring_system=="":
                    if random.random()<self.args.smiles_aug_proba and self.phase != "train":
                        smiles_result = smiles2token(self.tokenizer, ori_smiles, phase=self.phase, block_size=self.args.max_length, proba=0)
                    else:
                        random_smiles = get_random_smiles(Chem.MolFromSmiles(ori_smiles))
                        smiles_result = smiles2token(self.tokenizer, ori_smiles, phase=self.phase, block_size=self.args.max_length, proba=0)

                    temp_dict["smiles_result"] = smiles_result

                    ring_system_result = smiles2token(self.tokenizer, random_ring_system, phase=self.phase, block_size=self.args.max_length, proba=0)
                    ring_system_result["input_ids"] = [183] + ring_system_result["input_ids"][1:-1] + [184]
                    temp_dict["ring_system_result"] = ring_system_result
                    temp_dict["ring_system"] = random_ring_system
                
                else:
                    if random.random()<0.5:
                        # 创建分子对象
                        mol = Chem.MolFromSmiles(ori_smiles)
                        core = None
                        try:
                            core = Chem.MolFromSmiles(random_ring_system)
                        except Exception as e:
                            core = Chem.MolFromSmiles(random_ring_system, sanitize=False)
                        
                        if core is None:
                            core = Chem.MolFromSmiles(random_ring_system, sanitize=False)
                        
                        if core is not None:
                            # 查找子结构匹配
                            try:
                                matches = mol.GetSubstructMatches(core)
                            except Exception as e:
                                print("len(matches)==0")
                                matches = []
                            if len(matches)>0:
                                match_idx_list = list(random.choice(matches))
                                # # 构建新的原子顺序：环系统原子在前，其他原子在后
                                # all_atoms = list(range(mol.GetNumAtoms()))
                                # other_atoms = [i for i in all_atoms if i not in match_idx_list]
                                # new_order = match_idx_list + other_atoms
                                
                                # # 重新排序原子并生成SMILES
                                # mol_renumbered = Chem.RenumberAtoms(mol, new_order)
                                aligned_smiles = align_mol_with_order(mol, match_idx_list)

                                smiles_result = smiles2token(self.tokenizer, aligned_smiles, phase=self.phase, block_size=self.args.max_length, proba=0)
                                temp_dict["smiles_result"] = smiles_result

                                ring_system_result = smiles2token(self.tokenizer, random_ring_system, phase=self.phase, block_size=self.args.max_length, proba=0)
                                ring_system_result["input_ids"] = [183] + ring_system_result["input_ids"][1:-1] + [184]
                                temp_dict["ring_system_result"] = ring_system_result
                                temp_dict["ring_system"] = random_ring_system
                            else:
                                ## 退化到使用mcs
                                mcs = FindMCS([core, mol])
                                if mcs.numAtoms==core.GetNumAtoms():
                                    pattern = Chem.MolFromSmarts(mcs.smartsString)
                                    matches_in_core = core.GetSubstructMatches(pattern)
                                    core_match_idx_list = list(random.choice(matches_in_core))
                                    aligned_ring_system = align_mol_with_order(core, core_match_idx_list)

                                    matches_in_mol = mol.GetSubstructMatches(pattern)
                                    match_idx_list = list(random.choice(matches_in_mol))

                                    aligned_smiles = align_mol_with_order(mol, match_idx_list)

                                    smiles_result = smiles2token(self.tokenizer, aligned_smiles, phase=self.phase, block_size=self.args.max_length, proba=0)

                                    temp_dict["smiles_result"] = smiles_result

                                    ring_system_result = smiles2token(self.tokenizer, aligned_ring_system, phase=self.phase, block_size=self.args.max_length, proba=0)
                                    ring_system_result["input_ids"] = [183] + ring_system_result["input_ids"][1:-1] + [184]
                                    temp_dict["ring_system_result"] = ring_system_result
                                    temp_dict["ring_system"] = aligned_ring_system

                                else:
                                    if random.random()<self.args.smiles_aug_proba and self.phase != "train":
                                        smiles_result = smiles2token(self.tokenizer, ori_smiles, phase=self.phase, block_size=self.args.max_length, proba=0)
                                    else:
                                        random_smiles = get_random_smiles(Chem.MolFromSmiles(ori_smiles))
                                        smiles_result = smiles2token(self.tokenizer, ori_smiles, phase=self.phase, block_size=self.args.max_length, proba=0)

                                    temp_dict["smiles_result"] = smiles_result

                                    ring_system_result = smiles2token(self.tokenizer, random_ring_system, phase=self.phase, block_size=self.args.max_length, proba=0)
                                    ring_system_result["input_ids"] = [183] + ring_system_result["input_ids"][1:-1] + [184]
                                    temp_dict["ring_system_result"] = ring_system_result
                                    temp_dict["ring_system"] = random_ring_system

                                    temp_dict["aligned_flag"] = False
                        
                        else:
                            if random.random()<self.args.smiles_aug_proba and self.phase != "train":
                                smiles_result = smiles2token(self.tokenizer, ori_smiles, phase=self.phase, block_size=self.args.max_length, proba=0)
                            else:
                                random_smiles = get_random_smiles(Chem.MolFromSmiles(ori_smiles))
                                smiles_result = smiles2token(self.tokenizer, ori_smiles, phase=self.phase, block_size=self.args.max_length, proba=0)

                            temp_dict["smiles_result"] = smiles_result

                            ring_system_result = smiles2token(self.tokenizer, random_ring_system, phase=self.phase, block_size=self.args.max_length, proba=0)
                            ring_system_result["input_ids"] = [183] + ring_system_result["input_ids"][1:-1] + [184]
                            temp_dict["ring_system_result"] = ring_system_result
                            temp_dict["ring_system"] = random_ring_system

                            temp_dict["aligned_flag"] = False

                    else:
                        if random.random()<self.args.smiles_aug_proba and self.phase != "train":
                            smiles_result = smiles2token(self.tokenizer, ori_smiles, phase=self.phase, block_size=self.args.max_length, proba=0)
                        else:
                            random_smiles = get_random_smiles(Chem.MolFromSmiles(ori_smiles))
                            smiles_result = smiles2token(self.tokenizer, ori_smiles, phase=self.phase, block_size=self.args.max_length, proba=0)

                        temp_dict["smiles_result"] = smiles_result

                        ring_system_result = smiles2token(self.tokenizer, random_ring_system, phase=self.phase, block_size=self.args.max_length, proba=0)
                        ring_system_result["input_ids"] = [183] + ring_system_result["input_ids"][1:-1] + [184]
                        temp_dict["ring_system_result"] = ring_system_result
                        temp_dict["ring_system"] = random_ring_system

                        temp_dict["aligned_flag"] = False

                # random_ring_system_result = smiles2token(self.tokenizer, random_ring_system, phase=self.phase, block_size=self.args.max_length, proba=0)
                # random_ring_system_result["input_ids"] = [183] + random_ring_system_result["input_ids"][1:-1] + [184]
                # temp_dict["random_ring_system_result"] = random_ring_system_result

                # temp_dict["decoder_input_ids"] = smiles_result["input_ids"]
                # temp_dict["decoder_attention_mask"] = smiles_result["attention_mask"]
                if self.args.nmr_aug_scale>0 and self.phase == "train" and random.random()>self.args.nmr_aug_proba:
                    nmr_result = self.ori_NmrDataset.augment_item(ori_nmr)
                else:
                    nmr_result = self.ori_NmrDataset.fill_item(ori_nmr)
                
                temp_dict["nmr_result"] = nmr_result
                temp_dict["smiles"] = ori_smiles

            return temp_dict

        except Exception as e:
            print(e)
            import traceback
            traceback.print_exc()
            return self.__getitem__(idx+1)

    def collate_fn(self, batch):
        if self.args.use_extra_encoder is False:
            if self.args.use_ring_system is False:
                if self.args.use_molecule_formular is False:
                    max_encoder_length = max([len(batch[_]["input_ids"]) for _ in range(len(batch))])
                    max_decoder_length = max([len(batch[_]["decoder_input_ids"]) for _ in range(len(batch))])
                    for _ in range(len(batch)):
                        if len(batch[_]["input_ids"]) < max_encoder_length:
                            batch[_]["attention_mask"] = batch[_]["attention_mask"] + [0]*(max_encoder_length-len(batch[_]["input_ids"]))
                            batch[_]["input_ids"] = batch[_]["input_ids"] + [self.tokenizer.pad_token_id]*(max_encoder_length-len(batch[_]["input_ids"]))
                        if len(batch[_]["decoder_input_ids"]) < max_decoder_length:
                            batch[_]["decoder_attention_mask"] = batch[_]["decoder_attention_mask"] + [0]*(max_decoder_length-len(batch[_]["decoder_input_ids"]))
                            batch[_]["decoder_input_ids"] = batch[_]["decoder_input_ids"] + [self.tokenizer.pad_token_id]*(max_decoder_length-len(batch[_]["decoder_input_ids"]))

                    input_ids = torch.tensor([batch[_]["input_ids"] for _ in range(len(batch))]).long()[:, :self.args.max_length]
                    attention_mask = torch.tensor([batch[_]["attention_mask"] for _ in range(len(batch))]).long()[:, :self.args.max_length]
                    decoder_input_ids = torch.tensor([batch[_]["decoder_input_ids"] for _ in range(len(batch))]).long()
                    labels = decoder_input_ids[:,1:][:, :self.args.max_length]
                    decoder_input_ids=decoder_input_ids[:,:-1][:, :self.args.max_length]
                    # decoder_attention_mask = torch.tensor([batch[_]["decoder_attention_mask"] for _ in range(len(batch))]).long()
                    smiles_list = [batch[_]["smiles"] for _ in range(len(batch))]
                    return {
                        "input_ids":input_ids,
                        "attention_mask":attention_mask,
                        "decoder_input_ids":decoder_input_ids,
                        "labels":labels,
                        "smiles_list":smiles_list,
                    }
                else:
                    input_ids_list = []
                    attention_mask_list = []
                    decoder_input_ids_list = []
                    smiles_list = []
                    for i in range(len(batch)):
                        temp = batch[i]
                        # nmr2smiles
                        input_ids_list.append(temp["nmr_result"]["input_ids"])
                        attention_mask_list.append(temp["nmr_result"]["attention_mask"])
                        decoder_input_ids_list.append(temp["smiles_result"]["input_ids"])
                        smiles_list.append(temp["smiles"])
                        if temp["ring_system_result"] != {}:
                            if self.phase == "train" and ("multi" not in source or "Multi" not in source):
                                # nmr+frag2smiles
                                input_ids_list.append(temp["nmr_result"]["input_ids"] + temp["ring_system_result"]["input_ids"])
                                attention_mask_list.append(temp["nmr_result"]["attention_mask"] + temp["ring_system_result"]["attention_mask"])
                                decoder_input_ids_list.append(temp["smiles_result"]["input_ids"])
                                smiles_list.append(temp["smiles"])

                                # nmr2frag
                                input_ids_list.append(temp["nmr_result"]["input_ids"])
                                attention_mask_list.append(temp["nmr_result"]["attention_mask"])
                                decoder_input_ids_list.append(temp["ring_system_result"]["input_ids"])
                                smiles_list.append(temp["smiles"])
                    
                    max_encoder_length = max([len(input_ids_list[_]) for _ in range(len(input_ids_list))])
                    max_decoder_length = max([len(decoder_input_ids_list[_]) for _ in range(len(decoder_input_ids_list))])
                    for _ in range(len(input_ids_list)):
                        if len(input_ids_list[_]) < max_encoder_length:
                            attention_mask_list[_] = attention_mask_list[_] + [0]*(max_encoder_length-len(attention_mask_list[_]))
                            input_ids_list[_] = input_ids_list[_] + [self.tokenizer.pad_token_id]*(max_encoder_length-len(input_ids_list[_]))
                        if len(decoder_input_ids_list[_]) < max_decoder_length:
                            # batch[_]["decoder_attention_mask"] = batch[_]["decoder_attention_mask"] + [0]*(max_decoder_length-len(batch[_]["decoder_input_ids"]))
                            decoder_input_ids_list[_] = decoder_input_ids_list[_] + [self.tokenizer.pad_token_id]*(max_decoder_length-len(decoder_input_ids_list[_]))
                    
                    input_ids = torch.tensor(input_ids_list).long()[:, :self.args.max_length]
                    attention_mask = torch.tensor(attention_mask_list).long()[:, :self.args.max_length]
                    decoder_input_ids = torch.tensor(decoder_input_ids_list).long()
                    labels = decoder_input_ids[:,1:][:, :self.args.max_length]
                    decoder_input_ids=decoder_input_ids[:,:-1][:, :self.args.max_length]
                    # decoder_attention_mask = torch.tensor([batch[_]["decoder_attention_mask"] for _ in range(len(batch))]).long()
                    return {
                        "input_ids":input_ids,
                        "attention_mask":attention_mask,
                        "decoder_input_ids":decoder_input_ids,
                        "labels":labels,
                        "smiles_list":smiles_list,
                    }
            else:
                input_ids_list = []
                attention_mask_list = []
                decoder_input_ids_list = []
                smiles_list = []
                if self.phase == "train":
                    for i in range(len(batch)):
                        temp = batch[i]
                        # nmr2smiles
                        input_ids_list.append(temp["nmr_result"]["input_ids"])
                        attention_mask_list.append(temp["nmr_result"]["attention_mask"])
                        decoder_input_ids_list.append(temp["smiles_result"]["input_ids"])
                        smiles_list.append(temp["smiles"])
                        source = temp.get("source", "")
                        if temp["ring_system_result"] != {}:
                            ## 堆mextnova的数据不再生成
                            if self.phase == "train" and ("multi" not in source or "Multi" not in source):
                                # nmr+frag2smiles
                                if temp["aligned_flag"] is True:
                                    input_ids_list.append(temp["nmr_result"]["input_ids"] + temp["ring_system_result"]["input_ids"])
                                    attention_mask_list.append(temp["nmr_result"]["attention_mask"] + temp["ring_system_result"]["attention_mask"])
                                    decoder_input_ids_list.append(temp["smiles_result"]["input_ids"])
                                    smiles_list.append(temp["ring_system"])

                                # nmr2frag
                                input_ids_list.append(temp["nmr_result"]["input_ids"] )
                                attention_mask_list.append(temp["nmr_result"]["attention_mask"])
                                decoder_input_ids_list.append(temp["ring_system_result"]["input_ids"])
                                smiles_list.append(temp["smiles"])
                
                else:
                    for i in range(len(batch)):
                        temp = batch[i]
                        source = temp.get("source", "")
                        # nmr2smiles
                        if self.args.decode_mode == "smiles":
                            input_ids_list.append(temp["nmr_result"]["input_ids"])
                            attention_mask_list.append(temp["nmr_result"]["attention_mask"])
                            decoder_input_ids_list.append(temp["smiles_result"]["input_ids"])
                            smiles_list.append(temp["smiles"])
                        elif self.args.decode_mode == "frag":
                            # nmr2frag
                            input_ids_list.append(temp["nmr_result"]["input_ids"] )
                            attention_mask_list.append(temp["nmr_result"]["attention_mask"])
                            decoder_input_ids_list.append(temp["ring_system_result"]["input_ids"])
                            smiles_list.append(temp["smiles"])
                        else:
                            # nmr+frag; oracle_ring: label==1 附真实环系片段; sprompt: XGB票==True 附固定母核;
                            # 其余不给片段(退化为 smiles 输入)
                            _label = temp.get("label", [0])
                            if isinstance(_label, str):
                                _label = _label.strip("[]")
                            elif isinstance(_label, (list, tuple)):
                                _label = _label[0] if len(_label) > 0 else 0
                            try:
                                _label = int(_label)
                            except Exception:
                                _label = 0
                            _gate = temp.get("sprompt_gate", None)
                            _use_frag = not (self.args.oracle_ring and _label != 1)
                            if _gate is not None and not _gate:
                                _use_frag = False
                            if _use_frag:
                                input_ids_list.append(temp["nmr_result"]["input_ids"] + temp["ring_system_result"]["input_ids"])
                                attention_mask_list.append(temp["nmr_result"]["attention_mask"] + temp["ring_system_result"]["attention_mask"])
                            else:
                                input_ids_list.append(temp["nmr_result"]["input_ids"])
                                attention_mask_list.append(temp["nmr_result"]["attention_mask"])
                            decoder_input_ids_list.append(temp["smiles_result"]["input_ids"])
                            # oracle/sprompt 评测参考统一用完整分子(参考=环系时大量分子共享同一环系, 收尾去重会合并样本)
                            smiles_list.append(temp["smiles"] if (self.args.oracle_ring or _gate is not None) else temp["ring_system"])

                max_encoder_length = max([len(input_ids_list[_]) for _ in range(len(input_ids_list))])
                max_decoder_length = max([len(decoder_input_ids_list[_]) for _ in range(len(decoder_input_ids_list))])
                for _ in range(len(input_ids_list)):
                    if len(input_ids_list[_]) < max_encoder_length:
                        attention_mask_list[_] = attention_mask_list[_] + [0]*(max_encoder_length-len(attention_mask_list[_]))
                        input_ids_list[_] = input_ids_list[_] + [self.tokenizer.pad_token_id]*(max_encoder_length-len(input_ids_list[_]))
                    if len(decoder_input_ids_list[_]) < max_decoder_length:
                        # batch[_]["decoder_attention_mask"] = batch[_]["decoder_attention_mask"] + [0]*(max_decoder_length-len(batch[_]["decoder_input_ids"]))
                        decoder_input_ids_list[_] = decoder_input_ids_list[_] + [self.tokenizer.pad_token_id]*(max_decoder_length-len(decoder_input_ids_list[_]))
                
                input_ids = torch.tensor(input_ids_list).long()[:, :self.args.max_length]
                attention_mask = torch.tensor(attention_mask_list).long()[:, :self.args.max_length]
                decoder_input_ids = torch.tensor(decoder_input_ids_list).long()
                labels = decoder_input_ids[:,1:][:, :self.args.max_length]
                decoder_input_ids=decoder_input_ids[:,:-1][:, :self.args.max_length]
                # decoder_attention_mask = torch.tensor([batch[_]["decoder_attention_mask"] for _ in range(len(batch))]).long()
                return {
                    "input_ids":input_ids,
                    "attention_mask":attention_mask,
                    "decoder_input_ids":decoder_input_ids,
                    "labels":labels,
                    "smiles_list":smiles_list,
                }
        else:
            if self.args.use_ring_system is False:
                pass
            else:
                input_ids_list = []
                attention_mask_list = []
                extra_input_ids_list = []
                extra_attention_mask_list = []
                input_attention_mask_list = []
                decoder_input_ids_list = []
                smiles_list = []
                for i in range(len(batch)):
                    temp = batch[i]
                    ## nmr2smiles
                    input_ids_list.append(temp["nmr_result"]["input_ids"])
                    attention_mask_list.append(temp["nmr_result"]["attention_mask"])
                    extra_input_ids_list.append([183, 184])
                    extra_attention_mask_list.append([1, 1])
                    decoder_input_ids_list.append(temp["smiles_result"]["input_ids"])
                    smiles_list.append(temp["smiles"])
                    source = temp.get("source", "")
                    if temp["ring_system_result"] != {}:
                        ## 堆mextnova的数据不再生成
                        if self.phase == "train" and ("multi" not in source or "Multi" not in source):
                            ## nmr2frag
                            input_ids_list.append(temp["nmr_result"]["input_ids"])
                            attention_mask_list.append(temp["nmr_result"]["attention_mask"])
                            decoder_input_ids_list.append(temp["ring_system_result"]["input_ids"])
                            extra_input_ids_list.append([183, 184])
                            extra_attention_mask_list.append([1, 1])
                            smiles_list.append(temp["ring_system"])
                            
                            ## nmr+frag2smiles
                            input_ids_list.append(temp["nmr_result"]["input_ids"])
                            attention_mask_list.append(temp["nmr_result"]["attention_mask"])
                            extra_input_ids_list.append(temp["ring_system_result"]["input_ids"])
                            extra_attention_mask_list.append(temp["ring_system_result"]["attention_mask"])
                            decoder_input_ids_list.append(temp["smiles_result"]["input_ids"])
                            smiles_list.append(temp["smiles"])
                
                max_encoder_length = max([len(input_ids_list[_]) for _ in range(len(input_ids_list))])
                max_extra_encoder_length = max([len(extra_input_ids_list[_]) for _ in range(len(extra_input_ids_list))])
                max_decoder_length = max([len(decoder_input_ids_list[_]) for _ in range(len(decoder_input_ids_list))])
                for _ in range(len(input_ids_list)):
                    if len(input_ids_list[_]) < max_encoder_length:
                        attention_mask_list[_] = attention_mask_list[_] + [0]*(max_encoder_length-len(attention_mask_list[_]))
                        input_ids_list[_] = input_ids_list[_] + [self.tokenizer.pad_token_id]*(max_encoder_length-len(input_ids_list[_]))
                    
                    if len(extra_input_ids_list[_]) < max_extra_encoder_length:
                        extra_attention_mask_list[_] = extra_attention_mask_list[_] + [0]*(max_extra_encoder_length-len(extra_attention_mask_list[_]))
                        extra_input_ids_list[_] = extra_input_ids_list[_] + [self.tokenizer.pad_token_id]*(max_extra_encoder_length-len(extra_input_ids_list[_]))
                    
                    if len(decoder_input_ids_list[_]) < max_decoder_length:
                        # batch[_]["decoder_attention_mask"] = batch[_]["decoder_attention_mask"] + [0]*(max_decoder_length-len(batch[_]["decoder_input_ids"]))
                        decoder_input_ids_list[_] = decoder_input_ids_list[_] + [self.tokenizer.pad_token_id]*(max_decoder_length-len(decoder_input_ids_list[_]))
                
                input_ids = torch.tensor(input_ids_list).long()[:, :self.args.max_length]
                attention_mask = torch.tensor(attention_mask_list).long()[:, :self.args.max_length]
                extra_input_ids = torch.tensor(extra_input_ids_list).long()[:, :self.args.max_length]
                extra_attention_mask = torch.tensor(extra_attention_mask_list).long()[:, :self.args.max_length]
                decoder_input_ids = torch.tensor(decoder_input_ids_list).long()
                labels = decoder_input_ids[:,1:][:, :self.args.max_length]
                decoder_input_ids=decoder_input_ids[:,:-1][:, :self.args.max_length]
                # decoder_attention_mask = torch.tensor([batch[_]["decoder_attention_mask"] for _ in range(len(batch))]).long()
                return {
                    "input_ids":input_ids,
                    "extra_input_ids":extra_input_ids,
                    "extra_attention_mask":extra_attention_mask,
                    "attention_mask":attention_mask,
                    "decoder_input_ids":decoder_input_ids,
                    "labels":labels,
                    "smiles_list":smiles_list,
                }

class CustomDataModule(pl.LightningDataModule):
    def __init__(self, args, tokenizer=None):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer

        # 直接在__init__中创建数据集
        if args.train_file:
            self.train_dataset = CustomDataset(self.args, self.tokenizer, file_path=args.train_file, phase="train")
        if args.valid_file:
            self.val_dataset = CustomDataset(self.args, self.tokenizer, file_path=args.valid_file, phase="val")
        if args.test_file:
            self.test_dataset = CustomDataset(self.args, self.tokenizer, file_path=args.test_file, phase="test")

    def prepare_data(self):
        # 数据已在__init__中准备，这里可以为空
        pass

    def print_stats(self):
        ## 打印数据集
        print(f"Train dataset: {len(self.train_dataset)}")
        print(f"Valid dataset: {len(self.val_dataset)}")
        print(f"Test dataset: {len(self.test_dataset)}")

    def setup(self, stage: str = None):  # 必须包含 stage 参数
        pass
        # if stage == "fit" or stage is None:
        #     # 加载训练/验证数据
        #     self.train_dataset = ...
        #     self.val_dataset = ...
        # if stage == "test" or stage is None:
        #     # 加载测试数据
        #     self.test_dataset = ...

    def train_dataloader(self):
        if self.args.debug is False:
            ## 瓶颈不在数据
            # return torch.utils.data.DataLoader(
            #     self.dataset["train"], batch_size=self.args.batch_size, num_workers=self.args.num_workers,
            #     collate_fn=DataCollatorForSeq2Seq(self.tokenizer), prefetch_factor=2, persistent_workers=True, pin_memory=True,
            #     shuffle=True)
            return torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                collate_fn=self.train_dataset.collate_fn,
                shuffle=True,
                prefetch_factor=2, persistent_workers=True, pin_memory=True,
            )
        else:
            ## 单线程调试
            return torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=self.args.batch_size,
                num_workers=0,
                collate_fn=self.train_dataset.collate_fn,
                shuffle=True,
            )

    def val_dataloader(self):
        if self.args.debug is False:
            return torch.utils.data.DataLoader(
                self.val_dataset,
                batch_size=self.args.batch_size * 2,
                num_workers=self.args.num_workers,
                collate_fn=self.val_dataset.collate_fn,
            )
        else:
            ## 单线程调试
            return torch.utils.data.DataLoader(
                self.val_dataset,
                batch_size=self.args.batch_size * 2,
                num_workers=0,
                collate_fn=self.val_dataset.collate_fn,
            )

    def test_dataloader(self):
        if self.args.debug is False:
            return torch.utils.data.DataLoader(
                self.test_dataset,
                batch_size=self.args.batch_size * 2,
                num_workers=self.args.num_workers,
                collate_fn=self.test_dataset.collate_fn,
            )
        else:
            ## 单线程调试
            return torch.utils.data.DataLoader(
                self.test_dataset,
                batch_size=self.args.batch_size * 2,
                num_workers=0,
                collate_fn=self.test_dataset.collate_fn,
            )

def get_model(model_dir="./checkpoints/saved_hf_model_0910", use_lora=False, device = "cpu"):
    # flash-attn v2 可用时用 flash_attention_2, 否则退回 PyTorch 原生 sdpa (均支持训练/生成, 速度接近)
    attn_implementation = "flash_attention_2" if FLASH_ATTN_AVAILABLE else "sdpa"
    if attn_implementation != "flash_attention_2":
        print("⚠️ flash_attn 不可用, 改用 attn_implementation='sdpa'")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, attn_implementation=attn_implementation)
    model.to(device)
    return model

def get_tokenizer(model_dir="./checkpoints/saved_hf_model_0910"):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return tokenizer



class FlashCrossAttention(nn.Module):
    """基于FlashAttention官方CrossAttention实现的交叉注意力层，适配我们的架构需求"""
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0, norm_type: str = "pre"):
        super().__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout
        self.norm_type = norm_type  # "pre" or "post"

        # 投影层
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)

        # 层归一化 - 根据架构类型配置
        if self.norm_type == "pre":
            # Pre-norm架构：输入和输出都有归一化
            self.layer_norm_q = nn.LayerNorm(d_model)
            self.layer_norm_k = nn.LayerNorm(d_model)
            self.layer_norm_v = nn.LayerNorm(d_model)
            self.layer_norm_out = nn.LayerNorm(d_model)
        else:
            # Post-norm架构：只有输出归一化
            self.layer_norm_out = nn.LayerNorm(d_model)

        # 使用FlashAttention官方的CrossAttention实现
        self.flash_attn = FlashAttentionCrossAttention(
            causal=False,  # 交叉注意力不是因果的
            softmax_scale=None,  # 使用默认的1/sqrt(d_keys)
            attention_dropout=dropout
        )

        # Dropout层
        self.dropout_layer = nn.Dropout(dropout)

        # 初始化
        self._reset_parameters()

    def _reset_parameters(self):
        """初始化投影层的权重"""
        # 使用更小的初始化方差以提高稳定性
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1.0 / (self.num_heads ** 0.25))
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1.0 / (self.num_heads ** 0.25))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1.0)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        输入形状: [batch_size, seq_len, d_model]
        key_padding_mask: 形状 [batch_size, key_seq_len]，True/1表示需要被mask掉的padding位置。
        """
        batch_size, query_len, _ = query.shape
        key_len = key.size(1)

        # 保存残差连接
        residual = query

        if self.norm_type == "pre":
            # Pre-norm架构：先进行层归一化
            q = self.layer_norm_q(query)
            k = self.layer_norm_k(key)
            v = self.layer_norm_v(value)
        else:
            # Post-norm架构：不进行输入归一化
            q = query
            k = key
            v = value

        # 线性投影
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # 重排为多头形式 [batch, seq_len, d_model] -> [batch, seq_len, num_heads, head_dim]
        q = q.view(batch_size, query_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, key_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, key_len, self.num_heads, self.head_dim)

        # 准备KV张量给FlashAttention CrossAttention
        # FlashAttention期望的KV格式: [batch, key_len, 2, num_heads, head_dim]
        kv = torch.stack([k, v], dim=2)

        # 处理padding mask - FlashAttention期望True表示保留，False表示mask
        if key_padding_mask is not None:
            # 确保mask是bool类型
            if key_padding_mask.dtype != torch.bool:
                key_padding_mask = key_padding_mask > 0
            # 反转mask，因为FlashAttention的语义不同
            # 我们的mask: True表示padding，False表示有效
            # FlashAttention: True表示保留，False表示mask
            key_padding_mask = ~key_padding_mask

        try:
            # 使用FlashAttention官方实现
            attn_output = self.flash_attn(
                q=q,
                kv=kv,
                causal=False,
                key_padding_mask=key_padding_mask
            )
            # FlashAttention CrossAttention的输出已经是 [batch, query_len, num_heads, head_dim] 格式
            # 不需要额外的transpose操作
        except Exception as e:
            print(f"Error in FlashAttention CrossAttention: {e}")
            print(f"Falling back to standard attention computation")
            # 回退到标准注意力机制
            attn_output = self._fallback_attention(q, k, v, key_padding_mask)

        # FlashAttention的输出已经是正确的格式，直接进行输出投影
        # 确保输出维度正确：[batch, query_len, num_heads * head_dim] -> [batch, query_len, d_model]
        if attn_output.size(-1) != self.d_model:
            # 如果输出维度不匹配，可能需要reshape
            attn_output = attn_output.contiguous()
            if len(attn_output.shape) == 4:  # [batch, query_len, num_heads, head_dim]
                attn_output = attn_output.view(batch_size, query_len, self.d_model)

        # 输出投影
        attn_output = self.out_proj(attn_output)

        # 应用dropout
        attn_output = self.dropout_layer(attn_output)

        # 残差连接 + 层归一化
        if self.norm_type == "pre":
            output = self.layer_norm_out(residual + attn_output)
        else:
            output = residual + attn_output
            output = self.layer_norm_out(output)

        return output

    def _fallback_attention(self, q, k, v, key_padding_mask):
        """回退到标准注意力机制"""
        batch_size, query_len, num_heads, head_dim = q.shape
        key_len = k.shape[1]

        # 重排为标准注意力格式 [batch, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 计算注意力分数
        scale = 1.0 / math.sqrt(head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # 处理padding mask
        if key_padding_mask is not None:
            # 扩展mask到注意力分数的维度
            mask = key_padding_mask.unsqueeze(1).unsqueeze(1)  # [batch, 1, 1, key_len]
            mask = mask.expand(batch_size, num_heads, query_len, key_len)
            attn_scores = attn_scores.masked_fill(~mask, float('-inf'))

        # 计算注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1, dtype=v.dtype)
        attn_weights = self.dropout_layer(attn_weights)

        # 应用注意力权重
        attn_output = torch.matmul(attn_weights, v)

        # 合并多头输出 [batch, seq_len, num_heads, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output
        
class BartWithDoubleFlashCrossAttention(BartForConditionalGeneration):
    def __init__(self, config: BartConfig):
        super().__init__(config)

        self.config = config

        # 第一层交叉注意力（编码器内部信息融合）
        self.cross_attn_layer1 = FlashCrossAttention(
            d_model=config.d_model,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            norm_type="pre"  # 使用Pre-norm以获得更好的训练稳定性
        )

        # 第二层交叉注意力（外部信息融合）
        self.cross_attn_layer2 = FlashCrossAttention(
            d_model=config.d_model,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            norm_type="pre"  # 使用Pre-norm以获得更好的训练稳定性
        )
        
        # 外部信息投影层
        self.extra_projection = nn.Linear(config.d_model, config.d_model)
        
        # 中间增强层
        self.enhance_layer = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 4),
            nn.GELU(),
            nn.Linear(config.d_model * 4, config.d_model),
            nn.Dropout(config.dropout)
        )
    
        # 初始化权重
        self._reset_parameters()

        self.model = get_model()

    def _reset_parameters(self):
        """初始化新增组件的权重"""
        # 外部信息投影层
        nn.init.xavier_uniform_(self.extra_projection.weight)
        if self.extra_projection.bias is not None:
            nn.init.constant_(self.extra_projection.bias, 0.)

        # 中间增强层
        for layer in [self.enhance_layer[0], self.enhance_layer[2]]:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        extra_input_ids: Optional[torch.LongTensor] = None,
        extra_attention_mask: Optional[torch.Tensor] = None,
        extra_encoder_outputs: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            # if use_cache:
            #     print("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        if encoder_outputs is None:
            encoder_outputs = self.model.model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        
        encoder_hidden_states = encoder_outputs[0] # [batch, seq_len, d_model]
        
        if extra_encoder_outputs is None:
            extra_encoder_outputs = self.model.model.encoder(
                input_ids=extra_input_ids,
                attention_mask=extra_attention_mask,
            )
        
        extra_hidden_states = self.extra_projection(extra_encoder_outputs[0])  # [batch, seq_len, d_model]

        fused_output1 = self.cross_attn_layer1(
            query=encoder_hidden_states,
            key=extra_hidden_states,
            value=extra_hidden_states,
            key_padding_mask=extra_attention_mask
        )
        enhanced_output = self.enhance_layer(fused_output1)

        final_output = self.cross_attn_layer2(
            query=enhanced_output,
            key=extra_hidden_states,
            value=extra_hidden_states,
            key_padding_mask=extra_attention_mask
        )

        outputs = self.model.model.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=final_output,
            encoder_attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        lm_logits = self.lm_head(outputs[0])
        lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)

        masked_lm_loss = None
        if labels is not None:
            labels = labels.to(lm_logits.device)
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.hidden_states,
            decoder_attentions=outputs.attentions,
            encoder_last_hidden_state=final_output,
        )

    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        extra_input_ids: Optional[torch.LongTensor] = None,
        extra_attention_mask: Optional[torch.Tensor] = None,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        do_sample: bool = False,
        early_stopping: bool = False,
        num_beams: int = 1,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        length_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        encoder_no_repeat_ngram_size: int = 0,
        bad_words_ids: Optional[List[List[int]]] = None,
        num_return_sequences: int = 1,
        decoder_start_token_id: Optional[int] = None,
        use_cache: bool = True,
        output_scores: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict_in_generate: bool = False,
        **kwargs,
    ):
        """
        生成方法，支持双交叉注意力机制
        
        参数说明：
        - input_ids: 主输入序列
        - attention_mask: 主输入的注意力掩码
        - extra_input_ids: 额外输入序列（用于第二交叉注意力）
        - extra_attention_mask: 额外输入的注意力掩码
        - 其他参数与标准BART generate方法一致
        """
        # 确保至少有一个输入
        if input_ids is None and extra_input_ids is None:
            raise ValueError("必须提供至少input_ids或extra_input_ids中的一个")

        # 处理编码器输出
        encoder_outputs = None
        if input_ids is not None:
            encoder_outputs = self.model.model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            encoder_hidden_states = encoder_outputs[0]
        else:
            # 如果没有主输入，使用额外输入作为主输入
            encoder_hidden_states = torch.zeros(
                (extra_input_ids.size(0), 0, self.config.d_model),
                device=extra_input_ids.device
            )

        # 处理额外输入
        extra_hidden_states = None
        if extra_input_ids is not None:
            extra_encoder_outputs = self.model.model.encoder(
                input_ids=extra_input_ids,
                attention_mask=extra_attention_mask,
            )
            extra_hidden_states = self.extra_projection(extra_encoder_outputs[0])
        else:
            # 如果没有额外输入，使用主输入作为额外输入
            extra_hidden_states = encoder_hidden_states
            extra_attention_mask = attention_mask

        # 第一层交叉注意力
        fused_output1 = self.cross_attn_layer1(
            query=encoder_hidden_states,
            key=extra_hidden_states,
            value=extra_hidden_states,
            key_padding_mask=extra_attention_mask
        )

        # 中间增强
        enhanced_output = self.enhance_layer(fused_output1)

        # 第二层交叉注意力
        final_output = self.cross_attn_layer2(
            query=enhanced_output,
            key=extra_hidden_states,
            value=extra_hidden_states,
            key_padding_mask=extra_attention_mask
        )

        # 准备生成参数
        generation_config = self._prepare_generation_config(
            max_length=max_length,
            min_length=min_length,
            do_sample=do_sample,
            early_stopping=early_stopping,
            num_beams=num_beams,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
            bad_words_ids=bad_words_ids,
            num_return_sequences=num_return_sequences,
            decoder_start_token_id=decoder_start_token_id,
        )

        # 调用父类的generate方法
        return super().generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_outputs=(final_output,),
            use_cache=use_cache,
            output_scores=output_scores,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict_in_generate=return_dict_in_generate,
            **generation_config.to_dict(),
            **kwargs
        )

    def _prepare_generation_config(self, **kwargs):
        """准备生成配置"""
        from transformers import GenerationConfig
        
        # 获取默认配置
        config = GenerationConfig.from_model_config(self.config)
        
        # 更新用户提供的参数
        for key, value in kwargs.items():
            if value is not None and hasattr(config, key):
                setattr(config, key, value)
        
        return config

class CustomModel(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        self.args = args
        if self.args.use_extra_encoder is False:
            self.model = get_model(self.args.model_dir)
            self.freeze_base_epochs = 0
        else:
            temp_config = BartConfig.from_pretrained(self.args.model_dir)
            self.model = BartWithDoubleFlashCrossAttention(temp_config)
            self._freeze_base_model()
            self.freeze_base_epochs = 0
        

        print(print_model_size(self.model))
        self.tokenizer = get_tokenizer(self.args.model_dir)
        self.val_step_outputs = []
        self.val_loss = []
        self.test_step_outputs = []
        self.test_loss = []
    
    def _freeze_base_model(self):
        """冻结基础BART模型的参数"""
        for name, param in self.model.named_parameters():
            if not any(layer in name for layer in [
                'cross_attn_layer1', 'cross_attn_layer2', 
                'extra_projection', 'enhance_layer'
            ]):
                param.requires_grad = False
        print("✅ 基础模型参数已冻结，只训练新增模块")

    def _unfreeze_all(self):
        """解冻所有参数"""
        for param in self.model.parameters():
            param.requires_grad = True
        print("✅ 所有参数已解冻，进行联合训练")
    
    def on_train_epoch_start(self):
        """在每个训练epoch开始时检查是否需要解冻"""
        current_epoch = self.current_epoch
        if current_epoch == self.freeze_base_epochs:
            self._unfreeze_all()
        
    def training_step(self, batch, batch_idx):
        true_smiles = batch.pop("smiles_list")
        outputs = self.model(**batch)

        # 处理不同的输出格式
        if hasattr(outputs, 'loss'):
            loss = outputs.loss
        else:
            # 如果没有loss属性，需要手动计算（对于自定义模型）
            # 这里假设我们有一个labels在batch中，但看起来没有，需要添加
            labels = batch.get("labels")
            if labels is not None:
                # 计算loss - 这里需要根据实际的模型架构来调整
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            else:
                loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        self.log('train/loss', loss.item(), prog_bar=True)

        # 安全地记录学习率
        try:
            scheduler = self.lr_schedulers()
            if scheduler is not None:
                if hasattr(scheduler, 'get_lr'):
                    current_lr = scheduler.get_lr()[0]
                elif hasattr(scheduler, 'get_last_lr'):
                    current_lr = scheduler.get_last_lr()[0]
                else:
                    current_lr = self.args.lr
                self.log('lr', current_lr, prog_bar=True, logger=False)
            else:
                self.log('lr', self.args.lr, prog_bar=True, logger=False)
        except Exception as e:
            self.log('lr', self.args.lr, prog_bar=True, logger=False)
        return loss

    def validation_step(self, batch, batch_idx):
        true_smiles = batch.pop("smiles_list")
        with torch.no_grad():
            outputs = self.model(**batch)

            # 处理不同的输出格式
            if hasattr(outputs, 'loss'):
                loss = outputs.loss
            else:
                # 如果没有loss属性，需要手动计算
                labels = batch.get("labels")
                if labels is not None:
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
                else:
                    loss = torch.tensor(0.0, device=self.device)

            self.log('val/loss', loss.item(), prog_bar=True, rank_zero_only=True)

        #     result = self.model.generate(
        #             input_ids=batch["input_ids"],
        #             max_length=self.args.max_length,
        #             num_beams=1,
        #             num_return_sequences=1,
        #             bos_token_id=187,
        #             pad_token_id=1,
        #             eos_token_id=188,
        #             decoder_start_token_id=187
        #         )
        # pred_smiles = [self.tokenizer.decode(i) for i in result]
        # pred_smiles = [i.replace("<SMILES>", "").replace("</SMILES>", "").replace("<pad>", "").replace("</s>", "") for i in pred_smiles]
        # self.val_step_outputs.extend(list(zip(true_smiles, pred_smiles)))
        self.val_loss.append(loss.item())
        return loss
    
    def on_validation_epoch_end(self):
        if self.trainer.num_devices > 1:
            # gathered_outputs = [None for i in range(self.trainer.num_devices)]
            # dist.all_gather_object(gathered_outputs, self.val_step_outputs)
            # gathered_outputs = sum(gathered_outputs, [])
            gathered_losses =  [None for i in range(self.trainer.num_devices)]
            dist.all_gather_object(gathered_losses, self.val_loss)
            gathered_losses = sum(gathered_losses, [])
        else:
            # gathered_outputs = self.val_step_outputs
            gathered_losses = self.val_loss
        
        scores = [-np.mean(gathered_losses)]
        # scores = [0]
        
        # if self.trainer.is_global_zero:
        #     count = 0
        #     acc = 0
        #     smi = 0
        #     for (true_smiles, pred_smiles) in gathered_outputs:
        #         true_mol = None
        #         pred_mol = None
        #         try:
        #             true_mol = Chem.MolFromSmiles(true_smiles)
        #         except:
        #             pass
                    
        #         try:
        #             pred_mol = Chem.MolFromSmiles(pred_smiles)
        #         except:
        #             pass
                
        #         if (true_mol is None) or (pred_mol is None):
        #             pass
        #         else:
        #             if Chem.MolToSmiles(true_mol) == Chem.MolToSmiles(pred_mol):
        #                 acc = acc + 1
                    
        #             fp_true = AllChem.GetMorganFingerprintAsBitVect(true_mol, radius=2, nBits=2048, useChirality=True)
        #             fp_pred = AllChem.GetMorganFingerprintAsBitVect(pred_mol, radius=2, nBits=2048, useChirality=True)
        #             similarity = DataStructs.TanimotoSimilarity(fp_true, fp_pred)
        #             smi = smi + similarity
                
        #         count = count + 1
            
        #     with open(os.path.join(self.trainer.default_root_dir, f'val.json'), 'w') as f:
        #         json.dump(gathered_outputs, f)
            
        #     scores = [smi/count]

        dist.broadcast_object_list(scores)
        self.log('val/score', scores[0], prog_bar=True, rank_zero_only=True)

        self.val_step_outputs.clear()
        print("val epoch end end")

    def test_step(self, batch, batch_idx):
        true_smiles = batch.pop("smiles_list")
        with torch.no_grad():
            outputs = self.model(**batch)
            loss = outputs.loss
            self.log('test/loss', loss.item(), prog_bar=True, rank_zero_only=True)

            # Generate sequences with scores for Top-N analysis

            num_return = min(100, self.args.num_beams)  # Ensure we have at least 100 predictions for Top-N analysis

            if self.args.decode_mode in ["smiles", "nmr+frag"]:
                result = self.model.generate(
                    input_ids=batch["input_ids"],
                    max_length=self.args.max_length,
                    num_beams=self.args.num_beams,
                    num_return_sequences=num_return,
                    bos_token_id=187,
                    pad_token_id=1,
                    eos_token_id=188,
                    decoder_start_token_id=187,
                    output_scores=True,
                    return_dict_in_generate=True
                )
            else:
                result = self.model.generate(
                    input_ids=batch["input_ids"],
                    max_length=self.args.max_length,
                    num_beams=self.args.num_beams,
                    num_return_sequences=num_return,
                    bos_token_id=183,
                    pad_token_id=1,
                    eos_token_id=184,
                    decoder_start_token_id=183,
                    output_scores=True,
                    return_dict_in_generate=True
                )

            # Fix 1: Use 'sequences' instead of accessing sequences_scores
            generated_sequences = result.sequences
            generated_scores = result.scores

            # Convert to CPU and extract scores
            pred_sequences = [self.tokenizer.decode(seq, skip_special_tokens=True) for seq in generated_sequences]
            pred_sequences = [seq.replace("<fragment>","").replace("</fragment>","").replace("<SMILES>", "").replace("</SMILES>", "").replace("<pad>", "").replace("</s>", "").strip() for seq in pred_sequences]
            
            # Convert scores to numpy and handle different formats
            if isinstance(generated_scores, tuple):
                final_scores = generated_scores[-1].cpu().numpy() if generated_scores else np.zeros(len(generated_sequences))
            else:
                final_scores = generated_scores.cpu().numpy()

            # Simple score processing
            if len(final_scores.shape) > 1 and final_scores.shape[1] > 0:
                final_scores = final_scores[0, :len(generated_sequences)]

            # Group by input and store detailed results
            batch_size = len(true_smiles)
            sequences_per_input = len(pred_sequences) // batch_size if batch_size > 0 else len(pred_sequences)

            for i, true_smile in enumerate(true_smiles):
                start_idx = i * sequences_per_input
                end_idx = start_idx + sequences_per_input

                pred_smiles = pred_sequences[start_idx:end_idx]
                pred_scores = final_scores[start_idx:end_idx].tolist()

                # 分析参考分子
                true_mol_analysis = analyze_molecule(true_smile)
                canonical_true_smiles = ""
                if true_mol_analysis["is_valid"]:
                    try:
                        canonical_true_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(true_smile), canonical=True)
                    except:
                        canonical_true_smiles = true_smile
                else:
                    canonical_true_smiles = true_smile

                # 分析每个预测分子
                detailed_predictions = []
                for j, (pred_smile, pred_score) in enumerate(zip(pred_smiles, pred_scores)):
                    pred_analysis = analyze_molecule(pred_smile)

                    # 检查是否与参考分子完全匹配
                    is_exact_match = False
                    if pred_analysis["is_valid"] and true_mol_analysis["is_valid"]:
                        try:
                            canonical_pred_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(pred_smile), canonical=True)
                            is_exact_match = (canonical_pred_smiles == canonical_true_smiles)
                        except:
                            is_exact_match = (pred_smile == true_smile)

                    detailed_predictions.append({
                        "rank": j + 1,
                        "smiles": pred_smile,
                        "score": pred_score,
                        "is_valid": pred_analysis["is_valid"],
                        "is_exact_match": is_exact_match,
                        "has_core_structure": pred_analysis["has_core_structure"]
                    })

                # 保存详细结果
                detailed_result = {
                    "reference_smiles": true_smile,
                    "canonical_reference_smiles": canonical_true_smiles,
                    "reference_analysis": {
                        "is_valid": true_mol_analysis["is_valid"],
                        "has_core_structure": true_mol_analysis["has_core_structure"]
                    },
                    "predictions": detailed_predictions,
                    "summary": {
                        "total_predictions": len(detailed_predictions),
                        "valid_predictions": sum(1 for p in detailed_predictions if p["is_valid"]),
                        "exact_matches": sum(1 for p in detailed_predictions if p["is_exact_match"]),
                        "core_structure_predictions": sum(1 for p in detailed_predictions if p["has_core_structure"])
                    }
                }

                self.test_step_outputs.append(detailed_result)

        self.test_loss.append(loss.item())
        return loss

    def on_test_epoch_end(self):
        if self.trainer.num_devices > 1:
            # 各 rank 先把自己的结果落盘, NCCL 只汇总"文件路径"(几十字节)。
            # 原实现直接 all_gather_object 全部逐样本预测(全量beams10≈GB级), 2026-10-05
            # 的评测因此卡进 30 分钟 NCCL watchdog 超时并丢掉 103 分钟的全部结果;
            # 落盘的 shard 文件同时是保险: 后续合并/指标计算再出问题也能离线复算。
            shard_path = os.path.join(self.args.save_path, f'_test_shard_rank{self.global_rank}.pkl')
            with open(shard_path, 'wb') as f:
                pickle.dump(self.test_step_outputs, f)
            gathered_rank_shards = [None for i in range(self.trainer.num_devices)]
            dist.all_gather_object(gathered_rank_shards, shard_path)

            gathered_outputs = []
            if self.trainer.is_global_zero:
                for shard in gathered_rank_shards:
                    if shard and os.path.exists(shard):
                        with open(shard, 'rb') as f:
                            gathered_outputs.extend(pickle.load(f))

            # Apply deduplication for distributed processing (only on rank 0)
            if self.trainer.is_global_zero:
                print(f"🔄 Distributed processing detected ({self.trainer.num_devices} devices)")
                print(f"📊 Original sample count: {len(gathered_outputs)}")

                # Use dictionary to store unique results based on reference SMILES
                unique_results = {}
                duplicate_count = 0

                for output in gathered_outputs:
                    if isinstance(output, dict) and "reference_smiles" in output:
                        true_smiles = output["reference_smiles"]
                        if true_smiles not in unique_results:
                            unique_results[true_smiles] = output
                        else:
                            duplicate_count += 1
                            # Compare and keep the better result based on number of predictions
                            existing_output = unique_results[true_smiles]
                            if (len(output.get("predictions", [])) > len(existing_output.get("predictions", []))):
                                unique_results[true_smiles] = output

                # Convert back to list
                gathered_outputs = list(unique_results.values())

                print(f"✅ Deduplication complete: {len(gathered_outputs)} unique samples")
                print(f"🗑️  Removed {duplicate_count} duplicates")
        else:
            gathered_outputs = self.test_step_outputs

        scores = [0]

        if self.trainer.is_global_zero:

            # Calculate Top-N metrics from detailed analysis results
            # Define comprehensive Top-N values up to num_beams
            all_top_k_values = [1, 3, 5, 10, 20, 30, 50, 100]
            top_k_values = [k for k in all_top_k_values if k <= self.args.num_beams]  # Only calculate up to num_beams

            # Initialize metric containers
            top_accuracy_lists = {f'top_{k}': [] for k in top_k_values}
            top_core_accuracy_lists = {f'top_{k}': [] for k in top_k_values}
            top_validity_lists = {f'top_{k}': [] for k in top_k_values}

            # Process each sample using the detailed analysis
            print(f"🔍 Processing {len(gathered_outputs)} samples...")

            # Debug: Check data format
            dict_count = sum(1 for item in gathered_outputs if isinstance(item, dict))
            list_count = sum(1 for item in gathered_outputs if isinstance(item, list))
            print(f"📊 Data format: {dict_count} dict items, {list_count} list items")

            valid_sample_count = 0

            for i, detailed_result in enumerate(gathered_outputs):
                if not isinstance(detailed_result, dict):
                    if i < 3:  # Debug first few skipped items
                        print(f"  ❌ Skipping item {i}: type={type(detailed_result)}")
                    continue

                true_smiles = detailed_result.get("reference_smiles", "")
                predictions = detailed_result.get("predictions", [])
                reference_analysis = detailed_result.get("reference_analysis", {})

                # Debug only first few samples
                if i < 3:
                    print(f"  Sample {i}: true='{true_smiles[:20]}...', preds={len(predictions)}")

                # Check if reference is valid
                if not reference_analysis.get("is_valid", False):
                    print(f"    ❌ Invalid reference SMILES")
                    continue

                valid_sample_count += 1

                # Calculate metrics for each Top-N level
                for k in top_k_values:
                    if k <= len(predictions):
                        top_k_predictions = predictions[:k]

                        # 1. Top-N Accuracy: exact match
                        exact_matches = [p["is_exact_match"] for p in top_k_predictions if p.get("is_valid", False)]
                        top_accuracy_lists[f'top_{k}'].append(1 if any(exact_matches) else 0)

                        # 2. Top-N Core Accuracy: 对齐参考分子和预测分子的核心结构
                        reference_has_core = reference_analysis.get("has_core_structure", False)

                        if reference_has_core:
                            # 参考分子有母核：预测分子中至少有一个有母核则为True
                            core_predictions = [p for p in top_k_predictions if p.get("has_core_structure", False)]
                            top_core_accuracy_lists[f'top_{k}'].append(1 if len(core_predictions) > 0 else 0)
                        else:
                            # 参考分子没有母核：预测分子中至少有一个没有母核则为True
                            non_core_predictions = [p for p in top_k_predictions if not p.get("has_core_structure", False)]
                            top_core_accuracy_lists[f'top_{k}'].append(1 if len(non_core_predictions) > 0 else 0)

                        # 3. Top-N Validity: valid SMILES
                        valid_predictions = [p for p in top_k_predictions if p.get("is_valid", False)]
                        top_validity_lists[f'top_{k}'].append(1 if len(valid_predictions) > 0 else 0)

            # Calculate and display results
            print(f"\n📈 Final Results based on {valid_sample_count} valid samples:")
            print(f"{'='*70}")
            print(f"{'Top-N':<8} | {'Accuracy':<10} | {'Core Acc':<10} | {'Validity':<10}")
            print(f"{'='*70}")

            for k in top_k_values:
                if len(top_accuracy_lists[f'top_{k}']) > 0:
                    accuracy = np.mean(top_accuracy_lists[f'top_{k}'])
                    core_accuracy = np.mean(top_core_accuracy_lists[f'top_{k}'])
                    validity = np.mean(top_validity_lists[f'top_{k}'])
                    print(f"Top-{k:<6} | {accuracy:<10.4f} | {core_accuracy:<10.4f} | {validity:<10.4f}")

            print(f"{'='*70}")

            # Calculate similarity scores using Morgan fingerprints
            try:
                from rdkit.Chem import AllChem, DataStructs

                similarity_scores = []
                for detailed_result in gathered_outputs[:10]:  # Limit to first 10 samples for efficiency
                    if not isinstance(detailed_result, dict):
                        continue

                    true_smiles = detailed_result.get("reference_smiles", "")
                    predictions = detailed_result.get("predictions", [])

                    if not detailed_result.get("reference_analysis", {}).get("is_valid", False):
                        continue

                    try:
                        true_mol = Chem.MolFromSmiles(true_smiles)
                        if true_mol is None:
                            continue

                        # Calculate similarity for top-3 predictions
                        for pred in predictions[:3]:
                            if pred.get("is_valid", False):
                                pred_mol = Chem.MolFromSmiles(pred["smiles"])
                                if pred_mol is not None:
                                    fp_true = AllChem.GetMorganFingerprintAsBitVect(true_mol, radius=2, nBits=2048, useChirality=True)
                                    fp_pred = AllChem.GetMorganFingerprintAsBitVect(pred_mol, radius=2, nBits=2048, useChirality=True)
                                    similarity = DataStructs.TanimotoSimilarity(fp_true, fp_pred)
                                    similarity_scores.append(similarity)
                    except:
                        continue

                avg_similarity = np.mean(similarity_scores) if similarity_scores else 0.0
                print(f"Similarity Score: {avg_similarity:.4f}")
            except ImportError:
                print("Similarity Score: N/A (RDKit not available)")

            # Save only new format (detailed) results to JSON
            detailed_outputs = [item for item in gathered_outputs if isinstance(item, dict) and "reference_smiles" in item]
            test_set_size = len(detailed_outputs)
            test_filename = f'test_{self.args.decode_mode}_beams{self.args.num_beams}_samples{test_set_size}.json'
            test_path = Path(f'{self.args.save_path}/{test_filename}')
            test_path.parent.mkdir(parents=True, exist_ok=True)

            print(f"\n💾 Saving detailed results: {len(detailed_outputs)} samples")
            with open(test_path, 'w') as f:
                json.dump(detailed_outputs, f, indent=2, ensure_ascii=False)

            print(f"\n✅ Results saved to: {test_path}")
            print(f"📊 Total samples processed: {len(gathered_outputs)}")
            print(f"💾 Detailed results saved: {len(detailed_outputs)}")
            print(f"✅ Valid samples analyzed: {valid_sample_count}")

        # Print loss
        if self.test_loss:
            avg_loss = sum(self.test_loss) / len(self.test_loss)
            print(f"\n📊 Test Loss: {avg_loss:.4f}")

    def configure_optimizers(self):
        num_training_steps = self.trainer.num_training_steps
        self.print(f'Num training steps: {num_training_steps}')
        num_warmup_steps = int(num_training_steps * self.args.warmup_ratio)
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        scheduler = get_scheduler(self.args.scheduler, optimizer, num_warmup_steps, num_training_steps)
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}

    def on_train_epoch_end(self):
        # 训练结束时打印epoch信息
        print(f"Epoch {self.current_epoch} completed")

        # 如果需要，可以在这里保存额外的信息
        if hasattr(self, 'train_loss') and len(self.train_loss) > 0:
            avg_train_loss = sum(self.train_loss[-100:]) / min(len(self.train_loss), 100)  # 最近100个step的平均loss
            print(f"Recent average training loss: {avg_train_loss:.4f}")


class ModelCheckpoint(pl.callbacks.ModelCheckpoint):
    def _get_metric_interpolated_filepath_name(self, monitor_candidates, trainer, del_filepath=None) -> str:
        # 重写以支持更复杂的文件名格式
        return super()._get_metric_interpolated_filepath_name(monitor_candidates, trainer, del_filepath)


class SaveEveryEpoch(pl.callbacks.Callback):
    """每次 validation 结束存一份当前权重 (epoch_NN.ckpt, weights-only).
    save_last 在部分 run 里只在头 1-2 个 val 后停止更新, 导致训满权重不落盘;
    且 Lightning 限制同 trainer 至多 2 个 ModelCheckpoint, 故用普通 Callback 实现."""

    def __init__(self, dirpath):
        super().__init__()
        self.dirpath = dirpath
        os.makedirs(dirpath, exist_ok=True)

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking or not trainer.is_global_zero:
            return
        trainer.save_checkpoint(
            os.path.join(self.dirpath, f"epoch_{trainer.current_epoch:02d}.ckpt"),
            weights_only=True,
        )


def main():
    args = get_args()
    ## 让 --seed 真正生效 (此前该参数只定义未使用);
    ## workers=True 同时固定 DataLoader worker 的 RNG, 保证同 seed 可复现
    pl.seed_everything(args.seed, workers=True)

    tokenizer = get_tokenizer(args.model_dir)

    # 创建数据集
    dm = CustomDataModule(args, tokenizer)
    dm.prepare_data()
    dm.print_stats()

    try:
        import tokenizers
        import transformers
        from transformers import BartTokenizerFast
        ## torch2.6 引入的新机制
        ## ref:https://stackoverflow.com/questions/79584485/unable-to-torch-load-due-to-pickling-safety-error
        ## 注: 新版 transformers 已移除旧模块路径 tokenization_bart_fast(该 import 会失败,
        ##     导致整个白名单注册被 except 吞掉), 而老 ckpt 里 pickle 的是旧路径;
        ##     用 (对象, 旧路径) 别名注册才能通过默认 weights_only=True 的加载
        torch.serialization.add_safe_globals(
            [
                (BartTokenizerFast, "transformers.models.bart.tokenization_bart_fast.BartTokenizerFast"),
                tokenizers.Tokenizer,
                tokenizers.models.Model,
                tokenizers.AddedToken
            ]
        )
    except Exception as e:
        print(e)
        

    MODEL = CustomModel
    if args.do_train:
        model = MODEL(args)
        if args.ckpt is not None and os.path.exists(args.ckpt):
            model = MODEL.load_from_checkpoint(args.ckpt, strict=False, args=args)
    else:
        if args.ckpt is not None and os.path.exists(args.ckpt):
            model = MODEL.load_from_checkpoint(args.ckpt, strict=False, args=args)
            print("load ckpt")
        else:
            if os.path.exists(os.path.join(args.save_path, 'checkpoints/last.ckpt')):
                model = MODEL.load_from_checkpoint(os.path.join(args.save_path, 'checkpoints/last.ckpt'), strict=False, args=args)
                print("load successfully")
            else:
                model = MODEL(args)
                print("use default model")

    # 1. 确保所有目录存在
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(os.path.join(args.save_path, "step_checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, "best_models"), exist_ok=True)

    # 2. 配置检查点
    best_checkpoint = ModelCheckpoint(
        dirpath=os.path.join(args.save_path, "best_models"),
        monitor='val/score',
        mode='max',
        save_top_k=1,
        filename='best-{epoch:02d}-{step:06d}-{val/score:.4f}',
        save_last=True,
    )

    step_checkpoint = ModelCheckpoint(
        dirpath=os.path.join(args.save_path, "step_checkpoints"),
        filename='step-{epoch:02d}-{step:06d}-{train_loss:.4f}',  # 添加指标占位符
        monitor='train/loss',  # 添加监控指标
        mode='min',
        every_n_train_steps=50000,
        save_top_k=10,  # 保存train_loss最好的10个模型
        save_weights_only=True,
    )

    # 每 epoch 存档一次 (见 SaveEveryEpoch 说明)
    epoch_saver = SaveEveryEpoch(os.path.join(args.save_path, "checkpoints"))

    # 3. 其他配置
    lr_monitor = LearningRateMonitor(logging_interval='step')
    logger = pl.loggers.TensorBoardLogger(args.save_path, name='', version='')

    # 4. 训练器
    trainer = pl.Trainer(
        strategy=DDPStrategy(find_unused_parameters=False),
        accelerator='gpu',
        precision="bf16",
        devices=args.gpus,
        logger=logger,
        default_root_dir=args.save_path,
        callbacks=[best_checkpoint, step_checkpoint, epoch_saver, lr_monitor],
        max_epochs=args.epochs,
        gradient_clip_val=args.max_grad_norm,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        check_val_every_n_epoch=args.eval_per_epoch,
        log_every_n_steps=10,
        deterministic='warn'
    )

    if args.do_train:
        trainer.num_training_steps = math.ceil(
            len(dm.train_dataset) / (args.batch_size * args.gpus * args.gradient_accumulation_steps)) * args.epochs
        model.eval_dataset = dm.val_dataset
        ckpt_path = os.path.join(args.save_path, 'checkpoints/last.ckpt') if args.resume else None
        trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)
        model = MODEL.load_from_checkpoint(best_checkpoint.best_model_path, args=args)

    if args.do_valid:
        model.eval_dataset = dm.val_dataset
        trainer.validate(model, datamodule=dm)

    if args.do_test:
        model.eval_dataset = dm.test_dataset
        trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    main()
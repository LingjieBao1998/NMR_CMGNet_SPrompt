import os
import math
import json
import random
import argparse
import numpy as np

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
import lmdb
import pickle
import re

from chemicalgof import (
    Reduce2GoF,
    fragSMILES2GoF,
    Sequence2GoF,
    GoF2Mol,
    GoF2fragSMILES,
    encode,
    decode,
    split
)

# 屏蔽 RDKit 的警告（包括 DEPRECATION WARNING）
RDLogger.DisableLog('rdApp.warning')  # 屏蔽 RDKit 自己的 Logger 警告
warnings.filterwarnings("ignore", category=DeprecationWarning)  # 屏蔽 Python 的 DeprecationWarning

import lmdb
import pickle
import copy

import useful_rdkit_utils as uru
from rdkit.Chem import rdmolops
from rdkit.Chem import rdSubstructLibrary

def get_atom_count(mol):
    """计算分子中的原子数量"""
    if mol is None:
        return 0
    return mol.GetNumAtoms()

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


# def find_largest_smiles(ring_systems):
#     """
#     找到原子数量最多的SMILES字符串
#     如果多个分子具有相同的最大原子数，则根据非碳非氢原子数进行加权采样
#     """
#     max_atoms = 0
#     candidates = []  # 存储候选分子信息
    
#     for smiles in ring_systems:
#         mol = Chem.MolFromSmiles(smiles, sanitize=False)
#         if mol is None:
#             continue  # 跳过无效的SMILES
            
#         num_atoms = get_atom_count(mol)
        
#         if num_atoms > max_atoms:
#             # 发现更大的原子数，重置候选列表
#             max_atoms = num_atoms
#             candidates = [{
#                 "smiles": smiles,
#                 "nonCH_count": get_nonCH_atom_count(mol)
#             }]
            
#         elif num_atoms == max_atoms:
#             # 原子数相同，添加到候选列表
#             candidates.append({
#                 "smiles": smiles,
#                 "nonCH_count": get_nonCH_atom_count(mol)
#             })
    
#     # 如果没有候选分子
#     if not candidates:
#         return None
    
#     # 如果只有一个候选分子
#     if len(candidates) == 1:
#         return candidates[0]["smiles"]
    
#     # 多个候选分子时，根据非碳非氢原子数进行加权采样
#     weights = [c["nonCH_count"] + 1 for c in candidates]  # +1 避免权重为0
#     selected = random.choices(candidates, weights=weights, k=1)[0]
#     return selected["smiles"]

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
    优先考虑有芳香性的环，如果多个环都有芳香性，则按照(非CH原子个数 + 原子数量)作为权重进行采样
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
        return None
    
    # 如果只有一个候选分子
    if len(candidates) == 1:
        return candidates[0]["smiles"]
    
    # 第二步：优先考虑有芳香性的分子
    aromatic_candidates = [c for c in candidates if c["is_aromatic"]]
    non_aromatic_candidates = [c for c in candidates if not c["is_aromatic"]]
    
    # 如果有芳香性分子，只在芳香性分子中采样
    if aromatic_candidates:
        candidates_to_sample = aromatic_candidates
        # print(f"在 {len(aromatic_candidates)} 个芳香性分子中进行加权采样")
    else:
        candidates_to_sample = non_aromatic_candidates
        # print(f"在 {len(non_aromatic_candidates)} 个非芳香性分子中进行加权采样")
    
    # 第三步：计算权重 (非CH原子个数 + 原子数量)
    weights = []
    for candidate in candidates_to_sample:
        # 权重 = 非CH原子个数 + 原子数量
        weight = candidate["nonCH_count"] + candidate["atom_count"] + 1  # +1 避免权重为0
        
        weights.append(weight)
        
        # print(f"分子: {candidate['smiles']}, "
        #       f"原子数: {candidate['atom_count']}, "
        #       f"非CH原子: {candidate['nonCH_count']}, "
        #       f"芳香环数: {candidate['aromatic_count']}, "
        #       f"权重: {weight}")
    
    # 第四步：加权采样
    selected = random.choices(candidates_to_sample, weights=weights, k=1)[0]
    
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
    ring_system_finder = uru.RingSystemFinder()
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

def get_new_smiles(old_mol):
    old_mol = copy.deepcopy(old_mol)
    len_mol = old_mol.GetNumAtoms()
    index_list = list(range(len_mol))
    random.shuffle(index_list)
    li = []
    for index in random.sample(list(range(len_mol)),2):
        try:
            new_smiles = Chem.MolToSmiles(old_mol, rootedAtAtom=index)
            li.append(new_smiles)
        except:
            pass
        if len(li) == 2:
            break
    if len(li) >= 1:
        random.shuffle(li)
        return li[:10]
    else:
        return None

def get_random_smiles(old_mol):
    old_mol = copy.deepcopy(old_mol)
    len_mol = old_mol.GetNumAtoms()
    index_list = list(range(len_mol))
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

def count_parameters(model):
    params = model.parameters()
    num_params = sum(reduce(mul, p.size(), 1) for p in params)
    return num_params

def print_model_size(model):
    num_params = count_parameters(model)
    num_params_m = num_params / 1e6
    print(f"Model size: {num_params_m:.2f}M parameters")

def extract_angle_bracket_content(text):
    """
    提取文本中所有<...>格式的内容（包括尖括号），并替换为<pos_N>格式

    参数:
        text (str): 输入文本
    
    返回:
        tuple: (匹配到的<...>内容列表, 替换后的文本)
    """
    # 处理 <数字R/S> 格式
    replaced_text = re.sub(
        r'<(\d+)([RS])>',  # 匹配 <数字R/S>，分别捕获数字和手性标记
        lambda m: f'<<pos_{m.group(1)}><chirality_{m.group(2)}>>',  # 嵌套替换
        text
    )

    # 处理 <数字TET_CCW> 格式
    text = re.sub(
        r'<(\d+)TET_CCW>',
        lambda m: f'<<pos_{m.group(1)}><chirality_CCW>>',
        text
    )
    
    # 处理 <数字TET_CW> 格式
    text = re.sub(
        r'<(\d+)TET_CW>',
        lambda m: f'<<pos_{m.group(1)}><chirality_CW>>',
        text
    )

    # 处理纯数字格式 <数字>
    replaced_text = re.sub(
        r'<(\d+)>',  # 匹配 <数字R/S>，分别捕获数字和手性标记
        lambda m: f'<<pos_{m.group(1)}>>',  # 嵌套替换
        replaced_text
    )
    return replaced_text

def restore_angle_bracket_content(text):
    """
    将<pos_N>和<chirality_X>格式的内容恢复为原始<...>格式

    参数:
        text (str): 替换后的文本
    
    返回:
        str: 恢复原始格式的文本
    """
    # 处理嵌套格式 <<pos_N><chirality_X>>
    text = re.sub(
        r'<<pos_(\d+)><chirality_([RS])>>',
        lambda m: f'<{m.group(1)}{m.group(2)}>',
        text
    )
    
    text = re.sub(
        r'<<pos_(\d+)><chirality_(CCW)>>',
        lambda m: f'<{m.group(1)}TET_CCW>',
        text
    )
    
    text = re.sub(
        r'<<pos_(\d+)><chirality_(CW)>>',
        lambda m: f'<{m.group(1)}TET_CW>',
        text
    )

    # 处理纯数字格式 <<pos_N>>
    text = re.sub(
        r'<<pos_(\d+)>>',
        lambda m: f'<{m.group(1)}>',
        text
    )
    
    return text

def restore_stereo_chemistry(processed_smiles):
    """
    将处理后的立体化学标记恢复为原始SMILES格式
    
    参数:
        processed_smiles (str): 经过parse_stereo_chemistry处理的SMILES字符串
        
    返回:
        str: 恢复原始格式的SMILES字符串
    """
    if "|" not in processed_smiles:
        return processed_smiles
    
    # 分割SMILES字符串
    left_part, right_part = processed_smiles.split("|", 1)
    
    # 处理立体化学标记
    # 1. 恢复 <pos_N><chirality_X> 格式
    right_part = re.sub(
        r'<pos_(\d+)><chirality_([RS])>',
        lambda m: f'{m.group(1)}{m.group(2)}',
        right_part
    )
    
    # 2. 恢复 <pos_N><chirality_CCW> 格式
    right_part = re.sub(
        r'<pos_(\d+)><chirality_CCW>',
        lambda m: f'{m.group(1)}TET_CCW',
        right_part
    )
    
    # 3. 恢复 <pos_N><chirality_CW> 格式
    right_part = re.sub(
        r'<pos_(\d+)><chirality_CW>',
        lambda m: f'{m.group(1)}TET_CW',
        right_part
    )
    
    # 恢复原始的|R/|S标记
    if right_part.startswith("0R"):
        right_part = right_part.replace("0R", "R", 1)
    elif right_part.startswith("0S"):
        right_part = right_part.replace("0S", "S", 1)
    
    return f"{left_part}|{right_part}"

def parse_stereo_chemistry(smiles):
    """
    解析SMILES字符串中的立体化学标记
    
    参数:
        smiles (str): 包含立体化学标记的SMILES字符串
        
    返回:
        tuple: (拆分后的标记列表, 去除标记后的SMILES字符串)
    """
    # 替换
    if "|" in smiles:
        smiles = smiles.replace("|R", "|0R").replace("|S", "|0S")
        right_index = smiles.index("|")
        replaced_text = smiles[right_index+1:]

        replaced_text = re.sub(
            r'(\d+)([RS])',  # 匹配 <数字R/S>，分别捕获数字和手性标记
            lambda m: f'<pos_{m.group(1)}><chirality_{m.group(2)}>',  # 嵌套替换
            replaced_text
        )

        # 处理 <数字TET_CCW> 格式
        replaced_text = re.sub(
            r'(\d+)TET_CCW',
            lambda m: f'<pos_{m.group(1)}><chirality_CCW>',
            replaced_text
        )
        
        # 处理 <数字TET_CW> 格式
        replaced_text = re.sub(
            r'(\d+)TET_CW',
            lambda m: f'<pos_{m.group(1)}><chirality_CW>',
            replaced_text
        )

        return smiles[:right_index+1] + replaced_text
    
    else:
        return smiles


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
    parser.add_argument('--model_dir', default="./checkpoints/saved_hf_model_1015",type=str)

    # Training
    parser.add_argument('--use_lora', action='store_true')
    parser.add_argument('--use_molecule_formular', action='store_true')
    parser.add_argument('--use_clear_smiles', action='store_true') # 证明了效果确实不行
    parser.add_argument('--use_ring_system', action='store_true')
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
    parser.add_argument('--train_file', type=str, default=None)
    parser.add_argument('--valid_file', type=str, default=None)
    parser.add_argument('--test_file', type=str, default=None)
    # Inference
    parser.add_argument('--ckpt', type=str, default=None)
    
    ## data_augment
    parser.add_argument('--smiles_aug_proba', type=float, default=0.0)
    parser.add_argument('--nmr_aug_proba', type=float, default=0.0)
    parser.add_argument('--nmr_aug_scale', type=float, default=1.5)

    parser.add_argument('--lmdb_path', type=str, default="")
    parser.add_argument('--num_beams', type=int, default=1)

    args = parser.parse_args([]) if notebook else parser.parse_args()

    args.lmdb_path = args.lmdb_path.split(",")

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

class CustomDataset(Dataset):
    def __init__(self, args, tokenizer, file_path="", phase="valid"):
        """
        Args:
            data_list (list or array): 样本数据，比如特征列表 [样本1, 样本2, ...]
            labels (list or tensor, optional): 对应的标签，如 [0, 1, 0, ...]。可以为 None（如测试集）
        """
        self.args = args
        self.tokenizer = tokenizer
        self.phase = phase
        self.data = []
        if isinstance(file_path, str):
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    self.data = json.loads(f.read())
        elif isinstance(file_path, list):
            self.data = []
            for _ in file_path:
                if os.path.exists(_):
                    with open(_, "r") as f:
                        self.data_ = json.loads(f.read())
                        self.data.extend(self.data_)
        
        self.env_list = []
        self.key_list = []
        self.env_length_idx = []
        if isinstance(args.lmdb_path, str):
            if os.path.exists(args.lmdb_path):
                env = lmdb.open(
                        args.lmdb_path,
                        readonly=True,
                        lock=False,
                        readahead=False,
                        meminit=False,
                        max_readers=256,
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
        
        if self.args.debug:
            self.data = self.data[:1000]
        
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
        
        self.ori_NmrDataset = NmrDataset(augment_range=self.args.nmr_aug_scale)

        # self.data = [self.data[i] for i in range(len(self.data)) if self.data[i]["source"]!= "NMR BANK"]

    def __len__(self):
        return len(self.data) + len(self.key_list) # 数据集大小
    
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
    
    def get_frag_result(self, frag=""):
        temp_dict = {}
        output_list = []
        if frag != "":
            for _, temp_frag in enumerate(frag):
                if "<" in temp_frag:
                    replaced_frag = extract_angle_bracket_content(temp_frag)
                    output_list.extend(self.tokenizer(replaced_frag)["input_ids"][1:-1])
                elif "|" in temp_frag:
                    replaced_frag = parse_stereo_chemistry(temp_frag)
                    output_list.extend(self.tokenizer(replaced_frag)["input_ids"][1:-1])
                else:
                    output_list.extend(self.tokenizer(temp_frag)["input_ids"][1:-1])
                
                ## 添加分割符号
                if _ < len(frag) - 1:
                    output_list.extend(self.tokenizer("<sep>")["input_ids"][1:-1])
        
        temp_dict["input_ids"] = [4191] + output_list + [4192]
        temp_dict["attention_mask"] = [1 for _ in temp_dict["input_ids"]]
        return temp_dict

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
            
            mol = None
            try:
                mol = Chem.MolFromSmiles(ori_smiles)
            except Exception as e:
                print(e)
            if mol is None:
                return self.__getitem__(idx+1)
            number_carbon = count_carbon_atoms(mol)

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
                
                if self.phase == "train" and random.random()>self.args.nmr_aug_proba:
                    nmr_result = self.ori_NmrDataset.augment_item(ori_nmr)
                else:
                    nmr_result = self.ori_NmrDataset.fill_item(ori_nmr)

                temp_dict = {}
                if "source" in sample:
                    temp_dict["source"] = sample["source"]
                else:
                    temp_dict["source"] = ""
                temp_dict["smiles"] = ori_smiles
                temp_dict["nmr_result"] = nmr_result
                

                if self.phase == "train":
                    if random.random()>self.args.smiles_aug_proba:
                        if "frag_list" in sample and len(sample["frag_list"])>0:
                            frag = random.choice(sample["frag_list"])
                        else:
                            try:
                                frag = split(encode(ori_smiles, random=True))
                            except Exception as e:
                                print("random smiles is invalid")
                                frag = split(encode(ori_smiles))
                    else:
                        if "frag" in sample:
                            frag = sample["frag"]
                        else:
                            frag = split(encode(ori_smiles))
                else:
                    if "frag" in sample:
                        frag = sample["frag"]
                    else:
                        frag = split(encode(ori_smiles))
                
                frag_result = self.get_frag_result(frag)
                temp_dict["smiles_result"] = frag_result

                # if random_ring_system == "":
                #     random_ring_system_frag_result = self.get_frag_result(random_ring_system)
                # else:
                #     import ipdb
                #     ipdb.set_trace()
                #     if self.phase == "train" and random.random()>self.args.smiles_aug_proba:
                #         try:
                #             random_ring_system_frag = split(encode(random_ring_system, random=True))
                #         except Exception as e:
                #             print("random smiles is invalid")
                #             random_ring_system_frag = split(encode(random_ring_system))
                #     else:
                #         random_ring_system_frag = split(encode(random_ring_system))
                    
                #     random_ring_system_frag_result = self.get_frag_result(random_ring_system_frag)

                random_ring_system_frag_result = self.tokenizer(random_ring_system)
                    
                random_ring_system_frag_result["input_ids"] = [183] + random_ring_system_frag_result["input_ids"][1:-1] + [184]
                temp_dict["ring_system_result"] = random_ring_system_frag_result
                temp_dict["ring_system"] = random_ring_system
                
            else:
                if self.phase == "train" and random.random()>self.args.nmr_aug_proba:
                    nmr_result = self.ori_NmrDataset.augment_item(ori_nmr)
                else:
                    nmr_result = self.ori_NmrDataset.fill_item(ori_nmr)
                
                temp_dict = {}
                if "source" in sample:
                    temp_dict["source"] = sample["source"]
                else:
                    temp_dict["source"] = ""
                temp_dict["smiles"] = ori_smiles
                temp_dict["input_ids"] = nmr_result["input_ids"]
                temp_dict["attention_mask"] = nmr_result["attention_mask"]

                if self.phase == "train":
                    if random.random()>self.args.smiles_aug_proba:
                        if "frag_list" in sample and len(sample["frag_list"])>0:
                            frag = random.choice(sample["frag_list"])
                        else:
                            try:
                                frag = split(encode(get_random_smiles(Chem.MolFromSmiles(ori_smiles))))
                            except Exception as e:
                                print("random smiles is invalid")
                                frag = split(encode(ori_smiles))
                    else:
                        if "frag" in sample:
                            frag = sample["frag"]
                        else:
                            frag = split(encode(ori_smiles))
                else:
                    if "frag" in sample:
                        frag = sample["frag"]
                    else:
                        frag = split(encode(ori_smiles))
                
                output_list = []
                for _, temp_frag in enumerate(frag):
                    if "<" in temp_frag:
                        replaced_frag = extract_angle_bracket_content(temp_frag)
                        output_list.extend(self.tokenizer(replaced_frag)["input_ids"][1:-1])
                    elif "|" in temp_frag:
                        replaced_frag = parse_stereo_chemistry(temp_frag)
                        output_list.extend(self.tokenizer(replaced_frag)["input_ids"][1:-1])
                    else:
                        output_list.extend(self.tokenizer(temp_frag)["input_ids"][1:-1])
                    
                    ## 添加分割符号
                    if _ < len(frag) - 1:
                        output_list.extend(self.tokenizer("<sep>")["input_ids"][1:-1])
                
                temp_dict["decoder_input_ids"] = [4191] + output_list + [4192]
                temp_dict["decoder_attention_mask"] = [1 for _ in temp_dict["decoder_input_ids"]]
            
            return temp_dict

        except Exception as e:
            print(e)
            import traceback
            traceback.print_exc()
            return self.__getitem__(idx+1)
        
    def collate_fn(self, batch):
        if self.args.use_ring_system is False:
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
                source = temp.get("source", "")
                if temp["ring_system_result"] != {}:
                    ## 堆mextnova的数据不再生成
                    if self.phase == "train" and "multi" not in source:
                        # nmr+frag2smiles
                        input_ids_list.append(temp["nmr_result"]["input_ids"] + temp["ring_system_result"]["input_ids"])
                        attention_mask_list.append(temp["nmr_result"]["attention_mask"] + temp["ring_system_result"]["attention_mask"])
                        decoder_input_ids_list.append(temp["smiles_result"]["input_ids"])
                        smiles_list.append(temp["smiles"])
                        # nmr2frag
                        input_ids_list.append(temp["nmr_result"]["input_ids"])
                        attention_mask_list.append(temp["nmr_result"]["attention_mask"])
                        decoder_input_ids_list.append(temp["ring_system_result"]["input_ids"])
                        smiles_list.append(temp["ring_system"])
            
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



class CustomDataModule(pl.LightningDataModule):
    def __init__(self, args, tokenizer=None):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer

    def prepare_data(self):
        ## 导入数据
        self.train_dataset = CustomDataset(self.args, self.tokenizer, file_path=self.args.train_file, phase="train")
        self.val_dataset = CustomDataset(self.args, self.tokenizer, file_path=self.args.valid_file, phase="valid")
        self.test_dataset = CustomDataset(self.args, self.tokenizer, file_path=self.args.test_file, phase="test")

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

def get_model(model_dir="./saved_hf_model_0910", use_lora=False, device = "cpu"):
    try:
        from flash_attn import flash_attn_func  # noqa: F401  探测 flash-attn v2 是否可用
        attn_implementation = "flash_attention_2"
    except Exception:
        # flash-attn v2 不可用时退回 PyTorch 原生 sdpa (均支持训练/生成, 速度接近)
        print("⚠️ flash_attn 不可用, 改用 attn_implementation='sdpa'")
        attn_implementation = "sdpa"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, attn_implementation=attn_implementation)
    model.to(device)
    return model

def get_tokenizer(model_dir="./saved_hf_model_0910"):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return tokenizer

class CustomModel(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        self.args = args
        self.model = get_model(self.args.model_dir)
        print(print_model_size(self.model))
        self.tokenizer = get_tokenizer(self.args.model_dir)
        self.val_step_outputs = []
        self.val_loss = []
        self.test_step_outputs = []
        self.test_loss = []
        
    def training_step(self, batch, batch_idx):
        true_smiles = batch.pop("smiles_list")
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log('train/loss', loss.item(), prog_bar=True)

        self.log('lr', self.lr_schedulers().get_lr()[0], prog_bar=True, logger=False)
        return loss
    
    def decode_frag_smiles_fn(self, ori_pred_smiles_list=[]):
        pred_smiles_list = []
        for _ in range(len(ori_pred_smiles_list)):
            ori_pred_smiles = ori_pred_smiles[_]
            temp_list = ori_pred_smiles.split("<sep>")
            for __ in range(len(temp_list)):
                if "|" not in temp_list[__]:
                    temp_list[__] = restore_angle_bracket_content(temp_list[__])
                else:
                    temp_list[__] = restore_stereo_chemistry(temp_list[__])
            try:
                smiles = decode(temp_list)
            except Exception as e:
                print(e)
                import traceback
                traceback.print_exc()
                smiles = ""
            pred_smiles_list.append(smiles)
        
        return pred_smiles_list
    
    def validation_step(self, batch, batch_idx):
        true_smiles = batch.pop("smiles_list")
        with torch.no_grad():
            outputs = self.model(**batch)
            loss = outputs.loss
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
        # pred_smiles = self.decode_frag_smiles_fn(pred_smiles)

        # true_smiles_ = []
        # if self.args.num_beams == 1:
        #     true_smiles_ = true_smiles
        # else:
        #     for i in range(len(true_smiles)):
        #         true_smiles_.extend([true_smiles[i]]*self.args.num_beams)
        
        # self.val_step_outputs.extend(list(zip(true_smiles_, pred_smiles)))
        self.val_loss.append(loss.item())
        return loss
    
    def on_validation_epoch_end(self):
        if self.trainer.num_devices > 1:
            gathered_outputs = [None for i in range(self.trainer.num_devices)]
            dist.all_gather_object(gathered_outputs, self.val_step_outputs)
            gathered_outputs = sum(gathered_outputs, [])
            gathered_losses =  [None for i in range(self.trainer.num_devices)]
            dist.all_gather_object(gathered_losses, self.val_loss)
            gathered_losses = sum(gathered_losses, [])
        else:
            gathered_outputs = self.val_step_outputs
            gathered_losses = self.val_loss
        
        scores = [-np.mean(gathered_losses)]
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
            result = self.model.generate(
                    input_ids=batch["input_ids"],
                    max_length=self.args.max_length,
                    num_beams=self.args.num_beams,
                    num_return_sequences=self.args.num_beams,
                    bos_token_id=187,
                    pad_token_id=1,
                    eos_token_id=188,
                    decoder_start_token_id=187
                )
        pred_smiles = [self.tokenizer.decode(i) for i in result]
        pred_smiles = [i.replace("<SMILES>", "").replace("</SMILES>", "").replace("<pad>", "").replace("</s>", "") for i in pred_smiles]
        pred_smiles = self.decode_frag_smiles_fn(pred_smiles)

        true_smiles_ = []
        if self.args.num_beams == 1:
            true_smiles_ = true_smiles
        else:
            for i in range(len(true_smiles)):
                true_smiles_.extend([true_smiles[i]]*self.args.num_beams)
        
        self.test_step_outputs.extend(list(zip(true_smiles_, pred_smiles)))
        self.test_loss.append(loss.item())
        return loss

    def on_test_epoch_end(self):
        if self.trainer.num_devices > 1:
            gathered_outputs = [None for i in range(self.trainer.num_devices)]
            dist.all_gather_object(gathered_outputs, self.test_step_outputs)
            gathered_outputs = sum(gathered_outputs, [])
            gathered_losses =  [None for i in range(self.trainer.num_devices)]
            dist.all_gather_object(gathered_losses, self.test_loss)
            gathered_losses = sum(gathered_losses, [])
        else:
            gathered_outputs = self.val_step_outputs
            gathered_losses = self.val_loss
        
        scores = [0]
        
        if self.trainer.is_global_zero:

            count = 0
            acc = 0
            smi = 0
            for (true_smiles, pred_smiles) in gathered_outputs:
                true_mol = None
                pred_mol = None
                try:
                    true_mol = Chem.MolFromSmiles(true_smiles)
                except:
                    pass
                    
                try:
                    pred_mol = Chem.MolFromSmiles(pred_smiles)
                except:
                    pass
                
                if (true_mol is None) or (pred_mol is None):
                    pass
                else:
                    if Chem.MolToSmiles(true_mol) == Chem.MolToSmiles(pred_mol):
                        acc = acc + 1
                    
                    fp_true = AllChem.GetMorganFingerprintAsBitVect(true_mol, radius=2, nBits=2048, useChirality=True)
                    fp_pred = AllChem.GetMorganFingerprintAsBitVect(pred_mol, radius=2, nBits=2048, useChirality=True)
                    similarity = DataStructs.TanimotoSimilarity(fp_true, fp_pred)
                    smi = smi + similarity
                
                count = count + 1
            
            with open(os.path.join(self.trainer.default_root_dir, f'test.json'), 'w') as f:
                json.dump(gathered_outputs, f)
            
            scores = [smi/count]

        dist.broadcast_object_list(scores)
        self.log('test/score', scores[0], prog_bar=True, rank_zero_only=True)

        self.test_step_outputs.clear()
        print("test epoch end end")


    def configure_optimizers(self):
        num_training_steps = self.trainer.num_training_steps
        self.print(f'Num training steps: {num_training_steps}')
        num_warmup_steps = int(num_training_steps * self.args.warmup_ratio)
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        scheduler = get_scheduler(self.args.scheduler, optimizer, num_warmup_steps, num_training_steps)
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}
    
    def on_train_epoch_end(self):
        ## 保存模型
        if (self.current_epoch%5) == 0:
            self.trainer.save_checkpoint(os.path.join(self.args.save_path, f'checkpoints/epoch_{self.current_epoch}.ckpt'))


class ModelCheckpoint(pl.callbacks.ModelCheckpoint):
    def _get_metric_interpolated_filepath_name(self, monitor_candidates, trainer, del_filepath=None) -> str:
        filepath = self.format_checkpoint_name(monitor_candidates)
        return filepath


def main():
    args = get_args()
    pl.seed_everything(args.seed, workers=True)

    
    tokenizer = get_tokenizer(args.model_dir)
    # args.tokenizer = tokenizer

    dm = CustomDataModule(args, tokenizer)
    dm.prepare_data()
    dm.print_stats()

    try:
        import tokenizers
        import transformers
        from transformers.models.bart.tokenization_bart_fast import BartTokenizerFast
        ## torch2.6 引入的新机制
        ## ref:https://stackoverflow.com/questions/79584485/unable-to-torch-load-due-to-pickling-safety-error
        torch.serialization.add_safe_globals(
            [
                transformers.models.bart.tokenization_bart_fast.BartTokenizerFast,
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
            model = MODEL.load_from_checkpoint(os.path.join(args.save_path, 'checkpoints/last.ckpt'), strict=False, args=args)
            print("load successfully")

    checkpoint = ModelCheckpoint(monitor='val/score', mode='max', save_top_k=1, filename='best', save_last=True)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    logger = pl.loggers.TensorBoardLogger(args.save_path, name='', version='')

    trainer = pl.Trainer(
        strategy=DDPStrategy(find_unused_parameters=False),
        accelerator='gpu',
        precision = "bf16",
        devices=args.gpus,
        logger=logger,
        default_root_dir=args.save_path,
        callbacks=[checkpoint, lr_monitor],
        max_epochs=args.epochs,
        gradient_clip_val=args.max_grad_norm,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        check_val_every_n_epoch=args.eval_per_epoch,
        log_every_n_steps=10,
        deterministic='warn')

    if args.do_train:
        trainer.num_training_steps = math.ceil(
            len(dm.train_dataset) / (args.batch_size * args.gpus * args.gradient_accumulation_steps)) * args.epochs
        model.eval_dataset = dm.val_dataset
        ckpt_path = os.path.join(args.save_path, 'checkpoints/last.ckpt') if args.resume else None
        trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)
        model = MODEL.load_from_checkpoint(checkpoint.best_model_path, args=args)

    if args.do_valid:
        model.eval_dataset = dm.val_dataset
        trainer.validate(model, datamodule=dm)

    if args.do_test:
        model.eval_dataset = dm.test_dataset
        trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    main()

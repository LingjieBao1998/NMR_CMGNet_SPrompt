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
from torch.nn.utils.rnn import pad_sequence

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModel
from transformers import get_scheduler
import json
from functools import reduce
from operator import mul
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem import AllChem, MACCSkeys

import warnings
from rdkit import RDLogger
from rdkit.Chem import rdMolDescriptors
import copy
from rdkit.Chem import rdSubstructLibrary
import lmdb
import pickle

# 屏蔽 RDKit 的警告（包括 DEPRECATION WARNING）
RDLogger.DisableLog('rdApp.warning')  # 屏蔽 RDKit 自己的 Logger 警告
warnings.filterwarnings("ignore", category=DeprecationWarning)  # 屏蔽 Python 的 DeprecationWarning
import multiprocessing as mp

from new_RingSystemFinder import RingSystemFinder

from rdkit import Chem
from thermo.functional_groups import (
    identify_functional_groups, 
    ALL_FUNCTIONAL_GROUPS,
    FG_TO_FUNCTION
)

def get_all_functional_groups_presence(mol):
    """
    返回分子中所有官能团的存在情况
    
    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit分子对象
    
    Returns
    -------
    dict
        键为官能团名称，值为布尔值（是否存在）
    """
    if mol is None:
        raise ValueError("分子对象不能为None")
    
    # 使用thermo库的功能
    detected_groups = identify_functional_groups(mol)
    
    # 创建包含所有官能团状态的字典
    all_groups_presence = {}
    for fg_name in ALL_FUNCTIONAL_GROUPS:
        all_groups_presence[fg_name] = fg_name in detected_groups
    
    return all_groups_presence

def functional_groups_dict_to_binary_vector(presence_dict):
    """
    将官能团存在字典转换为固定顺序的二进制列表
    
    Parameters
    ----------
    presence_dict : dict
        官能团存在字典，键为官能团名称，值为布尔值
    
    Returns
    -------
    list
        固定顺序的二进制列表，0表示不存在，1表示存在
    """
    # 使用 ALL_FUNCTIONAL_GROUPS 定义的固定顺序
    fixed_order = [
        'acid', 'acyl_halide', 'alcohol', 'aldehyde', 'alkane', 'alkene',
        'alkylaluminium', 'alkyllithium', 'alkylmagnesium_halide', 'alkyne',
        'amide', 'amidine', 'amine', 'anhydride', 'aromatic', 'azide', 'azo',
        'borinic_acid', 'borinic_ester', 'boronic_acid', 'boronic_ester',
        'branched_alkane', 'bromoalkane', 'carbamate', 'carbodithio',
        'carbodithioic_acid', 'carbonate', 'carbothioic_o_acid',
        'carbothioic_s_acid', 'carboxylate', 'carboxylic_acid',
        'carboxylic_anhydride', 'chloroalkane', 'cyanate', 'cyanide',
        'cycloalkane', 'disulfide', 'ester', 'ether', 'fluoroalkane',
        'haloalkane', 'hydrocarbon', 'hydroperoxide', 'imide', 'imine',
        'inorganic', 'iodoalkane', 'isocyanate', 'isonitrile', 'isothiocyanate',
        'ketone', 'mercaptan', 'methylenedioxy', 'nitrate', 'nitrile',
        'nitrite', 'nitro', 'nitroso', 'organic', 'orthocarbonate_ester',
        'orthoester', 'oxime', 'peroxide', 'phenol', 'phosphate', 'phosphine',
        'phosphodiester', 'phosphonic_acid', 'polyol', 'primary_aldimine',
        'primary_amine', 'primary_ketimine', 'pyridyl', 'quat', 'radionuclide',
        'secondary_aldimine', 'secondary_amine', 'secondary_ketimine',
        'siloxane', 'silyl_ether', 'sulfide', 'sulfinic_acid',
        'sulfonate_ester', 'sulfone', 'sulfonic_acid', 'sulfoxide',
        'tertiary_amine', 'thial', 'thiocyanate', 'thioketone', 'thiolester',
        'thionoester'
    ]
    
    # 确保顺序正确
    binary_vector = []
    for fg_name in fixed_order:
        if fg_name in presence_dict:
            binary_vector.append(1 if presence_dict[fg_name] else 0)
        else:
            # 如果字典中缺少某个官能团，默认为0
            binary_vector.append(0)
    
    return binary_vector


ALLOWED_ELEMENTS_list = ['C', 'H', 'O', 'N', 'P', 'S', 'Cl', 'Br', 'I', 'B', 'F', "Si"]
def get_element_counts_list(mol, allowed_elements=ALLOWED_ELEMENTS_list):
    """
    计算分子中指定元素的原子个数，返回扁平化列表（无元素则填0）
    
    参数:
        mol: RDKit的Mol对象
        allowed_elements: 允许的元素列表（默认为ALLOWED_ELEMENTS_list）
    
    返回:
        list: 形如 ['C', 1, 'H', 0, 'O', 0, ...]，严格按allowed_elements顺序排列
    """
    # 添加氢原子（显式化）
    mol_hs = copy.deepcopy(mol)
    mol_hs = Chem.AddHs(mol_hs)
    
    # 初始化所有元素的计数为0
    element_counts = {elem: 0 for elem in allowed_elements}
    
    # 统计实际存在的元素
    for atom in mol_hs.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol in allowed_elements:
            element_counts[symbol] += 1
    
    # 转换为扁平化列表（交替元素符号和计数）
    result = []
    for elem in allowed_elements:
        result.extend([element_counts[elem]])
    
    return result

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

def get_atom_count(mol):
    """计算分子中的原子数量"""
    if mol is None:
        return 0
    return mol.GetNumAtoms()

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
    parser.add_argument('--use_clear_smiles', action='store_true') # 证明了效果确实不行
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
    parser.add_argument('--save_path', type=str, default='output/nmr_bert_1010')
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

    parser.add_argument('--mask_ratio', type=float, default=0.20)

    parser.add_argument('--use_smiles_encoder', action='store_true')
    parser.add_argument('--use_ring_system', action='store_true')

    parser.add_argument('--loss_type', type=str, default="siglip")

    parser.add_argument('--lmdb_path', type=str, default="")

    args = parser.parse_args([]) if notebook else parser.parse_args()
    args.lmdb_path = args.lmdb_path.split(",")

    return args

def smiles2token(tokenizer, smile_string, phase="train", block_size=512, proba=0.5):
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
    def __init__(self, args, tokenizer=None, smiles_tokenizer=None, file_path="", phase="val"):
        """
        Args:
            data_list (list or array): 样本数据，比如特征列表 [样本1, 样本2, ...]
            labels (list or tensor, optional): 对应的标签，如 [0, 1, 0, ...]。可以为 None（如测试集）
        """
        self.args = args
        self.tokenizer = tokenizer
        self.smiles_tokenizer = smiles_tokenizer
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
                    if temp_i>0:
                        key_list = [f'{count}'.encode('ascii') for count in range(length)]
                    else:
                        print("using 5000000 data as test data")
                        key_list = [f'{count}'.encode('ascii') for count in range(length) if count>5000000]
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
    
    def mask_nmr_token_list(self, nmr_data, mask_ratio=0.20, mask_token=4):
        """
        随机屏蔽 NMR 数据中的一部分值

        参数:
        nmr_data: 原始 NMR 数据列表
        mask_ratio: 屏蔽比例，默认15%

        返回:
        masked_data: 屏蔽后的数据列表（屏蔽的值用None表示）
        masked_indices: 被屏蔽的索引位置
        """
        if len(nmr_data)>1:
            data_length = len(nmr_data)
            num_to_mask = int(data_length * mask_ratio)

            # 随机选择要屏蔽的索引
            count = -1
            while count < 10:
                count = count + 1
                masked_indices = random.sample(range(data_length), num_to_mask)
                if len(masked_indices)>0:
                    break

            # 创建屏蔽后的数据副本
            masked_data = nmr_data.copy()

            # 屏蔽选中的值
            for idx in masked_indices:
                masked_data[idx] = mask_token  # 或者可以用其他标记值，如 "MASKED"

            return masked_data, masked_indices
        else:
            return nmr_data, []
    
    def check_tensor(self, t):
        """检查张量是否包含NaN或inf"""
        if isinstance(t, torch.Tensor):
            if torch.isnan(t).any() or torch.isinf(t).any():
                return True
        return False
    
    def post_process_data(self, raw_item):
        # 转换所有需要的数据为tensor
        item = {
            "idx": raw_item["idx"],
            "smiles": raw_item["smiles"],
            "nmr_result": {
                "input_ids": torch.tensor(raw_item["nmr_result"]["input_ids"]).long(),
                "attention_mask": torch.tensor(raw_item["nmr_result"]["attention_mask"]).long()
            },
            "masked_nmr_result": {
                "input_ids": torch.tensor(raw_item["masked_nmr_result"]["input_ids"]).long(),
                "attention_mask": torch.tensor(raw_item["masked_nmr_result"]["attention_mask"]).long()
            },
            # 提前合并指纹特征并转换为tensor
            "combined_fp": torch.from_numpy(
                np.concatenate([
                    raw_item["ecfp"], 
                    raw_item["maccs"], 
                    raw_item["function_list"], 
                    raw_item["label"]
                ])
            ).long(),
            "element_list": torch.from_numpy(raw_item["element_list"]).long()
        }
        
        if self.args.use_smiles_encoder:
            item["smiles_result"] = {
                "input_ids": torch.tensor(raw_item["smiles_result"]["input_ids"]).long(),
                "attention_mask": torch.tensor(raw_item["smiles_result"]["attention_mask"]).long()
            }
        
        def recursive_check(d):
            for k, v in d.items():
                if isinstance(v, dict):
                    if recursive_check(v):
                        return True
                elif self.check_tensor(v):
                    print(f"WARNING: Found NaN/inf in {k} at index {idx}")
                    return True
            return False
        
        if recursive_check(item):
            return self.__getitem__(raw_item["idx"]+1)
            # return None
        
        return item

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
                try:
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
                except Exception as e:
                    print(e)
                    import traceback
                    traceback.print_exc()
                    # return self.__getitem__(idx+1)
                    return None
            
            # if len(ori_nmr)<2:
            #     # return self.__getitem__(idx+1)
            #     return None
            
            mol = None
            try:
                mol = Chem.MolFromSmiles(ori_smiles)  # 以乙醇为例
            except Exception as e:
                print(e)
            if mol is None:
                print("mol is invalid")
                return self.__getitem__(idx+1)
                # return None

            random_ring_system = ""
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
            
            if self.phase == "train" and random.random()<self.args.smiles_aug_proba:
                smiles = get_random_smiles(mol)
            
            temp_dict = {}
            masked_indices = []
            if self.phase == "train" and random.random()<self.args.nmr_aug_proba:
                ori_nmr_result = self.ori_NmrDataset.augment_item(ori_nmr)
            else:
                ori_nmr_result = self.ori_NmrDataset.fill_item(ori_nmr)
            
            temp_dict["nmr_result"] = ori_nmr_result
            if "ecfp" not in sample:
                ecfp = AllChem.GetMorganFingerprintAsBitVect(
                    mol,
                    radius=2,          # ECFP4 的半径为2
                    nBits=2048,        # 2048维
                    useChirality=True  # 关键：启用手性感知
                )
                ecfp = np.array(ecfp)
                temp_dict["ecfp"] = ecfp
            else:
                temp_dict["ecfp"] = sample["ecfp"]
            
            if self.args.use_smiles_encoder:
                if self.args.use_ring_system is False:
                    smiles_result = self.smiles_tokenizer(ori_smiles)
                    temp_dict["smiles_result"] = smiles_result
                    temp_dict["smiles"] = ori_smiles
                else:
                    smiles_result = self.smiles_tokenizer(random_ring_system)
                    temp_dict["smiles_result"] = smiles_result
                    temp_dict["smiles"] = random_ring_system

            if "maccs" not in sample:
                maccs_fp = MACCSkeys.GenMACCSKeys(mol)
                fp_bitstring = maccs_fp.ToBitString()
                maccs_fp = np.array([int(bit) for bit in fp_bitstring])
                temp_dict["maccs"] = maccs_fp
            else:
                temp_dict["maccs"] = sample["maccs"]

            if "label" not in sample:
                mols = rdSubstructLibrary.CachedSmilesMolHolder()
                fps = rdSubstructLibrary.TautomerPatternHolder(1024)
                mols.AddSmiles(ori_smiles)
                fp = Chem.PatternFingerprint(mol,fpSize=1024,tautomerFingerprints=True)
                fps.AddFingerprint(fp)
                library = rdSubstructLibrary.SubstructLibrary(mols,fps)
                indices_1 = library.GetMatches(Chem.MolFromSmiles('O=C1C(C(O)=O)=CNC2=CC=CC=C12'),maxResults=50000)
                indices_2 = library.GetMatches(Chem.MolFromSmiles('O=C1C(C(O)=O)=CNC2=NC=CC=C12'),maxResults=50000)

                if len(indices_1)>0 or len(indices_2)>0:
                    temp_dict["label"] = np.array([1])
                else:
                    temp_dict["label"] = np.array([0])
            else:
                temp_dict["label"] = sample["label"]
            
            nmr_result = copy.deepcopy(ori_nmr_result)
            nmr_result["input_ids"][1:-1], masked_indices = self.mask_nmr_token_list(nmr_result["input_ids"][1:-1], mask_ratio=self.args.mask_ratio)
            masked_indices = [_ + 1 for _ in masked_indices]
            temp_dict["masked_nmr_result"] = nmr_result
            # temp_dict["masked_indices"] = masked_indices
            temp_dict["idx"] = idx

            if "element_list" not in sample:
                temp_dict["element_list"] = np.array(get_element_counts_list(mol))
            else:
                temp_dict["element_list"] = sample["element_list"]
            if "function_list" not in sample:
                temp_dict["function_list"] = np.array(functional_groups_dict_to_binary_vector(get_all_functional_groups_presence(mol)))
            else:
                temp_dict["function_list"] = sample["function_list"]

            return self.post_process_data(temp_dict)

        except Exception as e:
            print(e)
            import traceback
            traceback.print_exc()
            return self.__getitem__(idx+1)
            # return None
    
    def collate_fn(self, batch):
        # 初始化容器，直接收集已转换的tensor
        batch_dict = {
            "index_list": [],
            "smiles_list": [],
            "mask_nmr_inputs": {
                "input_ids": [],
                "attention_mask": [],
                "labels": []
            },
            "nmr_inputs": {
                "input_ids": [],
                "attention_mask": []
            },
            "smiles_inputs": {
                "input_ids": [],
                "attention_mask": []
            } if self.args.use_smiles_encoder else None,
            "fp_tensor": [],
            "element_tensor": []
        }
        
        batch = [b for b in batch if b is not None]  # 过滤无效样本
        if len(batch)==0:
            print("batch is zero")
            batch = [self.__getitem__(0)]

        # 直接收集已处理好的tensor
        for item in batch:
            batch_dict["index_list"].append(item["idx"])
            batch_dict["smiles_list"].append(item["smiles"])
            
            batch_dict["mask_nmr_inputs"]["input_ids"].append(item["masked_nmr_result"]["input_ids"])
            batch_dict["mask_nmr_inputs"]["attention_mask"].append(item["masked_nmr_result"]["attention_mask"])
            batch_dict["mask_nmr_inputs"]["labels"].append(item["nmr_result"]["input_ids"])
            
            batch_dict["nmr_inputs"]["input_ids"].append(item["nmr_result"]["input_ids"])
            batch_dict["nmr_inputs"]["attention_mask"].append(item["nmr_result"]["attention_mask"])
            
            if self.args.use_smiles_encoder:
                batch_dict["smiles_inputs"]["input_ids"].append(item["smiles_result"]["input_ids"])
                batch_dict["smiles_inputs"]["attention_mask"].append(item["smiles_result"]["attention_mask"])
            
            batch_dict["fp_tensor"].append(item["combined_fp"])
            batch_dict["element_tensor"].append(item["element_list"])

        # 填充序列
        batch_dict["nmr_inputs"]["input_ids"] = pad_sequence(
            batch_dict["nmr_inputs"]["input_ids"], 
            batch_first=True, 
            padding_value=1
        )[:, :self.args.max_length]
        
        batch_dict["nmr_inputs"]["attention_mask"] = pad_sequence(
            batch_dict["nmr_inputs"]["attention_mask"],
            batch_first=True,
            padding_value=0
        )[:, :self.args.max_length]

        batch_dict["mask_nmr_inputs"]["input_ids"] = pad_sequence(
            batch_dict["mask_nmr_inputs"]["input_ids"],
            batch_first=True,
            padding_value=1
        )[:, :self.args.max_length]
        
        batch_dict["mask_nmr_inputs"]["attention_mask"] = pad_sequence(
            batch_dict["mask_nmr_inputs"]["attention_mask"],
            batch_first=True,
            padding_value=0
        )[:, :self.args.max_length]
        
        batch_dict["mask_nmr_inputs"]["labels"] = pad_sequence(
            batch_dict["mask_nmr_inputs"]["labels"],
            batch_first=True,
            padding_value=1
        )[:, :self.args.max_length]
        
        # 处理特殊标记
        batch_dict["mask_nmr_inputs"]["labels"][
            batch_dict["mask_nmr_inputs"]["input_ids"] != 4
        ] = -100

        if self.args.use_smiles_encoder:
            batch_dict["smiles_inputs"]["input_ids"] = pad_sequence(
                batch_dict["smiles_inputs"]["input_ids"],
                batch_first=True,
                padding_value=2
            )[:, :self.args.max_length]
            
            batch_dict["smiles_inputs"]["attention_mask"] = pad_sequence(
                batch_dict["smiles_inputs"]["attention_mask"],
                batch_first=True,
                padding_value=0
            )[:, :self.args.max_length]

        # 堆叠fp和element tensor
        fp_tensor = torch.stack(batch_dict["fp_tensor"]).long()
        element_tensor = torch.stack(batch_dict["element_tensor"]).long()
        element_tensor.clamp_(min=0, max=120)

        return (
            batch_dict["index_list"],
            batch_dict["smiles_list"],
            batch_dict["mask_nmr_inputs"],
            batch_dict["nmr_inputs"],
            batch_dict["smiles_inputs"],
            fp_tensor,
            element_tensor
        )



class CustomDataModule(pl.LightningDataModule):
    def __init__(self, args, tokenizer=None, smiles_tokenizer=None):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.smiles_tokenizer = smiles_tokenizer

    def prepare_data(self):
        ## 导入数据
        self.train_dataset = CustomDataset(self.args, self.tokenizer, self.smiles_tokenizer, file_path=self.args.train_file, phase="train")
        self.val_dataset = CustomDataset(self.args, self.tokenizer, self.smiles_tokenizer, file_path=self.args.valid_file, phase="val")
        self.test_dataset = CustomDataset(self.args, self.tokenizer, self.smiles_tokenizer, file_path=self.args.test_file, phase="test")

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
            # sampler = DistributedSampler(
            #     self.train_dataset,
            #     num_replicas=self.trainer.world_size,
            #     rank=self.trainer.global_rank,
            #     shuffle=True,
            #     drop_last=True  # 避免最后批次尺寸不一致
            # )
            return torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                collate_fn=self.train_dataset.collate_fn,
                persistent_workers=True,  # 保持worker存活
                prefetch_factor=2,        # 减少预取数量
                pin_memory=True,         # 关闭pin_memory测试
                # timeout=240,               # 增加超时时间
                # shuffle=True,
                # sampler=sampler,
                # multiprocessing_context='spawn'  # 使用spawn方法
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
                # persistent_workers=True,  # 保持worker存活
                # prefetch_factor=2,        # 减少预取数量
                # pin_memory=False,         # 关闭pin_memory测试
                # timeout=60,               # 增加超时时间
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
                # persistent_workers=True,  # 保持worker存活
                # prefetch_factor=2,        # 减少预取数量
                # pin_memory=False,         # 关闭pin_memory测试
                # timeout=60,               # 增加超时时间
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
    encoder = model.get_encoder()
    del model
    return encoder

def get_tokenizer(model_dir="./saved_hf_model_0910"):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return tokenizer

def con_loss(logits):
    targets = torch.arange(logits.size(0)).to(logits.device)
    loss_images = F.cross_entropy(logits, targets)
    loss_texts = F.cross_entropy(logits.t(), targets)
    return (loss_images + loss_texts) / 2


def siglip_loss(logits):
    n = logits.size(0)
    # -1 for off-diagonals and 1 for diagonals
    labels = 2 * torch.eye(n, device=logits.device) - 1
    # pairwise sigmoid loss
    return -torch.sum(F.logsigmoid(labels * logits)) / n

class CustomModel(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        self.args = args
        self.model = get_model(self.args.model_dir)
        print_model_size(self.model)
        if args.use_smiles_encoder:
            self.smiles_encoder = AutoModel.from_pretrained("./checkpoints/molformer", trust_remote_code=True)
        else:
            self.smiles_encoder = None
        if self.smiles_encoder is not None:
            print_model_size(self.smiles_encoder)
        self.tokenizer = get_tokenizer(self.args.model_dir)
        self.val_step_outputs = []
        self.val_loss = []
        self.test_step_outputs = []
        self.test_loss = []

        self.mask_linear = nn.Sequential(
                            nn.Linear(768, 768//2),
                            nn.GELU(),  # 或 nn.ReLU()
                            nn.Dropout(0.1),
                            nn.Linear(768//2, len(self.tokenizer))
                        )
        self.fp_linear = nn.Sequential(
                            nn.Linear(768, 768//2),
                            nn.GELU(),  # 或 nn.ReLU()
                            nn.Dropout(0.1),
                            nn.Linear(768//2, 2308)
                        )
        self.ele_linear = nn.Sequential(
                            nn.Linear(768, 768//2),
                            nn.GELU(),  # 或 nn.ReLU()
                            nn.Dropout(0.1),
                            nn.Linear(768//2, 12*121))

        ## ref:https://github.com/ramanakshay/clip/blob/e030d1b17296fcd766bdda16089ddb4f917710b5/src/model/clip/clip.py#L22
        if self.args.loss_type == 'clip':
            init_temp, init_bias = np.log(5), 0
            self.con_loss_fn = con_loss
        else:
            init_temp, init_bias = np.log(10), -10
            self.con_loss_fn = siglip_loss

        self.temp = nn.Parameter(torch.ones([]) * init_temp)
        self.bias = nn.Parameter(torch.ones([]) * init_bias)

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        pos_weight = torch.ones(2308)
        pos_weight[-1] = 100
        self.fp_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    def forward_fn(self, index_list, smiles_list, mask_nmr_inputs, nmr_inputs, smiles_inputs, fp_tensor, element_tensor):
        mask_nmr_labels = mask_nmr_inputs.pop("labels")
        # true_smiles = batch.pop("smiles_list")
        nmr_outputs = self.model(**nmr_inputs) # nmr_outputs.last_hidden_state.shape, torch.Size([128, 15, 768])
        nmr_encodings = F.normalize(nmr_outputs.last_hidden_state[:,0,:], p=2, dim=-1)
        mask_nmr_outputs = self.model(**mask_nmr_inputs) # mask_nmr_outputs.last_hidden_state.shape, torch.Size([128, 15, 768])
        if self.args.use_smiles_encoder:
            smiles_outputs = self.smiles_encoder(**smiles_inputs) # smiles_outputs.last_hidden_state.shape

            smiles_encodings = F.normalize(smiles_outputs.last_hidden_state[:,0,:], p=2, dim=-1)
            
            sim_matrix = smiles_encodings @ nmr_encodings.t() * self.temp.exp() + self.bias
            
            con_loss = self.con_loss_fn(sim_matrix)
        else:
            con_loss = 0
        
        mask_out = self.mask_linear(nmr_outputs.last_hidden_state).permute(0,2,1) # torch.Size([128, 4191, 15])
        fp_out = self.fp_linear(nmr_encodings)
        ele_out = self.ele_linear(nmr_encodings).reshape((nmr_encodings.shape[0], 121, -1))

        mask_loss = self.loss_fn(mask_out, mask_nmr_labels)
        # print(fp_out.shape, fp_tensor.shape)
        fp_loss = self.fp_loss_fn(fp_out, fp_tensor.float())
        ele_loss = self.loss_fn(ele_out, element_tensor)
        return con_loss, mask_loss, fp_loss, ele_loss
        
    def training_step(self, batch, batch_idx):
        index_list, smiles_list, mask_nmr_inputs, nmr_inputs, smiles_inputs, fp_tensor, element_tensor = batch
        con_loss, mask_loss, fp_loss, ele_loss = self.forward_fn(index_list, smiles_list, mask_nmr_inputs, nmr_inputs, smiles_inputs, fp_tensor, element_tensor)
        
        loss = 2*con_loss + mask_loss + 5*fp_loss + ele_loss

        self.log('train/con_loss', con_loss.item(), prog_bar=True, rank_zero_only=True)
        self.log('train/mask_loss', mask_loss.item(), prog_bar=True, rank_zero_only=True)
        self.log('train/fp_loss', fp_loss.item(), prog_bar=True, rank_zero_only=True)
        self.log('train/ele_loss', ele_loss.item(), prog_bar=True, rank_zero_only=True)
        # self.log('train/loss', loss.item(), prog_bar=True, rank_zero_only=True)
        self.log('lr', self.lr_schedulers().get_lr()[0], prog_bar=True, logger=False)
        return loss
    
    def validation_step(self, batch, batch_idx):
        index_list, smiles_list, mask_nmr_inputs, nmr_inputs, smiles_inputs, fp_tensor, element_tensor = batch
        with torch.no_grad():
            con_loss, mask_loss, fp_loss, ele_loss = self.forward_fn(index_list, smiles_list, mask_nmr_inputs, nmr_inputs, smiles_inputs, fp_tensor, element_tensor)
            
            loss = con_loss + mask_loss + fp_loss + ele_loss

            self.log('val/con_loss', con_loss.item(), prog_bar=True, rank_zero_only=True)
            self.log('val/mask_loss', mask_loss.item(), prog_bar=True, rank_zero_only=True)
            self.log('val/fp_loss', fp_loss.item(), prog_bar=True, rank_zero_only=True)
            self.log('val/ele_loss', ele_loss.item(), prog_bar=True, rank_zero_only=True)
            # self.log('val/loss', loss.item(), prog_bar=True, rank_zero_only=True)

        self.val_step_outputs.append(loss)
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
        tindex_list, smiles_list, mask_nmr_inputs, nmr_inputs, smiles_inputs, fp_tensor, element_tensor = batch
        with torch.no_grad():
            con_loss, mask_loss, fp_loss, ele_loss = self.forward_fn(index_list, smiles_list, mask_nmr_inputs, nmr_inputs, smiles_inputs, fp_tensor, element_tensor)
            
            loss = con_loss + mask_loss + fp_loss + ele_loss

            self.log('test/con_loss', con_loss.item(), prog_bar=True, rank_zero_only=True)
            self.log('test/mask_loss', mask_loss.item(), prog_bar=True, rank_zero_only=True)
            self.log('test/fp_loss', fp_loss.item(), prog_bar=True, rank_zero_only=True)
            self.log('test/ele_loss', ele_loss.item(), prog_bar=True, rank_zero_only=True)
            # self.log('val/loss', loss.item(), prog_bar=True, rank_zero_only=True)

        self.test_step_outputs.append(loss)
        return loss

    def on_test_epoch_end(self):
        if self.trainer.num_devices > 1:
            gathered_outputs = [None for i in range(self.trainer.num_devices)]
            dist.all_gather_object(gathered_outputs, self.test_step_outputs)
            gathered_outputs = sum(gathered_outputs, [])
            gathered_losses =  [None for i in range(self.trainer.num_devices)]
            dist.all_gather_object(gathered_losses, self.val_loss)
            gathered_losses = sum(gathered_losses, [])
        else:
            gathered_outputs = self.test_step_outputs
            gathered_losses = self.val_loss
        
        scores = [-np.mean(gathered_losses)]

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
        if (self.current_epoch%1) == 0:
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
    if args.use_smiles_encoder:
        smiles_tokenizer = AutoTokenizer.from_pretrained("./checkpoints/molformer", trust_remote_code=True)
    else:
        smiles_tokenizer = None
    # args.smiles_tokenizer = smiles_tokenizer

    dm = CustomDataModule(args, tokenizer, smiles_tokenizer)
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
    # mp.set_start_method('spawn')
    # mp.set_start_method('spawn', force=True) 
    main()

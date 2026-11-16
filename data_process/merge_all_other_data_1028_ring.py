import lmdb
from multiprocessing import Pool, Queue, Process
import time
import os
from tqdm import tqdm
import pickle
from rdkit import Chem
from rdkit.Chem import Descriptors, QED
import json
import copy
import random
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem import AllChem, MACCSkeys
import copy
from thermo.functional_groups import (
    identify_functional_groups, 
    ALL_FUNCTIONAL_GROUPS,
    FG_TO_FUNCTION
)
from rdkit.Chem import rdSubstructLibrary
import numpy as np
from func_timeout import func_set_timeout, FunctionTimedOut
import func_timeout
import gc
import psutil
import useful_rdkit_utils as uru
from rdkit.Chem import rdmolops

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

from rdkit.Chem import rdSubstructLibrary

start = time.time()
lmdb_path = "data/NMR_other_data_1020/new_db.lmdb"
ori_env = lmdb.open(
            lmdb_path,
            map_size=100*(1024*1024*1024),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
        )

# ==================== 第二部分：创建新的可写LMDB ====================
new_lmdb_path = "data/NMR_other_data_1020_add_ring_system_1028/new_db.lmdb"  # 请修改为你的新路径
os.makedirs(os.path.dirname(new_lmdb_path), exist_ok=True)
map_size = 1099511627776  # 1TB初始空间（根据数据量调整）

# 创建优化配置的新LMDB环境
new_env = lmdb.open(
    new_lmdb_path,
    map_size=100*(1024*1024*1024),
)
txn_writer = new_env.begin(write=True)

def get_cano_smiles(smiles):
    """
    将输入的 SMILES 字符串转换为规范的（Canonical）SMILES 表示。

    参数:
        smiles (str): 输入的 SMILES 字符串

    返回:
        str: 规范化的 SMILES，如果输入无效则返回 None
    """
    try:
        # 从 SMILES 创建分子对象
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None  # 无法解析的 SMILES
        
        ## 清除去自由基
        for atom in mol.GetAtoms():
            if atom.GetNumRadicalElectrons() > 0:
                print(f"发现自由基原子 {atom.GetSymbol()}，有 {atom.GetNumRadicalElectrons()} 个自由电子")
                # 加一个氢原子（简化处理，真实情况可能需要更复杂的逻辑）
                # atom.SetNumExplicitHs(atom.GetTotalNumHs() + 1)
                if atom.GetSymbol() in ["C","N","O"]:
                    atom.SetNumRadicalElectrons(0)  # 手动清除自由基
                else:
                    return None
        
        # 生成 Canonical SMILES
        cano_smiles = Chem.MolToSmiles(mol, canonical=True)
        return cano_smiles
    except Exception as e:
        # 捕获可能的异常（如输入不是字符串等）
        print(f"[Error] 处理 SMILES 时发生异常: {e}")
        return None

with ori_env.begin() as txn:
    length = txn.stat()["entries"]  # 关键统计操作

end = time.time()
print(f"time consumed:{end-start:.3f}")
print(f"LMDB contains {length} entries")

# ✅ 只允许的元素符号
ALLOWED_ELEMENTS = {'C', 'H', 'O', 'N', 'P', 'S', 'Cl', 'Br', 'I', 'B', "F", 'Si'} # 'Li', 'Na', 'K', 'Si'

# ======================
# 2. 检查分子是否符合条件
# ======================
def is_molecule_valid(mol: Chem.rdchem.Mol) -> bool:
    if mol is None:
        return False

    # 1. 检查分子量 16 <= MW <= 700
    mw = Descriptors.MolWt(mol)
    if mw < 16 or mw > 800:
        return False

    # 2. 检查所有原子是否属于允许的元素
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol not in ALLOWED_ELEMENTS:
            return False
    
    # 3. 检查是否存在自由基（任意原子的 radical electrons > 0）
    for atom in mol.GetAtoms():
        if atom.GetNumRadicalElectrons() > 0:
            return False

    return True

def get_new_smiles(old_mol):
    old_mol = copy.deepcopy(old_mol)
    len_mol = old_mol.GetNumAtoms()
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

def count_carbon_atoms(mol):
    return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)

func_set_timeout(120)
def process_single(data):
    key = data
    try:
        with ori_env.begin() as txn:
            sample = pickle.loads(txn.get(key))
            if "canonical_smiles" in sample:
                smiles = sample["canonical_smiles"]
            else:
                smiles = sample["cano_smiles"]
            
            # smiles = get_cano_smiles(smiles)
            if "." in smiles:
                return None
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            if is_molecule_valid(mol) is False:
                return None
            
            number_carbon = count_carbon_atoms(mol)

            if "C_nmr" in sample:
                ori_nmr = sample["C_nmr"]
            else:
                ori_nmr = sample["NMR"]
            
            # if len(ori_nmr)>number_carbon+2:
            #     print("too long ori_nmr")
            #     return None
            
            mols = rdSubstructLibrary.CachedSmilesMolHolder()
            fps = rdSubstructLibrary.TautomerPatternHolder(1024)
            mols.AddSmiles(smiles)
            fp = Chem.PatternFingerprint(mol,fpSize=1024,tautomerFingerprints=True)
            fps.AddFingerprint(fp)
            library = rdSubstructLibrary.SubstructLibrary(mols,fps)
            indices_1 = library.GetMatches(Chem.MolFromSmiles('O=C1C(C(O)=O)=CNC2=CC=CC=C12'),maxResults=50000)
            indices_2 = library.GetMatches(Chem.MolFromSmiles('O=C1C(C(O)=O)=CNC2=NC=CC=C12'),maxResults=50000)

            if "label" not in sample:
                if len(indices_1)>0 or len(indices_2)>0:
                    label = np.array([1])
                else:
                    label = np.array([0])
                
                sample["label"] = label
            
            label = sample["label"]

            label = int(label)

            if len(indices_1)==0 and len(indices_2)==0:
                sample["extra_ring_system"] = []
            
            elif len(indices_1) > 0 and len(indices_2) > 0 :
                sample["extra_ring_system"] = ['O=C1C(C(O)=O)=CNC2=CC=CC=C12', 'O=C1C(C(O)=O)=CNC2=NC=CC=C12']
            
            elif len(indices_1) > 0 and len(indices_2) == 0 :
                sample["extra_ring_system"] = ['O=C1C(C(O)=O)=CNC2=CC=CC=C12']
            else:
                sample["extra_ring_system"] = ['O=C1C(C(O)=O)=CNC2=NC=CC=C12']
            
            sample["extra_ring_system"] = [Chem.MolToSmiles(Chem.MolFromSmiles(_)) for _ in sample["extra_ring_system"]]
            
            ring_system_finder = uru.RingSystemFinder()
            ring_system_list_with_Rgroups = ring_system_finder.find_ring_systems(mol, as_mols=True)

            ring_system_list_with_Rgroups = [Chem.MolToSmiles(remove_specific_terminal_carbons(_)) for _ in ring_system_list_with_Rgroups]
            
            sample["ring_system"] = ring_system_list_with_Rgroups


            if psutil.virtual_memory().percent > 92.5:  # 内存使用超过80%
                gc.collect()
            
            return sample

    except Exception as e:
        print(e)
        import traceback
        traceback.print_exc()
        return None

def wrapper(data):
    try:
        return process_single(data)
    except func_timeout.exceptions.FunctionTimedOut as e:
        return None
    except Exception as e:
        print(e)
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    num_processes = 100 #cpu_count()  # 或者设置一个数字，比如 8
    # with open("/home/baolingjie/git_download/NMR_PubChem/nmr_pubchem_lmdb_key.pkl", "rb") as f:
    #     key_list = pickle.loads(f.read())
    key_list = [f'{count}'.encode('ascii') for count in range(length)]
    print(f"Using {num_processes} processes to process {length} rows")
    
    ring_system_list = []
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap_unordered(wrapper, key_list),
            total=len(key_list)
        ))
        
        count = txn_writer.stat()['entries']
        print(f"count:{count}")
        
        for i in tqdm(range(len(results))):
            if results[i] is not None:
                if "ring_system" in results[i]:
                    ring_system_list.extend(results[i]["ring_system"])
                txn_writer.put(f'{count}'.encode('ascii'), pickle.dumps(results[i], protocol=-1))
                count = count + 1
        
        with open("data/NMR_other_data_1020_add_ring_system_1028/ring_system_list.pickle" , 'wb') as f:  # 注意是二进制写入模式 'wb'
            pickle.dump(list(set(ring_system_list)), f)
        
        txn_writer.commit()

        del results
        if psutil.virtual_memory().percent > 92.5:  # 内存使用超过80%
            gc.collect()
        
        print(f"count:{count}")
    
    new_env.close()
import lmdb
from multiprocessing import Pool, Queue, Process
import time
import os
from tqdm import tqdm
import pickle
from rdkit import Chem
from rdkit.Chem import Descriptors, QED
import json
from rdkit import DataStructs
from rdkit.Chem import AllChem, MACCSkeys
import copy
from thermo.functional_groups import (
    identify_functional_groups, 
    ALL_FUNCTIONAL_GROUPS,
    FG_TO_FUNCTION
)
from rdkit.Chem import rdSubstructLibrary

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
import numpy as np
import pandas as pd

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



# ==================== 第二部分：创建新的可写LMDB ====================
new_lmdb_path = "data/NMR_other_data_1022/new_db.lmdb"  # 请修改为你的新路径
os.makedirs(os.path.dirname(new_lmdb_path), exist_ok=True)
# map_size = 1099511627776  # 1TB初始空间（根据数据量调整）

# 创建优化配置的新LMDB环境
new_env = lmdb.open(
    new_lmdb_path,
    map_size=100*(1024*1024*1024),
)
txn_writer = new_env.begin(write=True)

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
        print(f"smiles:{smiles}[Error] 处理 SMILES 时发生异常: {e}")
        return None

def standardize_smiles(smiles):
    """标准化SMILES字符串"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            # 标准化分子（移除氢，添加氢，清洁等）
            mol = Chem.AddHs(mol)
            mol = Chem.RemoveHs(mol)
            # 生成标准SMILES
            standardized_smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
            return standardized_smiles
        else:
            return None
    except:
        return None



with open("/home/baolingjie/git_download/NMR_Core_Mol_Detection/data/data_valid_test_1020/ex_train.json", "r") as f:
    ref_test_data = json.loads(f.read())

ref_test_dict = {
    ref_test_data[_]["cano_smiles"]:"" for _ in range(len(ref_test_data))
}
df0 = pd.read_csv("data/test_core_mol_clean.csv")
# 对DataFrame中的每个SMILES进行标准化
df0['cano_smiles'] = df0['SMILES'].apply(standardize_smiles)
for cano_smiles in df0['cano_smiles']:
    ref_test_dict[cano_smiles] = ""

def process_single(data):
    index, sample = data
    try:
        smiles = sample["cano_smiles"]
        if "." in smiles:
            return None
        smiles = get_cano_smiles(smiles)
        if smiles is None:
            print("smiles is None")
            return None
        if smiles in ref_test_dict:
            return None
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        if is_molecule_valid(mol) is False:
            return None
        # frag = split(encode(smiles))
        # frag_list = [split(encode(_)) for _ in temp_smiles_list]

        ecfp = AllChem.GetMorganFingerprintAsBitVect(
                mol,
                radius=2,          # ECFP4 的半径为2
                nBits=2048,        # 2048维
                useChirality=True  # 关键：启用手性感知
            )
        ecfp = np.array(ecfp)

        maccs_fp = MACCSkeys.GenMACCSKeys(mol)
        fp_bitstring = maccs_fp.ToBitString()
        maccs_fp = np.array([int(bit) for bit in fp_bitstring])

        mols = rdSubstructLibrary.CachedSmilesMolHolder()
        fps = rdSubstructLibrary.TautomerPatternHolder(1024)
        mols.AddSmiles(smiles)
        fp = Chem.PatternFingerprint(mol,fpSize=1024,tautomerFingerprints=True)
        fps.AddFingerprint(fp)
        library = rdSubstructLibrary.SubstructLibrary(mols,fps)
        indices_1 = library.GetMatches(Chem.MolFromSmiles('O=C1C(C(O)=O)=CNC2=CC=CC=C12'),maxResults=50000)
        indices_2 = library.GetMatches(Chem.MolFromSmiles('O=C1C(C(O)=O)=CNC2=NC=CC=C12'),maxResults=50000)

        if len(indices_1)>0 or len(indices_2)>0:
            label = np.array([1])
        else:
            label = np.array([0])
        
        element_list = np.array(get_element_counts_list(mol))
        function_list = np.array(functional_groups_dict_to_binary_vector(get_all_functional_groups_presence(mol)))

        split_ = "test"
        if "split" in sample:
            split_ = sample["split"]
        return {
                "key":index,
                'canonical_smiles': smiles,
                'C_nmr':sample["NMR"],
                # "frag":frag,
                # "frag_list":frag_list,
                "source":sample["source"],
                "split":split_,
                "ecfp":ecfp,
                "maccs":maccs_fp,
                "function_list":function_list,
                "label":label,
                "element_list":element_list,
                }
    except Exception as e:
        print(e)
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":

    save_dir = "/home/baolingjie/git_download/NMR_Core_Mol_Detection/data/acd_lotusv9_processed"
    with open(os.path.join(save_dir, "acd_lotusv9_processed.json"), "r") as f:
        acd_lotusv9_data = json.loads(f.read())
    
    save_dir = "/home/baolingjie/git_download/NMR_Core_Mol_Detection/data/bmrb_processed"
    with open(os.path.join(save_dir, "bmrb_processed.json"), "r") as f:
        bmrb_data = json.loads(f.read())
    
    save_dir = "/home/baolingjie/git_download/NMR_Core_Mol_Detection/data/c13_naproc_processed"
    with open(os.path.join(save_dir, "c13_naproc_processed.json"), "r") as f:
        c13_naproc_data = json.loads(f.read())
    
    # save_dir = "/home/baolingjie/git_download/NMR_Core_Mol_Detection/data/ch-nmr-np_processed"
    # with open(os.path.join(save_dir, "ch-nmr-np_processed.json"), "r") as f:
    #     ch_nmr_np_data = json.loads(f.read())
    
    save_dir = "/home/baolingjie/git_download/NMR_Core_Mol_Detection/data/GlycoNMR_processed"
    with open(os.path.join(save_dir, "GlycoNMR_processed.json"), "r") as f:
        GlycoNMR_data = json.loads(f.read())
    
    save_dir = '/home/baolingjie/git_download/NMR_Core_Mol_Detection/data/hmdb_processed/'
    with open(os.path.join(save_dir, "hmdb_processed.json"), "r") as f:
        hmdb_data = json.loads(f.read())
    
    # save_dir = '/home/baolingjie/git_download/NMR_Core_Mol_Detection/data/NMRShifted_v2_processed/'
    # with open(os.path.join(save_dir, "nmrshiftdb2_v2_processed.json"), "r") as f:
    #     nmrshiftdb2_v2_data = json.loads(f.read())
    
    save_dir = "/home/baolingjie/git_download/NMR_Core_Mol_Detection/data/pnmrnp2_processed"
    with open(os.path.join(save_dir, "pnmrnp2_processed.json"), "r") as f:
        pnmrnp2_data = json.loads(f.read())
    
    save_dir = "/home/baolingjie/git_download/NMR_Core_Mol_Detection/data/Spektraris_NMR_processed"
    with open(os.path.join(save_dir, "Spektraris_NMR_processed.json"), "r") as f:
        Spektraris_NMR_data = json.loads(f.read())
    
    save_dir = "/home/baolingjie/git_download/NMR_Core_Mol_Detection/data/data_1392980"
    with open(os.path.join(save_dir, "data_1014.json"), "r") as f:
        data_1014 = json.loads(f.read())
    
    with open("data/data_1600000_0928/train_0928.json", "r") as f:
        train_data = json.loads(f.read())
    with open("data/data_1600000_0928/val_0928.json", "r") as f:
        val_data = json.loads(f.read())
    with open("data/data_1600000_0928/test_0928.json", "r") as f:
        test_data = json.loads(f.read())
    
    train_data = [train_data[_] for _ in range(len(train_data)) if train_data[_]["source"]!="nmrshiftdb"]
    val_data = [val_data[_] for _ in range(len(val_data)) if val_data[_]["source"]!="nmrshiftdb"]
    test_data = [test_data[_] for _ in range(len(test_data)) if test_data[_]["source"]!="nmrshiftdb"]


    data_all = []
    data_all.extend(acd_lotusv9_data)
    data_all.extend(bmrb_data)
    data_all.extend(c13_naproc_data)
    data_all.extend(GlycoNMR_data)
    data_all.extend(hmdb_data)
    data_all.extend(pnmrnp2_data)
    data_all.extend(Spektraris_NMR_data)
    data_all.extend(data_1014)
    data_all.extend(train_data)
    data_all.extend(val_data)
    data_all.extend(test_data)

    process_single((-1, data_1014[-1]))

    num_processes = 80
    length = len(data_all)

    print(f"Using {num_processes} processes to process {length} rows...")

    data_list = []
    for i, data in tqdm(enumerate(data_all)):
        if "split" in data:
            split_ = data["split"]
        else:
            split_ = "train"
        if split_ == "train":
            data_list.append((i,data)) #str(key).encode('ascii')
        # if i == 1000:
        #     break
    
    ## 单线程
    # for data in data_list:
    #     process_single(data)

    temp_temp_dict = {}
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_single, data_list),
            total=len(data_list)
        ))
    # # 4. 收集有效的 key
    # valid_results = [result for idx, result in enumerate(results) if result is not None]
    key_list = []
    # frag_dict = {}
    count = 0
    for index, i in tqdm(enumerate(range(len(results)))):
        if results[i] is not None:
            key = results[i]["key"]
            key_list.append(key)
            value = results[i]
            # if "key" in value:
            #     value.pop("key")
            # frag = value["frag"]
            # for _ in frag:
            #     if _ not in frag_dict:
            #         frag_dict[_] = 1
            #     else:
            #         frag_dict[_] = frag_dict[_] + 1
            txn_writer.put(f'{count}'.encode('ascii'), pickle.dumps(value, protocol=-1))
            count = count +  1

    txn_writer.commit()
    new_env.close()
    # with open("data/NMR_PubChem_1014_con/frag_dict.json", "w") as f:
    #     f.write(json.dumps(frag_dict))
    print(count)

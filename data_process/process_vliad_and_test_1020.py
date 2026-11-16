import os
from rdkit import Chem
import json
import copy
from rdkit.Chem import Descriptors, QED
from tqdm import tqdm
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




if __name__ == "__main__":
    # valid_data_path = "/home/baolingjie/git_download/NMR_Core_Mol_Detection/data/15495052/validation_55167_upload/valid00_55167.json"
    # valid_data = []
    # with open(valid_data_path, "r") as f:
    #     for line in f.readlines():
    #         valid_data.append(json.loads(line))
    #     # valid_data = json.loads(f.read())
    
    # test_data_path = "/home/baolingjie/git_download/NMR_Core_Mol_Detection/data/15495052/test_55150_upload/test00_55150.json"
    # test_data = []
    # with open(test_data_path, "r") as f:
    #     for line in f.readlines():
    #         test_data.append(json.loads(line))

    save_dir = "/home/baolingjie/git_download/NMR_Core_Mol_Detection/data/data_1392980"
    with open(os.path.join(save_dir, "data_1014.json"), "r") as f:
        data_1014 = json.loads(f.read())
    
    valid_data = [data_1014[_] for _ in range(len(data_1014)) if "val" in data_1014[_]["split"]]
    valid_data_ = []
    for i,data in enumerate(valid_data):
        try:
            smiles = data["cano_smiles"]
            if "." in smiles:
                continue
            smiles = get_cano_smiles(smiles)
            if smiles is None:
                continue
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            if is_molecule_valid(mol) is False:
                continue
            # ecfp = AllChem.GetMorganFingerprintAsBitVect(
            #     mol,
            #     radius=2,          # ECFP4 的半径为2
            #     nBits=2048,        # 2048维
            #     useChirality=True  # 关键：启用手性感知
            # )
            # ecfp = np.array(ecfp)

            # maccs_fp = MACCSkeys.GenMACCSKeys(mol)
            # fp_bitstring = maccs_fp.ToBitString()
            # maccs_fp = np.array([int(bit) for bit in fp_bitstring])

            # mols = rdSubstructLibrary.CachedTrustedSmilesMolHolder()
            # fps = rdSubstructLibrary.TautomerPatternHolder(1024)
            # mols.AddSmiles(smiles)
            # fp = Chem.PatternFingerprint(mol,fpSize=1024,tautomerFingerprints=True)
            # fps.AddFingerprint(fp)
            # library = rdSubstructLibrary.SubstructLibrary(mols,fps)
            # indices_1 = library.GetMatches(Chem.MolFromSmiles('O=C1C(C(O)=O)=CNC2=CC=CC=C12'),maxResults=50000)
            # indices_2 = library.GetMatches(Chem.MolFromSmiles('O=C1C(C(O)=O)=CNC2=NC=CC=C12'),maxResults=50000)

            # if len(indices_1)>0 or len(indices_2)>0:
            #     label = np.array([1])
            # else:
            #     label = np.array([0])
            
            # element_list = np.array(get_element_counts_list(mol))
            # function_list = np.array(functional_groups_dict_to_binary_vector(get_all_functional_groups_presence(mol)))
            
            # valid_data[i]["ecfp"] = ecfp
            # valid_data[i]["maccs"] = maccs_fp
            # valid_data[i]["label"] = label
            # valid_data[i]["element_list"] = element_list
            # valid_data[i]["function_list"] = function_list
            valid_data_.append(valid_data[i])
        except Exception as e:
            print(e)
            import traceback
            traceback.print_exc()
        
        
    test_data = [data_1014[_] for _ in range(len(data_1014)) if "test" in data_1014[_]["split"]]
    test_data_ = []
    for i, data in enumerate(test_data):
        try:
            smiles = data["cano_smiles"]
            if "." in smiles:
                continue
            smiles = get_cano_smiles(smiles)
            if smiles is None:
                continue
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            if is_molecule_valid(mol) is False:
                continue
            # ecfp = AllChem.GetMorganFingerprintAsBitVect(
            #     mol,
            #     radius=2,          # ECFP4 的半径为2
            #     nBits=2048,        # 2048维
            #     useChirality=True  # 关键：启用手性感知
            # )
            # ecfp = np.array(ecfp)

            # maccs_fp = MACCSkeys.GenMACCSKeys(mol)
            # fp_bitstring = maccs_fp.ToBitString()
            # maccs_fp = np.array([int(bit) for bit in fp_bitstring])

            # mols = rdSubstructLibrary.CachedTrustedSmilesMolHolder()
            # fps = rdSubstructLibrary.TautomerPatternHolder(1024)
            # mols.AddSmiles(smiles)
            # fp = Chem.PatternFingerprint(mol,fpSize=1024,tautomerFingerprints=True)
            # fps.AddFingerprint(fp)
            # library = rdSubstructLibrary.SubstructLibrary(mols,fps)
            # indices_1 = library.GetMatches(Chem.MolFromSmiles('O=C1C(C(O)=O)=CNC2=CC=CC=C12'),maxResults=50000)
            # indices_2 = library.GetMatches(Chem.MolFromSmiles('O=C1C(C(O)=O)=CNC2=NC=CC=C12'),maxResults=50000)

            # if len(indices_1)>0 or len(indices_2)>0:
            #     label = np.array([1])
            # else:
            #     label = np.array([0])
            
            # element_list = np.array(get_element_counts_list(mol))
            # function_list = np.array(functional_groups_dict_to_binary_vector(get_all_functional_groups_presence(mol)))
            
            # test_data[i]["ecfp"] = ecfp
            # test_data[i]["maccs"] = maccs_fp
            # test_data[i]["label"] = label
            # test_data[i]["element_list"] = element_list
            # test_data[i]["function_list"] = function_list
            test_data_.append(test_data[i])
        except Exception as e:
            print(e)
            import traceback
            traceback.print_exc()
    
    train_data_path = "/home/baolingjie/git_download/NMR_Core_Mol_Detection/data/NMRShiftDB2/train.json"
    with open(train_data_path, "r") as f:
        ex_test_data = json.loads(f.read())
        for _ in range(len(ex_test_data)):
            ex_test_data[_]["split"] = "train"
    
    test_data_path_2 = "/home/baolingjie/git_download/NMR_Core_Mol_Detection/data/NMRShiftDB2/val.json"
    with open(test_data_path_2, "r") as f:
        ex_test_data_2 = json.loads(f.read())
        for _ in range(len(ex_test_data_2)):
            ex_test_data_2[_]["split"] = "val"
    
    ex_test_data.extend(ex_test_data_2)
    
    ex_test_data_list = []
    for i in tqdm(range(len(ex_test_data))):
        try:
            split = ex_test_data[i]["split"]
            source = "nmrshiftdb"
            smiles = ex_test_data[i]["smiles"][0]
            if "." in smiles:
                continue
            smiles = get_cano_smiles(smiles)
            if smiles is None:
                continue
            nmr = ex_test_data[i]["nmr"][0]
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            if is_molecule_valid(mol) is False:
                continue

            # ecfp = AllChem.GetMorganFingerprintAsBitVect(
            #     mol,
            #     radius=2,          # ECFP4 的半径为2
            #     nBits=2048,        # 2048维
            #     useChirality=True  # 关键：启用手性感知
            # )
            # ecfp = np.array(ecfp)

            # maccs_fp = MACCSkeys.GenMACCSKeys(mol)
            # fp_bitstring = maccs_fp.ToBitString()
            # maccs_fp = np.array([int(bit) for bit in fp_bitstring])

            # mols = rdSubstructLibrary.CachedTrustedSmilesMolHolder()
            # fps = rdSubstructLibrary.TautomerPatternHolder(1024)
            # mols.AddSmiles(smiles)
            # fp = Chem.PatternFingerprint(mol,fpSize=1024,tautomerFingerprints=True)
            # fps.AddFingerprint(fp)
            # library = rdSubstructLibrary.SubstructLibrary(mols,fps)
            # indices_1 = library.GetMatches(Chem.MolFromSmiles('O=C1C(C(O)=O)=CNC2=CC=CC=C12'),maxResults=50000)
            # indices_2 = library.GetMatches(Chem.MolFromSmiles('O=C1C(C(O)=O)=CNC2=NC=CC=C12'),maxResults=50000)

            # if len(indices_1)>0 or len(indices_2)>0:
            #     label = np.array([1])
            # else:
            #     label = np.array([0])
            
            # element_list = np.array(get_element_counts_list(mol))
            # function_list = np.array(functional_groups_dict_to_binary_vector(get_all_functional_groups_presence(mol)))

            temp_dict = {
                    "key":i,
                    # "ecfp":ecfp,
                    # "maccs":maccs_fp,
                    # "label":label,
                    # "element_list":element_list,
                    # "function_list":function_list,
                    'cano_smiles': smiles,
                    'C_nmr':nmr,
                    "source":source,
                    "split":split,
                    }

            ex_test_data_list.append(temp_dict)
        
        except Exception as e:
            print(e)
            import traceback
            traceback.print_exc()
    
    save_dir = "/home/baolingjie/git_download/NMR_Core_Mol_Detection/data/data_valid_test_1020"
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "val.json"), "w") as f:
        data = f.write(json.dumps(valid_data_))
    
    with open(os.path.join(save_dir, "test.json"), "w") as f:
        data = f.write(json.dumps(test_data_))
    
    with open(os.path.join(save_dir, "ex_test.json"), "w") as f:
        data = f.write(json.dumps(ex_test_data_list))
    
    with open(os.path.join(save_dir, "ex_train.json"), "w") as f:
        data = f.write(json.dumps([ex_test_data_list[_] for _ in range(len(ex_test_data_list)) if ex_test_data_list[_]["split"]=="train"]))
    
    with open(os.path.join(save_dir, "ex_ex_test.json"), "w") as f:
        data = f.write(json.dumps([ex_test_data_list[_] for _ in range(len(ex_test_data_list)) if ex_test_data_list[_]["split"]!="train"]))
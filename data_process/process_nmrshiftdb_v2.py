import json
from rdkit import Chem
import copy
import os
from tqdm import tqdm

def extract_nmr_shifts(nmr_data_str, return_dict=False):
    """
    从 NMREDATA_1D_13C 格式的字符串中提取化学位移（ppm）和标签（可选）。
    
    参数:
        nmr_data_str (str): NMREDATA_1D_13C 格式的字符串
        return_dict (bool): 是否返回字典（带标签），默认返回列表（仅化学位移）
    
    返回:
        如果 return_dict=False -> List[float] (化学位移列表)
        如果 return_dict=True -> Dict[str, float] (标签到化学位移的映射)
    """
    lines = nmr_data_str.split("\n")
    chemical_shifts = []
    shift_dict = {}
    
    for line in lines:
        # 跳过元数据行（如 Spectrum_Location, Larmor）
        if line.startswith(("Spectrum_Location", "Larmor")):
            continue
        
        parts = line.split(",")
        if len(parts) >= 1:
            shift_str = parts[0].strip()
            if shift_str:  # 确保不是空字符串
                try:
                    shift = float(shift_str)
                    chemical_shifts.append(shift)
                    
                    # 如果存在标签（如 "L=s0"），提取它
                    if len(parts) >= 2 and return_dict:
                        label = parts[1].strip().replace("L=", "")
                        shift_dict[label] = shift
                except ValueError:
                    continue  # 如果无法转换为 float，跳过
    
    return shift_dict if return_dict else chemical_shifts

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


suppl = Chem.SDMolSupplier('/home/baolingjie/git_download/NMR_Core_Mol_Detection/data/NMRShifted_v2/nmrshiftdb2.nmredata.sd')
save_dir = '/home/baolingjie/git_download/NMR_Core_Mol_Detection/data/NMRShifted_v2_processed/'
os.makedirs(save_dir, exist_ok=True)
if __name__ == "__main__":
    data_list = []
    for i, mol in tqdm(enumerate(suppl)):
        if mol is None:
            continue
        props = mol.GetPropsAsDict()
        if "NMREDATA_1D_13C" in props:
            shift_list = extract_nmr_shifts(props["NMREDATA_1D_13C"])
            if len(shift_list)==0:
                continue
            smiles = Chem.MolToSmiles(mol)
            smiles = get_cano_smiles(smiles)
            temp_dict = {}
            temp_dict["index"] = i
            temp_dict["cano_smiles"] = smiles
            temp_dict["NMR"] = shift_list
            temp_dict["source"] = "nmrshiftdb_v2"
            data_list.append(copy.deepcopy(temp_dict))
    
    print(len(data_list))
    with open(os.path.join(save_dir, "nmrshiftdb2_v2_processed.json"), "w") as f:
        f.write(json.dumps(data_list, indent=4))
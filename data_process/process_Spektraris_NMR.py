import os
from rdkit import Chem
from rdkit.Chem import PandasTools
import pandas as pd
import glob
import copy
import random
from tqdm import tqdm
import json

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

save_dir = "/home/baolingjie/git_download/NMR_Core_Mol_Detection/data/Spektraris_NMR_processed"
os.makedirs(save_dir, exist_ok=True)
if __name__ == "__main__":
    df0 = pd.read_csv("/home/baolingjie/git_download/NMR_Core_Mol_Detection/data/Spektraris_NMR/record_processed.csv")
    df1 = pd.read_csv("/home/baolingjie/git_download/NMR_Core_Mol_Detection/data/Spektraris_NMR/carbon.tsv",sep="\t")
    data_list = []
    for i in tqdm(df0.index):
        lab_num = df0.loc[i, "lab_num"]
        smiles = df0.loc[i, "smiles"]
        smiles = get_cano_smiles(smiles)
        if smiles is not None:
            float_positions = [float(x) for x in df1[df1["record_id"] == lab_num]["ppm"]]
            temp_dict = {}
            temp_dict["cano_smiles"] = smiles
            temp_dict["NMR"] = float_positions
            temp_dict["source"] = "Spektraris_NMR"
            data_list.append(copy.deepcopy(temp_dict))

    with open(os.path.join(save_dir, "Spektraris_NMR_processed.json"), "w") as f:
        f.write(json.dumps(data_list, indent=4))

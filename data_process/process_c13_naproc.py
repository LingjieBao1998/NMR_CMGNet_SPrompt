import json
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import copy
import ast

def get_new_smiles(old_mol):
    old_mol = copy.deepcopy(old_mol)
    len_mol = old_mol.GetNumAtoms()
    li = []
    for index in range(len_mol):
        try:
            new_smiles = Chem.MolToSmiles(old_mol, rootedAtAtom=index)
            li.append(new_smiles)
        except:
            pass
    if len(li) >= 1:
        random.shuffle(li)
        return li
    else:
        return None

save_dir = "/home/baolingjie/git_download/NMR_Core_Mol_Detection/data/c13_naproc_processed"
os.makedirs(save_dir, exist_ok=True)
if __name__ == "__main__":
    df0 = pd.read_csv("/home/baolingjie/git_download/NMR_Core_Mol_Detection/data/c13_naproc_interim/naproc_curated.csv")
    # df0["ID"] = df0["ID"].astype(np.int32)
    # df0 = df0.set_index("ID")
    with open("/home/baolingjie/git_download/NMR_Core_Mol_Detection/data/c13_naproc_interim/c13_naproc.json", "r") as f:
        ori_data = json.loads(f.read())

    data = []
    for _ in tqdm(ori_data):
        s = _["shift"]

        # ✅ 使用 ast.literal_eval 安全地将其转为 Python 列表
        try:
            original_list = ast.literal_eval(s)
        except Exception as e:
            original_list = _["shift"]

        # ✅ 再将列表中的每个元素（字符串）转为 float
        nmr_list = [float(x) for x in original_list if x!=""]
        smiles = df0.loc[_["index"], "SMILES_chiral"]
        try:
            temp_smiles_list = get_new_smiles(mol)
        except Exception as e:
            temp_smiles_list = []

        temp_dict = {}
        temp_dict["cano_smiles"] = smiles
        temp_dict["NMR"] = list(set(nmr_list))
        temp_dict["smiles_list"] = temp_smiles_list
        temp_dict["index"] = _["index"]
        temp_dict["source"] = "c13_naproc"
        data.append(copy.deepcopy(temp_dict))
    

    print(len(data))
    with open(os.path.join(save_dir, "c13_naproc_processed.json"), "w") as f:
        f.write(json.dumps(data, indent=4))
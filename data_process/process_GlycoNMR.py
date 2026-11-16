from rdkit import Chem
from rdkit.Chem import Draw
import os
from tqdm import tqdm
import copy
import random
import json
import pandas as pd
import numpy as np
from openbabel import openbabel

obConversion = openbabel.OBConversion()
# obConversion.SetInAndOutFormats("pdb", "mol2")
obConversion.SetInAndOutFormats("pdb", "smi") 

# 你的 SDF 文件路径
read_dir1 = "/home/baolingjie/git_download/NMR_Core_Mol_Detection/data/GlycoNMR/GlycoNMR.Exp_processed"
read_dir2 = "/home/baolingjie/git_download/NMR_Core_Mol_Detection/data/GlycoNMR/GlycoNMR.Exp_raw_files"
save_dir = "/home/baolingjie/git_download/NMR_Core_Mol_Detection/data/GlycoNMR_processed"
# os.makedirs(save_dir, exist_ok=True)

temp_dir = "temp_dir"
os.makedirs(temp_dir, exist_ok=True)

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

if __name__ == "__main__":
    data = []
    for i, file_name in tqdm(enumerate(os.listdir(read_dir2))):
        if file_name[-4:] == ".pdb":
            pdb_file = os.path.join(read_dir2, file_name)
            try:
                if os.path.exists(pdb_file):
                    mol = openbabel.OBMol()
                    success = obConversion.ReadFile(mol, pdb_file) 

                    if success:
                        smiles = obConversion.WriteString(mol).strip().split("\t")[0]
                        mol = Chem.MolFromSmiles(smiles)
                        try:
                            temp_smiles_list = get_new_smiles(mol)
                        except Exception as e:
                            temp_smiles_list = []

                        data_file = os.path.join(read_dir2, ".".join(file_name.split(".")[:-1]+["csv"]))
                        df = pd.read_csv(data_file) #skiprows=3
                        # header_row = df.iloc[0]  # 第 0 行是列名
                        # df = df[1:]             # 去掉第 0 行（现在是数据的第一行）
                        # df.columns = header_row # 设置列名

                        # 重置索引（可选）
                        df = df.reset_index(drop=True)
                        
                        temp_df = df[df.iloc[:,2].astype(str).str.match(r'^ C\d+$')]
                        nmr_list = list(set(temp_df.iloc[:,3].astype(np.float32).tolist()))

                        temp_dict = {}
                        temp_dict["cano_smiles"] = smiles
                        temp_dict["NMR"] = nmr_list
                        temp_dict["smiles_list"] = temp_smiles_list
                        temp_dict["index"] = i//2
                        temp_dict["source"] = "GlycoNMR"
                        temp_dict["file_name"] = os.path.basename(file_name)
                        data.append(copy.deepcopy(temp_dict))
            except Exception as e:
                print(e)
                import traceback
                traceback.print_exc()
                import ipdb
                ipdb.set_trace()
            
    print(len(data))
    with open(os.path.join(save_dir, "GlycoNMR_processed.json"), "w") as f:
        f.write(json.dumps(data, indent=4))
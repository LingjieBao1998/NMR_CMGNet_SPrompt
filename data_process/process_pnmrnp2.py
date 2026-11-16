from rdkit import Chem
from rdkit.Chem import Draw
import os
from tqdm import tqdm
import copy
import random
import json

# 你的 SDF 文件路径
sdf_file_path = "/home/baolingjie/git_download/NMR_Core_Mol_Detection/data/pnmrnp2/pnmrnp2.sdf"
save_dir = "/home/baolingjie/git_download/NMR_Core_Mol_Detection/data/pnmrnp2_processed"
os.makedirs(save_dir, exist_ok=True)


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
    # 检查文件是否存在
    if not os.path.exists(sdf_file_path):
        print(f"错误：文件不存在，请检查路径：{sdf_file_path}")
    else:
        # 使用 RDKit 的 SDMolSupplier 读取 SDF 文件
        suppl = Chem.SDMolSupplier(sdf_file_path)

        print(f"开始读取 SDF 文件：{sdf_file_path}")
        print(f"总共读取到 {len(suppl)} 个分子。\n")  # 注意：这个长度可能不准确，建议用循环计数

        valid_molecules = 0
        for i, mol in tqdm(enumerate(suppl)):
            if mol is not None:
                nmr_data_str = mol.GetProp('NMREDATA_ASSIGNMENT')
                lines = nmr_data_str.split(" \\\n")
                nmr_data = []
                for line in lines:
                    nmr_data.append(eval(line.split(", ")[1]))
                
                
                smiles = Chem.MolToSmiles(mol)
                temp_smiles_list = get_new_smiles(mol)
                
                temp_dict = {}
                temp_dict["cano_smiles"] = smiles
                temp_dict["NMR"] = list(set(nmr_data))
                temp_dict["smiles_list"] = temp_smiles_list
                temp_dict["index"] = i
                temp_dict["source"] = "pnmrnp2"
                data.append(copy.deepcopy(temp_dict))
        
        print(len(data))
        with open(os.path.join(save_dir, "pnmrnp2_processed.json"), "w") as f:
            f.write(json.dumps(data, indent=4))
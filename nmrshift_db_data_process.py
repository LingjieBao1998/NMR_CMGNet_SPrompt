import numpy as np
import os
import pickle
import argparse
from tqdm import tqdm
import json
from rdkit import Chem


if __name__ == "__main__":
    # 1. 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="这是一个简单的命令行工具示例")

    # 2. 添加参数
    parser.add_argument("--file_path", default="/mnt/e/image_caption/NMRDataBase/data/nmrshiftdb2_2018/mol_nmrshift_nmrshiftdb2_2018_C_train.pkl", help="")  # 位置参数
    parser.add_argument("--save_path", default="./data/mol_nmrshift_nmrshiftdb2_2018_C_train_processed.sdf", help="")  # 位置参数

    # 3. 解析参数
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    with open(args.file_path, "rb") as f:
        data = pickle.load(f) #dict
    print(f"data.keys():{data.keys()}")
    #dict_keys(['mol', 'atom_target', 'atom_mask', 'smiles', 'db_id', 'Inchi'])
    print(f"len(data['mol']):{len(data['mol'])}")

    mol_list = []
    for i in tqdm(range(len(data['mol']))):
        mol = data['mol'][i]
        if mol is not None:
            atom_target = data['atom_target'][i][1:]
            atom_mask = data['atom_mask'][i][1:]
            # smiles = data['smiles'][i]
            # Inchi = data['Inchi'][i]

            temp_dict = {}
            for temp_i, atom in enumerate(mol.GetAtoms()):
                if atom.GetSymbol() == "C":
                    if atom_mask[temp_i] != 0:
                        temp_dict[temp_i] = atom_target[temp_i]
                        atom.SetProp("shfit", f'{atom_target[temp_i]}')
            
            if len(temp_dict)>0:
                mol.SetProp("shfit", f'{json.dumps(temp_dict)}')
                # mol.SetProp("smiles", f'{smiles}')
                # mol.SetProp("Inchi", f'{Inchi}')
                try:
                    mol.SetProp("smiles", f'{Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(mol)))}')
                    mol_list.append(mol)
                except Exception as e:
                    print(e)
    
    print(f'len(mol_list):{len(mol_list)}')
    with Chem.SDWriter(args.save_path) as writer:
        for mol in mol_list:
            if mol is not None:  # 安全判断，避免写入无效分子
                writer.write(mol)




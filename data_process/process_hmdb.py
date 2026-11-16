import os
from rdkit import Chem
from rdkit.Chem import PandasTools
import pandas as pd
import glob
import copy
import random
from tqdm import tqdm
import json

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

def get_all_carbons_have_h_fn(mol):
    all_carbons_have_h = True  # 先假设所有碳都有氢

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 6:  # 是碳原子
            h_count = atom.GetTotalNumHs()  # 获取连接的氢总数（显式+隐式）
            if h_count == 0:
                # print(f"❌ 碳原子 (idx={atom.GetIdx()}) 没有连接任何氢原子")
                all_carbons_have_h = False
                break
            # else:
            #     print(f"✅ 碳原子 (idx={atom.GetIdx()}, H数={h_count}) 有氢")
    return all_carbons_have_h


# 你的 SDF 文件路径
sdf_file = '/home/baolingjie/git_download/NMR_Core_Mol_Detection/data/hmdb/structures.sdf'
csv_file = "/home/baolingjie/git_download/NMR_Core_Mol_Detection/data/hmdb/structures.csv"
search_dir = '/home/baolingjie/git_download/NMR_Core_Mol_Detection/data/hmdb/'
save_dir = '/home/baolingjie/git_download/NMR_Core_Mol_Detection/data/hmdb_processed/'
os.makedirs(save_dir, exist_ok=True)

if __name__ == "__main__":
    # 方法 1：使用 PandasTools.LoadSDF —— 直接转成 DataFrame（推荐！）
    # 这个函数会读取 sdf 中的分子和大部分属性，生成一个 pandas DataFrame
    # df_molecules = PandasTools.LoadSDF(
    #                 sdf_file,
    #                 includeFingerprints=False # 是否包含指纹（一般False即可）
    #                 )
    
    # supplier = Chem.SDMolSupplier(sdf_file)
    # df_all = pd.read_csv(csv_file)
    # print(len(supplier))

    df_all = pd.read_csv(csv_file)
    print(len(df_all))
    
    database_id_list = []
    for i in tqdm(df_all.index):
    
        database_id_list.append(df_all.loc[i, "DATABASE_ID"])
        # all_carbons_have_h = get_all_carbons_have_h_fn(mol)
        # if all_carbons_have_h:
            # database_id_list.append(database_id)

    with open(os.path.join(save_dir, "database_id_list.json"), "w") as f:
        f.write(json.dumps(database_id_list, indent=4))


    data = []
    for i in tqdm(df_all.index):

        database_id = df_all.loc[i, "DATABASE_ID"]
        matching_files = glob.glob(os.path.join(search_dir, f"{database_id}_nmrtwod_*"))
        if len(matching_files) == 0:
            continue
        try:
            smiles = df_all.loc[i, "SMILES"]
            mol = Chem.MolFromSmiles(smiles)
            Chem.RemoveHs(mol)
        except Exception as e:
            mol = None
        
        smiles = Chem.MolToSmiles(mol)
        
        if mol is not None:
            all_carbons_have_h = get_all_carbons_have_h_fn(mol)
            if all_carbons_have_h:
                try:
                    df = pd.read_csv(matching_files[0],sep="\t")
                except Exception as e:
                    continue
                if df.values.shape[-1] == 1:
                    nmr_data = []
                    with open(matching_files[0], "r") as f:
                        for line in f.readlines():
                            print(line)
                            temp_list = line.split()
                            if len(temp_list)==6:
                                try:
                                    nmr_data.append(float(temp_list[-2]))
                                except Exception as e:
                                    print(e)
                    
                    if len(nmr_data) == 0:
                        with open(matching_files[0], "r") as f:
                            for line in f.readlines():
                                print(line)
                                temp_list = line.split()
                                if len(temp_list)==4:
                                    try:
                                        nmr_data.append(float(temp_list[-2]))
                                    except Exception as e:
                                        print(e)
                    if len(nmr_data) == 0:
                        import ipdb
                        ipdb.set_trace()
                else:
                    try:
                        nmr_data = list(set(df["F1ppm"].tolist()))
                    except Exception as e:
                        print(e)
                        nmr_data = list(set(df[" F1ppm"].tolist()))


                # smiles = Chem.MolToSmiles(mol)
                temp_smiles_list = get_new_smiles(mol)
                
                temp_dict = {}
                temp_dict["cano_smiles"] = smiles
                temp_dict["NMR"] = list(set(nmr_data))
                temp_dict["smiles_list"] = temp_smiles_list
                temp_dict["index"] = i
                temp_dict["source"] = "hmdb"
                data.append(copy.deepcopy(temp_dict))
        
    print(len(data))
    with open(os.path.join(save_dir, "hmdb_processed.json"), "w") as f:
        f.write(json.dumps(data, indent=4))
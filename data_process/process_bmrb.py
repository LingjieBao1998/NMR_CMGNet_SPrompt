import json
import rdkit
from rdkit import Chem
import os
from tqdm import tqdm
import pandas as pd
import copy
import random

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


sdf_dir = "/home/baolingjie/git_download/NMR_Core_Mol_Detection/data/bmrb"
save_dir = "/home/baolingjie/git_download/NMR_Core_Mol_Detection/data/bmrb_processed"
os.makedirs(save_dir, exist_ok=True)
if __name__ == "__main__":

    # with open("/home/baolingjie/git_download/NMR_Core_Mol_Detection/data/bmrb_interim/bmrb.json", "r") as f:
    #     ori_data = json.loads(f.read())
    # ori_data = [_ for _ in ori_data if _ != None]
    data = []
    for i, file_name in tqdm(enumerate(os.listdir(sdf_dir))):
        try:
            if file_name[-4:] == ".sdf":
                continue
            
            temp_df = pd.read_csv(os.path.join(sdf_dir, file_name))
            temp_df_ = pd.DataFrame(json.loads(temp_df["temp_df"].loc[0]))
            temp_df_ = temp_df_[temp_df_.iloc[:,0].astype(str).str.match(r'^C\d+$')]
            sdf_path = os.path.join(sdf_dir, temp_df["mol_file"].loc[0])
            suppl = Chem.SDMolSupplier(sdf_path)
            if len(temp_df_)>0 and len(suppl)==1:
                nmr_list = list(set(temp_df_.iloc[:,2].tolist()))
                mol = next(suppl)
                smiles = Chem.MolToSmiles(mol)
                temp_smiles_list = get_new_smiles(mol)

                temp_dict = {}
                temp_dict["cano_smiles"] = smiles
                temp_dict["NMR"] = list(set(nmr_list))
                temp_dict["smiles_list"] = temp_smiles_list
                temp_dict["index"] = i
                temp_dict["source"] = "bmrb"
                temp_dict["file_name"] = file_name
                data.append(copy.deepcopy(temp_dict))
                
        except Exception as e:
            print(e)
            import traceback
            traceback.print_exc()
        
    print(len(data))
    with open(os.path.join(save_dir, "bmrb_processed.json"), "w") as f:
        f.write(json.dumps(data, indent=4))


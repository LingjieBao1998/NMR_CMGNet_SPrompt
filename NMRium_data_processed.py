import os
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from pandarallel import pandarallel
pandarallel.initialize(30) 
import time

from rdkit.Chem import rdSubstructLibrary

def get_library(df):
    data = []
    t1 = time.time()
    for idx in tqdm(df.index):
        smi = df.loc[idx, "clean_cano_smiles"]
        mol = Chem.MolFromSmiles(smi)
        if not ((idx+1)%50000):
            print(f"Processed {idx+1} molecules in {(time.time()-t1):.1f} seconds")
        if mol is None: #or mol.GetNumAtoms()>50
            continue
        fp = Chem.PatternFingerprint(mol,fpSize=1024,tautomerFingerprints=True)
        data.append((smi,fp))

    print(f"Processed all molecules in {(time.time()-t1):.1f} seconds")
    mols = rdSubstructLibrary.CachedSmilesMolHolder()
    fps = rdSubstructLibrary.TautomerPatternHolder(1024)
    for smi,fp in data:
        mols.AddSmiles(smi)
        fps.AddFingerprint(fp)
    library = rdSubstructLibrary.SubstructLibrary(mols,fps)
    return library

def strard_mol_fn(smiles):
    try:
        new_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(Chem.MolFromSmiles(smiles))))
        return new_smiles
    except Exception as e:
        print(e)
        return ""

def standardize_mol(smiles):
    try:
        # follows the steps in
        # https://github.com/greglandrum/RSC_OpenScience_Standardization_202104/blob/main/MolStandardize%20pieces.ipynb
        # as described **excellently** (by Greg) in
        # https://www.youtube.com/watch?v=eWTApNX8dJQ
        mol = Chem.MolFromSmiles(smiles)
        
        # removeHs, disconnect metal atoms, normalize the molecule, reionize the molecule
        clean_mol = rdMolStandardize.Cleanup(mol) 
        
        # if many fragments, get the "parent" (the actual mol we are interested in) 
        parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)
        
        # try to neutralize molecule
        uncharger = rdMolStandardize.Uncharger() # annoying, but necessary as no convenience method exists
        uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)
        
        # note that no attempt is made at reionization at this step
        # nor at ionization at some pH (rdkit has no pKa caculator)
        # the main aim to to represent all molecules from different sources
        # in a (single) standard way, for use in ML, catalogue, etc.
        
        return Chem.MolToSmiles(uncharged_parent_clean_mol)

        # te = rdMolStandardize.TautomerEnumerator() # idem
        # taut_uncharged_parent_clean_mol = te.Canonicalize(uncharged_parent_clean_mol)
        
        # return Chem.MolToSmiles(taut_uncharged_parent_clean_mol)
    except Exception as e:
        return ""


if __name__ == "__main__":
    data = []
    file_dir = "./data/15495052/train_2137771_upload/"
    for file_name in os.listdir(file_dir):
        file_path = os.path.join(file_dir, file_name)
        with open(file_path, "r") as f:
            for temp_data in tqdm(f.readlines()):
                temp_data = json.loads(temp_data)
                carbon_data = temp_data["13C_NMR"]
                float_values = [round(float(item.replace('C_', '')), 1) for item in carbon_data]
                data.append(
                    {
                    "smiles":temp_data["smiles"],
                    "NMR":float_values,
                    }
                )

        break
    
    ms = []
    df0 = pd.DataFrame(data)
    df0["clean_cano_smiles"] = df0["smiles"].parallel_apply(standardize_mol)
    df0 = df0[df0["clean_cano_smiles"]!=""]
    df1 = df0.drop_duplicates(subset="clean_cano_smiles")
    df1 = df1.reset_index(drop=True)

    df1.to_csv("/home/baolingjie/git_download/NMR_Core_Mol_Detection/data/15495052_v1.csv")

    library = get_library(df1)

    qry = Chem.MolFromSmarts('OC(C1=CNC2=[N,C]C=CC=C2C1=O)=O')
    qry = Chem.MolFromSmarts(Chem.MolToSmarts(Chem.MolFromSmarts(Chem.MolToSmarts(qry))))

    mids = library.GetMatches(qry, numThreads=10)
    ms += [library.GetMolHolder().GetMol(x) for x in mids]

    qry = Chem.MolFromSmiles('OC(C1=CNC2=CC=CC=C2C1=O)=O')
    qry = Chem.MolFromSmiles(Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(qry))))
    mids = library.GetMatches(qry, numThreads=10)
    ms += [library.GetMolHolder().GetMol(x) for x in mids]

    qry = Chem.MolFromSmiles('OC(C1=CNC2=NC=CC=C2C1=O)=O')
    qry = Chem.MolFromSmiles(Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(qry))))
    mids = library.GetMatches(qry, numThreads=10)
    ms += [library.GetMolHolder().GetMol(x) for x in mids]


    from rdkit import Chem

    # 假设你的 Mol 对象列表叫做 mols
    # mols = [mol1, mol2, mol3, ...]  # 这是你贴出来的那些 Mol 对象

    # 指定输出的 SDF 文件名，比如：
    output_sdf_path = "/home/baolingjie/git_download/NMR_Core_Mol_Detection/data/15495052_v1_query.sdf"

    # 创建一个 SDWriter 对象，用于写入 SDF 文件
    writer = Chem.SDWriter(output_sdf_path)

    # 遍历每个 Mol 对象，并写入文件
    for mol in ms:
        writer.write(mol)

    # 关闭 writer，确保所有数据写入磁盘
    writer.close()

    print(f"✅ 已成功将 {len(ms)} 个分子保存到 {output_sdf_path}")
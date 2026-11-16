import os
import json
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from tqdm import tqdm
import time
import pickle

from rdkit.Chem import rdSubstructLibrary

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
        
        return Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(uncharged_parent_clean_mol)))

        # te = rdMolStandardize.TautomerEnumerator() # idem
        # taut_uncharged_parent_clean_mol = te.Canonicalize(uncharged_parent_clean_mol)
        
        # return Chem.MolToSmiles(taut_uncharged_parent_clean_mol)
    except Exception as e:
        return ""

def kekulize_fn(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        else:
            return Chem.MolToSmiles(mol, kekuleSmiles=True)
    except Exception as e:
        return ""

def get_library(df):
    data = []
    index_list = []
    t1 = time.time()
    for idx in tqdm(df.index):
        smi = df.loc[idx, "kekule_smiles"]
        mol = Chem.MolFromSmiles(smi)
        if not ((idx+1)%50000):
            print(f"Processed {idx+1} molecules in {(time.time()-t1):.1f} seconds")
        if mol is None: #or mol.GetNumAtoms()>50
            index_list.append(idx)
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
    return library, index_list

if __name__ == "__main__":

    if True:#os.path.exists("/home/baolingjie/git_download/NMR_Core_Mol_Detection/data/df_all_0813.csv") is False:
        ## 训练集
        df0 = pd.read_csv("./data/positive_remove_duplicated_nmr_processed.csv", index_col=0)
        df0["clean_cano_smiles"] = df0["Cano_SMILES"].apply(standardize_mol)
        df0 = df0[df0["clean_cano_smiles"]!=""]
        df0 = df0.reset_index(drop=True)
        df0["source"] = "NMRium"
        df1 = pd.read_csv("./data/15495052_v1.csv", index_col=0)
        df1["source"] = "NMRium"

        ## 测试集
        ## 单独阳性测试集
        df2 = pd.read_csv("./data/test_core_mol_nmr_mae.csv", index_col=0)
        df2["clean_cano_smiles"] = df2["SMILES"].apply(standardize_mol)
        df2 = df2[df2["clean_cano_smiles"]!=""]
        df2 = df2.reset_index(drop=True)
        df2["source"] = "experiment"
        
        df3 = pd.read_json("./data/train.json")
        df3["smiles_first"] = df3["smiles"].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None)
        df3["NMR"] = df3["nmr"].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None)
        df3["clean_cano_smiles"] = df3["smiles_first"].apply(standardize_mol)
        df3 = df3[df3["clean_cano_smiles"]!=""]
        df3 = df3.reset_index(drop=True)
        df4 = pd.read_json("./data/val.json")
        df4["smiles_first"] = df4["smiles"].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None)
        df4["NMR"] = df4["nmr"].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None)
        df4["clean_cano_smiles"] = df4["smiles_first"].apply(standardize_mol)
        df4 = df4[df4["clean_cano_smiles"]!=""]
        df4 = df4.reset_index(drop=True)
        
        df3["source"]="NMRShiftDB"
        df4["source"]="NMRShiftDB"

        df0_ = df0[["clean_cano_smiles", "NMR(prediction)", "source"]]
        df0_.columns = ["clean_cano_smiles", "NMR", "source"]

        df1_ = df1[["clean_cano_smiles", "NMR", "source"]]
        df1_.columns = ["clean_cano_smiles", "NMR", "source"]

        df2_ = df2[["clean_cano_smiles", "NMR1_Peaks", "source"]]
        df2_.columns = ["clean_cano_smiles", "NMR", "source"]

        df3_ = df3[["clean_cano_smiles", "NMR", "source"]]
        df3_.columns = ["clean_cano_smiles", "NMR", "source"]

        df4_ = df4[["clean_cano_smiles", "NMR", "source"]]
        df4_.columns = ["clean_cano_smiles", "NMR", "source"]

        df_all = pd.concat([df0_, df1_, df3_, df4_, df2_])
        df_all.to_csv("/home/baolingjie/git_download/NMR_Core_Mol_Detection/data/df_all_0813.csv")

        df_all = df_all.drop_duplicates(subset="clean_cano_smiles", keep="last")
        df_all = df_all.reset_index(drop=True)
        df_all.to_csv("/home/baolingjie/git_download/NMR_Core_Mol_Detection/data/df_all_0813.csv")
    
    else:
        df_all = pd.read_csv("/home/baolingjie/git_download/NMR_Core_Mol_Detection/data/df_all_0813.csv")
    
    df_all["kekule_smiles"] = df_all["clean_cano_smiles"].apply(kekulize_fn)
    df_all = df_all[df_all["kekule_smiles"]!=""]
    df_all = df_all.reset_index(drop=True)
    df_all.to_csv("/home/baolingjie/git_download/NMR_Core_Mol_Detection/data/df_all_0813.csv")
    
    library_save_path = "/home/baolingjie/git_download/NMR_Core_Mol_Detection/data/df_all_0813_ssslib.pkl"
    if True:
        library, index_list = get_library(df_all)
        df_all = df_all.loc[[i for i in df_all.index if i not in index_list]]
        df_all = df_all.reset_index(drop=True)
        df_all.to_csv("/home/baolingjie/git_download/NMR_Core_Mol_Detection/data/df_all_0813.csv")
        
        pickle.dump(library,open(library_save_path,'wb+'))
    else:
        with open(library_save_path,'rb') as file:
            library = pickle.load(file)

    ms = []
    qry = Chem.MolFromSmarts('OC(C1=CNC2=[N,C]C=CC=C2C1=O)=O')
    qry = Chem.MolFromSmarts(Chem.MolToSmarts(Chem.MolFromSmarts(Chem.MolToSmarts(qry))))
    mids = library.GetMatches(qry, numThreads=10)
    ms += [Chem.MolToSmiles(library.GetMolHolder().GetMol(x)) for x in mids]

    qry = Chem.MolFromSmiles('OC(C1=CNC2=CC=CC=C2C1=O)=O')
    mids = library.GetMatches(qry, numThreads=10)
    ms += [Chem.MolToSmiles(library.GetMolHolder().GetMol(x)) for x in mids]

    qry = Chem.MolFromSmiles('OC(C1=CNC2=CC=CC=C2C1=O)=O')
    qry = Chem.MolFromSmiles(Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(qry))))
    mids = library.GetMatches(qry, numThreads=10)
    ms += [Chem.MolToSmiles(library.GetMolHolder().GetMol(x)) for x in mids]

    qry = Chem.MolFromSmiles('OC(C1=CNC2=NC=CC=C2C1=O)=O')
    qry = Chem.MolFromSmiles(Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(qry))))
    mids = library.GetMatches(qry, numThreads=10)
    ms += [Chem.MolToSmiles(library.GetMolHolder().GetMol(x)) for x in mids]

    qry = Chem.MolFromSmiles('OC(C1=CNC2=NC=CC=C2C1=O)=O')
    mids = library.GetMatches(qry, numThreads=10)
    ms += [Chem.MolToSmiles(library.GetMolHolder().GetMol(x)) for x in mids]

    qry = Chem.MolFromSmarts('O=C(O)c1c(=O)c(cccn2)c2nc1')
    # qry = Chem.MolFromSmiles(Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(qry))))
    mids = library.GetMatches(qry, numThreads=10)
    ms += [Chem.MolToSmiles(library.GetMolHolder().GetMol(x)) for x in mids]

    qry = Chem.MolFromSmarts("O=C(c1cnc2c(CC=CC2)c1=O)O")
    mids = library.GetMatches(qry, numThreads=10)
    ms += [Chem.MolToSmiles(library.GetMolHolder().GetMol(x)) for x in mids]

    ms = list(set(ms))

    df_all['label'] = df_all['kekule_smiles'].apply(lambda x: 1 if x in ms else 0)
    df_all.to_csv("/home/baolingjie/git_download/NMR_Core_Mol_Detection/data/df_all_0813.csv")
    import ipdb
    ipdb.set_trace()
    ## 测试集去重    
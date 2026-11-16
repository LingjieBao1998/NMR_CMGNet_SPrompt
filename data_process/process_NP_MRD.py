import os
import pandas as pd
import numpy as np
from rdkit import Chem
import pandas as pd
import copy
from tqdm import tqdm

# 指定 SDF 文件路径
save_dir = "data/NP_MRD"
os.makedirs(save_dir, exist_ok=True)
if __name__ == "__main__":
    data_list = []
    index = 0
    error_list = []
    for i in range(7):
        sdf_file = f"/home/baolingjie/git_download/NMR_Core_Mol_Detection/data/structures_NP00{i*50000+1:05d}_NP00{(i+1)*50000:05d}.sdf"
        filr_dir = f"/home/baolingjie/git_download/NMR_Core_Mol_Detection/data/NP-MRD_nmr_peak_lists_NP00{i*50000+1:05d}_NP00{(i+1)*50000:05d}"
        # 使用 SDMolSupplier 读取 SDF
        supplier = Chem.SDMolSupplier(sdf_file)
        for mol in tqdm(supplier):
            try:
                prop_dict = mol.GetPropsAsDict()
                if 'NP-MRD_ID' in prop_dict:
                    NP_MRD_ID = prop_dict['NP-MRD_ID']
                    smiles = prop_dict['SMILES']
                    smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
                    temp_file_path_list = [file_path for file_path in os.listdir(filr_dir) if f"{NP_MRD_ID}_nmroned" in file_path]
                    for temp_file_path in temp_file_path_list:
                        if temp_file_path in ['NP0000040_nmroned_5185_2733540.txt']:
                            continue
                        if  temp_file_path.split(".")[-1].lower()=="csv":
                            df0 = pd.read_csv(os.path.join(filr_dir, temp_file_path))
                            if "ppm" in df0.columns:
                                if  df0["ppm"].max()>20:
                                    print(1)
                                    import ipdb
                                    ipdb.set_trace()
                            else:
                                print(2)
                                import ipdb
                                ipdb.set_trace()
                        else:
                            try:
                                df0 = pd.read_csv(os.path.join(filr_dir, temp_file_path), header=None)
                            except:
                                df0 = pd.read_csv(os.path.join(filr_dir, temp_file_path))
                            if len(df0.columns) == 5:
                                df0.columns = ["element", "index", "ppm", "peak", "None"]
                                C_NMR = df0[df0["element"]=="C"]["ppm"].tolist()
                                
                                temp_dict = {}
                                temp_dict["index"] = index
                                temp_dict["source"] = "NP-MRD"
                                temp_dict["smiles"] = smiles
                                temp_dict["NP_MRD_ID"] = NP_MRD_ID
                                temp_dict["C_NMR"] = C_NMR
                                data_list.append(copy.deepcopy(temp_dict))

                                index = index + 1
                            elif len(df0.columns) == 1:
                                df0 = pd.read_csv(os.path.join(filr_dir, temp_file_path), header=None, sep="\t")
                                if "shift" not in df0.columns and "ppm" not in df0.columns:
                                    if "shift" not in df0.loc[0] or "ppm" not in df0.loc[0]:
                                        df0 = df0.set_axis(df0.iloc[0], axis=1).drop(df0.index[0]).reset_index(drop=True)

                                        if "shift" in df0.columns:
                                            C_NMR = df0["shift"].tolist()
                                            C_NMR = [float(_) for _ in C_NMR]
                                            if np.max(C_NMR)>20:
                                                C_NMR = df0[df0["element"]=="C"]["ppm"].tolist()
                                                temp_dict = {}
                                                temp_dict["index"] = index
                                                temp_dict["source"] = "NP-MRD"
                                                temp_dict["smiles"] = smiles
                                                temp_dict["NP_MRD_ID"] = NP_MRD_ID
                                                temp_dict["C_NMR"] = C_NMR
                                                data_list.append(copy.deepcopy(temp_dict))
                                                index = index + 1
                                            else:
                                                pass
                                        
                                        elif "ppm" in df0.columns:
                                            C_NMR = df0["shift"].tolist()
                                            if np.max(C_NMR)>20:
                                                C_NMR = df0[df0["element"]=="C"]["ppm"].tolist()
                                                temp_dict = {}
                                                temp_dict["index"] = index
                                                temp_dict["source"] = "NP-MRD"
                                                temp_dict["smiles"] = smiles
                                                temp_dict["NP_MRD_ID"] = NP_MRD_ID
                                                temp_dict["C_NMR"] = C_NMR
                                                data_list.append(copy.deepcopy(temp_dict))
                                                index = index + 1
                                            else:
                                                pass
                                        else:
                                            print(5)
                                            import ipdb
                                            ipdb.set_trace(0)

                                    else:
                                        print(6)
                                        import ipdb
                                        ipdb.set_trace()


                                else:
                                    print(4)
                                    import ipdb
                                    ipdb.set_trace()
                            else:
                                print(3)
                                import ipdb
                                ipdb.set_trace()

            except Exception as e:
                print(e)
                import traceback
                traceback.print_exc()
                temp_dict["index"] = index
                temp_dict["source"] = "NP-MRD"
                temp_dict["smiles"] = smiles
                temp_dict["NP_MRD_ID"] = NP_MRD_ID
                temp_dict["file_path"] = os.path.join(filr_dir, temp_file_path)
                error_list.append(copy.deepcopy(temp_dict))
                print(os.path.join(filr_dir, temp_file_path))

                # import ipdb
                # ipdb.set_trace()
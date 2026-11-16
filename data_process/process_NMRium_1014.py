import json
from rdkit import Chem
from tqdm import tqdm
import copy
import random
from rdkit.Chem import rdSubstructLibrary
from multiprocessing import Pool, cpu_count
import os

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

temp_dict_list = []
for file_path in [
    # "/home/baolingjie/git_download/NMR_Core_Mol_Detection/data/15495052/train_2137771_upload/train01_427554.json",
    "/home/baolingjie/git_download/NMR_Core_Mol_Detection/data/15495052/train_2137771_upload/train02_427554.json",
    "/home/baolingjie/git_download/NMR_Core_Mol_Detection/data/15495052/train_2137771_upload/train03_427554.json",
    "/home/baolingjie/git_download/NMR_Core_Mol_Detection/data/15495052/train_2137771_upload/train04_427555.json",
    "/home/baolingjie/git_download/NMR_Core_Mol_Detection/data/15495052/validation_55167_upload/valid00_55167.json",
    "/home/baolingjie/git_download/NMR_Core_Mol_Detection/data/15495052/test_55150_upload/test00_55150.json",
]:
    with open(file_path, "r") as f:
        for line in tqdm(f.readlines()):
            temp_dict_list.append(json.loads(line))
            if "valid" in file_path:
                temp_dict_list[-1]["split"] = "valid"
            elif "test" in file_path:
                temp_dict_list[-1]["split"] = "test"
            else:
                temp_dict_list[-1]["split"] = "train"


def process_single(batch):
    index, data = batch
    try:
        smiles = data["smiles"]
        mol = None
        try:
            mol = Chem.MolFromSmiles(smiles)
        except Exception as e:
            print(e)
            return None
        
        if mol is None:
            return None
        
        smiles = get_cano_smiles(smiles)
        # temp_smiles_list = get_new_smiles(mol)
        NMR = data["13C_NMR"]
        split = data["split"]
        NMR = [float(_.split("_")[1]) for _ in NMR]
        # mols = rdSubstructLibrary.CachedTrustedSmilesMolHolder()
        # fps = rdSubstructLibrary.TautomerPatternHolder(1024)
        # mols.AddSmiles(smiles)
        # fp = Chem.PatternFingerprint(mol,fpSize=1024,tautomerFingerprints=True)
        # fps.AddFingerprint(fp)
        # library = rdSubstructLibrary.SubstructLibrary(mols,fps)
        # indices_1 = library.GetMatches(Chem.MolFromSmiles('O=C1C(C(O)=O)=CNC2=CC=CC=C12'),maxResults=50000)
        # indices_2 = library.GetMatches(Chem.MolFromSmiles('O=C1C(C(O)=O)=CNC2=NC=CC=C12'),maxResults=50000)

        # if len(indices_1)>0 or len(indices_2):
        #     label = 1
        # else:
        #     label = 0
        return {
            "index":index,
            "cano_smiles":smiles,
            "NMR":NMR,
            # "smiles_list":temp_smiles_list,
            "source":"NMRium",
            # "label":label,
            "split":split
        }
    except Exception as e:
        print(e)
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # result_0 = process_single((0, temp_dict_list[0]))
    # import ipdb
    # ipdb.set_trace()

    # 使用多进程 Pool
    num_processes = 140 #cpu_count()  # 或者设置一个数字，比如 8
    print(f"Using {num_processes} processes to process {len(temp_dict_list)} rows...")
    data_list = []
    for i, data in enumerate(temp_dict_list):
        data_list.append((i, data))


    # 方法：使用 Pool.imap_unordered + tqdm 显示进度条（推荐）
    with Pool(processes=num_processes) as pool:
        # 使用 imap_unordered 更快，但顺序可能乱（如果顺序重要，用 imap）
        # results = list(tqdm(pool.imap_unordered(
        #     lambda i: process_row(i), indices
        # ), total=len(indices)))

        results = list(tqdm(
            pool.imap_unordered(process_single, data_list),
            total=len(temp_dict_list)
        ))

    last_result = []
    for res in results:
        if res is not None:
            last_result.append(res)

    save_dir = "data/data_1392980"
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "data_1014.json"), "w") as f:
        f.write(json.dumps(last_result, indent=4))


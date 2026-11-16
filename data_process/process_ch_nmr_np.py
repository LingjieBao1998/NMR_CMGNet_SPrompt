import json
import os
from rdkit import Chem
import rdkit
import os
import requests
from tqdm import tqdm
import random
import copy
import pandas as pd

from urllib.request import urlopen
from urllib.error import URLError, HTTPError
import cirpy


def get_smiles_from_cas_cactus(cas):
    """
    根据 CAS 号，使用 Cactus NCI 服务获取对应的 SMILES 字符串
    
    参数:
        cas (str): 化合物的 CAS 号，例如 '108-95-2'
    
    返回:
        str: 对应的 SMILES，如果请求失败或 CAS 无效则返回 None
    """
    base_url = 'http://cactus.nci.nih.gov/chemical/structure/'
    url = f"{base_url}{cas}/smiles"
    
    try:
        with urlopen(url) as response:
            smiles = response.read().decode('utf8').strip()
            return smiles
    except HTTPError as e:
        print(f"HTTP 错误，CAS {cas} 可能无效或服务不可用: {e.code} - {e.reason}")
    except URLError as e:
        print(f"URL 请求错误（网络或服务问题）: {e.reason}")
    except Exception as e:
        print(f"获取 SMILES 时发生未知错误: {e}")
    
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


def get_cid_from_cas(cas):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{cas}/cids/JSON"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        cids = data.get('IdentifierList', {}).get('CID', [])
        if cids:
            return cids[0]  # 返回第一个 CID
        else:
            return None
    except Exception as e:
        print(f"Error querying PubChem for CAS {cas}: {e}")
        return None

def get_smiles_from_cid(cid):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/CanonicalSMILES/JSON"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        smiles = data.get('PropertyTable', {}).get('Properties', [{}])[0].get('ConnectivitySMILES')
        return smiles
    except Exception as e:
        print(f"Error getting SMILES for CID {cid}: {e}")
        return None

def get_smiles_from_cas(cas):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{cas}/property/CanonicalSMILES/JSON"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        smiles = data.get('PropertyTable', {}).get('Properties', [{}])[0].get('ConnectivitySMILES')
        return smiles
    except Exception as e:
        print(f"Error getting SMILES for CID {cid}: {e}")
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


data_file = "/home/baolingjie/git_download/NMR_Core_Mol_Detection/data/ch-nmr-np_interim/ch-nmr-np.json"
save_dir = "/home/baolingjie/git_download/NMR_Core_Mol_Detection/data/ch-nmr-np_processed"
os.makedirs(save_dir, exist_ok=True)


if __name__ == "__main__":
    # smiles = get_smiles_from_cid('55950-33-9')
    # result = get_smiles_from_cid(get_cid_from_cas('55950-33-9'))
    with open(data_file, "r") as f:
        data_ori = json.loads(f.read()) 

    data = []
    error_list = []

    for i, _ in tqdm(enumerate(data_ori)):
        try:
            cas = _['CAS Registry No.']
            smiles = get_smiles_from_cas(cas)
        except Exception as e:
            if "Chemical Name" in _:
                if  _['Chemical Name'].split("\n")[0] == '3beta,22beta-dihydroxy-olean-12-en-29-oic acid':
                    smiles = "CC1(C)[C@@H](O)CC[C@]2(C)[C@H]3CC=C4[C@@H]5C[C@](C)(C(=O)O)C[C@@H](O)[C@]5(C)CC[C@@]4(C)[C@]3(C)CC[C@@H]12"
                else:
                    print(f"error {i}")
                    error_list.append(_)
                    continue
            else:
                if _["Name"] == 'diosgenyl 3,4,6-tri-O-acetyl-2-deoxy-2-tetrachrolophthalimido-beta-D-glucopyranoside':
                    smiles = "CC(=O)OCC1OC(OC2C(OCC3=CC=CC=C3)OC3COC(OC3C2OCC2=CC=CC=C2)C2=CC=CC=C2)C(C(OC(C)=O)C1OC(C)=O)N1C(=O)C2=C(C=CC=C2)C1=O"
                elif _["Name"] == "Brazilide A":
                    smiles = "C=12C(=CC(O)=CC1)OC[C@]3(C2=C4C(C3)=CC(=O)C(=C4)O)O"
                elif _["Name"] == 'Secoorthosiphol A':
                    smiles = "C[C@@]12[C@@]3([C@@](O)([C@H](OC(C)=O)C[C@]1(C(C)(C)[C@H](O)[C@H](OC(C)=O)[C@@H]2OC(=O)C4=CC=CC=C4)[H])C(=O)[C@](C=C)(C)C[C@H]3OC(=O)C5=CC=CC=C5)[H]"
                else:
                    print(f"error {i}")
                    error_list.append(_)
                    continue
        mol = Chem.MolFromSmiles(smiles)
        
        # smiles = Chem.MolToSmiles(mol)
        # temp_smiles_list = get_new_smiles(mol)

        temp_df = pd.DataFrame(json.loads(_["nmr"]))
        nmr_data = temp_df['13C Shift /ppm'].dropna().tolist()
        
        temp_dict = {}
        temp_dict["cano_smiles"] = smiles
        temp_dict["NMR"] = list(set(nmr_data))
        # temp_dict["smiles_list"] = temp_smiles_list
        temp_dict["index"] = i
        temp_dict["source"] = "ch-nmr-np"
        data.append(copy.deepcopy(temp_dict))
        # import ipdb
        # ipdb.set_trace()
        
    print(len(data))
    with open(os.path.join(save_dir, "ch-nmr-np_processed.json"), "w") as f:
        f.write(json.dumps(data, indent=4))
    
    print(len(error_list))
    with open(os.path.join(save_dir, "ch-nmr-np_processed_error.json"), "w") as f:
        f.write(json.dumps(error_list, indent=4))


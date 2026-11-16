import os
import numpy as np
import lmdb
from tqdm import tqdm
from multiprocessing import Pool, Queue, Process
from rdkit import Chem
import pickle


env_list = []
key_list = []
env_lenght_list = []
for i in [0, 5]:
    lmdb_path = f"/home/baolingjie/git_download/NMR_Core_Mol_Detection/data/NMR_merged_1022_add_fp_script_{i}/merged_NMR.lmdb"
    env = lmdb.open(
        lmdb_path,
        map_size= 1000 * 1024**3,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1024,
    )
    length = env.stat()['entries']
    env_list.append(env)
    key_list.extend([f'{count}'.encode('ascii') for count in range(length)])
    env_lenght_list.append(len(key_list))
idx_list = np.arange(len(key_list))

new_lmdb_path = "data/enumerate_1b_group/new_db.lmdb"  # 请修改为你的新路径
os.makedirs(os.path.dirname(new_lmdb_path), exist_ok=True)
# map_size = 1099511627776  # 1TB初始空间（根据数据量调整）

# 创建优化配置的新LMDB环境
new_env = lmdb.open(
    new_lmdb_path,
    map_size=100*(1024*1024*1024),
)
txn_writer = new_env.begin(write=True)

fixed_group = [
        'acid', 'acyl_halide', 'alcohol', 'aldehyde', 'alkane', 'alkene',
        'alkylaluminium', 'alkyllithium', 'alkylmagnesium_halide', 'alkyne',
        'amide', 'amidine', 'amine', 'anhydride', 'aromatic', 'azide', 'azo',
        'borinic_acid', 'borinic_ester', 'boronic_acid', 'boronic_ester',
        'branched_alkane', 'bromoalkane', 'carbamate', 'carbodithio',
        'carbodithioic_acid', 'carbonate', 'carbothioic_o_acid',
        'carbothioic_s_acid', 'carboxylate', 'carboxylic_acid',
        'carboxylic_anhydride', 'chloroalkane', 'cyanate', 'cyanide',
        'cycloalkane', 'disulfide', 'ester', 'ether', 'fluoroalkane',
        'haloalkane', 'hydrocarbon', 'hydroperoxide', 'imide', 'imine',
        'inorganic', 'iodoalkane', 'isocyanate', 'isonitrile', 'isothiocyanate',
        'ketone', 'mercaptan', 'methylenedioxy', 'nitrate', 'nitrile',
        'nitrite', 'nitro', 'nitroso', 'organic', 'orthocarbonate_ester',
        'orthoester', 'oxime', 'peroxide', 'phenol', 'phosphate', 'phosphine',
        'phosphodiester', 'phosphonic_acid', 'polyol', 'primary_aldimine',
        'primary_amine', 'primary_ketimine', 'pyridyl', 'quat', 'radionuclide',
        'secondary_aldimine', 'secondary_amine', 'secondary_ketimine',
        'siloxane', 'silyl_ether', 'sulfide', 'sulfinic_acid',
        'sulfonate_ester', 'sulfone', 'sulfonic_acid', 'sulfoxide',
        'tertiary_amine', 'thial', 'thiocyanate', 'thioketone', 'thiolester',
        'thionoester'
    ]

all_other_group = [
    'alkane',           # 烷烃 (R-CH₃, R₂CH₂, R₃CH) - δc 8~60 ppm
    'branched_alkane',  # 支链烷烃
    'cycloalkane',      # 环烷烃
    'alcohol',           # 醇 R-OH
    'polyol',           # 多元醇（含多个OH）
    'phenol',           # 酚 Ar-OH
    'ether',            # 醚 R-O-R'
    'methylenedioxy',   # 亚甲二氧基 -O-CH₂-O-
    'aldehyde',         # 醛 R-CHO - δc 185~220 ppm
    'ketone',
    # 酸类
    'acid',
    # 酰胺类
    'amide',
    # 酯类
    'ester',
    # 酸酐类
    'anhydride',
    'alkyne',
    'alkene',
    'aromatic', 
    'pyridyl',
]

## 
alkane_groups = [
    'alkane',           # 烷烃 (R-CH₃, R₂CH₂, R₃CH) - δc 8~60 ppm
    'branched_alkane',  # 支链烷烃
    'cycloalkane',      # 环烷烃
]
alkane_index = [fixed_group.index(_) for _ in alkane_groups if _ in fixed_group] # [4, 21, 35]
mask_alkane_index = [_ for _ in range(len(fixed_group)) if _ not in alkane_index and _ in all_other_group]

ether_groups = [
    'alcohol',           # 醇 R-OH
    'polyol',           # 多元醇（含多个OH）
    'phenol',           # 酚 Ar-OH
    'ether',            # 醚 R-O-R'
    'methylenedioxy',   # 亚甲二氧基 -O-CH₂-O-
]

ether_index = [fixed_group.index(_) for _ in ether_groups if _ in fixed_group] # [2, 68, 63, 38, 52]
mask_ether_index = [_ for _ in range(len(fixed_group)) if _ not in ether_index and _ in all_other_group]


aldehyde_ketone_groups = [
    'aldehyde',         # 醛 R-CHO - δc 185~220 ppm
    'ketone',
]

aldehyde_ketone_index = [fixed_group.index(_) for _ in aldehyde_ketone_groups if _ in fixed_group] # [3, 50]
mask_aldehyde_ketone_index = [_ for _ in range(len(fixed_group)) if _ not in aldehyde_ketone_index and _ in all_other_group]

acid_groups = [
    # 酸类
    'acid',
    # 酰胺类
    'amide',
    # 酯类
    'ester',
    # 酸酐类
    'anhydride',
]

acid_index = [fixed_group.index(_) for _ in acid_groups if _ in fixed_group] # [0, 10, 37, 13]
mask_acid_index = [_ for _ in range(len(fixed_group)) if _ not in acid_index and _ in all_other_group]

alkyne_group = ['alkyne',]
alkyne_index = [fixed_group.index(_) for _ in alkyne_group if _ in fixed_group] # [9]
mask_alkyne_index = [_ for _ in range(len(fixed_group)) if _ not in alkyne_index and _ in all_other_group]

alkene_group = ['alkene',]
alkene_index = [fixed_group.index(_) for _ in alkene_group if _ in fixed_group] # [5]
mask_alkene_index = [_ for _ in range(len(fixed_group)) if _ not in alkene_index and _ in all_other_group]


aromatic_group = ['aromatic', 
                    'pyridyl',]
aromatic_index = [fixed_group.index(_) for _ in aromatic_group if _ in fixed_group] # [14, 72]
mask_aromatic_index = [_ for _ in range(len(fixed_group)) if _ not in aromatic_index and _ in all_other_group]


def process_single(batch):
    idx = batch
    for env_idx in range(len(env_lenght_list)):
        if idx < env_lenght_list[env_idx]:
            break
    env = env_list[env_idx]
    key = key_list[idx]
    try:
        with env.begin() as txn:
            sample = pickle.loads(txn.get(key))
            if sample["function_list"][alkane_index].sum()>0 and sample["function_list"][mask_alkane_index].sum()==0:
                return sample
            elif sample["function_list"][ether_index].sum()>0 and sample["function_list"][mask_ether_index].sum()==0:
                return sample
            elif sample["function_list"][aldehyde_ketone_index].sum()>0 and sample["function_list"][mask_aldehyde_ketone_index].sum()==0:
                return sample
            elif sample["function_list"][acid_index].sum()>0 and sample["function_list"][mask_acid_index].sum()==0:
                return sample
            elif sample["function_list"][alkyne_index].sum()>0 and sample["function_list"][mask_alkyne_index].sum()==0:
                return sample
            elif sample["function_list"][alkene_index].sum()>0 and sample["function_list"][mask_alkene_index].sum()==0:
                return sample
            elif sample["function_list"][aromatic_index].sum()>0 and sample["function_list"][mask_aromatic_index].sum()==0:
                return sample
            else:
                return None
    except Exception as e:
        print(e)
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    num_processes = 50
    print(f"len(idx_list):{len(idx_list)}")
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_single, idx_list),
            total=len(idx_list)
        ))

    count = 0
    for i in tqdm(range(len(results))):
        if results[i] is not None:
            txn_writer.put(f'{count}'.encode("ascii"), pickle.dumps(results[i], protocol=-1))
            count = count + 1

    print(f"count:{count}")
    txn_writer.commit()
    new_env.close()
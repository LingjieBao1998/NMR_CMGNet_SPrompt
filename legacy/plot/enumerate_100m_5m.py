import lmdb
import pickle
import numpy as np
import multiprocessing
from tqdm import tqdm
import os

lmdb_path = "/home/baolingjie/git_download/NMR_Core_Mol_Detection/data/NMR_merged_1022_add_fp_script_0/merged_NMR.lmdb"
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
key_list = [f'{count}'.encode('ascii') for count in range(length) if count<=5000000]
query_dict = {
    # 'alkene': 5,
    'alcohol': 2,
    'amide': 10,
    'ester': 37,
    # 'ketone': 50,
    'sulfide': 80,
    'ether': 38,
    'pyridyl': 72,
    'hydrocarbon': 41,
    'amine': 12,
    'sulfone': 83,
    'aldehyde': 3,
    'polyol': 68,
    'nitro': 56,
    #  'secondary_amine': 76,
    'haloalkane': 40,
    #  'alkyne': 9,
    #  'tertiary_amine': 86
}

query_list = [v for v in query_dict.values()]
print(query_list)


def process_single(batch):
    key = batch
    try:
        with env.begin() as txn:
            sample = pickle.loads(txn.get(key))
            if sample["function_list"][query_list].sum()==1:
                return sample
            else:
                return None
    except Exception as e:
        print(e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # process_single(key_list[0])
    # import ipdb
    # ipdb.set_trace()

    # 创建进程池，processes 参数指定进程数，通常设为 CPU 核心数或稍多
    with multiprocessing.Pool(processes=100) as pool:
        # 使用 map 方法将 key_list 中的每个元素应用于 process_single 函数
        # map 会阻塞直到所有任务完成，并返回结果列表
        # results = pool.map(process_single, key_list)
        results = list(tqdm(
                pool.imap_unordered(process_single, key_list),
                total=len(key_list)
            ))
    
    last_results = [result for result in results if result is not None]
    print(f'len(last_results):{len(last_results)}')

    save_dir = "data/enumerate_100m_5m_1_group_restricted"
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir,'last_results.pkl'), 'wb') as file: # 注意 'wb' 表示用二进制模式写入
        pickle.dump(last_results, file)
    
    import ipdb
    ipdb.set_trace()
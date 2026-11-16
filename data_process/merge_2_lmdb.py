import lmdb
import shutil
import os
from tqdm import tqdm
import pickle
from tqdm import tqdm


def merge_lmdb(db1_path, db2_path, output_path, map_size=300 * 1024**3):
    """
    安全合并两个LMDB数据库
    
    参数:
        db1_path: 第一个数据库路径
        db2_path: 第二个数据库路径
        output_path: 合并后输出路径
        map_size: 合并后数据库大小 (默认100GB)
    """
    # 0. 检查路径是否存在
    for path in [db1_path, db2_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"数据库不存在: {path}")
    
    # 1. 创建新环境
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    env_new = lmdb.open(
        output_path,
        map_size=map_size,
        writemap=True  # 提高写入性能
    )

    env1 = lmdb.open(
                    db1_path,
                    readonly=True,
                    map_size= 100 * 1024**3,
                    lock=False,
                    readahead=False,
                    meminit=False,
                    max_readers=256,
                )
    length1 = env1.stat()['entries']
    key_list1 = [f'{count}'.encode('ascii') for count in range(length1)]

    env2 = lmdb.open(
                    db2_path,
                    readonly=True,
                    map_size= 100 * 1024**3,
                    lock=False,
                    readahead=False,
                    meminit=False,
                    max_readers=256,
                )
    length2 = env2.stat()['entries']
    key_list2 = [f'{count}'.encode('ascii') for count in range(length2)]

    count = -1
    
    # 2. 合并过程 (原子操作)
    with env_new.begin(write=True) as txn_new:
        # 合并第一个数据库
        for i, key in tqdm(enumerate(key_list1)):
            with env1.begin() as txn:
                value = pickle.loads(txn.get(key))
                if value is not None:
                    count = count + 1
                    txn_new.put(f'{count}'.encode('ascii'), pickle.dumps(value, protocol=-1))
            
            # if i == 1000:
            #     break
        
        for i, key in tqdm(enumerate(key_list2)):
            with env2.begin() as txn:
                value = pickle.loads(txn.get(key))
                if value is not None:
                    count = count + 1
                    txn_new.put(f'{count}'.encode('ascii'), pickle.dumps(value, protocol=-1))
            
            # if i == 1000:
            #     break
    
    print(f"合并完成 → 输出路径: {output_path}")

if __name__ == "__main__":
    # 使用示例
    merge_lmdb(
        db1_path="data/NMR_other_data_1020/new_db.lmdb",
        db2_path="data/NMR_PubChem_1020/new_db.lmdb",
        output_path="data/NMR_merged_1020/merged_NMR.lmdb"
    )
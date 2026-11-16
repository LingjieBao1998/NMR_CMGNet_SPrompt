#!/usr/bin/env python3
"""
从LMDB数据库获取分子详细信息并计算Wasserstein距离
"""

import lmdb
import pickle
import numpy as np
from scipy import stats
import json
from tqdm import tqdm

def load_search_results(results_file):
    """加载搜索结果"""
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def wasserstein_distance(nmr1, nmr2):
    """计算两个NMR分布之间的Wasserstein距离"""
    nmr1 = np.array(nmr1)
    nmr2 = np.array(nmr2)
    
    # 确保数据不为空
    if len(nmr1) == 0 or len(nmr2) == 0:
        return float('inf')
    
    try:
        # 计算Wasserstein距离 (Earth Mover's Distance)
        distance = stats.wasserstein_distance(nmr1, nmr2)
        return distance
    except Exception as e:
        print(f"计算Wasserstein距离失败: {e}")
        return float('inf')

def get_molecule_info_from_lmdb(keys, lmdb_path="/home/lingjiebao/project/NMR_Core_Mol_Detection/data/NMR_other_data_1020/new_db.lmdb"):
    """从LMDB数据库获取分子信息"""
    
    # 打开LMDB数据库
    env = lmdb.open(
        lmdb_path,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1024,
    )
    
    molecules_info = []
    
    print(f"从LMDB数据库获取 {len(keys)} 个分子的详细信息...")
    
    for key in tqdm(keys, desc="获取分子信息"):
        try:
            # 确保key是bytes格式
            if isinstance(key, str):
                key_bytes = key.encode('ascii')
            else:
                key_bytes = key
            
            with env.begin() as txn:
                data = txn.get(key_bytes)
                
                if data is None:
                    print(f"警告: 键 {key} 未找到")
                    continue
                
                # 反序列化数据
                sample = pickle.loads(data)
                
                # 提取分子信息
                mol_info = {
                    'key': key,
                    'smiles': sample.get('canonical_smiles', ''),
                    'C_nmr': sample.get('C_nmr', []),
                    'NMR': sample.get('NMR', []),
                    'source': sample.get('source', ''),
                    'split': sample.get('split', '')
                }
                
                molecules_info.append(mol_info)
                
        except Exception as e:
            print(f"获取分子 {key} 信息失败: {type(e).__name__}: {str(e)}")
            continue
    
    env.close()
    return molecules_info

def main():
    # 加载搜索结果
    print("加载搜索结果...")
    search_data = load_search_results('search_results_1754.json')
    
    # 获取前20个最相似分子的键
    top_keys = [result['key'] for result in search_data['search_results'][:20]]
    
    # 查询NMR数据
    query_nmr = search_data['query_info']['file']
    with open(query_nmr, 'r', encoding='utf-8') as f:
        query_data = json.load(f)
    query_nmr_data = query_data['nmr_data']
    
    print(f"查询NMR数据: {query_nmr_data}")
    print(f"数据点数: {len(query_nmr_data)}")
    
    # 从LMDB获取分子详细信息
    molecules_info = get_molecule_info_from_lmdb(top_keys)
    
    # 计算Wasserstein距离
    print("\n计算Wasserstein距离...")
    
    results_with_distance = []
    for mol_info in molecules_info:
        # 优先使用C_nmr，其次使用NMR
        if mol_info['C_nmr']:
            mol_nmr = mol_info['C_nmr']
        elif mol_info['NMR']:
            mol_nmr = mol_info['NMR']
        else:
            print(f"分子 {mol_info['key']} 没有NMR数据")
            continue
        
        # 计算Wasserstein距离
        w_dist = wasserstein_distance(query_nmr_data, mol_nmr)
        
        # 添加到结果
        result_entry = {
            'rank': len(results_with_distance) + 1,
            'key': mol_info['key'],
            'smiles': mol_info['smiles'],
            'C_nmr': mol_info['C_nmr'],
            'NMR': mol_info['NMR'],
            'source': mol_info['source'],
            'split': mol_info['split'],
            'cuvs_similarity': search_data['search_results'][len(results_with_distance)]['similarity'],
            'wasserstein_distance': w_dist,
            'nmr_points_count': len(mol_nmr)
        }
        
        results_with_distance.append(result_entry)
    
    # 按Wasserstein距离排序
    results_with_distance.sort(key=lambda x: x['wasserstein_distance'])
    
    # 保存结果
    output_data = {
        'query_info': {
            'nmr_data': query_nmr_data,
            'nmr_count': len(query_nmr_data)
        },
        'results': results_with_distance
    }
    
    with open('molecules_with_wasserstein.json', 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    # 显示结果
    print(f"\n=== 前10个最相似的分子 (按Wasserstein距离排序) ===")
    for i, result in enumerate(results_with_distance[:10]):
        print(f"\n排名 {i+1}:")
        print(f"  键: {result['key']}")
        print(f"  SMILES: {result['smiles']}")
        print(f"  数据源: {result['source']}")
        print(f"  数据集: {result['split']}")
        print(f"  NMR数据点数: {result['nmr_points_count']}")
        print(f"  cuVS相似度: {result['cuvs_similarity']:.4f}")
        print(f"  Wasserstein距离: {result['wasserstein_distance']:.4f}")
        print(f"  C-NMR: {result['C_nmr'][:10]}..." if len(result['C_nmr']) > 10 else f"  C-NMR: {result['C_nmr']}")
    
    print(f"\n结果已保存到: molecules_with_wasserstein.json")
    print(f"总共处理了 {len(results_with_distance)} 个分子")

if __name__ == "__main__":
    main()
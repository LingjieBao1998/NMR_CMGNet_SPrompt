#!/usr/bin/env python3
"""
NMR数据搜索接口
支持输入NMR数据，使用cuVS索引进行相似性搜索
"""

import cuvs
import cuvs.neighbors.ivf_pq as ivf_pq
import cupy as cp
import torch
import numpy as np
import pickle
import os
import time
from infer import ModelInference


class NMRSearchInterface:
    """
    NMR数据搜索接口
    支持从NMR数据到相似分子搜索的完整流程
    """
    
    def __init__(self, index_path="data/query_dataset/cuvs_768d_lmdb_index.pkl",
                 keys_path="data/query_dataset/cuvs_768d_lmdb_keys.pkl"):
        """
        初始化搜索接口
        
        Args:
            index_path: cuVS索引文件路径
            keys_path: 索引元数据文件路径
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {self.device}")
        
        # 初始化模型推理器
        self.model_inference = ModelInference(
            config_path="models/2_5_w_model/8.json",
            pretrain_model_path="models/2_5_w_model/8.pth",
            device=self.device
        )
        
        # 加载索引
        self.index = None
        self.valid_keys = []
        self.feature_dim = 768
        self.search_params = None
        
        if self.load_index(index_path, keys_path):
            print("✅ 搜索接口初始化成功")
        else:
            print("❌ 索引加载失败")
    
    def load_index(self, index_path, keys_path):
        """加载预构建的cuVS索引"""
        try:
            if not os.path.exists(index_path) or not os.path.exists(keys_path):
                print(f"索引文件不存在: {index_path} 或 {keys_path}")
                return False
            
            print(f"加载索引: {index_path}")
            self.index = ivf_pq.load(index_path)
            
            print(f"加载元数据: {keys_path}")
            with open(keys_path, 'rb') as f:
                metadata = pickle.load(f)
            
            self.valid_keys = metadata['valid_keys']
            self.feature_dim = metadata['feature_dim']
            
            # 设置搜索参数
            self.search_params = ivf_pq.SearchParams(n_probes=100)
            
            print(f"索引加载成功: {len(self.valid_keys):,} 条数据")
            return True
            
        except Exception as e:
            print(f"索引加载失败: {type(e).__name__}: {str(e)}")
            return False
    
    def encode_nmr_data(self, nmr_data):
        """
        将NMR数据编码为768维特征向量
        
        Args:
            nmr_data: NMR数据，可以是以下格式:
                     - list/tuple: NMR化学位移值列表
                     - numpy.ndarray: NMR数组
                     - dict: 包含'C_nmr'或'NMR'键的字典
        
        Returns:
            numpy.ndarray: 768维特征向量，失败时返回None
        """
        try:
            # 处理不同输入格式
            if isinstance(nmr_data, dict):
                if 'C_nmr' in nmr_data:
                    nmr_list = nmr_data['C_nmr']
                elif 'NMR' in nmr_data:
                    nmr_list = nmr_data['NMR']
                else:
                    print("错误: 字典中未找到'C_nmr'或'NMR'键")
                    return None
            elif isinstance(nmr_data, (list, tuple, np.ndarray)):
                nmr_list = list(nmr_data)
            else:
                print(f"错误: 不支持的NMR数据类型: {type(nmr_data)}")
                return None
            
            # 检查NMR数据有效性
            if not nmr_list or len(nmr_list) == 0:
                print("错误: NMR数据为空")
                return None
            
            # 确保数据类型正确
            if not all(isinstance(x, (int, float)) for x in nmr_list):
                print("错误: NMR数据包含非数值类型")
                return None
            
            # 使用模型进行特征编码
            print(f"编码NMR数据: {len(nmr_list)} 个化学位移值")
            nmr_feature = self.model_inference.nmr_encode(nmr_list)
            
            if nmr_feature is not None:
                feature_array = nmr_feature.cpu().numpy()
                print(f"特征编码成功: {feature_array.shape}")
                return feature_array
            else:
                print("特征编码失败")
                return None
                
        except Exception as e:
            print(f"NMR编码错误: {type(e).__name__}: {str(e)}")
            return None
    
    def search_similar_nmr(self, nmr_data, top_k=10, n_probes=None):
        """
        搜索相似的NMR数据
        
        Args:
            nmr_data: 输入的NMR数据
            top_k: 返回最相似的k个结果
            n_probes: 搜索的聚类数量，None时使用默认值
        
        Returns:
            list: 搜索结果列表，每个元素包含:
                  - rank: 排名
                  - key: 数据库中的键
                  - similarity: 相似度/距离
                  - metadata: 额外元数据
        """
        if self.index is None:
            print("错误: 索引未加载")
            return []
        
        # 编码NMR数据
        query_features = self.encode_nmr_data(nmr_data)
        if query_features is None:
            return []
        
        # 确保特征维度正确
        if len(query_features.shape) == 1:
            query_features = query_features.reshape(1, -1)
        
        # 设置搜索参数
        search_params = self.search_params
        if n_probes is not None:
            search_params = ivf_pq.SearchParams(n_probes=n_probes)
        
        print(f"搜索参数: top_k={top_k}, n_probes={search_params.n_probes}")
        
        try:
            # 将查询向量传输到GPU
            query_gpu = cp.asarray(query_features.astype('float32'))
            
            # 执行搜索
            start_time = time.time()
            distances, indices = ivf_pq.search(search_params, self.index, query_gpu, k=top_k)
            search_time = time.time() - start_time
            
            # 将结果传回CPU
            distances_cpu = cp.asnumpy(distances).flatten()
            indices_cpu = cp.asnumpy(indices).flatten()
            
            print(f"搜索完成，耗时: {search_time:.4f}秒")
            
            # 构建结果列表
            results = []
            for i, (idx, dist) in enumerate(zip(indices_cpu, distances_cpu)):
                if idx < len(self.valid_keys):
                    result = {
                        'rank': i + 1,
                        'key': self.valid_keys[idx].decode('utf-8') if isinstance(self.valid_keys[idx], bytes) else str(self.valid_keys[idx]),
                        'similarity': float(dist),
                        'distance': float(dist),
                        'index': int(idx)
                    }
                    results.append(result)
            
            return results
            
        except Exception as e:
            print(f"搜索失败: {type(e).__name__}: {str(e)}")
            return []
    
    def search_with_metadata(self, nmr_data, top_k=10, n_probes=None, lmdb_path="/home/lingjiebao/project/NMR_Core_Mol_Detection/data/NMR_other_data_1020/new_db.lmdb"):
        """
        搜索相似的NMR数据并返回完整的元数据
        
        Args:
            nmr_data: 输入的NMR数据
            top_k: 返回最相似的k个结果
            n_probes: 搜索的聚类数量
            lmdb_path: LMDB数据库路径
        
        Returns:
            list: 包含完整元数据的搜索结果
        """
        # 执行基础搜索
        search_results = self.search_similar_nmr(nmr_data, top_k, n_probes)
        
        if not search_results:
            return []
        
        # 从LMDB获取详细元数据
        try:
            import lmdb
            env = lmdb.open(lmdb_path, readonly=True, lock=False)
            
            enhanced_results = []
            with env.begin() as txn:
                for result in search_results:
                    key = result['key'].encode('utf-8') if isinstance(result['key'], str) else result['key']
                    
                    # 获取完整数据
                    data = txn.get(key)
                    if data:
                        sample = pickle.loads(data)
                        
                        # 构建增强结果
                        enhanced_result = result.copy()
                        enhanced_result.update({
                            'canonical_smiles': sample.get('canonical_smiles', ''),
                            'source': sample.get('source', ''),
                            'split': sample.get('split', ''),
                            'ecfp': sample.get('ecfp', ''),
                            'maccs': sample.get('maccs', ''),
                            'function_list': sample.get('function_list', []),
                            'label': sample.get('label', ''),
                            'element_list': sample.get('element_list', [])
                        })
                        enhanced_results.append(enhanced_result)
            
            env.close()
            return enhanced_results
            
        except Exception as e:
            print(f"获取元数据失败: {type(e).__name__}: {str(e)}")
            return search_results
    
    def get_index_info(self):
        """获取索引信息"""
        if self.index is None:
            return "索引未加载"
        
        return {
            '总向量数': f"{len(self.valid_keys):,}",
            '特征维度': self.feature_dim,
            '索引类型': 'cuVS.IVF_PQ',
            '设备': self.device,
            '搜索参数': f"n_probes={self.search_params.n_probes}"
        }


def demo_search():
    """搜索演示"""
    print("=" * 60)
    print("NMR数据搜索演示")
    print("=" * 60)
    
    # 初始化搜索接口
    search_interface = NMRSearchInterface()
    
    # 显示索引信息
    info = search_interface.get_index_info()
    print("索引信息:")
    for k, v in info.items():
        print(f"  {k}: {v}")
    print()
    
    # 示例NMR数据 (碳-13 NMR化学位移)
    example_nmr_data = [
        192.6, 128.68, 135.12, 121.45, 118.23, 
        156.78, 132.45, 127.89, 145.67, 110.34
    ]
    
    print(f"查询NMR数据: {example_nmr_data}")
    print()
    
    # 执行搜索
    print("执行相似性搜索...")
    results = search_interface.search_similar_nmr(example_nmr_data, top_k=5)
    
    if results:
        print("搜索结果:")
        for result in results:
            print(f"排名 {result['rank']}: 键={result['key']}, 距离={result['similarity']:.4f}")
    else:
        print("未找到相似结果")
    
    print()
    
    # 带元数据的搜索
    print("执行带元数据的搜索...")
    detailed_results = search_interface.search_with_metadata(example_nmr_data, top_k=3)
    
    if detailed_results:
        print("详细搜索结果:")
        for result in detailed_results:
            print(f"排名 {result['rank']}:")
            print(f"  键: {result['key']}")
            print(f"  距离: {result['similarity']:.4f}")
            print(f"  SMILES: {result.get('canonical_smiles', 'N/A')}")
            print(f"  来源: {result.get('source', 'N/A')}")
            print()
    else:
        print("未找到详细结果")


if __name__ == "__main__":
    demo_search()
#!/usr/bin/env python3
"""
NMR数据搜索接口 - CPU版本 (FAISS备用方案)
支持输入NMR数据，使用FAISS索引进行相似性搜索
"""

import faiss
import torch
import numpy as np
import pickle
import os
import time
import lmdb
from infer import ModelInference


class NMRSearchInterfaceCPU:
    """
    NMR数据搜索接口 - CPU版本
    当GPU搜索不可用时的备用方案
    """
    
    def __init__(self, use_faiss=True):
        """
        初始化搜索接口
        
        Args:
            use_faiss: 是否使用FAISS，False时使用暴力搜索
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {self.device}")
        
        # 初始化模型推理器
        self.model_inference = ModelInference(
            config_path="models/2_5_w_model/8.json",
            pretrain_model_path="models/2_5_w_model/8.pth",
            device=self.device
        )
        
        self.use_faiss = use_faiss
        self.index = None
        self.valid_keys = []
        self.feature_dim = 768
        self.feature_matrix = None
        
        print(f"搜索模式: {'FAISS索引' if use_faiss else '暴力搜索'}")
    
    def load_from_npy_data(self, features_path="data/query_dataset/cuvs_768d_lmdb_index_features.npy", 
                          keys_path="data/query_dataset/cuvs_768d_lmdb_index_keys.pkl"):
        """
        从保存的numpy数据加载特征
        
        Args:
            features_path: 特征矩阵文件路径
            keys_path: 键列表文件路径
        """
        try:
            if os.path.exists(features_path) and os.path.exists(keys_path):
                print(f"加载特征矩阵: {features_path}")
                self.feature_matrix = np.load(features_path)
                
                print(f"加载键列表: {keys_path}")
                with open(keys_path, 'rb') as f:
                    self.valid_keys = pickle.load(f)
                
                print(f"数据加载成功: {self.feature_matrix.shape}")
                
                # 如果使用FAISS，构建索引
                if self.use_faiss:
                    self.build_faiss_index()
                
                return True
            else:
                print(f"数据文件不存在: {features_path} 或 {keys_path}")
                return False
                
        except Exception as e:
            print(f"数据加载失败: {type(e).__name__}: {str(e)}")
            return False
    
    def build_faiss_index(self):
        """构建FAISS索引"""
        try:
            print("构建FAISS索引...")
            
            # 创建索引
            self.index = faiss.IndexFlatL2(self.feature_dim)
            
            # 添加向量到索引
            self.index.add(self.feature_matrix.astype('float32'))
            
            print(f"FAISS索引构建完成: {self.index.ntotal} 个向量")
            return True
            
        except Exception as e:
            print(f"FAISS索引构建失败: {type(e).__name__}: {str(e)}")
            print("切换到暴力搜索模式")
            self.use_faiss = False
            return False
    
    def encode_nmr_data(self, nmr_data):
        """
        将NMR数据编码为768维特征向量
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
    
    def search_similar_nmr(self, nmr_data, top_k=10):
        """
        搜索相似的NMR数据
        """
        if self.feature_matrix is None:
            print("错误: 数据未加载")
            return []
        
        # 编码NMR数据
        query_features = self.encode_nmr_data(nmr_data)
        if query_features is None:
            return []
        
        # 确保特征维度正确
        if len(query_features.shape) == 1:
            query_features = query_features.reshape(1, -1)
        
        query_features = query_features.astype('float32')
        
        print(f"搜索参数: top_k={top_k}")
        
        try:
            start_time = time.time()
            
            if self.use_faiss and self.index is not None:
                # 使用FAISS搜索
                distances, indices = self.index.search(query_features, top_k)
                search_time = time.time() - start_time
                print(f"FAISS搜索完成，耗时: {search_time:.4f}秒")
                
                # 构建结果列表
                results = []
                for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
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
            else:
                # 暴力搜索
                print("使用暴力搜索...")
                distances = np.linalg.norm(self.feature_matrix - query_features, axis=1)
                top_indices = np.argsort(distances)[:top_k]
                search_time = time.time() - start_time
                print(f"暴力搜索完成，耗时: {search_time:.4f}秒")
                
                results = []
                for i, idx in enumerate(top_indices):
                    if idx < len(self.valid_keys):
                        result = {
                            'rank': i + 1,
                            'key': self.valid_keys[idx].decode('utf-8') if isinstance(self.valid_keys[idx], bytes) else str(self.valid_keys[idx]),
                            'similarity': float(distances[idx]),
                            'distance': float(distances[idx]),
                            'index': int(idx)
                        }
                        results.append(result)
                
                return results
                
        except Exception as e:
            print(f"搜索失败: {type(e).__name__}: {str(e)}")
            return []
    
    def get_metadata_for_keys(self, keys, lmdb_path="/home/lingjiebao/project/NMR_Core_Mol_Detection/data/NMR_other_data_1020/new_db.lmdb"):
        """
        为给定的键列表获取元数据
        """
        try:
            env = lmdb.open(lmdb_path, readonly=True, lock=False)
            metadata_dict = {}
            
            with env.begin() as txn:
                for key in keys:
                    key_bytes = key.encode('utf-8') if isinstance(key, str) else key
                    data = txn.get(key_bytes)
                    if data:
                        sample = pickle.loads(data)
                        metadata_dict[key] = {
                            'canonical_smiles': sample.get('canonical_smiles', ''),
                            'source': sample.get('source', ''),
                            'split': sample.get('split', ''),
                            'C_nmr': sample.get('C_nmr', []),
                            'ecfp': sample.get('ecfp', ''),
                            'maccs': sample.get('maccs', ''),
                            'function_list': sample.get('function_list', []),
                            'label': sample.get('label', ''),
                            'element_list': sample.get('element_list', '')
                        }
            
            env.close()
            return metadata_dict
            
        except Exception as e:
            print(f"获取元数据失败: {type(e).__name__}: {str(e)}")
            return {}
    
    def search_with_metadata(self, nmr_data, top_k=10):
        """
        搜索相似的NMR数据并返回完整的元数据
        """
        # 执行基础搜索
        search_results = self.search_similar_nmr(nmr_data, top_k)
        
        if not search_results:
            return []
        
        # 获取元数据
        keys = [result['key'] for result in search_results]
        metadata_dict = self.get_metadata_for_keys(keys)
        
        # 合并结果
        enhanced_results = []
        for result in search_results:
            enhanced_result = result.copy()
            metadata = metadata_dict.get(result['key'], {})
            enhanced_result.update(metadata)
            enhanced_results.append(enhanced_result)
        
        return enhanced_results
    
    def save_features(self, features_path="data/query_dataset/nmr_features.npy", 
                     keys_path="data/query_dataset/nmr_keys.pkl"):
        """
        保存特征矩阵和键列表
        """
        if self.feature_matrix is not None:
            np.save(features_path, self.feature_matrix)
            with open(keys_path, 'wb') as f:
                pickle.dump(self.valid_keys, f)
            print(f"特征数据已保存到: {features_path}")
            print(f"键列表已保存到: {keys_path}")
    
    def get_index_info(self):
        """获取索引信息"""
        if self.feature_matrix is None:
            return "数据未加载"
        
        info = {
            '总向量数': f"{len(self.valid_keys):,}",
            '特征维度': self.feature_dim,
            '搜索模式': 'FAISS' if self.use_faiss else '暴力搜索',
            '设备': self.device
        }
        
        if self.use_faiss and self.index is not None:
            info['索引类型'] = type(self.index).__name__
            info['索引大小'] = f"{self.index.ntotal:,}"
        
        return info


def demo_search():
    """搜索演示"""
    print("=" * 60)
    print("NMR数据搜索演示 - CPU版本")
    print("=" * 60)
    
    # 初始化搜索接口
    search_interface = NMRSearchInterfaceCPU(use_faiss=True)
    
    # 尝试从之前保存的数据加载
    if not search_interface.load_from_npy_data():
        print("无法加载预保存的特征数据")
        print("请先运行cuvs_search.py生成特征数据")
        return
    
    # 显示索引信息
    info = search_interface.get_index_info()
    print("索引信息:")
    for k, v in info.items():
        print(f"  {k}: {v}")
    print()
    
    # 示例NMR数据
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
            print(f"  C_nmr: {result.get('C_nmr', [])[:5]}...")  # 只显示前5个
            print()
    else:
        print("未找到详细结果")


if __name__ == "__main__":
    demo_search()
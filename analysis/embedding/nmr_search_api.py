#!/usr/bin/env python3
"""
NMR搜索API - 简易接口
提供简单易用的NMR相似性搜索API
"""

from enhanced_nmr_search import EnhancedNMRSearchInterface
from typing import List, Dict, Union, Optional, Tuple
import numpy as np
import json


class NMRSimpleAPI:
    """
    简易NMR搜索API
    提供最简单易用的搜索接口
    """
    
    def __init__(self):
        """初始化API"""
        self.search_interface = EnhancedNMRSearchInterface()
        print("NMR搜索API初始化完成")
    
    def search(self, nmr_data: Union[List, Tuple, np.ndarray, Dict], 
               k: int = 10, 
               use_case: str = "general") -> List[Dict]:
        """
        简单搜索接口
        
        Args:
            nmr_data: NMR数据
            k: 返回结果数量
            use_case: 使用场景 ("general", "diverse", "precise", "exploratory", "research")
        
        Returns:
            搜索结果列表
        """
        # 根据用例推荐k值
        recommended_k = self.search_interface.recommend_k(use_case)
        if k == 10:  # 使用默认值时，应用推荐值
            k = recommended_k
        
        print(f"NMR搜索: k={k}, 用例={use_case}")
        
        # 执行搜索并返回带元数据的结果
        return self.search_interface.search_with_metadata(nmr_data, k=k)
    
    def search_diverse(self, nmr_data: Union[List, Tuple, np.ndarray, Dict], 
                      k: int = 20) -> List[Dict]:
        """搜索多样化的结果"""
        return self.search(nmr_data, k, "diverse")
    
    def search_precise(self, nmr_data: Union[List, Tuple, np.ndarray, Dict], 
                      k: int = 5) -> List[Dict]:
        """搜索精确的结果"""
        return self.search(nmr_data, k, "precise")
    
    def search_exploratory(self, nmr_data: Union[List, Tuple, np.ndarray, Dict], 
                          k: int = 50) -> List[Dict]:
        """探索性搜索"""
        return self.search(nmr_data, k, "exploratory")
    
    def batch_search(self, nmr_data_list: List[Union[List, Tuple, np.ndarray, Dict]], 
                     k: int = 10,
                     use_case: str = "general") -> List[List[Dict]]:
        """批量搜索"""
        # 根据用例推荐k值
        recommended_k = self.search_interface.recommend_k(use_case)
        if k == 10:
            k = recommended_k
        
        return self.search_interface.search_batch(nmr_data_list, k)
    
    def get_info(self) -> Dict:
        """获取系统信息"""
        return {
            "total_molecules": self.search_interface.performance_stats.get("total_data", 0),
            "feature_dimension": self.search_interface.feature_dim,
            "device": self.search_interface.device,
            "search_stats": self.search_interface.get_performance_stats()
        }


# 全局API实例
_api_instance = None

def get_api():
    """获取API实例（单例模式）"""
    global _api_instance
    if _api_instance is None:
        _api_instance = NMRSimpleAPI()
    return _api_instance


# 便捷函数
def search_nmr(nmr_data: Union[List, Tuple, np.ndarray, Dict], 
                 k: int = 10, 
                 use_case: str = "general") -> List[Dict]:
    """
    搜索相似的NMR数据
    
    Args:
        nmr_data: NMR数据 (化学位移列表或字典)
        k: 返回结果数量
        use_case: 使用场景
        
    Returns:
        搜索结果列表
        
    Example:
        >>> results = search_nmr([192.6, 128.68, 135.12], k=5)
        >>> print(f"找到 {len(results)} 个相似分子")
    """
    api = get_api()
    return api.search(nmr_data, k, use_case)


def search_nmr_diverse(nmr_data: Union[List, Tuple, np.ndarray, Dict], 
                      k: int = 20) -> List[Dict]:
    """搜索多样化的相似NMR数据"""
    return search_nmr(nmr_data, k, "diverse")


def search_nmr_precise(nmr_data: Union[List, Tuple, np.ndarray, Dict], 
                      k: int = 5) -> List[Dict]:
    """搜索精确的相似NMR数据"""
    return search_nmr(nmr_data, k, "precise")


def search_nmr_exploratory(nmr_data: Union[List, Tuple, np.ndarray, Dict], 
                          k: int = 50) -> List[Dict]:
    """探索性搜索相似NMR数据"""
    return search_nmr(nmr_data, k, "exploratory")


def batch_search_nmr(nmr_data_list: List[Union[List, Tuple, np.ndarray, Dict]], 
                     k: int = 10,
                     use_case: str = "general") -> List[List[Dict]]:
    """批量搜索相似NMR数据"""
    api = get_api()
    return api.batch_search(nmr_data_list, k, use_case)


def format_results(results: List[Dict], 
                  max_display: int = 5,
                  show_smiles: bool = True,
                  show_nmr: bool = False) -> str:
    """
    格式化搜索结果为可读字符串
    
    Args:
        results: 搜索结果列表
        max_display: 最大显示数量
        show_smiles: 是否显示SMILES
        show_nmr: 是否显示NMR数据
        
    Returns:
        格式化的结果字符串
    """
    if not results:
        return "未找到相似结果"
    
    output = [f"找到 {len(results)} 个相似分子:\n"]
    
    for i, result in enumerate(results[:max_display]):
        output.append(f"排名 {result['rank']}:")
        output.append(f"  键: {result['key']}")
        output.append(f"  距离: {result.get('distance', 'N/A'):.4f}")
        
        if show_smiles and result.get('canonical_smiles'):
            output.append(f"  SMILES: {result['canonical_smiles']}")
        
        if show_nmr and result.get('C_nmr'):
            nmr = result['C_nmr'][:5]  # 只显示前5个
            output.append(f"  NMR: {nmr}...")
        
        output.append("")
    
    if len(results) > max_display:
        output.append(f"... 还有 {len(results) - max_display} 个结果")
    
    return "\n".join(output)


def demo_simple_api():
    """简易API演示"""
    print("=" * 60)
    print("简易NMR搜索API演示")
    print("=" * 60)
    
    # 示例NMR数据
    nmr_query = [192.6, 128.68, 135.12, 121.45, 118.23]
    
    print(f"查询NMR: {nmr_query}")
    print()
    
    # 使用不同搜索方式
    print("1. 一般搜索 (k=5):")
    results = search_nmr(nmr_query, k=5)
    print(format_results(results, max_display=3))
    
    print("2. 精确搜索 (k=3):")
    results = search_nmr_precise(nmr_query, k=3)
    print(format_results(results, max_display=2))
    
    print("3. 多样化搜索 (k=8):")
    results = search_nmr_diverse(nmr_query, k=8)
    print(format_results(results, max_display=3))
    
    print("4. 探索性搜索 (k=15):")
    results = search_nmr_exploratory(nmr_query, k=15)
    print(format_results(results, max_display=3))
    
    # 批量搜索演示
    print("5. 批量搜索:")
    query_list = [
        [192.6, 128.68],  # 短查询
        [105.7, 114.2, 118.7],  # 中等查询
        nmr_query  # 完整查询
    ]
    
    batch_results = batch_search_nmr(query_list, k=3)
    for i, results in enumerate(batch_results):
        print(f"  查询 {i+1}: {len(results)} 个结果")


# 使用示例
if __name__ == "__main__":
    demo_simple_api()
    
    print("\n" + "=" * 60)
    print("API使用示例")
    print("=" * 60)
    
    # 示例1: 基本搜索
    print("# 基本搜索示例")
    print("from nmr_search_api import search_nmr")
    print("results = search_nmr([192.6, 128.68, 135.12], k=5)")
    print()
    
    # 示例2: 不同用例
    print("# 不同用例示例")
    print("from nmr_search_api import search_nmr_precise, search_nmr_diverse")
    print("precise_results = search_nmr_precise([192.6, 128.68, 135.12])")
    print("diverse_results = search_nmr_diverse([192.6, 128.68, 135.12])")
    print()
    
    # 示例3: 批量搜索
    print("# 批量搜索示例")
    print("from nmr_search_api import batch_search_nmr")
    print("queries = [[192.6, 128.68], [105.7, 114.2, 118.7]]")
    print("batch_results = batch_search_nmr(queries, k=5)")
    print()
    
    # 示例4: 字典格式
    print("# 字典格式示例")
    print("nmr_dict = {'C_nmr': [192.6, 128.68, 135.12]}")
    print("results = search_nmr(nmr_dict, k=10)")
"""
增强的numpy兼容性模块
解决numpy 1.x和2.x之间的数据序列化兼容性问题
"""

import numpy as np
import sys
import types
import io
import pickle


class NumpyCompatibilityManager:
    """管理numpy版本兼容性的类"""
    
    def __init__(self):
        self.setup_compatibility_layers()
    
    def setup_compatibility_layers(self):
        """设置兼容性层"""
        numpy_version = np.__version__
        print(f"检测到NumPy版本: {numpy_version}")
        
        # 检查是否需要创建numpy._core兼容性层
        if not hasattr(np, '_core'):
            self._create_numpy_core_compatibility()
        
        # 如果是numpy 1.x，创建额外的兼容性函数
        if numpy_version.startswith('1.'):
            self._create_numpy1x_compatibility()
        
        print("NumPy兼容性层已设置完成")
    
    def _create_numpy_core_compatibility(self):
        """创建numpy._core模块"""
        try:
            # 创建假的numpy._core模块结构
            np_core = types.ModuleType('numpy._core')
            
            # 创建numeric模块
            np_core_numeric = types.ModuleType('numpy._core.numeric')
            
            # 从当前numpy复制基本函数
            basic_functions = [
                'asarray', 'array', 'zeros', 'ones', 'empty', 'full',
                'arange', 'linspace', 'logspace', 'geomspace',
                'frombuffer', 'fromfile', 'fromstring', 'fromiter',
                'concatenate', 'stack', 'vstack', 'hstack', 'dstack',
                'split', 'array_split', 'hsplit', 'vsplit', 'dsplit',
                'transpose', 'swapaxes', 'reshape', 'ravel', 'flatten',
                'squeeze', 'expand_dims', 'atleast_1d', 'atleast_2d', 'atleast_3d'
            ]
            
            for func_name in basic_functions:
                if hasattr(np, func_name):
                    setattr(np_core_numeric, func_name, getattr(np, func_name))
            
            # 添加特殊的内部函数
            np_core_numeric._frombuffer = np.frombuffer
            np_core_numeric._fromstring = np.fromstring
            np_core_numeric._fromfile = np.fromfile
            
            # 添加类型对象
            if hasattr(np, 'dtype'):
                np_core_numeric.dtype = np.dtype
            if hasattr(np, 'ndarray'):
                np_core_numeric.ndarray = np.ndarray
            
            # 创建multiarray_umath模块
            np_core_multiarray = types.ModuleType('numpy._core.multiarray')
            if hasattr(np, 'ndarray'):
                np_core_multiarray.ndarray = np.ndarray
            if hasattr(np, 'dtype'):
                np_core_multiarray.dtype = np.dtype
                
            # 创建_internal模块
            np_core_internal = types.ModuleType('numpy._core._internal')
            if hasattr(np, 'ndarray'):
                np_core_internal.ndarray = np.ndarray
            
            # 添加到sys.modules
            sys.modules['numpy._core'] = np_core
            sys.modules['numpy._core.numeric'] = np_core_numeric
            sys.modules['numpy._core.multiarray'] = np_core_multiarray
            sys.modules['numpy._core._multiarray_umath'] = np_core_multiarray
            sys.modules['numpy._core._internal'] = np_core_internal
            
            # 添加到numpy模块
            np._core = np_core
            np_core.numeric = np_core_numeric
            np_core.multiarray = np_core_multiarray
            np_core._internal = np_core_internal
            
            print("✅ numpy._core 兼容性层创建成功")
            
        except Exception as e:
            print(f"⚠️ numpy._core 兼容性层创建失败: {e}")
    
    def _create_numpy1x_compatibility(self):
        """为numpy 1.x创建额外的兼容性"""
        try:
            # 创建一些numpy 2.x中存在但1.x中不存在的函数
            if not hasattr(np, 'format_float_positional'):
                np.format_float_positional = lambda x, **kwargs: str(float(x))
            
            print("✅ numpy 1.x 兼容性增强完成")
            
        except Exception as e:
            print(f"⚠️ numpy 1.x 兼容性增强失败: {e}")
    
    def safe_pickle_loads(self, data):
        """安全地加载pickle数据，处理numpy兼容性问题"""
        try:
            # 首先尝试标准pickle
            return pickle.loads(data)
        except (ModuleNotFoundError, AttributeError) as e:
            if 'numpy._core' in str(e):
                print(f"检测到numpy兼容性问题，尝试修复...")
                # 重新设置兼容性层
                self.setup_compatibility_layers()
                try:
                    return pickle.loads(data)
                except Exception as e2:
                    print(f"修复后仍然失败: {e2}")
                    # 尝试其他方法
                    return self._alternative_load(data)
            else:
                raise e
    
    def _alternative_load(self, data):
        """替代的数据加载方法"""
        try:
            # 尝试使用io.BytesIO
            with io.BytesIO(data) as f:
                # 设置不同的pickle协议
                for protocol in [2, 3, 4, 5]:
                    try:
                        f.seek(0)
                        # 尝试不同的unpickler
                        import pickle
                        unpickler = pickle.Unpickler(f)
                        unpickler.encoding = 'latin1'
                        return unpickler.load()
                    except:
                        continue
                        
        except Exception as e:
            print(f"所有替代方法都失败了: {e}")
            raise


# 全局兼容性管理器
_compatibility_manager = None

def get_compatibility_manager():
    """获取兼容性管理器实例"""
    global _compatibility_manager
    if _compatibility_manager is None:
        _compatibility_manager = NumpyCompatibilityManager()
    return _compatibility_manager

def setup_numpy_compatibility():
    """设置numpy兼容性的快捷函数"""
    manager = get_compatibility_manager()
    return manager

def safe_loads(data):
    """安全加载数据的快捷函数"""
    manager = get_compatibility_manager()
    return manager.safe_pickle_loads(data)


# 自动设置兼容性
if __name__ != "__main__":
    setup_numpy_compatibility()
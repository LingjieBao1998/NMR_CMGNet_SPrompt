import faiss
import torch
import numpy as np
import cv2
from PIL import Image
import os
import pickle
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import gc
from pathlib import Path
import time
import psutil
import lmdb
from infer import ModelInference
import pickle
import copy
from torch.utils.data import Dataset


def get_feature_from_json(json_list,
                          save_name,
                          model_inference,
                          n=256,
                          type="nmr",
                          flag_get_value=False):
    context = []
    print("start parse json")
    for file in json_list:
        with open(file, "r") as f:
            context_tmp = json.loads(f.read())
            context_tmp = [
                i[type][0] for i in tqdm(context_tmp) if len(i[type]) > 0
            ]
        context += context_tmp
    print("Size of the library: ", len(context))
    if flag_get_value == "only":
        return context
    if type == "nmr":
        fn = model_inference.nmr_encode
    elif type == "smiles":
        fn = model_inference.smiles_encode
    contexts = []
    print("start load batch")
    for i in range(0, len(context), n):
        contexts.append(context[i:i + n])
    print("start encode batch")
    result = [fn(i) for i in tqdm(contexts)]
    result = torch.cat(result, 0)
    if flag_get_value is True:
        if save_name is not None:
            torch.save((result, context), save_name)
        return result, context

    if save_name is not None:
        torch.save(result, save_name)
    return result


def get_topK_result(nmr_feature, smiles_feature, topK):
    indices = []
    scores = []
    with torch.no_grad():
        for i in tqdm(nmr_feature):
            nmr_smiles_distances_tmp = (
                i.unsqueeze(0) @ smiles_feature.t()).cpu()
        scores_, indices_ = nmr_smiles_distances_tmp.topk(topK,
                                                          dim=1,
                                                          largest=True,
                                                          sorted=True)
        indices.append(indices_)
        scores.append(scores_)
    indices = torch.cat(indices, 0)
    scores = torch.cat(scores, 0)
    return indices, scores

env = lmdb.open(
                    "/home/lingjiebao/project/NMR_Core_Mol_Detection/data/NMR_other_data_1020/new_db.lmdb",
                    readonly=True,
                    map_size= 1000 * 1024**3,
                    lock=False,
                    readahead=False,
                    meminit=False,
                    max_readers=1024,
                )
length = env.stat()['entries']                    
key_list = [f'{count}'.encode('ascii') for count in range(length) if count < 10000]
print(f'len(key_list):{len("key_list")}')



class LargeScale768DImageSearchEngine:
    def __init__(self, model_name="openai/clip-vit-large-patch14"):
        """
        初始化768维特征的大规模GPU图像搜索引擎
        """
        # 检查GPU可用性
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {self.device}")
        if torch.cuda.is_available():
            gpu_props = torch.cuda.get_device_properties(0)
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU内存: {gpu_props.total_memory / 1024**3:.1f} GB")
        
        config_path = "models/2_5_w_model/8.json"
        pretrain_model_path = "models/2_5_w_model/8.pth"
        self.model_inference = ModelInference(config_path=config_path,
                                        pretrain_model_path=pretrain_model_path,
                                        device=self.device)
        
        
        
        # 特征维度
        self.feature_dim = 768
        print(f"特征维度: {self.feature_dim}")
        
        # FAISS索引配置
        self.index = None
        self.image_paths = []
        
        # GPU资源配置
        self.res = None  # Will be initialized when needed
        
        # 针对768维优化的参数
        self.batch_size = 256  # 768维特征需要更多内存，减小batch_size
        self.chunk_size = 50000  # 数据块大小
        
        # 索引配置 (减少聚类中心数量以适应测试数据)
        self.nlist = 1000  # 适中的聚类中心数量
        self.m = 16  # 子量化器数量（768/16=48，每个子量化器处理48维）
        self.bits = 8  # 每个子量化器的比特数
        self.nprobe = 100  # 默认搜索的聚类中心数量
    
    def clear_gpu_cache(self):
        """清理GPU缓存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def get_memory_info(self):
        """获取内存使用信息"""
        process = psutil.Process()
        memory_usage = process.memory_info().rss / 1024 ** 3  # GB
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024 ** 3  # GB
            gpu_max = torch.cuda.max_memory_allocated() / 1024 ** 3  # GB
            return f"内存: {memory_usage:.1f}GB, GPU: {gpu_memory:.1f}GB (峰值: {gpu_max:.1f}GB)"
        return f"内存使用: {memory_usage:.1f}GB"
    
    def extract_features_batch(self, image_paths):
        """
        批量提取768维特征（内存优化版本）
        """
        features_list = []
        valid_paths = []
        failed_count = 0
        
        for i in tqdm(range(0, len(image_paths), self.batch_size), desc="提取768维特征"):
            batch_paths = image_paths[i:i + self.batch_size]
            batch_images = []
            batch_valid_paths = []
            
            # 加载图像批次
            for img_path in batch_paths:
                try:
                    image = Image.open(img_path).convert('RGB')
                    # 对于768维特征，保持较高分辨率以获得更好效果
                    if max(image.size) > 224:  # CLIP模型输入尺寸
                        image = image.resize((224, 224))
                    batch_images.append(image)
                    batch_valid_paths.append(img_path)
                except Exception as e:
                    failed_count += 1
                    continue
            
            if not batch_images:
                continue
            
            # 预处理和特征提取
            try:
                inputs = self.processor(images=batch_images, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # 提取768维特征
                with torch.no_grad():
                    batch_features = self.model.get_image_features(**inputs)
                    # L2归一化（重要：768维特征需要归一化）
                    batch_features = torch.nn.functional.normalize(batch_features, p=2, dim=1)
                
                features_list.append(batch_features.cpu().numpy().astype('float32'))
                valid_paths.extend(batch_valid_paths)
                
                # 清理内存
                del inputs, batch_features, batch_images
                if i % 1000 == 0:  # 定期清理
                    self.clear_gpu_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("GPU内存不足，减小batch_size")
                    self.batch_size = max(32, self.batch_size // 2)
                    print(f"新的batch_size: {self.batch_size}")
                    self.clear_gpu_cache()
                    # 重新尝试当前批次
                    continue
                else:
                    print(f"特征提取错误: {e}")
                    failed_count += len(batch_paths)
        
        if failed_count > 0:
            print(f"跳过 {failed_count} 张无法处理的图像")
        
        if features_list:
            all_features = np.vstack(features_list)
            print(f"特征提取完成: {all_features.shape}")
            return all_features, valid_paths
        return None, []
    
    # def collate_fn(self, batch):
    #     max_encoder_length = max([len(batch[_]["input_ids"]) for _ in range(len(batch))])
    #     idx_list = [batch[_]["idx"] for _ in range(len(batch))]
    #     for _ in range(len(batch)):
    #         if len(batch[_]["input_ids"]) < max_encoder_length:
    #             batch[_]["attention_mask"] = batch[_]["attention_mask"] + [0]*(max_encoder_length-len(batch[_]["input_ids"]))
    #             batch[_]["input_ids"] = batch[_]["input_ids"] + [self.tokenizer.pad_token_id]*(max_encoder_length-len(batch[_]["input_ids"]))

    #     input_ids = torch.tensor([batch[_]["input_ids"] for _ in range(len(batch))]).long()[:, :self.max_length]
    #     attention_mask = torch.tensor([batch[_]["attention_mask"] for _ in range(len(batch))]).long()[:, :self.max_length]
        
    #     return {
    #         "input_ids":input_ids,
    #         "attention_mask":attention_mask,
    #         "idx_list":idx_list
    #     }
       
    
    def build_index_large_scale(self, key_list=[], index_path="data/query_dataset/large_scale_768d_index.pkl", key_list_path="data/query_dataset/valid_keylist.pickle"):
        """
        构建300万数据量的768维特征索引（基于LMDB数据库）
        """
        print("开始构建768维特征的大规模索引...")
        print(f"索引配置: nlist={self.nlist}, m={self.m}, bits={self.bits}")
        start_time = time.time()
        
        print(f"找到 {len(key_list):,} 条数据")

        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        
        if len(key_list) == 0:
            raise ValueError("未找到任何数据")
        
        # 分块处理数据
        all_features = []
        all_valid_keys = []
        total_processed = 0
        failed_count = 0
        
        # 分块处理，避免内存溢出
        chunk_size = 50000  # 每个块处理5万条数据
        total_chunks = (len(key_list) - 1) // chunk_size + 1
        
        for chunk_idx in range(total_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, len(key_list))
            chunk_keys = key_list[start_idx:end_idx]
            
            print(f"\n=== 处理数据块 {chunk_idx + 1}/{total_chunks} ===")
            print(f"数据块大小: {len(chunk_keys):,}")
            print(self.get_memory_info())
            
            chunk_features = []
            chunk_valid_keys = []

              
            # 处理当前块的数据 - 处理前100个用于调试
            debug_keys = chunk_keys[:100]
            for i, key in enumerate(tqdm(debug_keys, desc=f"处理块 {chunk_idx + 1} (调试模式)")):
                try:
                    with env.begin() as txn:
                        data = txn.get(key)
                        try:
                            sample = pickle.loads(data)
                        except (TypeError, AttributeError) as pickle_e:
                            print(f"Unpickle error: {type(pickle_e).__name__}: {str(pickle_e)[:100]}")
                            # Try alternative unpickling methods
                            import pickle5
                            try:
                                sample = pickle5.loads(data)
                            except:
                                # If that fails, skip this sample
                                raise pickle_e
                    
                    # 提取NMR数据
                    if "C_nmr" in sample:
                        ori_nmr = sample["C_nmr"]
                    else:
                        ori_nmr = sample["NMR"]
                    
                    if len(ori_nmr) > 0:
                        # 使用模型进行特征提取
                        try:
                            nmr_feature = self.model_inference.nmr_encode(ori_nmr)
                        except Exception as model_e:
                            print(f"模型推理错误: {type(model_e).__name__}: {str(model_e)[:100]}")
                            raise
                            
                        # 确保特征维度正确
                        if nmr_feature is not None:
                            try:
                                feature_array = nmr_feature.cpu().numpy()
                            except Exception as numpy_e:
                                print(f"numpy转换错误: {type(numpy_e).__name__}: {str(numpy_e)[:100]}")
                                raise
                            
                            # 如果是1D数组，转换为2D
                            if len(feature_array.shape) == 1:
                                feature_array = feature_array.reshape(1, -1)
                            
                            # 验证维度
                            if feature_array.shape[1] != self.feature_dim:
                                print(f"警告: 特征维度不匹配，期望{self.feature_dim}，实际{feature_array.shape[1]}")
                                # 如果维度不匹配，尝试调整或跳过
                                if feature_array.shape[1] > self.feature_dim:
                                    feature_array = feature_array[:, :self.feature_dim]
                                else:
                                    # 填充或跳过
                                    continue
                            
                            chunk_features.append(feature_array)
                            chunk_valid_keys.append(key)
                            
                except Exception as e:
                    failed_count += 1
                    if failed_count % 100 == 0:
                        print(f"跳过失败数量: {failed_count}, 最新错误: {type(e).__name__}: {str(e)[:50]}")
                    continue
            
            # 合并当前块的特征
            if chunk_features:
                chunk_features_array = np.vstack(chunk_features)
                all_features.append(chunk_features_array)
                all_valid_keys.extend(chunk_valid_keys)
                total_processed += len(chunk_valid_keys)
                
                print(f"数据块 {chunk_idx + 1} 完成，处理 {len(chunk_valid_keys):,} 条数据")
                print(f"累计处理: {total_processed:,} 条数据")
                print(f"特征维度: {chunk_features_array.shape}")
            
            # 定期清理内存
            if chunk_idx % 5 == 0:  # 每5个块清理一次
                gc.collect()
                self.clear_gpu_cache()
        
        # 构建索引
        if all_features:
            print("\n=== 开始构建FAISS索引 ===")
            print(f"总共处理了 {total_processed:,} 条有效数据")
            
            # 合并所有特征
            if len(all_features) > 1:
                features_array = np.vstack(all_features)
            else:
                features_array = all_features[0]
            
            print(f"最终特征矩阵形状: {features_array.shape}")
            
            # 构建索引
            self._build_faiss_index_768d_single_batch(features_array, all_valid_keys)
            
            # 保存索引
            self.save_index(index_path)
            
            ## 保存lmdb的索引
            with open(key_list_path, 'wb') as f:
                pickle.dump(all_valid_keys, f)
            
            total_time = time.time() - start_time
            hours = total_time // 3600
            minutes = (total_time % 3600) // 60
            seconds = total_time % 60
            
            print(f"\n=== 索引构建完成 ===")
            print(f"总耗时: {hours:.0f}小时{minutes:.0f}分{seconds:.0f}秒")
            print(f"索引包含 {self.index.ntotal:,} 个768维向量")
            print(f"失败处理数量: {failed_count}")
            
            # 估算索引大小
            if hasattr(self.index, 'ntotal'):
                raw_size_gb = self.index.ntotal * self.feature_dim * 4 / 1024**3
                print(f"索引大小估算: {raw_size_gb:.2f} GB (原始)")
        else:
            raise ValueError("没有成功提取到任何特征")

    def _build_faiss_index_768d_single_batch(self, features, valid_keys):
        """构建768维特征的FAISS索引（单批次版本）"""
        print("初始化768维FAISS索引...")
        
        # 创建针对768维优化的IVFPQ索引
        quantizer = faiss.IndexFlatIP(self.feature_dim)  # 使用内积距离
        
        # 使用IVFPQ索引节省内存并保持精度
        self.index = faiss.IndexIVFPQ(quantizer, self.feature_dim, self.nlist, self.m, self.bits)
        
        # 配置编码器
        self.index.verbose = True
        
        # 移动到GPU
        print("将索引移动到GPU...")
        try:
            gpu_index = faiss.index_cpu_to_all_gpus(self.index)
            self.index = gpu_index
        except Exception as e:
            print(f"GPU move failed: {e}, using CPU")
        
        # 训练索引（使用部分数据）
        print("训练索引...")
        training_samples_needed = min(100000, self.nlist * 10)  # 至少10倍nlist
        
        if features.shape[0] > training_samples_needed:
            # 随机采样训练数据
            indices = np.random.choice(features.shape[0], training_samples_needed, replace=False)
            training_data = features[indices]
        else:
            training_data = features
        
        print(f"使用 {training_data.shape[0]:,} 个样本进行训练")
        self.index.train(training_data)
        
        # 添加所有数据到索引
        print("添加数据到索引...")
        
        # 分批添加以避免内存问题
        add_batch_size = 20000  # 768维特征需要更小的批次
        total_vectors = features.shape[0]
        
        for i in range(0, total_vectors, add_batch_size):
            batch_end = min(i + add_batch_size, total_vectors)
            batch_features = features[i:batch_end]
            self.index.add(batch_features)
            
            if i % 50000 == 0:
                print(f"已添加 {self.index.ntotal:,} 个向量")
                self.clear_gpu_cache()
        
        # 保存有效的key列表
        self.valid_keys = valid_keys
        
        print(f"索引构建完成，总计 {self.index.ntotal:,} 个768维向量")

    def search_similar_nmr(self, query_nmr, top_k=10, nprobe=None):
        """
        搜索相似NMR数据
        """
        if self.index is None:
            raise ValueError("请先构建索引")
        
        # 设置搜索参数
        if nprobe is None:
            nprobe = self.nprobe
        self.index.nprobe = nprobe
        
        print(f"搜索参数: top_k={top_k}, nprobe={nprobe}")
        
        # 提取查询NMR特征
        query_features = self.model_inference.nmr_encode(query_nmr)
        if query_features is None:
            return []
        
        query_features = query_features.cpu().numpy().astype('float32')
        
        # 确保维度正确
        if len(query_features.shape) == 1:
            query_features = query_features.reshape(1, -1)
        
        # 搜索相似数据
        distances, indices = self.index.search(query_features, top_k)
        
        # 获取结果
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if 0 <= idx < len(self.valid_keys):
                results.append({
                    'key': self.valid_keys[idx],
                    'similarity': float(distance),  # 内积距离，值越大越相似
                    'rank': i + 1
                })
        
        return results

    def save_index(self, index_path):
        """保存索引"""
        print("保存768维特征索引...")
        
        # 将索引移回CPU
        index_cpu = faiss.index_gpu_to_cpu(self.index)
        
        data = {
            'index': index_cpu,
            'valid_keys': self.valid_keys,
            'feature_dim': self.feature_dim,
            'config': {
                'nlist': self.nlist,
                'm': self.m,
                'bits': self.bits
            }
        }
        
        with open(index_path, 'wb') as f:
            pickle.dump(data, f)
        
        # 估算索引大小
        index_size_mb = os.path.getsize(index_path) / 1024**2
        print(f"索引已保存到: {index_path}")
        print(f"索引文件大小: {index_size_mb:.2f} MB")

    def load_index(self, index_path):
        """加载索引"""
        print("加载768维特征索引...")
        
        with open(index_path, 'rb') as f:
            data = pickle.load(f)
        
        try:
            self.index = faiss.index_cpu_to_all_gpus(data['index'])
        except Exception as e:
            print(f"GPU move failed: {e}, using CPU")
            self.index = data['index']
        self.valid_keys = data['valid_keys']
        self.feature_dim = data['feature_dim']
        
        print(f"索引加载完成，包含 {len(self.valid_keys):,} 条数据")
        print(f"特征维度: {self.feature_dim}")
    
    def get_index_info(self):
        """获取索引信息"""
        if self.index is None:
            return "索引未加载"
        
        info = {
            '总向量数': f"{self.index.ntotal:,}",
            '特征维度': self.feature_dim,
            '图像数量': f"{len(self.image_paths):,}",
            '索引类型': type(self.index).__name__,
            'nlist': self.nlist,
            'm': self.m,
            'bits': self.bits
        }
        
        return info


# 使用示例
def main():
    # 初始化768维搜索引擎
    search_engine = LargeScale768DImageSearchEngine()
    
    # 图像文件夹路径
    # image_folder = "path/to/your/3million/images"  # 替换为实际路径
    # index_path = "large_scale_768d_index.pkl"

    # Create test data instead of using problematic LMDB
    print("创建测试数据...")
    test_features = np.random.random((10000, 768)).astype('float32')
    test_keys = [f'test_key_{i}'.encode() for i in range(10000)]
    
    print("构建测试索引...")
    search_engine._build_faiss_index_768d_single_batch(test_features, test_keys)
    
    # 构建或加载索引
    # if not os.path.exists(index_path):
    #     print("构建新索引...")
    #     search_engine.build_index_large_scale()
    # else:
    #     print("加载现有索引...")
    #     search_engine.load_index(index_path)
    
    # 显示索引信息
    info = search_engine.get_index_info()
    print("\n索引信息:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # # 搜索示例
    # query_image = "path/to/query/image.jpg"
    # print(f"\n搜索相似图像: {query_image}")
    
    # results = search_engine.search_similar_images(query_image, top_k=5, nprobe=200)
    
    # print("搜索结果:")
    # for result in results:
    #     print(f"排名 {result['rank']}: {result['image_path']} (相似度: {result['similarity']:.4f})")
    
    # # 文本搜索示例
    # query_text = "a beautiful sunset with mountains"
    # print(f"\n文本搜索: '{query_text}'")
    
    # text_results = search_engine.search_by_text(query_text, top_k=3)
    
    # print("文本搜索结果:")
    # for result in text_results:
    #     print(f"排名 {result['rank']}: {result['image_path']} (相似度: {result['similarity']:.4f})")

if __name__ == "__main__":
    main()
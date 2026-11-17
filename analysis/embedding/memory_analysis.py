#!/usr/bin/env python3
"""
300万条NMR数据的索引策略分析
"""

def analyze_memory_requirements():
    """分析不同规模的内存需求"""
    
    # 基本参数
    feature_dim = 768
    bytes_per_float = 4
    
    # 不同数据规模
    test_sizes = [10000, 100000, 500000, 1000000, 2000000, 3000000]
    
    print("=" * 80)
    print("NMR数据索引策略内存需求分析")
    print("=" * 80)
    print(f"特征维度: {feature_dim}")
    print(f"GPU内存: 23.5 GB (RTX 4090)")
    print(f"可用内存 (80%): {23.5 * 0.8:.1f} GB")
    print()
    
    for size in test_sizes:
        print(f"数据规模: {size:,} 条")
        print("-" * 60)
        
        # 精确索引内存需求
        exact_memory_gb = size * feature_dim * bytes_per_float / (1024**3)
        
        # PQ压缩索引内存需求
        pq_bits_per_vector = 16 * 8  # 16 subquantizers × 8 bits
        pq_memory_gb = size * pq_bits_per_vector / 8 / (1024**3)
        
        # IVF聚类中心内存
        n_lists = min(10000, max(1000, size // 300))
        centroids_memory_gb = n_lists * feature_dim * bytes_per_float / (1024**3)
        
        total_pq_memory = pq_memory_gb + centroids_memory_gb
        
        print(f"精确索引 (IndexFlatL2):")
        print(f"  内存需求: {exact_memory_gb:.2f} GB")
        print(f"  搜索速度: 最快 (精确搜索)")
        print(f"  压缩比: 1.0x")
        
        print(f"\nPQ压缩索引 (IndexIVFPQ):")
        print(f"  向量存储: {pq_memory_gb:.2f} GB")
        print(f"  聚类中心: {centroids_memory_gb:.2f} GB")
        print(f"  总内存: {total_pq_memory:.2f} GB")
        print(f"  搜索速度: 很快 (近似搜索)")
        print(f"  压缩比: {exact_memory_gb / total_pq_memory:.1f}x")
        
        # 推荐
        available_gpu_memory = 23.5 * 0.8
        
        if exact_memory_gb <= available_gpu_memory:
            recommendation = "🟢 精确索引 (IndexFlatL2)"
            reason = "GPU内存充足，推荐精确搜索以获得最佳效果"
        elif total_pq_memory <= available_gpu_memory:
            recommendation = "🟡 PQ压缩索引 (IndexIVFPQ)"
            reason = "GPU内存适中，PQ压缩在性能和精度间平衡"
        else:
            recommendation = "🔴 内存不足"
            reason = "需要分批处理或使用CPU索引"
        
        print(f"\n推荐策略: {recommendation}")
        print(f"原因: {reason}")
        
        # 搜索性能预期
        if exact_memory_gb <= available_gpu_memory:
            print(f"预期搜索性能: < 1ms (GPU精确搜索)")
        elif total_pq_memory <= available_gpu_memory:
            print(f"预期搜索性能: 1-5ms (GPU近似搜索)")
        else:
            print(f"预期搜索性能: 需要分批处理")
        
        print("=" * 80)
    
    # 针对300万条数据的详细分析
    size = 3000000
    exact_memory_gb = size * feature_dim * bytes_per_float / (1024**3)
    pq_bits_per_vector = 16 * 8
    pq_memory_gb = size * pq_bits_per_vector / 8 / (1024**3)
    n_lists = min(10000, max(1000, size // 300))
    centroids_memory_gb = n_lists * feature_dim * bytes_per_float / (1024**3)
    total_pq_memory = pq_memory_gb + centroids_memory_gb
    
    print("\n" + "=" * 80)
    print("300万条数据详细分析")
    print("=" * 80)
    print(f"精确索引内存需求: {exact_memory_gb:.2f} GB")
    print(f"PQ压缩索引内存需求: {total_pq_memory:.2f} GB")
    print(f"内存节省: {exact_memory_gb - total_pq_memory:.2f} GB")
    print(f"压缩比: {exact_memory_gb / total_pq_memory:.1f}x")
    
    print(f"\n针对300万条数据的建议:")
    if exact_memory_gb <= 18.8:  # 23.5 * 0.8
        print("✅ 使用精确索引 (IndexFlatL2)")
        print("   优势: 搜索精度最高，速度最快")
        print("   劣势: 占用GPU内存最多")
        print("   适用场景: 对搜索精度要求极高的应用")
    else:
        print("✅ 使用PQ压缩索引 (IndexIVFPQ)")
        print("   优势: 大幅减少内存使用")
        print("   劣势: 轻微的搜索精度损失")
        print("   适用场景: 大规模生产环境推荐")
    
    print(f"\nPQ压缩参数建议:")
    print(f"  聚类数量 (n_lists): {n_lists:,}")
    print(f"  子量化器数量 (pq_dim): 16")
    print(f"  量化比特数 (pq_bits): 8")
    print(f"  搜索聚类数 (n_probes): {min(1000, n_lists//10):,}")
    
    # 索引构建时间估算
    print(f"\n性能预期:")
    print(f"  索引构建时间: 2-5分钟 (GPU加速)")
    print(f"  单次搜索时间: 1-3ms")
    print(f"  QPS (每秒查询): 300-1000")


if __name__ == "__main__":
    analyze_memory_requirements()
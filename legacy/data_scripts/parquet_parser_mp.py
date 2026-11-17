import os
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# 定义函数：单个进程处理一个文件
def process_file(file_name):
    file_path = os.path.join(file_dir, file_name)
    try:
        df = pd.read_parquet(file_path)
        df_selected = df[["smiles", "c_nmr_peaks"]]  # 只选取需要的列
        return df_selected
    except Exception as e:
        print(f"Error processing file {file_name}: {e}")
        return pd.DataFrame()  # 出错时返回空的 DataFrame

if __name__ == "__main__":
    # 定义路径
    file_dir = "./data/multimodal_spectroscopic_dataset/"
    save_dir = "./data/multimodal_spectroscopic_dataset_processed/"

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 获取所有 parquet 文件名（可根据需要过滤，比如只处理 .parquet 后缀）
    file_names = [f for f in sorted(os.listdir(file_dir)) if f.endswith('.parquet')]

    file_names = file_names[:5]

    # 多进程池处理：一般设置为 CPU 核心数
    num_processes = min(cpu_count(), len(file_names), 5)  # 不超过文件数和 CPU 核数
    print(f"Using {num_processes} processes to process {len(file_names)} files...")

    # 使用进程池并行处理
    with Pool(processes=num_processes) as pool:
        # 使用 tqdm 显示进度条
        results = list(tqdm(pool.imap(process_file, file_names), total=len(file_names)))

    # 拼接所有结果
    df_all = pd.concat(results, ignore_index=True)
    df_all = df_all.reset_index(drop=True)

    # 保存为 CSV
    output_path = os.path.join(save_dir, "multimodal_spectroscopic_dataset_5.csv")
    df_all.to_csv(output_path, index=False)
    print(f"Saved combined CSV to: {output_path}")
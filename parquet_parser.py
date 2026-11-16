import os
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm


if __name__ == "__main__":

    # file_path = "./data/multimodal_spectroscopic_dataset/aligned_chunk_142.parquet"
    # df = pd.read_parquet(file_path)
    # print(df.head())     # 查看前5行
    # print(df.info())     # 查看列名、数据类型、非空情况
    # print(df.columns)    # 查看所有列名
    # df["c_nmr_peaks"][0]
    #array([{'delta (ppm)': 167.7979990389225, 'integral': 0.0018874415298058, 'intensity': 0.1005720495247954, 'width (ppm)': 0.0119376908527277},
    #   {'delta (ppm)': 164.9977172163624, 'integral': 0.000943720628302, 'intensity': 0.050286021632343, 'width (ppm)': 0.0119376908527277},
    #   {'delta (ppm)': 35.818503984910606, 'integral': 0.0022649298713791, 'intensity': 0.1206864644921272, 'width (ppm)': 0.0119376908527277},
    #   {'delta (ppm)': 14.457771117725173, 'integral': 0.0029255342981253, 'intensity': 0.1558866705798354, 'width (ppm)': 0.0119376908527277}],
    #  dtype=object)
    
    df_list = []
    file_dir = "./data/multimodal_spectroscopic_dataset/"
    save_dir = "./data/multimodal_spectroscopic_dataset_processed/"
    for file_name in tqdm(os.listdir(file_dir)):
        file_path = os.path.join(file_dir, file_name)
        df = pd.read_parquet(file_path)
        df_ = df[["smiles", "c_nmr_peaks"]]
        df_list.append(df_)
    
    df_all = pd.concat(df_list)
    df_all.to_csv(os.path.join(save_dir, "multimodal_spectroscopic_dataset.csv"))
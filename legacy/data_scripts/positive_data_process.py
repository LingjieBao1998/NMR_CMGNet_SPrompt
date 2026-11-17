import os
import numpy as np
import pandas as pd
import json

def process_ori_result(text):
    temp_dict = json.loads(text)
    temp_list = []
    try:
        for i in range(len(temp_dict["data"]["signals"])):
            temp_list.append(temp_dict["data"]["signals"][i]["delta"])
        return temp_list
    except Exception as e:
        return None

if __name__ == "__main__":
    df0 = pd.read_csv("./data/positive_remove_duplicated_nmr.csv", index_col=0)
    df0["NMR(prediction)"] = df0["ori_result"].apply(process_ori_result)
    del df0["ori_result"]
    df0.to_csv("./data/positive_remove_duplicated_nmr_processed.csv")

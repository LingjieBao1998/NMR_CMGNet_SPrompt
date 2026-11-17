import torch
import json
from tqdm import tqdm
from infer import ModelInference
import numpy as np
import pandas as pd


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


if __name__ == "__main__":
    # Load the model
    config_path = "models/2_5_w_model/8.json"
    pretrain_model_path = "models/2_5_w_model/8.pth"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_inference = ModelInference(config_path=config_path,
                                     pretrain_model_path=pretrain_model_path,
                                     device=device)
    
    df0 = pd.read_csv("/home/baolingjie/git_download/NMR_Core_Mol_Detection/data/extest_data_0916.csv", index_col=0)
    data_dict = {}
    for i in tqdm(df0.index):
        if df0.loc[i, "nmr"] != df0.loc[i, "nmr"]:
            continue
        try:
            nmr_list = json.loads(df0.loc[i, "nmr"].replace("None,","").replace("None", ""))
            nmr_feature = model_inference.nmr_encode(nmr_list)
            data_dict[i] = nmr_feature
        except Exception as e:
            print(e)
            import traceback
            traceback.print_exc()
    torch.save(data_dict, "./data/extest_data_0916_dict.pt")

import torch
import json
from tqdm import tqdm
from infer import ModelInference
import numpy as np
import pandas as pd
import pickle


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

    with open("/home/baolingjie/git_download/NMR_Core_Mol_Detection/data/enumerate_100m_5m_1_group_restricted/last_results.pkl", "rb") as f:
        data = pickle.loads(f.read())
    
    data_ = [_ for _ in data if len(list(set(_["C_nmr"])))>0]
    data_dict = {}
    with torch.no_grad():
        for temp_data in tqdm(data_):
            try:
                nmr_list = ist(set(temp_data["C_nmr"]))
                # nmr_list = list(map(float, df0.loc[i, "clean_NMR2"].rstrip(".").rstrip(",").rstrip(";").replace(",,", ",").split(", ")))
                nmr_feature = model_inference.nmr_encode(nmr_list)
                data_dict[i] = nmr_feature.cpu().numpy()
            except Exception as e:
                print(e)
                import traceback
                traceback.print_exc()
                import ipdb
                ipdb.set_trace()
    torch.save(data_dict, "/home/baolingjie/git_download/NMR_Core_Mol_Detection/data/enumerate_100m_5m_1_group_restricted/last_results_processed.pt")
    
    import ipdb
    ipdb.set_trace()
    
    # df0 = pd.read_csv("data/test_core_mol_clean.csv", index_col=0)
    # data_dict = {}
    # for i in tqdm(df0.index):
    #     if df0.loc[i, "NMR1"] != df0.loc[i, "NMR1"]:
    #         continue
    #     try:
    #         nmr_list = list(map(float, df0.loc[i, "NMR1"].rstrip(".").rstrip(";").split(", ")))
    #         nmr_feature = model_inference.nmr_encode(nmr_list)
    #         data_dict[i] = nmr_feature
    #     except Exception as e:
    #         print(e)
    #         import traceback
    #         traceback.print_exc()
    #         import ipdb
    #         ipdb.set_trace()
    # torch.save(data_dict, "./data/positive_58_no_processed.pt")
    # data_dict = {}
    # for i in tqdm(df0.index):
    #     if df0.loc[i, "clean_NMR2"] != df0.loc[i, "clean_NMR2"]:
    #         continue
    #     try:
    #         nmr_list = list(map(float, df0.loc[i, "clean_NMR2"].rstrip(".").rstrip(",").rstrip(";").replace(",,", ",").split(", ")))
    #         nmr_feature = model_inference.nmr_encode(nmr_list)
    #         data_dict[i] = nmr_feature
    #     except Exception as e:
    #         print(e)
    #         import traceback
    #         traceback.print_exc()
    #         import ipdb
    #         ipdb.set_trace()
    # torch.save(data_dict, "./data/positive_58_processed.pt")
    # import ipdb
    # ipdb.set_trace()
    

    # # # C(O)C1=CC2(C)CCC3C(C)CCC32C1C
    # # nmr_list = [
    # #     17.7, 20.0, 22.9, 28.9, 29.9, 35.8, 37.6, 39.7, 50.9, 57.3, 61.3, 64.1,
    # #     64.9, 134.0, 146.7
    # # ]

    # # # Extract NMR spectral feature vector 
    # # nmr_feature = model_inference.nmr_encode(nmr_list)
    

    # # # Construct a reference library by extracting structural feature vectors from SMILES strings
    # # # This might take a long time
    # # smiles_feature, smiles_list = get_feature_from_json(
    # #     json_list=["data/val.json", "data/train.json"],
    # #     model_inference=model_inference,
    # #     n=64,
    # #     save_name=None,
    # #     type="smiles",
    # #     flag_get_value=True)

    # # # Get top10 candidates by searching library 
    # # indices, scores = get_topK_result(nmr_feature, smiles_feature, 10)

    # # # Print the result
    # # for (sco, idx) in zip(scores, indices):
    # #     for ii, i in enumerate(idx):
    # #         print("top:", ii, "scores:", sco[ii].item(), "smiles:",
    # #               smiles_list[i])

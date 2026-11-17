import torch
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModel
from tqdm import tqdm
import torch
import torch.nn.functional as F

import sys
sys.path.append("/home/baolingjie/.cache/huggingface/modules")

from torch.utils.data import DataLoader, Dataset

try:
    import tokenizers
    import transformers
    from transformers.models.bart.tokenization_bart_fast import BartTokenizerFast
    ## torch2.6 引入的新机制
    ## ref:https://stackoverflow.com/questions/79584485/unable-to-torch-load-due-to-pickling-safety-error
    torch.serialization.add_safe_globals(
        [
            transformers.models.bart.tokenization_bart_fast.BartTokenizerFast,
            tokenizers.Tokenizer,
            tokenizers.models.Model,
            tokenizers.AddedToken
        ]
    )
except Exception as e:
    print(e)

def get_model(model_dir="./checkpoints/saved_hf_model_0910", use_lora=False, device = "cpu"):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    model.to(device)
    encoder = model.get_encoder()
    del model
    return encoder

def get_tokenizer(model_dir="./checkpoints/saved_hf_model_0910"):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return tokenizer

class NmrDataset(Dataset):
    def __init__(self, **kwargs):
        super(NmrDataset, self).__init__()
        self.scale = kwargs.pop(
            'scale') if 'scale' in kwargs.keys() else 10
        self.min_value = kwargs.pop(
            'min_value') if 'min_value' in kwargs.keys() else -50
        self.max_value = kwargs.pop(
            'max_value') if 'max_value' in kwargs.keys() else 350
        self.augment_prob = kwargs.pop(
            'augment_prob') if 'augment_prob' in kwargs.keys() else 0.5
        # augment_range unit ±ppm
        self.augment_range = kwargs.pop(
            'augment_range') if 'augment_range' in kwargs.keys() else 1.5
        self.units = (self.max_value - self.min_value) * self.scale
        self.start = kwargs.pop(
            'start') if 'start' in kwargs.keys() else 189
        self.end = kwargs.pop(
            'end') if 'end' in kwargs.keys() else 190
        self.number_start = kwargs.pop(
            'number_start') if 'number_start' in kwargs.keys() else 191

    def get_value(self, value, augment_prob):
        if random.random() < augment_prob:
            return value
        else:
            return 0

    def fill_item(self, nmr, want_token=True):
        nmr = list(set(nmr))
        nmr = [round((value - self.min_value) * self.scale) for value in nmr]
        nmr = [min(max(0, value), self.units-1) for value in nmr]
        nmr.sort()
        if want_token:
            nmr = [189] + [value + self.number_start for value in nmr] + [190]
            item = {"input_ids": nmr,
                    "attention_mask": [1 for _ in range(len(nmr))]}
            return item
        else:
            item = np.zeros(self.units)
            item[nmr] = 1
            item = torch.from_numpy(item).to(torch.float32)
            item_ = {"input_ids": [189, 191, 190],
                     "attention_mask": [1, 1, 1]}
            return item, item_

    def augment_item(self, nmr, want_token=True):
        nmr = list(set(nmr))
        offset = np.random.normal(
            loc=0.0, scale=self.augment_range/2, size=len(nmr))
        offset = [min(max(-self.augment_range, i), self.augment_range)
                  for i in offset]
        offset = [self.get_value(i, self.augment_prob) for i in offset]
        new_nmr = [nmr_item + offset_item for nmr_item,
                   offset_item in zip(nmr, offset)]
        return self.fill_item(new_nmr, want_token=want_token)


if __name__ == "__main__":
    model = get_model()
    model = model.to("cuda:0")
    tokenizer = get_tokenizer()

    checkpoints = torch.load("/home/baolingjie/git_download/NMR_Core_Mol_Detection/output/train_nmr_bert_all_1022/checkpoints/last_ep_6.ckpt", map_location="cpu", weights_only=False)

    from collections import OrderedDict

    new_state_dict = OrderedDict()

    # 遍历原始 state_dict 的所有 key-value
    for k, v in checkpoints["state_dict"].items():
        # 去掉 key 的前缀 "model." 或 "model.model."
        if k.startswith("model."):
            new_key = k[6:]  # 去掉前 6 个字符 "model."
        elif k.startswith("model.model."):
            new_key = k[12:]  # 去掉前 12 个字符 "model.model."
        else:
            new_key = k  # 如果没有前缀，保持不变

        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict, strict=False)  # strict=True 表示严格匹配所有 key
    with open("/home/baolingjie/git_download/NMR_Core_Mol_Detection/data/enumerate_100m_5m_1_group_restricted/last_results.pkl", "rb") as f:
        data = pickle.loads(f.read())
    
    nmr_process = NmrDataset()
    
    data_ = [_ for _ in data if len(list(set(_["C_nmr"])))>0]
    data_dict = {}
    with torch.no_grad():
        for i, temp_data in tqdm(enumerate(data_)):
            try:
                nmr_list = list(set(temp_data["C_nmr"]))
                # nmr_list = list(map(float, df0.loc[i, "clean_NMR2"].rstrip(".").rstrip(",").rstrip(";").replace(",,", ",").split(", ")))
                nmr_feature = nmr_process.fill_item(nmr_list)
                nmr_feature["input_ids"] = torch.tensor(nmr_feature["input_ids"]).long().to("cuda:0").reshape(1,-1)
                nmr_feature["attention_mask"] = torch.tensor(nmr_feature["attention_mask"]).long().to("cuda:0").reshape(1,-1)
                nmr_outputs = model(**nmr_feature) # nmr_outputs.last_hidden_state.shape, torch.Size([128, 15, 768])
                nmr_encodings = F.normalize(nmr_outputs.last_hidden_state[:,0,:], p=2, dim=-1)
                data_dict[i] = nmr_encodings.cpu().numpy()
            except Exception as e:
                print(e)
                import traceback
                traceback.print_exc()
    torch.save(data_dict, "/home/baolingjie/git_download/NMR_Core_Mol_Detection/data/enumerate_100m_5m_1_group_restricted/last_results_processed_new_1024_1b_ep6.pt")
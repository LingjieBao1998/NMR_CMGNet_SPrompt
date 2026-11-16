import os
import math
import json
import random
import argparse
import numpy as np

import time
import copy

import torch.distributed as dist
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, LightningDataModule
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.strategies.ddp import DDPStrategy
from torch import nn
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import get_scheduler
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
)
import json
from functools import reduce
from operator import mul
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

import warnings
from rdkit import RDLogger

# 屏蔽 RDKit 的警告（包括 DEPRECATION WARNING）
RDLogger.DisableLog('rdApp.warning')  # 屏蔽 RDKit 自己的 Logger 警告
warnings.filterwarnings("ignore", category=DeprecationWarning)  # 屏蔽 Python 的 DeprecationWarning

from dataclasses import dataclass
from torch import FloatTensor, LongTensor
from torch.nn.utils import clip_grad_norm_
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from typing import Tuple
import gc

@dataclass
class BatchRewards:
    rewards: FloatTensor


@dataclass
class GRPOOutput:
    loss: FloatTensor
    reward: FloatTensor
    kl: FloatTensor
    
def count_parameters(model):
    params = model.parameters()
    num_params = sum(reduce(mul, p.size(), 1) for p in params)
    return num_params

def print_model_size(model):
    num_params = count_parameters(model)
    num_params_m = num_params / 1e6
    print(f"Model size: {num_params_m:.2f}M parameters")


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

def get_new_smiles(old_mol):
    len_mol = old_mol.GetNumAtoms()
    li = []
    for index in range(len_mol):
        try:
            new_smiles = Chem.MolToSmiles(old_mol, rootedAtAtom=index)
            li.append(new_smiles)
        except:
            pass
    if len(li) >= 1:
        return random.choice(li)
    else:
        return None

def get_args(notebook=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--debug', action='store_true')

    ## model_dir 
    parser.add_argument('--model_dir', default="./checkpoints/saved_hf_model_0910",type=str)

    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--max_grad_norm', type=float, default=5.)
    parser.add_argument('--scheduler', type=str, choices=['cosine', 'constant'], default='cosine')
    parser.add_argument('--warmup_ratio', type=float, default=0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)

    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--load_encoder_only', action='store_true')
    parser.add_argument('--train_steps_per_epoch', type=int, default=-1)
    parser.add_argument('--eval_per_epoch', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='output/')
    parser.add_argument('--save_mode', type=str, default='best', choices=['best', 'all', 'last'])
    parser.add_argument('--load_ckpt', type=str, default='best')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--num_train_example', type=int, default=None)
    parser.add_argument('--label_smoothing', type=float, default=0.0)

    ## dataset
    parser.add_argument('--train_file', type=str, default=None)
    parser.add_argument('--valid_file', type=str, default=None)
    parser.add_argument('--test_file', type=str, default=None)
    # Inference

    parser.add_argument('--ckpt', type=str, default=None)
    
    ## data_augment
    parser.add_argument('--smiles_aug_proba', type=float, default=0.0)
    parser.add_argument('--nmr_aug_scale', type=float, default=1.5)

    ## grpo
    parser.add_argument('--grpo_lower_epsilon', type=float, default=0.1)
    parser.add_argument('--grpo_higher_epsilon', type=float, default=0.1)
    parser.add_argument('--grpo_beta', type=float, default=0.0)
    parser.add_argument('--loss_type', type=str, default="gspo_token")
    parser.add_argument('--group_size', type=int, default=16)
    parser.add_argument('--update_old_after', type=int, default=100)

    ## lora
    parser.add_argument('--use_lora', action='store_true')
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.1)

    args = parser.parse_args([]) if notebook else parser.parse_args()

    return args

def grpo(
    generated_ids: LongTensor,
    old_scores: FloatTensor,
    current_scores: FloatTensor,
    reference_scores: FloatTensor,
    labels: LongTensor,
    tokenizer: PreTrainedTokenizer,
    lower_epsilon: float,
    higher_epsilon: float,
    beta: float,
    loss_type: str = "grpo"
) -> GRPOOutput:
    """
    Compute the loss of Group Relative Policy Optimization (GRPO) on the given inputs.
    Args:
        generated_ids (LongTensor): Tensor of generated token IDs.
        old_scores (FloatTensor): Tensor of old policy scores.
        current_scores (FloatTensor): Tensor of current policy scores.
        reference_scores (FloatTensor): Tensor of reference policy scores.
        truths (LongTensor): Tensor of ground truth token IDs.
        tokenizer (PreTrainedTokenizer): Tokenizer used for encoding/decoding.
        epsilon (float): Clipping parameter for policy ratios.
        beta (float): Weighting factor for the Kullback-Leibler divergence term.
    Returns:
        GRPOOutput: A dataclass containing the mean loss, rewards and KL divergences.
    """
    losses = torch.zeros(generated_ids.shape[0])
    rewards = torch.zeros(generated_ids.shape[0])
    kls = torch.zeros(generated_ids.shape[0])

    if reference_scores is None:
        reference_scores = [None] * len(generated_ids)

    for idx, (
        group_ids,
        group_labels,
        group_old_scores,
        group_current_scores,
        group_reference_scores,
    ) in enumerate(
        zip(generated_ids, labels, old_scores, current_scores, reference_scores)
    ):
        # Compute advantages
        group_rewards = compute_rewards(group_ids, group_labels, tokenizer) #计算奖励
        mean = group_rewards.rewards.mean()
        centered = group_rewards.rewards - mean
        std = group_rewards.rewards.std()
        if std < 1e-8:
            advantages = torch.zeros_like(centered)
        else:
            advantages = centered / (std + 1e-8)

        # Store the mean of each rewards for the group
        rewards[idx] = group_rewards.rewards.mean()

        # Compute the ratios(log_importance_weights)
        if loss_type == "grpo":
            log_ratio = group_current_scores - group_old_scores
            log_ratio = torch.clamp(log_ratio, max=10.0)
            ratios = torch.exp(log_ratio)
        elif loss_type == "gspo":
            log_ratio = group_current_scores - group_old_scores
            log_ratio = torch.clamp(log_ratio, max=10.0)
            completion_mask = group_ids[:, 1:] != tokenizer.pad_token_id
            seq_weight = (log_ratio * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1e-8)
            log_importance_weights = seq_weight.unsqueeze(-1)
            log_importance_weights = torch.clamp(log_importance_weights, max=10.0)
            ratios = torch.exp(log_importance_weights)
        elif loss_type == "gspo_token":
            log_ratio = group_current_scores - group_old_scores
            log_ratio = torch.clamp(log_ratio, max=10.0)
            completion_mask = group_ids[:, 1:] != tokenizer.pad_token_id
            seq_weight = (log_ratio * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1e-8)
            log_importance_weights = seq_weight.detach().unsqueeze(-1) + (group_current_scores - group_current_scores.detach())
            log_importance_weights = torch.clamp(log_importance_weights, max=10.0)
            ratios = torch.exp(log_importance_weights)


        # Compute the clipped ratios
        clipped_ratios = torch.clamp(
            ratios, min= 1.0 - lower_epsilon, max= 1.0 + higher_epsilon
        )

        # Compute kullback-leibler divergence between reference and current policy
        if group_reference_scores is not None:
            kl = (
                torch.exp(group_reference_scores - group_current_scores)
                - (group_reference_scores - group_current_scores)
                - 1
            )
            kls[idx] = kl.mean()
        else:
            kl = 0
            kls[idx] = 0
        
        if group_reference_scores is not None:
            loss = (
                torch.min(
                    ratios * advantages.unsqueeze(-1),
                    clipped_ratios * advantages.unsqueeze(-1),
                )
                - beta * kl
            )
            
            # Compute mean loss of the group
            completion_mask = group_ids[:, 1:] != tokenizer.pad_token_id
            loss = -(loss * completion_mask).sum() / completion_mask.sum()
            losses[idx] = loss
        else:
            loss = (
                torch.min(
                    ratios * advantages.unsqueeze(-1),
                    clipped_ratios * advantages.unsqueeze(-1),
                )
            )
            completion_mask = group_ids[:, 1:] != tokenizer.pad_token_id
            loss = -(loss * completion_mask).sum() / completion_mask.sum()
            losses[idx] = loss

    return GRPOOutput(
        loss=losses.mean(),
        reward=rewards.mean(),
        kl=kls.mean(),
    )

def postprocess_text(
    preds: list[str], labels: list[str]
) -> Tuple[list[str], list[str]]:
    """
    Post-processes the predicted and label texts,
    formatting them for ROUGE-L summarization evaluation.
    Args:
        preds (list[str]): List of predicted text strings.
        labels (list[str]): List of label text strings.
    Returns:
        Tuple[list[str], list[str]]: A tuple containing two lists:
            - The first list contains the post-processed predicted texts.
            - The second list contains the post-processed label texts.
    """

    preds = [pred.strip().replace("<SMILES>", "").replace("</SMILES>", "").replace("<pad>", "").replace("</s>", "") for pred in preds]
    labels = [label.strip().replace("<SMILES>", "").replace("</SMILES>", "").replace("<pad>", "").replace("</s>", "") for label in labels]

    return preds, labels

def selective_log_softmax(
    logits: FloatTensor, index: LongTensor
) -> FloatTensor:
    """
    Computes the log softmax of the input logits selectively based on the provided indices.
    This function performs the same operation as applying `log_softmax` on the logits tensor
    along the last dimension and then gathering the results based on the provided indices.
    However, it processes the logits row by row to save memory by leveraging PyTorch internals.
    Taken from https://www.tylerromero.com/posts/2025-02-selective-log-softmax/
    Args:
        logits (FloatTensor): A tensor of shape (batch_size, num_classes) containing the raw
                              logits for each class.
        index (LongTensor): A tensor of shape (batch_size, num_indices) containing the indices
                            of the classes for which to compute the log softmax.
    Returns:
        FloatTensor: A tensor of shape (batch_size, num_indices) containing the log softmax
                     values for the specified indices.
    """

    token_logprobs = []
    for logits_row, index_row in zip(logits, index):
        logprobs_row = logits_row.log_softmax(dim=-1)
        token_logprobs_row = torch.gather(
            logprobs_row, dim=-1, index=index_row.unsqueeze(-1)
        ).squeeze(-1)
        token_logprobs.append(token_logprobs_row)
    return torch.stack(token_logprobs)

def gather_token_scores(
    logits: FloatTensor, generated_ids: LongTensor
) -> FloatTensor:
    """
    Gathers token scores from logits based on generated token IDs.
    Args:
        logits (FloatTensor): The logits output from the model. It can be a tuple of tensors or a single tensor.
        generated_ids (LongTensor): The IDs of the generated tokens.
    Returns:
        FloatTensor: The token scores after applying a selective log softmax on the logits.
    """

    if isinstance(logits, tuple):
        # Stack the logits (batch_size*group_size, output_length, vocab)
        logits = torch.stack(logits, axis=0).permute((1, 0, 2))

    # Logsoftmax the logits
    token_scores = selective_log_softmax(logits, generated_ids)

    return token_scores


def compute_token_scores(
    model: PreTrainedModel,
    encoder_input_ids: LongTensor,
    encoder_attention_mask: LongTensor,
    decoder_input_ids: LongTensor,
    decoder_attention_mask: LongTensor,
    batch_size: int,
    group_size: int,
) -> FloatTensor:
    """
    Computes token scores for a given batch of input sequences using a pre-trained model.
    Args:
        model (PreTrainedModel): The pre-trained model to use for generating logits.
        encoder_input_ids (LongTensor): Tensor containing input IDs for the encoder.
        encoder_attention_mask (LongTensor): Tensor containing attention masks for the encoder inputs.
        decoder_input_ids (LongTensor): Tensor containing input IDs for the decoder.
        decoder_attention_mask (LongTensor): Tensor containing attention masks for the decoder inputs.
        batch_size (int): The size of the batch.
        group_size (int): The size of the group.
    Returns:
        FloatTensor: A tensor containing the computed token scores, reshaped to (batch_size, group_size, -1).
    """
    ## 常规的forward过程
    logits = model(
        input_ids=encoder_input_ids,
        attention_mask=encoder_attention_mask,
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attention_mask,
    ).logits # [batch_size * training_args.group_size,  decoder_input_ids.shape[0], num_vocab] 
    
    ## 这里可以理解为计算-loss(损失函数的负数)
    scores = gather_token_scores(logits[:, :-1], decoder_input_ids[:, 1:]) 
    scores = scores.view(batch_size, group_size, -1)
    del logits
    torch.cuda.empty_cache()
    return scores



def smiles2token(tokenizer, smile_string, phase="train", block_size=512, proba=0.5):
    smile_string_tmp = smile_string
    if phase == "train":
        if random.random()<proba:
            old_mol = Chem.MolFromSmiles(smile_string)
            if old_mol is not None:
                try:
                    smile_string_tmp = Chem.MolToSmiles(
                        old_mol, isomericSmiles=False, doRandom=True, canonical=False)
                except:
                    smile_string_tmp2 = get_new_smiles(old_mol)
                    if smile_string_tmp2 is not None:
                        smile_string_tmp = smile_string_tmp2

    tmp = tokenizer([smile_string_tmp], max_length=block_size)
    tmp["input_ids"] = [[187] + i[1:-1] + [188]
                        for i in tmp["input_ids"]]
    tmp["input_ids"] = tmp["input_ids"][0]
    tmp["attention_mask"] = tmp["attention_mask"][0]
    return tmp

def similarity_reward(pred_smiles_list, true_smiles_list):
    sim = 0
    count = 0
    for i in range(len(true_smiles_list)):
        true_smiles = true_smiles_list[i]
        pred_smiles = pred_smiles_list[i]
        count = count + 1
        mol1 = None
        mol2 = None
        try:
            mol1 = Chem.MolFromSmiles(pred_smiles)
            mol2 = Chem.MolFromSmiles(true_smiles)
        except Exception as e:
            pass
        if mol1 is None or mol2 is None:
            continue

        # 生成MACCS指纹
        fp1 = AllChem.GetMACCSKeysFingerprint(mol1)
        fp2 = AllChem.GetMACCSKeysFingerprint(mol2)

        # 计算Tanimoto相似性
        similarity_1 = DataStructs.TanimotoSimilarity(fp1, fp2)
        if similarity_1 == 1:
            similarity_1 = 2
        fp3 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=2, nBits=2048, useChirality=True)
        fp4 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=2, nBits=2048, useChirality=True)
        # 计算 Tanimoto 相似性
        similarity_2 = DataStructs.TanimotoSimilarity(fp3, fp4)
        if similarity_2 == 1:
            similarity_2 = 2

        sim = sim + similarity_1 + similarity_2
    return sim/(count+1e-7)


def compute_rewards(
    token_ids: LongTensor, labels: LongTensor, tokenizer: PreTrainedTokenizer
) -> BatchRewards:
    """
    Compute rewards based on the ROUGE avg score between generated completions and reference summaries.
    Args:
        token_ids (LongTensor): Tensor containing token IDs of the generated completions.
        labels (LongTensor): Tensor containing token IDs of the reference summaries.
        tokenizer (PreTrainedTokenizer): Tokenizer used to decode the token IDs.
    Returns:
        BatchRewards: A tensor containing the computed rewards for each completion.
    """
    labels[labels == -100] = tokenizer.pad_token_id
    completions = tokenizer.batch_decode(token_ids, skip_special_tokens=True)
    summaries = tokenizer.batch_decode(labels, skip_special_tokens=True)
    completions, summaries = postprocess_text(completions, summaries)

    rewards = torch.zeros(token_ids.shape[0], device=token_ids.device)
    for idx, (completion, summary) in enumerate(zip(completions, summaries)):
        rouge_score = similarity_reward(
            [completion], [summary]
        )
        rewards[idx] = rouge_score

    return BatchRewards(rewards)

class CustomDataset(Dataset):
    def __init__(self, args, tokenizer, file_path="", phase="val"):
        """
        Args:
            data_list (list or array): 样本数据，比如特征列表 [样本1, 样本2, ...]
            labels (list or tensor, optional): 对应的标签，如 [0, 1, 0, ...]。可以为 None（如测试集）
        """
        self.args = args
        self.tokenizer = tokenizer
        self.phase = phase
        if isinstance(file_path, str):
            with open(file_path, "r") as f:
                self.data = json.loads(f.read())
        elif isinstance(file_path, list):
            self.data = []
            for _ in file_path:
                with open(file_path, "r") as f:
                    self.data_ = json.loads(f.read())
                    self.data.extend(self.data)
        
        if self.args.debug:
            self.data = self.data[:1000]
        
        if self.phase == "train":
            self.positive_data = []
            self.negtive_data = []
            for i in range(len(self.data)):
                if self.data[i]["label"] == 1:
                    self.positive_data.append(self.data[i])
                else:
                    self.negtive_data.append(self.data[i])

            print("len(self.positive_data)",len(self.positive_data))
            print("len(self.negtive_data)",len(self.negtive_data))

        
        self.ori_NmrDataset = NmrDataset(augment_range=self.args.nmr_aug_scale)

    def __len__(self):
        if self.phase != "train":
            return len(self.data)  # 数据集大小
        else:
            return len(self.positive_data) + int(0.1*len(self.negtive_data))  # 数据集大小

    def __getitem__(self, idx):
        if self.phase != "train":
            sample = self.data[idx]  # 获取第 idx 个样本
        else:
            if idx >= len(self.positive_data):
                sample = random.choice(self.negtive_data)
            else:
                sample = self.positive_data[idx]
        
        ori_nmr = sample["NMR"]
        ori_smiles = sample["cano_smiles"]
        temp_dict = {}
        smiles_result = smiles2token(self.tokenizer, ori_smiles, phase=self.phase, block_size=self.args.max_length, proba=self.args.smiles_aug_proba)
        temp_dict["decoder_input_ids"] = smiles_result["input_ids"]
        temp_dict["decoder_attention_mask"] = smiles_result["attention_mask"]
        if self.args.nmr_aug_scale>0 and self.phase == "train" and random.random()>0.5:
            nmr_result = self.ori_NmrDataset.augment_item(ori_nmr)
        else:
            nmr_result = self.ori_NmrDataset.fill_item(ori_nmr)
        
        temp_dict["input_ids"] = nmr_result["input_ids"]
        temp_dict["attention_mask"] = nmr_result["attention_mask"]
        temp_dict["smiles"] = ori_smiles

        return temp_dict
    
    def collate_fn(self, batch):
        max_encoder_length = max([len(batch[_]["input_ids"]) for _ in range(len(batch))])
        max_decoder_length = max([len(batch[_]["decoder_input_ids"]) for _ in range(len(batch))])
        for _ in range(len(batch)):
            if len(batch[_]["input_ids"]) < max_encoder_length:
                batch[_]["attention_mask"] = batch[_]["attention_mask"] + [0]*(max_encoder_length-len(batch[_]["input_ids"]))
                batch[_]["input_ids"] = batch[_]["input_ids"] + [self.tokenizer.pad_token_id]*(max_encoder_length-len(batch[_]["input_ids"]))
            if len(batch[_]["decoder_input_ids"]) < max_decoder_length:
                batch[_]["decoder_attention_mask"] = batch[_]["decoder_attention_mask"] + [0]*(max_decoder_length-len(batch[_]["decoder_input_ids"]))
                batch[_]["decoder_input_ids"] = batch[_]["decoder_input_ids"] + [self.tokenizer.pad_token_id]*(max_decoder_length-len(batch[_]["decoder_input_ids"]))

        input_ids = torch.tensor([batch[_]["input_ids"] for _ in range(len(batch))]).long()[:, :self.args.max_length]
        attention_mask = torch.tensor([batch[_]["attention_mask"] for _ in range(len(batch))]).long()[:, :self.args.max_length]
        decoder_input_ids = torch.tensor([batch[_]["decoder_input_ids"] for _ in range(len(batch))]).long()
        labels = decoder_input_ids[:,1:][:, :self.args.max_length]
        decoder_input_ids=decoder_input_ids[:,:-1][:, :self.args.max_length]
        # decoder_attention_mask = torch.tensor([batch[_]["decoder_attention_mask"] for _ in range(len(batch))]).long()
        smiles_list = [batch[_]["smiles"] for _ in range(len(batch))]
        return {
            "input_ids":input_ids,
            "attention_mask":attention_mask,
            "decoder_input_ids":decoder_input_ids,
            "labels":labels,
            "smiles_list":smiles_list,
        }

class CustomDataModule(pl.LightningDataModule):
    def __init__(self, args, tokenizer=None):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer

    def prepare_data(self):
        ## 导入数据
        self.train_dataset = CustomDataset(self.args, self.tokenizer, file_path=self.args.train_file, phase="train")
        self.val_dataset = CustomDataset(self.args, self.tokenizer, file_path=self.args.valid_file, phase="val")
        self.test_dataset = CustomDataset(self.args, self.tokenizer, file_path=self.args.test_file, phase="test")

    def print_stats(self):
        ## 打印数据集
        print(f"Train dataset: {len(self.train_dataset)}")
        print(f"Valid dataset: {len(self.val_dataset)}")
        print(f"Test dataset: {len(self.test_dataset)}")

    def setup(self, stage: str = None):  # 必须包含 stage 参数
        pass
        # if stage == "fit" or stage is None:
        #     # 加载训练/验证数据
        #     self.train_dataset = ...
        #     self.val_dataset = ...
        # if stage == "test" or stage is None:
        #     # 加载测试数据
        #     self.test_dataset = ...

    def train_dataloader(self):
        if self.args.debug is False:
            ## 瓶颈不在数据
            # return torch.utils.data.DataLoader(
            #     self.dataset["train"], batch_size=self.args.batch_size, num_workers=self.args.num_workers,
            #     collate_fn=DataCollatorForSeq2Seq(self.tokenizer), prefetch_factor=2, persistent_workers=True, pin_memory=True,
            #     shuffle=True)
            return torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                collate_fn=self.train_dataset.collate_fn,
                shuffle=True,
            )
        else:
            ## 单线程调试
            return torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=self.args.batch_size,
                num_workers=0,
                collate_fn=self.train_dataset.collate_fn,
                shuffle=True,
            )

    def val_dataloader(self):
        if self.args.debug is False:
            return torch.utils.data.DataLoader(
                self.val_dataset,
                batch_size=self.args.batch_size * 4,
                num_workers=self.args.num_workers,
                collate_fn=self.val_dataset.collate_fn,
            )
        else:
            ## 单线程调试
            return torch.utils.data.DataLoader(
                self.val_dataset,
                batch_size=self.args.batch_size * 4,
                num_workers=0,
                collate_fn=self.val_dataset.collate_fn,
            )

    def test_dataloader(self):
        if self.args.debug is False:
            return torch.utils.data.DataLoader(
                self.test_dataset,
                batch_size=self.args.batch_size * 2,
                num_workers=self.args.num_workers,
                collate_fn=self.test_dataset.collate_fn,
            )
        else:
            ## 单线程调试
            return torch.utils.data.DataLoader(
                self.test_dataset,
                batch_size=self.args.batch_size * 2,
                num_workers=0,
                collate_fn=self.test_dataset.collate_fn,
            )

def get_model(model_dir="./saved_hf_model_0910", use_lora=False, device = "cpu", args=None):
    try:
        from flash_attn import flash_attn_func  # noqa: F401  探测 flash-attn v2 是否可用
        attn_implementation = "flash_attention_2"
    except Exception:
        # flash-attn v2 不可用时退回 PyTorch 原生 sdpa (均支持训练/生成, 速度接近)
        print("⚠️ flash_attn 不可用, 改用 attn_implementation='sdpa'")
        attn_implementation = "sdpa"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, attn_implementation=attn_implementation)
    if use_lora:
        print("use lora")
        from peft import LoraConfig, TaskType, get_peft_model, get_peft_model_state_dict, PeftModel
        peft_config = LoraConfig(
                r=args.lora_r, 
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                task_type=TaskType.SEQ_2_SEQ_LM, 
                inference_mode=False,
                target_modules=["q_proj", "v_proj", "k_proj"],  # 编码器和解码器的注意力层模块名称,#
                bias="none",
            )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    model.to(device)
    return model

def get_tokenizer(model_dir="./saved_hf_model_0910"):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return tokenizer

class CustomModel(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        self.args = args
        self.model = get_model(self.args.model_dir, use_lora=self.args.use_lora, args=self.args)
        print(print_model_size(self.model))
        self.old_model = copy.deepcopy(self.model)
        self.old_model.eval()
        for name, param in self.old_model.named_parameters():
            param.requires_grad = False
        if self.args.grpo_beta != 0:
            self.reference_model = copy.deepcopy(self.model)
            self.reference_model.eval()
            for name, param in self.reference_model.named_parameters():
                param.requires_grad = False
        else:
            self.reference_model = None
        self.tokenizer = get_tokenizer(self.args.model_dir)
        self.val_step_outputs = []
        self.val_loss = []
        self.test_step_outputs = []
        self.test_loss = []
        
    def training_step(self, batch, batch_idx):
        temp_i = 0
        while True:
            if ((self.global_step ) * self.trainer.num_devices  <= self.args.update_old_after * temp_i) and \
                ((self.global_step + 1 ) * self.trainer.num_devices  > self.args.update_old_after * temp_i):
                self.old_model.load_state_dict(self.model.state_dict(), strict=False)
                torch.cuda.empty_cache()
                break
            elif self.args.update_old_after * temp_i >= (self.global_step + 1 ) * self.trainer.num_devices:
                break
            else:
                temp_i = temp_i + 1
        
        self.old_model.eval()

        # Prepare the batch data
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        effective_batch_size = input_ids.shape[0]

        # Generate ids with the old policy
        generated_ids = self.old_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.args.max_length,
            do_sample=True,
            num_beams=self.args.group_size, #default=5
            num_return_sequences=self.args.group_size,
        ) # [batch_size * args.group_size, max_len]

        # Prepare attention mask for computing current
        # and reference logits on the generated ids
        decoder_attention_mask = generated_ids != self.tokenizer.pad_token_id # [batch_size * args.group_size, max_len]

        # Interleave input_ids and attention_mask to have
        # the same shape than the generated completions
        repeated_input_ids = input_ids.repeat_interleave(
            repeats=self.args.group_size, dim=0
        ) # [batch_size, max_len] -> [batch_size * args.group_size, max_len], 将input_ids沿着某个维度进行复制

        repeated_attention_mask = attention_mask.repeat_interleave(
            repeats=self.args.group_size, dim=0
        ) # [batch_size, max_len] -> [batch_size * args.group_size, max_len], 将attention_mask沿着某个维度进行复制

        with torch.inference_mode():
            old_scores = compute_token_scores(
                self.old_model,
                encoder_input_ids=repeated_input_ids,
                encoder_attention_mask=repeated_attention_mask,
                decoder_input_ids=generated_ids,
                decoder_attention_mask=decoder_attention_mask,
                batch_size=effective_batch_size,
                group_size=self.args.group_size,
            ).detach() # [batch_size, args.group_size, max_len]
        
        
        self.model.eval()
        current_scores = compute_token_scores(
            self.model,
            encoder_input_ids=repeated_input_ids,
            encoder_attention_mask=repeated_attention_mask,
            decoder_input_ids=generated_ids,
            decoder_attention_mask=decoder_attention_mask,
            batch_size=effective_batch_size,
            group_size=self.args.group_size,
        )
        self.model.train()

        if self.args.grpo_beta != 0:
            with torch.inference_mode():
                reference_scores = compute_token_scores(
                    self.reference_model,
                    encoder_input_ids=repeated_input_ids,
                    encoder_attention_mask=repeated_attention_mask,
                    decoder_input_ids=generated_ids,
                    decoder_attention_mask=decoder_attention_mask,
                    batch_size=effective_batch_size,
                    group_size=self.args.group_size,
                ).detach() 
        else:
            reference_scores = None
        
        # Group the generated ids (batch_size, group_size, output_length)
        generated_ids = generated_ids.view(
            effective_batch_size, self.args.group_size, -1
        )

        # Repeat the labels and group (batch_size, group_size)
        labels = labels.repeat_interleave(
            repeats=self.args.group_size, dim=0
        ).view(effective_batch_size, self.args.group_size, -1)
        
        ## 新旧模型的权重还是一致的
        # Compute GRPO objective
        # 核心
        grpo_output = grpo(
            generated_ids,
            old_scores,
            current_scores,
            reference_scores,
            labels,
            self.tokenizer,
            self.args.grpo_lower_epsilon,
            self.args.grpo_higher_epsilon,
            self.args.grpo_beta,
            self.args.loss_type,
        )
        self.log('train/loss', grpo_output.loss, prog_bar=True)
        self.log('train/reward', grpo_output.reward, prog_bar=True)
        # self.log('train/kl', grpo_output.kl, prog_bar=True)
        self.log('train/completion_length', decoder_attention_mask.sum(-1).float().mean(), prog_bar=True)
        self.log('lr', self.lr_schedulers().get_lr()[0], prog_bar=True, logger=False)

        # Free GPU memory at the end
        del (
            generated_ids,
            old_scores,
            input_ids,
            attention_mask,
            repeated_input_ids,
            repeated_attention_mask,
            current_scores,
            reference_scores,
            labels,
        )
        torch.cuda.empty_cache()
        gc.collect()
        return grpo_output.loss
    
    def validation_step(self, batch, batch_idx):
        true_smiles = batch.pop("smiles_list")
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=self.args.max_length,
                do_sample=True,
                num_beams=1,
                num_return_sequences=1,
                bos_token_id=187,
                pad_token_id=1,
                eos_token_id=188,
                decoder_start_token_id=187,
            )
        result = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        result = [i.replace("<SMILES>", "").replace("</SMILES>", "").replace("<pad>", "").replace("</s>", "")  for i in result]
        batch["labels"][batch["labels"]==-100]=self.tokenizer.pad_token_id
        label = self.tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)
        label = [i.replace("<SMILES>", "").replace("</SMILES>", "").replace("<pad>", "").replace("</s>", "")  for i in label]
        zipped = list(zip(result, label))
        self.val_step_outputs.append(list(zip(result, label)))
        return None
    
    def on_validation_epoch_end(self):
        if self.trainer.num_devices > 1:
            gathered_outputs = [None for i in range(self.trainer.num_devices)]
            dist.all_gather_object(gathered_outputs, self.val_step_outputs)
            gathered_outputs = np.concatenate([np.concatenate(gathered_outputs[i]).reshape(-1,2) for i in range(len(gathered_outputs))]).reshape(-1,2)
        else:
            gathered_outputs = self.val_step_outputs
            gathered_outputs = np.concatenate(gathered_outputs).reshape(-1,2)
        
        # scores = [-np.mean(gathered_losses)]
        self.val_step_outputs.clear()
        if self.current_epoch<=0:
            print("gathered_outputs", gathered_outputs.shape)
        scores = [0]
        if self.trainer.is_global_zero:
            score = similarity_reward(gathered_outputs[:,0].tolist(), gathered_outputs[:,1].tolist())
            scores = [score]
        
        dist.broadcast_object_list(scores)
        self.log('val/score', scores[0], prog_bar=True, rank_zero_only=True, sync_dist=True)

        print("val epoch end end")

    def test_step(self, batch, batch_idx):
        true_smiles = batch.pop("smiles_list")
        with torch.no_grad():
            outputs = self.model(**batch)
            loss = outputs.loss
            self.log('val/loss', loss.item(), prog_bar=True, rank_zero_only=True)

            result = self.model.generate(
                    input_ids=batch["input_ids"],
                    max_length=self.args.max_length,
                    num_beams=1,
                    num_return_sequences=1,
                    bos_token_id=187,
                    pad_token_id=1,
                    eos_token_id=188,
                    decoder_start_token_id=187
                )
        pred_smiles = [self.tokenizer.decode(i) for i in result]
        pred_smiles = [i.replace("<SMILES>", "").replace("</SMILES>", "").replace("<pad>", "").replace("</s>", "") for i in pred_smiles]
        self.test_step_outputs.extend(list(zip(true_smiles, pred_smiles)))
        self.test_loss.append(loss.item())
        return loss

    def on_test_epoch_end(self):
        if self.trainer.num_devices > 1:
            gathered_outputs = [None for i in range(self.trainer.num_devices)]
            dist.all_gather_object(gathered_outputs, self.test_step_outputs)
            gathered_outputs = sum(gathered_outputs, [])
        else:
            gathered_outputs = self.test_step_outputs
        
        scores = [0]
        
        if self.trainer.is_global_zero:

            count = 0
            acc = 0
            smi = 0
            for (true_smiles, pred_smiles) in gathered_outputs:
                true_mol = None
                pred_mol = None
                try:
                    true_mol = Chem.MolFromSmiles(true_smiles)
                except:
                    pass
                    
                try:
                    pred_mol = Chem.MolFromSmiles(pred_smiles)
                except:
                    pass
                
                if (true_mol is None) or (pred_mol is None):
                    pass
                else:
                    if Chem.MolToSmiles(true_mol) == Chem.MolToSmiles(pred_mol):
                        acc = acc + 1
                    
                    fp_true = AllChem.GetMorganFingerprintAsBitVect(true_mol, radius=2, nBits=2048, useChirality=True)
                    fp_pred = AllChem.GetMorganFingerprintAsBitVect(pred_mol, radius=2, nBits=2048, useChirality=True)
                    similarity = DataStructs.TanimotoSimilarity(fp_true, fp_pred)
                    smi = smi + similarity
                
                count = count + 1
            
            with open(os.path.join(self.trainer.default_root_dir, f'test.json'), 'w') as f:
                json.dump(gathered_outputs, f)
            
            scores = [smi/count]

        dist.broadcast_object_list(scores)
        self.log('test/score', scores[0], prog_bar=True, rank_zero_only=True)

        self.test_step_outputs.clear()
        print("test epoch end end")


    def configure_optimizers(self):
        num_training_steps = self.trainer.num_training_steps
        self.print(f'Num training steps: {num_training_steps}')
        num_warmup_steps = int(num_training_steps * self.args.warmup_ratio)
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        scheduler = get_scheduler(self.args.scheduler, optimizer, num_warmup_steps, num_training_steps)
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}
    
    def on_train_epoch_end(self):
        ## 保存模型
        if (self.current_epoch%5) == 0:
            self.trainer.save_checkpoint(os.path.join(self.args.save_path, f'checkpoints/epoch_{self.current_epoch}.ckpt'))


class ModelCheckpoint(pl.callbacks.ModelCheckpoint):
    def _get_metric_interpolated_filepath_name(self, monitor_candidates, trainer, del_filepath=None) -> str:
        filepath = self.format_checkpoint_name(monitor_candidates)
        return filepath


def main():
    args = get_args()
    pl.seed_everything(args.seed, workers=True)

    
    tokenizer = get_tokenizer(args.model_dir)
    args.tokenizer = tokenizer

    dm = CustomDataModule(args, tokenizer)
    dm.prepare_data()
    dm.print_stats()

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
        

    MODEL = CustomModel
    if args.do_train:
        model = MODEL(args)
        if args.ckpt is not None and os.path.exists(args.ckpt):
            model = MODEL.load_from_checkpoint(args.ckpt, strict=False, args=args)
    else:
        if args.ckpt is not None and os.path.exists(args.ckpt):
            model = MODEL.load_from_checkpoint(args.ckpt, strict=False, args=args)
            print("load ckpt")
        else:
            model = MODEL.load_from_checkpoint(os.path.join(args.save_path, 'checkpoints/last.ckpt'), strict=False, args=args)
            print("load successfully")

    import ipdb
    ipdb.set_trace()
    checkpoint = ModelCheckpoint(monitor='val/score', mode='max', save_top_k=1, filename='best', save_last=True)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    logger = pl.loggers.TensorBoardLogger(args.save_path, name='', version='')

    trainer = pl.Trainer(
        strategy=DDPStrategy(find_unused_parameters=False),
        accelerator='gpu',
        precision = "bf16",
        devices=args.gpus,
        logger=logger,
        default_root_dir=args.save_path,
        callbacks=[checkpoint, lr_monitor],
        max_epochs=args.epochs,
        gradient_clip_val=args.max_grad_norm,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        check_val_every_n_epoch=args.eval_per_epoch,
        log_every_n_steps=10,
        deterministic='warn')

    if args.do_train:
        trainer.num_training_steps = math.ceil(
            len(dm.train_dataset) / (args.batch_size * args.gpus * args.gradient_accumulation_steps)) * args.epochs
        model.eval_dataset = dm.val_dataset
        ckpt_path = os.path.join(args.save_path, 'checkpoints/last.ckpt') if args.resume else None
        trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)
        model = MODEL.load_from_checkpoint(checkpoint.best_model_path, args=args)

    if args.do_valid:
        model.eval_dataset = dm.val_dataset
        trainer.validate(model, datamodule=dm)

    if args.do_test:
        model.eval_dataset = dm.test_dataset
        trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    main()

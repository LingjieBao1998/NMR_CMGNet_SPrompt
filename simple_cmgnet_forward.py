#!/usr/bin/env python3
"""
简化版CMGNet Forward Loss计算
只计算给定NMR-SMILES对的forward loss
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class SimpleCMGNetForward:
    """简化的CMGNet Forward计算"""

    def __init__(self, model_path: str = None, device: str = "cuda:0"):
        """
        初始化CMGNet模型

        Args:
            model_path: 模型路径
            device: 设备 (cuda:0 或 cpu)
        """
        self.device = device
        self.max_length = 512

        # 设置默认模型路径
        project_dir = Path(__file__).parent
        if model_path is None:
            model_path = str(project_dir / "models/cmgnet/train_cmgnet_ring_system_3.2_1029_v2_best.ckpt")

        # 加载模型和分词器
        try:
            model_path_obj = Path(model_path)

            if model_path_obj.suffix == '.ckpt':
                # PyTorch Lightning checkpoint
                print("   检测到PyTorch Lightning checkpoint文件")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    str(project_dir / "checkpoints/saved_hf_model_0910")
                )

                # 加载checkpoint
                import torch
                torch.serialization.add_safe_globals([
                    'transformers.models.bart.tokenization_bart_fast.BartTokenizerFast'
                ])

                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

                # 检查checkpoint结构
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint

                # 创建模型并加载权重
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    str(project_dir / "checkpoints/saved_hf_model_0910")
                )

                # 加载权重
                self.model.load_state_dict(state_dict, strict=False)
                print("   ✅ Checkpoint权重加载完成")
            else:
                raise ValueError(f"不支持的模型路径格式: {model_path}")

            # 强制使用eager attention以支持output_attentions
            self.model.config._attn_implementation = "eager"

            self.model = self.model.to(self.device)
            self.model.eval()

            print(f"   ✅ 模型加载成功")
            print("=" * 80)

        except Exception as e:
            print(f"   ❌ 模型加载失败: {e}")
            raise

    def process_nmr(self, nmr_shifts: List[float]) -> Dict:
        """
        处理NMR化学位移数据

        Args:
            nmr_shifts: NMR化学位移列表

        Returns:
            处理后的NMR数据字典
        """
        min_value = -50
        max_value = 350
        scale = 10
        units = (max_value - min_value) * scale
        start = 189
        end = 190
        number_start = 191

        # 去重和排序
        nmr_unique = sorted(list(nmr_shifts), reverse=True)

        # 转换为模型输入格式
        nmr_scaled = [round((value - min_value) * scale) for value in nmr_unique]
        nmr_scaled = [min(max(0, value), units-1) for value in nmr_scaled]
        nmr_scaled.sort()

        # 添加特殊标记
        input_ids = [start] + [value + number_start for value in nmr_scaled] + [end]
        attention_mask = [1 for _ in range(len(input_ids))]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "original_shifts": nmr_unique
        }

    def smiles2token(self, smile_string: str) -> Dict:
        """将SMILES转换为token"""
        tmp = self.tokenizer([smile_string], max_length=self.max_length)
        tmp["input_ids"] = [[187] + i[1:-1] + [188] for i in tmp["input_ids"]]
        tmp["input_ids"] = tmp["input_ids"][0]
        tmp["attention_mask"] = tmp["attention_mask"][0]
        return tmp

    def get_cross_attention_matrix(self, result: Dict, layer_idx: int = 4, head_idx: int = None):
        """
        从结果中提取cross attention矩阵
        
        Args:
            result: compute_forward_loss的结果
            layer_idx: 层索引 (默认第5层，索引4)
            head_idx: 头部索引，如果为None则返回所有头部
            
        Returns:
            numpy数组或None
        """
        if result.get('cross_attentions') is None:
            return None
            
        cross_attn = result['cross_attentions']
        
        if layer_idx >= len(cross_attn):
            print(f"警告: 层索引{layer_idx}超出范围 [0, {len(cross_attn)-1}]")
            return None
            
        layer_attn = cross_attn[layer_idx]  # [batch, num_heads, seq_len, nmr_len]
        
        # 移除batch维度
        if layer_attn.size(0) == 1:
            layer_attn = layer_attn.squeeze(0)  # [num_heads, seq_len, nmr_len]
        
        # 转换为numpy
        layer_attn_numpy = layer_attn.cpu().numpy()
        
        # 如果指定了头部，返回单个头部
        if head_idx is not None:
            if head_idx >= layer_attn_numpy.shape[0]:
                print(f"警告: 头部索引{head_idx}超出范围 [0, {layer_attn_numpy.shape[0]-1}]")
                return None
            return layer_attn_numpy[head_idx]  # [seq_len, nmr_len]
        
        # 返回所有头部
        return layer_attn_numpy  # [num_heads, seq_len, nmr_len]

    def select_best_head(self, result: Dict, layer_idx: int = 4):
        """
        智能选择最佳头部 - 基于综合评分
        
        Args:
            result: compute_forward_loss的结果
            layer_idx: 要分析的层索引 (默认第5层，索引4)
            
        Returns:
            (best_head_idx, scores_dict) 最佳头部索引和评分字典
        """
        if result.get('cross_attentions') is None:
            return None, None
            
        cross_attn = result['cross_attentions']
        
        if layer_idx >= len(cross_attn):
            print(f"警告: 层索引{layer_idx}超出范围")
            return None, None
            
        # 获取指定层的attention
        layer_attn = cross_attn[layer_idx]  # [batch, num_heads, seq_len, nmr_len]
        
        # 移除batch维度
        if layer_attn.size(0) == 1:
            layer_attn = layer_attn.squeeze(0)  # [num_heads, seq_len, nmr_len]
        
        num_heads = layer_attn.size(0)
        head_scores = {}
        
        print(f"\n🤖 智能选择最佳头部 (第{layer_idx}层，共{num_heads}个头)...")
        print("=" * 60)
        
        for h in range(num_heads):
            attn = layer_attn[h]  # [seq_len, nmr_len]
            
            # 指标1: 最大注意力权重（峰值响应）
            max_response = attn.max().item()
            
            # 指标2: 注意力熵（低熵 = 集中对齐）
            attn_safe = attn + 1e-9
            entropy = -(attn_safe * torch.log(attn_safe)).sum(dim=-1).mean().item()
            
            # 指标3: 稀疏度
            sorted_attn, _ = attn.sort(dim=-1, descending=True)
            cumsum = sorted_attn.cumsum(dim=-1)
            weights = torch.arange(1, attn.size(-1) + 1, dtype=torch.float32, device=attn.device) / attn.size(-1)
            sparsity = (1 - 2 * (cumsum * weights).sum(dim=-1).mean()).item()
            
            # 综合评分
            score = max_response - 0.5 * entropy + 0.3 * sparsity
            
            head_scores[h] = {
                'max_response': max_response,
                'entropy': entropy,
                'sparsity': sparsity,
                'composite_score': score
            }
            
            print(f"头部 {h:2d}: 最大响应={max_response:.4f}, 熵={entropy:.4f}, "
                  f"稀疏度={sparsity:.4f}, 综合评分={score:.4f}")
        
        # 按综合评分排序
        sorted_heads = sorted(head_scores.items(), key=lambda x: x[1]['composite_score'], reverse=True)
        
        print("\n" + "=" * 60)
        best_head_idx = sorted_heads[0][0]
        best_score = sorted_heads[0][1]['composite_score']
        
        print(f"🏆 智能选择结果: 头部 {best_head_idx}")
        print(f"📊 综合评分: {best_score:.4f}")
        print(f"🔍 最大响应: {sorted_heads[0][1]['max_response']:.4f}")
        print(f"📉 熵值: {sorted_heads[0][1]['entropy']:.4f}")
        print(f"💎 稀疏度: {sorted_heads[0][1]['sparsity']:.4f}")
        print("=" * 60)
        
        return best_head_idx, head_scores

    def compute_forward_loss(self, nmr_shifts: List[float], smiles: str) -> Dict:
        """
        计算NMR-SMILES对的forward loss并返回cross attention

        Args:
            nmr_shifts: NMR化学位移列表
            smiles: SMILES字符串

        Returns:
            计算结果字典，包含loss和cross attention等
        """
        print(f"\n🎯 计算Forward Loss + Cross Attention")
        print(f"   NMR峰数量: {len(nmr_shifts)}")
        print(f"   SMILES: {smiles}")

        try:
            # 1. 处理NMR数据
            nmr_data = self.process_nmr(nmr_shifts)

            # 2. 处理SMILES数据
            smiles_data = self.smiles2token(smiles)

            # 3. 准备模型输入
            inputs = {
                "input_ids": torch.tensor([nmr_data["input_ids"]]).to(self.device),
                "attention_mask": torch.tensor([nmr_data["attention_mask"]]).to(self.device),
                "decoder_input_ids": torch.tensor([smiles_data["input_ids"][:-1]]).to(self.device),
                "labels": torch.tensor([smiles_data["input_ids"][1:]]).to(self.device),
                "output_attentions": True,  # 输出attention
                "output_hidden_states": True,
                "return_dict": True
            }

            print(f"   📊 输入形状:")
            print(f"      input_ids: {inputs['input_ids'].shape}")
            print(f"      decoder_input_ids: {inputs['decoder_input_ids'].shape}")
            print(f"      labels: {inputs['labels'].shape}")

            # 4. 执行forward pass
            print(f"\n🧠 执行Forward Pass...")
            with torch.no_grad():
                outputs = self.model(**inputs)

            # 5. 获取loss
            if hasattr(outputs, 'loss') and outputs.loss is not None:
                loss = outputs.loss.item()
                print(f"   📉 Forward Loss: {loss:.4f}")
            else:
                # 如果模型没有直接计算loss，手动计算
                logits = outputs.logits
                labels = inputs["labels"]

                logits = logits.view(-1, logits.size(-1))
                labels = labels.view(-1)
                
                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(logits, labels).item()
                print(f"   📉 手动计算Loss: {loss:.4f}")

            # 6. 获取cross attention分数
            cross_attentions = None
            if hasattr(outputs, 'cross_attentions') and outputs.cross_attentions is not None:
                cross_attentions = outputs.cross_attentions
                print(f"   🎯 Cross Attention获取成功")
                print(f"      - 层数: {len(cross_attentions)}")
                if len(cross_attentions) > 0:
                    layer_attn = cross_attentions[0]
                    print(f"      - 每层形状: {layer_attn.shape} [batch, heads, seq_len, nmr_len]")
                    print(f"      - Attention范围: [{layer_attn.min():.4f}, {layer_attn.max():.4f}]")
            else:
                print(f"   ⚠️  无法获取cross_attentions")

            return {
                'loss': loss,
                'cross_attentions': cross_attentions,
                'nmr_count': len(nmr_shifts),
                'smiles_length': len(smiles),
                'success': True
            }

        except Exception as e:
            print(f"❌ Forward计算失败: {e}")
            import traceback
            traceback.print_exc()
            return {
                'loss': None,
                'cross_attentions': None,
                'error': str(e),
                'success': False
            }


def main():
    """主函数 - 简单的forward loss计算演示"""

    print("=" * 80)
    print("🚀 简化CMGNet Forward Loss计算")
    print("=" * 80)

    try:
        # 1. 初始化模型
        print("步骤1: 初始化CMGNet模型...")
        forward_model = SimpleCMGNetForward()

        # 2. 测试数据
        print("\n步骤2: 准备测试数据...")
        nmr_shifts = [
            176.4, 166.1, 155.5, 146.2, 140.2, 132.1, 124.8, 119.6, 106.6, 103.3,
            68.0, 55.3, 54.8, 50.1, 46.0, 17.9
        ]
        nmr_shifts.sort()
        # new_nmr_shifts = [14.67, 46.7, 49.9, 51.47, 56.8, 69.15, 106.2,
        #     107.98, 119.75, 131.53, 133.8, 140.1, 141.72, 157.6,
        #     166.02, 173.99]
        new_nmr_shifts = [
            14.67,   # C 1
            51.47,   # C 2
            69.15,   # C 3
            140.10,  # C 5
            133.80,  # C 6
            49.90,   # C 8
            56.80,   # C 9
            46.70,   # C11
            56.80,   # C12
            49.90,   # C13
            157.60,  # C14
            106.20,  # C16
            119.75,  # C17
            173.99,  # C18
            107.98,  # C20
            166.02,  # C21
            141.72,  # C24
            131.53   # C26
        ]
        new_nmr_shifts.sort()
        smiles = "CC1COc2c(N3CCN(C)CC3)c(F)cc3c(=O)c(C(=O)O)cn1c23"
        
        # 找到new_nmr_shifts中每个值在nmr_shifts中最近邻的索引
        print("\n寻找最近邻索引...")
        closest_indices = []
        for new_shift in new_nmr_shifts:
            distances = [abs(new_shift - original_shift) for original_shift in nmr_shifts]
            closest_idx = distances.index(min(distances))
            closest_indices.append(closest_idx)
            print(f"new_shift: {new_shift:.2f} -> nmr_shifts[{closest_idx}] = {nmr_shifts[closest_idx]:.2f} (距离: {distances[closest_idx]:.2f})")
        
        print(f"\n最近邻索引列表: {closest_indices}")
        print(f"对应的原始化学位移: {[nmr_shifts[i] for i in closest_indices]}")

        align_nmr_shfit = [nmr_shifts[_] for _ in closest_indices]
        align_nmr_shfit.sort()
        # import ipdb
        # ipdb.set_trace()

        # nmr_shifts = [
        #     176.4, 166.1, 155.5, 146.2, 140.2, 132.1, 124.8, 119.6, 106.6, 103.3,
        #     68.0, 55.3, 54.8, 50.1, 46.0, 17.9
        # ]
        # smiles = "CC1COc2c(N3CCN(C)CC3)c(F)cc3c(=O)c(C(=O)O)cn1c23"
        # smiles = "O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O"
        # nmr_shifts = [
        #     176.4,
        #     165.8,
        #     152.9,
        #     148.2,
        #     144.1,
        #     139.1,
        #     119.4,
        #     111.2,
        #     106.9,
        #     106.9,
        #     46.3,
        #     42.5,
        #     36.0,
        #     7.6
        #     ]

        # 3. 计算forward loss
        print("\n步骤3: 计算Forward Loss...")
        result = forward_model.compute_forward_loss(align_nmr_shfit, smiles)

        # 4. 显示结果
        print("\n步骤4: 结果...")
        if result['success']:
            print(f"   ✅ 计算成功!")
            print(f"   📉 Loss: {result['loss']:.4f}")
            print(f"   📊 NMR峰数量: {result['nmr_count']}")
            print(f"   📊 SMILES长度: {result['smiles_length']}")
            
            # 显示cross attention信息
            if result.get('cross_attentions') is not None:
                cross_attn = result['cross_attentions']
                print(f"   🎯 Cross Attention:")
                print(f"      - 层数: {len(cross_attn)}")
                if len(cross_attn) > 0:
                    layer0 = cross_attn[0]
                    print(f"      - 第0层形状: {layer0.shape}")
                    print(f"      - 第0层范围: [{layer0.min():.4f}, {layer0.max():.4f}]")
                    print(f"      - 第0层均值: {layer0.mean():.4f}")
                    
                    # 提取第5层(倒数第二层)的attention
                    if len(cross_attn) > 4:
                        layer4 = cross_attn[4]
                        print(f"      - 第5层形状: {layer4.shape}")
                        print(f"      - 第5层范围: [{layer4.min():.4f}, {layer4.max():.4f}]")
                        print(f"      - 第5层均值: {layer4.mean():.4f}")
                        
                        # 转换为numpy格式便于使用
                        layer4_numpy = layer4[0].cpu().numpy()  # [num_heads, seq_len, nmr_len]
                        print(f"      - 第5层numpy形状: {layer4_numpy.shape}")
            else:
                print(f"   ⚠️  Cross Attention为空")
        else:
            print(f"   ❌ 计算失败: {result.get('error', 'Unknown error')}")

        print("\n" + "=" * 80)
        print("✅ Forward Loss计算完成!")
        print("=" * 80)
        
        # 5. 智能选择最佳头部
        print("\n💡 智能Head选择演示:")
        print("-" * 80)
        
        best_head_idx, head_scores = forward_model.select_best_head(result, layer_idx=4)
        
        if best_head_idx is not None:
            # 使用智能选择的最佳头部
            layer5_best_head = forward_model.get_cross_attention_matrix(result, layer_idx=4, head_idx=best_head_idx)
            if layer5_best_head is not None:
                print(f"\n🎯 智能选择的最佳头部{best_head_idx}信息:")
                print(f"   形状: {layer5_best_head.shape}")
                print(f"   范围: [{layer5_best_head.min():.4f}, {layer5_best_head.max():.4f}]")
                print(f"   均值: {layer5_best_head.mean():.4f}")
                print(f"   标准差: {layer5_best_head.std():.4f}")
                
                # 显示前3x3子矩阵
                print(f"   前3x3子矩阵:")
                print(f"   {layer5_best_head[:3, :3]}")
        
        layer5_best_head = forward_model.get_cross_attention_matrix(result, layer_idx=4, head_idx=best_head_idx)
        layer5_best_head = layer5_best_head[:-1, 1:-1]
        
        # 只选择碳原子，排除其他原子如O
        c_index_list = []
        for i, char in enumerate(smiles):
            if char in ["C", "c"]:  # 只包含大写和小写的碳原子
                c_index_list.append(i)
        
        # 先筛选碳原子，再根据closest_indices筛选对应的NMR列
        new_layer5_best_head = layer5_best_head[c_index_list, :]
        
        print(f"📊 原始注意力矩阵形状: {layer5_best_head.shape}")
        print(f"📊 筛选后的矩阵形状: {new_layer5_best_head.shape}")
        print(f"📊 筛选后的碳原子数量: {len(c_index_list)}")
        print(f"📊 筛选后的NMR数量: {len(closest_indices)}")
        print(f"📊 碳原子在SMILES中的位置: {c_index_list}")
        print(f"📊 选择的NMR索引: {closest_indices}")
        
        # 贪婪匹配算法：结合注意力和化学经验（碳原子键类型）
        print("\n🔍 执行经验增强的贪婪匹配算法...")
        print(f"矩阵形状: {new_layer5_best_head.shape} (原子数 × NMR数)")
        
        try:
            import numpy as np
            from rdkit import Chem
            from rdkit.Chem import rdmolops
            
            # 创建分子对象用于分析键类型
            mol = Chem.MolFromSmiles(smiles)
            
            # 分析每个碳原子的键类型
            carbon_bond_types = []
            for atom in mol.GetAtoms():
                if atom.GetSymbol() == 'C':
                    # 检查碳原子的键类型
                    has_double_bond = False
                    has_triple_bond = False
                    has_aromatic_bond = False
                    
                    for bond in atom.GetBonds():
                        bond_type = bond.GetBondType()
                        if bond_type == Chem.rdchem.BondType.DOUBLE:
                            has_double_bond = True
                        elif bond_type == Chem.rdchem.BondType.TRIPLE:
                            has_triple_bond = True
                        elif bond_type == Chem.rdchem.BondType.AROMATIC:
                            has_aromatic_bond = True
                    
                    # 判断是否有多重键
                    has_multiple_bonds = has_double_bond or has_triple_bond or has_aromatic_bond
                    carbon_bond_types.append(has_multiple_bonds)
                else:
                    carbon_bond_types.append(None)  # 非碳原子
            
            print(f"📊 分析了{len([bt for bt in carbon_bond_types if bt is not None])}个碳原子的键类型")
            
            # 创建经验增强的注意力矩阵
            attention_matrix = new_layer5_best_head.copy()
            num_atoms, num_nmr = attention_matrix.shape
            
            # 应用化学经验规则，直接将不匹配的设置为无效
            enhanced_attention = attention_matrix.copy()
            
            for atom_idx in range(num_atoms):
                for nmr_idx in range(num_nmr):
                    nmr_value = align_nmr_shfit[nmr_idx]
                    
                    # 获取对应碳原子的键类型信息
                    # 需要找到SMILES中第atom_idx个碳原子对应的RDKit原子索引
                    carbon_count = 0
                    rdkit_carbon_idx = None
                    for atom in mol.GetAtoms():
                        if atom.GetSymbol() == 'C':
                            if carbon_count == atom_idx:
                                rdkit_carbon_idx = atom.GetIdx()
                                break
                            carbon_count += 1
                    
                    if rdkit_carbon_idx is not None:
                        has_multiple_bonds = carbon_bond_types[rdkit_carbon_idx]
                        
                        # 经验规则：如果只有单键，NMR应<80ppm；如果有双键/三键/芳香键，NMR应>80ppm
                        if has_multiple_bonds and nmr_value < 80:
                            # 有多重键但NMR<80ppm：直接设置无效
                            enhanced_attention[atom_idx, nmr_idx] = -1
                        elif not has_multiple_bonds and nmr_value > 80:
                            # 只有单键但NMR>80ppm：直接设置无效
                            enhanced_attention[atom_idx, nmr_idx] = -1
                        # 其他情况保持原注意力值
            
            # 存储匹配结果
            atom_indices = []
            nmr_indices = []
            attention_scores = []
            match_types = []  # 记录匹配类型
            
            max_matches = min(num_atoms, num_nmr)
            
            print(f"开始经验增强的贪婪匹配，最多匹配{max_matches}对...")
            
            for iteration in range(max_matches):
                # 找到经验增强矩阵中最大的注意力值
                max_idx = np.unravel_index(np.argmax(enhanced_attention), enhanced_attention.shape)
                atom_idx, nmr_idx = max_idx
                max_attention = enhanced_attention[atom_idx, nmr_idx]
                original_attention = attention_matrix[atom_idx, nmr_idx]
                
                # 判断是否是经验规则过滤掉的匹配
                is_experience_filtered = (max_attention != original_attention)
                
                # 如果最大值为负数或很小，停止匹配
                if max_attention <= 0:
                    print(f"第{iteration+1}轮: 最大增强注意力值{max_attention:.4f} <= 0，停止匹配")
                    break
                
                # 记录这次匹配
                atom_indices.append(atom_idx)
                nmr_indices.append(nmr_idx)
                attention_scores.append(original_attention)
                
                # 获取对应的原子字符和NMR值
                atom_char = smiles[c_index_list[atom_idx]]
                nmr_value = align_nmr_shfit[nmr_idx]
                
                # 判断匹配类型
                match_type = "✅经验一致" if not is_experience_filtered else "🚫经验过滤"
                match_types.append(match_type)
                
                print(f"第{iteration+1}轮: 原子{atom_idx}({atom_char}) <-> NMR{nmr_idx}({nmr_value:.2f}) "
                      f"| 原始注意力: {original_attention:.4f} | {match_type}")
                
                # 将对应的行和列设置为-1，表示已匹配
                enhanced_attention[atom_idx, :] = -1
                enhanced_attention[:, nmr_idx] = -1
                attention_matrix[atom_idx, :] = -1
                attention_matrix[:, nmr_idx] = -1
            
            # 转换为numpy数组
            atom_indices = np.array(atom_indices)
            nmr_indices = np.array(nmr_indices)
            attention_scores = np.array(attention_scores)
            
            print(f"\n🎯 经验增强匹配结果:")
            print(f"原子索引: {atom_indices}")
            print(f"NMR索引: {nmr_indices}")
            print(f"成功匹配数: {len(atom_indices)}")
            
            # 计算总注意力和平均注意力
            total_attention = np.sum(attention_scores)
            avg_attention = np.mean(attention_scores) if len(attention_scores) > 0 else 0
            
            # 统计匹配类型
            consistent_matches = sum(1 for mt in match_types if "经验一致" in mt)
            filtered_matches = sum(1 for mt in match_types if "经验过滤" in mt)
            
            print(f"\n📊 匹配统计:")
            print(f"  总注意力: {total_attention:.4f}")
            print(f"  平均注意力: {avg_attention:.4f}")
            print(f"  经验一致匹配: {consistent_matches}/{len(match_types)} ({consistent_matches/len(match_types)*100:.1f}%)")
            print(f"  经验过滤匹配: {filtered_matches}/{len(match_types)} ({filtered_matches/len(match_types)*100:.1f}%)")
            print(f"  最高注意力: {np.max(attention_scores):.4f}" if len(attention_scores) > 0 else "  最高注意力: N/A")
            print(f"  最低注意力: {np.min(attention_scores):.4f}" if len(attention_scores) > 0 else "  最低注意力: N/A")
            
            # 保存匹配结果
            matching_result = {
                'atom_indices': atom_indices,
                'nmr_indices': nmr_indices,
                'attention_scores': attention_scores.tolist(),
                'total_attention': float(total_attention),
                'avg_attention': float(avg_attention),
                'atom_chars': [smiles[c_index_list[idx]] for idx in atom_indices],
                'nmr_values': [align_nmr_shfit[idx] for idx in nmr_indices],
                'attention_matrix': new_layer5_best_head,
                'match_types': match_types,
                'algorithm': 'experience_enhanced_greedy',
                'consistent_matches': consistent_matches,
                'filtered_matches': filtered_matches
            }
            
        except Exception as e:
            print(f"❌ 经验增强匹配算法执行失败: {e}")
            import traceback
            traceback.print_exc()
            matching_result = None
        
        # RDKit可视化：在原子位置添加NMR位移
        if matching_result is not None:
            print("\n🎨 RDKit分子结构可视化...")
            try:
                from rdkit import Chem
                from rdkit.Chem import Draw
                
                # 创建分子对象
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    print(f"   📊 分子原子数: {mol.GetNumAtoms()}")
                    print(f"   📊 SMILES字符数: {len(smiles)}")
                    print(f"   📊 碳原子在SMILES中的位置: {c_index_list}")
                    
                    # 创建原子到NMR的映射 - 需要找到SMILES字符索引到RDKit原子索引的正确映射
                    atom_to_nmr = {}
                    
                    # 使用RDKit的原子映射来找到正确的对应关系
                    for atom_idx, nmr_idx in zip(matching_result['atom_indices'], 
                                                 matching_result['nmr_indices']):
                        # 获取SMILES中的碳原子位置
                        smiles_char_idx = c_index_list[atom_idx]
                        
                        # 找到对应的RDKit原子索引（只考虑碳原子）
                        carbon_count = 0
                        rdkit_atom_idx = None
                        
                        for atom in mol.GetAtoms():
                            if atom.GetSymbol() == 'C':  # 只处理碳原子
                                if carbon_count == atom_idx:
                                    rdkit_atom_idx = atom.GetIdx()
                                    break
                                carbon_count += 1
                        
                        if rdkit_atom_idx is not None and rdkit_atom_idx < mol.GetNumAtoms():
                            nmr_value = matching_result['nmr_values'][nmr_idx]
                            atom_to_nmr[rdkit_atom_idx] = nmr_value
                            atom = mol.GetAtomWithIdx(rdkit_atom_idx)
                            print(f"   映射: 碳原子{atom_idx} -> RDKit碳原子[{rdkit_atom_idx}] -> NMR:{nmr_value:.2f}")
                        else:
                            print(f"   ⚠️  跳过无效映射: 碳原子{atom_idx} -> RDKit[{rdkit_atom_idx}]")
                    
                    # 使用SetProp为每个原子设置NMR标签
                    for atom in mol.GetAtoms():
                        atom_idx = atom.GetIdx()
                        if atom_idx in atom_to_nmr:
                            nmr_value = atom_to_nmr[atom_idx]
                            atom.SetProp('atomNote', f'{nmr_value:.1f}')
                        else:
                            # 清除现有标签
                            if atom.HasProp('atomNote'):
                                atom.ClearProp('atomNote')
                    
                    print(f"   📊 已设置NMR标签的原子数: {sum(1 for atom in mol.GetAtoms() if atom.HasProp('atomNote'))}")
                    
                    # 使用MolDraw2DCairo创建高质量的分子图
                    from rdkit.Chem.Draw import MolDraw2DCairo
                    
                    # 创建标准分辨率图像
                    d2d = MolDraw2DCairo(800, 600)
                    dopts = d2d.drawOptions()
                    dopts.baseFontSize = 0.5  # 调整基础字体大小
                    dopts.annotationFontScale = 0.7  # 调整注释字体大小
                    
                    d2d.DrawMolecule(mol)
                    d2d.FinishDrawing()
                    
                    # 保存标准分辨率图像
                    output_path = "nmr_atom_matching.png"
                    with open(output_path, 'wb') as f:
                        f.write(d2d.GetDrawingText())
                    print(f"   ✅ 可视化图像已保存: {output_path}")
                    
                    # 创建高分辨率图像
                    d2d_hq = MolDraw2DCairo(1200, 900)
                    dopts_hq = d2d_hq.drawOptions()
                    dopts_hq.baseFontSize = 0.6  # 更大的字体
                    dopts_hq.annotationFontScale = 0.8  # 更大的注释字体
                    
                    d2d_hq.DrawMolecule(mol)
                    d2d_hq.FinishDrawing()
                    
                    # 保存高分辨率图像
                    hq_output_path = "nmr_atom_matching_hq.png"
                    with open(hq_output_path, 'wb') as f:
                        f.write(d2d_hq.GetDrawingText())
                    print(f"   ✅ 高分辨率图像已保存: {hq_output_path}")
                    
                    # 额外创建一个超大高分辨率版本用于细节查看
                    d2d_ultra = MolDraw2DCairo(1600, 1200)
                    dopts_ultra = d2d_ultra.drawOptions()
                    dopts_ultra.baseFontSize = 0.7
                    dopts_ultra.annotationFontScale = 1.0
                    
                    d2d_ultra.DrawMolecule(mol)
                    d2d_ultra.FinishDrawing()
                    
                    ultra_output_path = "nmr_atom_matching_ultra.png"
                    with open(ultra_output_path, 'wb') as f:
                        f.write(d2d_ultra.GetDrawingText())
                    print(f"   ✅ 超高分辨率图像已保存: {ultra_output_path}")
                    
                    # 打印匹配总结
                    print(f"\n📋 匹配总结:")
                    print(f"   分子式: {Chem.MolToSmiles(mol)}")
                    print(f"   原子数: {mol.GetNumAtoms()}")
                    print(f"   NMR峰数: {len(matching_result['nmr_values'])}")
                    print(f"   成功匹配数: {len(atom_to_nmr)}")
                    
                    # 打印详细的原子-NMR对应关系（只显示碳原子）
                    print(f"\n🔍 详细碳原子-NMR匹配信息:")
                    carbon_match_count = 0
                    for rdkit_idx, nmr_value in sorted(atom_to_nmr.items()):
                        atom = mol.GetAtomWithIdx(rdkit_idx)
                        if atom.GetSymbol() == 'C':  # 只显示碳原子
                            carbon_match_count += 1
                            print(f"   碳原子{rdkit_idx} -> NMR: {nmr_value:.2f} ppm")
                    
                    print(f"\n📊 匹配统计:")
                    print(f"   总碳原子数: {sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C')}")
                    print(f"   已匹配碳原子: {carbon_match_count}")
                    print(f"   匹配率: {carbon_match_count}/{len(matching_result['atom_indices'])} ({carbon_match_count/len(matching_result['atom_indices'])*100:.1f}%)")
                    
                else:
                    print("   ❌ 无法从SMILES创建分子对象")
                    
            except ImportError:
                print("   ❌ 未安装RDKit，无法进行可视化")
                print("   安装命令: pip install rdkit")
            except Exception as e:
                print(f"   ❌ 可视化失败: {e}")
                import traceback
                traceback.print_exc()
        
        # import ipdb
        # ipdb.set_trace()
        
        print("-" * 80)
        print("📚 使用方法总结:")
        print("  1. 计算forward loss并获取cross attention:")
        print("     result = forward_model.compute_forward_loss(nmr_shifts, smiles)")
        print()
        print("  2. 智能选择最佳头部:")
        print("     best_head_idx, scores = forward_model.select_best_head(result, layer_idx=4)")
        print()
        print("  3. 获取智能选择的最佳头部attention:")
        print("     best_head_attn = forward_model.get_cross_attention_matrix(result, layer_idx=4, head_idx=best_head_idx)")
        print()
        print("  4. 或者直接访问任意头部:")
        print("     head0_attn = forward_model.get_cross_attention_matrix(result, layer_idx=4, head_idx=0)")
        print("=" * 80)

        # 6. 直接返回最优head的交叉注意力矩阵
        print("\n🎯 返回最优Head的交叉注意力矩阵:")
        print("-" * 80)
        
        if best_head_idx is not None:
            # 直接获取最优head的attention矩阵
            best_head_matrix = forward_model.get_cross_attention_matrix(result, layer_idx=4, head_idx=best_head_idx)
            
            print(f"✅ 最优Head: 第{best_head_idx}号头部")
            print(f"📊 Attention矩阵形状: {best_head_matrix.shape}")
            print(f"📈 数值范围: [{best_head_matrix.min():.4f}, {best_head_matrix.max():.4f}]")
            print(f"📊 统计信息:")
            print(f"   - 均值: {best_head_matrix.mean():.4f}")
            print(f"   - 标准差: {best_head_matrix.std():.4f}")
            print(f"   - 中位数: {np.median(best_head_matrix):.4f}")
            
            # 添加到result中方便访问
            result['best_head_idx'] = best_head_idx
            result['best_head_matrix'] = best_head_matrix
            result['head_scores'] = head_scores
            
            print(f"\n💾 访问方式:")
            print(f"   result['best_head_matrix']  # 最优head的attention矩阵")
            print(f"   result['best_head_idx']     # 最优head的索引")
            print(f"   result['head_scores']       # 所有head的评分")
        else:
            print("❌ 无法获取最优head")
            result['best_head_matrix'] = None
            result['best_head_idx'] = None
            result['head_scores'] = None
        
        print("=" * 80)

        return result

    except Exception as e:
        print(f"\n❌ 程序执行失败: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    result = main()
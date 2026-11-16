#!/usr/bin/env python3
"""
完整CMGNet集成工具 - 真实模型加载、推理和智能头部选择
包含真实的NMR-SMILES对齐分析
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Draw, AllChem
import json

class CompleteCMGNetAnalyzer:
    """完整的CMGNet分析器 - 包含模型加载、推理和智能分析"""

    def __init__(self, model_path: str = None, device: str = "cuda:0"):
        """
        初始化CMGNet分析器

        Args:
            model_path: 模型路径
            device: 设备 (cuda:0 或 cpu)
        """
        self.device = device
        self.max_length = 512  # 参照main_cmgnet.py的设置

        # 设置默认模型路径
        project_dir = Path(__file__).parent
        if model_path is None:
            # 使用正确的模型权重路径
            model_path = str(project_dir / "models/cmgnet/train_cmgnet_ring_system_3.2_1029_v2_best.ckpt")

        print("=" * 80)
        print("🚀 初始化完整CMGNet分析器")
        print("=" * 80)
        print(f"📁 模型路径: {model_path}")
        print(f"🔧 设备: {device}")

        # 加载模型和分词器
        print("\n📦 加载CMGNet模型...")
        try:
            # 检查模型路径类型
            model_path_obj = Path(model_path)

            if model_path_obj.is_dir():
                # 如果是目录，使用HuggingFace格式
                print("   检测到HuggingFace格式模型目录")
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            elif model_path_obj.suffix == '.ckpt':
                # 如果是.ckpt文件，使用PyTorch Lightning格式
                print("   检测到PyTorch Lightning checkpoint文件")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    str(project_dir / "checkpoints/saved_hf_model_0910")
                )

                # 加载checkpoint
                from transformers import AutoModelForSeq2SeqLM
                import torch

                # 添加安全全局变量
                torch.serialization.add_safe_globals([
                    'transformers.models.bart.tokenization_bart_fast.BartTokenizerFast'
                ])

                # 使用weights_only=False加载完整checkpoint
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

            # 获取模型配置信息
            self.config = self.model.config
            self.num_layers = self.config.num_hidden_layers
            self.num_heads_per_layer = self.config.num_attention_heads

            print(f"   ✅ 模型加载成功")
            print(f"   📊 模型信息:")
            print(f"      - 层数: {self.num_layers}")
            print(f"      - 每层注意力头数: {self.num_heads_per_layer}")

        except Exception as e:
            print(f"   ❌ 模型加载失败: {e}")
            raise

        # NMR数据处理类
        self.nmr_processor = self.NmrProcessor()
        
        # SMILES处理函数 - 参照main_cmgnet.py
        self.smiles2token = lambda smile_string, phase="val": self._smiles2token(smile_string, phase)

        print(f"   ✅ CMGNet分析器初始化完成")
        print("=" * 80)

    def _smiles2token(self, smile_string: str, phase: str = "val"):
        """参照main_cmgnet.py的smiles2token函数"""
        tmp = self.tokenizer([smile_string], max_length=self.max_length)
        tmp["input_ids"] = [[187] + i[1:-1] + [188] for i in tmp["input_ids"]]
        tmp["input_ids"] = tmp["input_ids"][0]
        tmp["attention_mask"] = tmp["attention_mask"][0]
        return tmp

    class NmrProcessor:
        """NMR数据处理器"""

        def __init__(self):
            self.min_value = -50
            self.max_value = 350
            self.scale = 10
            self.units = (self.max_value - self.min_value) * self.scale
            self.start = 189
            self.end = 190
            self.number_start = 191

        def process_nmr(self, nmr_shifts: List[float]) -> Dict:
            """
            处理NMR化学位移数据

            Args:
                nmr_shifts: NMR化学位移列表

            Returns:
                处理后的NMR数据字典
            """
            # 去重和排序
            nmr_unique = sorted(list(set(nmr_shifts)), reverse=True)

            # 转换为模型输入格式
            nmr_scaled = [round((value - self.min_value) * self.scale) for value in nmr_unique]
            nmr_scaled = [min(max(0, value), self.units-1) for value in nmr_scaled]
            nmr_scaled.sort()

            # 添加特殊标记
            input_ids = [self.start] + [value + self.number_start for value in nmr_scaled] + [self.end]
            attention_mask = [1 for _ in range(len(input_ids))]

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "original_shifts": nmr_unique,
                "scaled_values": nmr_scaled
            }

    def explain_attention_structure(self, cross_attentions) -> None:
        """
        详细解释attention分数的结构和访问方式
        
        Args:
            cross_attentions: 模型输出的cross_attentions
        """
        if cross_attentions is None:
            print("❌ Cross_attentions is None")
            return
            
        print("\n" + "=" * 80)
        print("📚 Cross-Attention分数结构详解")
        print("=" * 80)
        
        print(f"1️⃣  总体结构:")
        print(f"   cross_attentions类型: {type(cross_attentions)}")
        print(f"   层数: {len(cross_attentions)} (CMGNet有6层cross-attention)")
        
        print(f"\n2️⃣  每层结构:")
        for layer_idx in range(min(3, len(cross_attentions))):  # 显示前3层
            layer_attn = cross_attentions[layer_idx]
            print(f"   第{layer_idx}层: shape = {layer_attn.shape}")
            print(f"            [batch_size={layer_attn.shape[0]}, num_heads={layer_attn.shape[1]}, "
                  f"seq_len={layer_attn.shape[2]}, nmr_len={layer_attn.shape[3]}]")
        
        print(f"\n3️⃣  访问特定层的attention:")
        layer5_attn = cross_attentions[4]  # 第5层(索引4)
        print(f"   layer5_attn = cross_attentions[4]")
        print(f"   layer5_attn.shape = {layer5_attn.shape}")
        
        print(f"\n4️⃣  访问特定头部的attention:")
        head0 = layer5_attn[0, 0, :, :]  # 第5层第0个头
        head1 = layer5_attn[0, 1, :, :]  # 第5层第1个头
        print(f"   head0 = layer5_attn[0, 0, :, :]  # 第5层第0个头")
        print(f"   head0.shape = {head0.shape}  # [seq_len, nmr_len]")
        print(f"   head0范围: [{head0.min():.4f}, {head0.max():.4f}]")
        print(f"   head0均值: {head0.mean():.4f}")
        
        print(f"\n5️⃣  具体attention分数示例:")
        # 显示几个具体的attention值
        print(f"   head0[0, 0] = {head0[0, 0]:.4f}  # 第1个SMILES token对第1个NMR峰的attention")
        print(f"   head0[0, 1] = {head0[0, 1]:.4f}  # 第1个SMILES token对第2个NMR峰的attention")
        print(f"   head0[1, 0] = {head0[1, 0]:.4f}  # 第2个SMILES token对第1个NMR峰的attention")
        
        print(f"\n6️⃣  可视化attention矩阵:")
        print(f"   - 横轴: NMR化学位移 (ppm)")
        print(f"   - 纵轴: SMILES字符")
        print(f"   - 颜色: attention权重 (越高表示对齐越强)")
        
        print("=" * 80)

    def select_responsive_heads(self, cross_attentions, threshold: float = 0.1, top_k: int = 1):
        """
        选择对 NMR 输入响应最强烈的头 - 基于6层所有头部的全面分析
        
        Args:
            cross_attentions: [num_layers, num_heads, seq_len, nmr_len] 或 tuple格式
            threshold: 响应阈值
            top_k: 返回top-k个最佳头部
        
        Returns:
            top_k个最佳头部的列表，格式: [(layer_idx, head_idx, score), ...]
        """
        print("🔍 调试cross_attentions类型:", type(cross_attentions))
        print("🔍 cross_attentions长度:", len(cross_attentions) if hasattr(cross_attentions, '__len__') else "N/A")

        # 处理不同的cross_attentions格式
        if isinstance(cross_attentions, tuple):
            # 在生成模式下，cross_attentions是tuple of tuples
            if len(cross_attentions) > 0:
                # 使用最后一个生成步骤的注意力
                last_step_attention = cross_attentions[-1]
                
                if isinstance(last_step_attention, tuple):
                    # 格式: (layer0_attn, layer1_attn, ..., layer5_attn)
                    # 每个layer_attn形状: [batch_size, num_heads, seq_len, nmr_len]
                    all_layers = []
                    for layer_attn in last_step_attention:
                        # 确保每个layer_attn都移除了batch维度
                        if layer_attn.dim() == 4 and layer_attn.size(0) == 1:
                            layer_attn = layer_attn.squeeze(0)  # [num_heads, seq_len, nmr_len]
                        all_layers.append(layer_attn)
                    
                    # 堆叠为 [num_layers, num_heads, seq_len, nmr_len]
                    cross_attentions_tensor = torch.stack(all_layers, dim=0)
                else:
                    # 已经是处理过的格式
                    if last_step_attention.dim() == 4 and last_step_attention.size(0) == 1:
                        last_step_attention = last_step_attention.squeeze(0)
                    cross_attentions_tensor = last_step_attention
            else:
                raise ValueError("Empty cross_attentions tuple")
        else:
            # 直接是tensor的情况
            if cross_attentions.dim() == 4:  # [batch, num_layers, num_heads, seq_len, nmr_len] 或类似
                if cross_attentions.size(0) == 1:
                    cross_attentions_tensor = cross_attentions.squeeze(0)
                else:
                    cross_attentions_tensor = cross_attentions
            else:
                cross_attentions_tensor = cross_attentions

        print(f"🔍 cross_attentions_tensor形状: {cross_attentions_tensor.shape}")
        print(f"🔍 预期形状: [num_layers, num_heads, seq_len, nmr_len]")

        num_layers = cross_attentions_tensor.size(0)
        # 检查维度是否正确
        if cross_attentions_tensor.dim() != 4:
            raise ValueError(f"cross_attentions_tensor维度错误，期望4维，实际{cross_attentions_tensor.dim()}维")
        
        # 第二维应该是num_heads，如果不是则进行转置或重新排列
        if cross_attentions_tensor.size(1) > 20:  # 如果第二维远大于正常的num_heads(通常是8或16)
            print(f"⚠️  检测到异常维度，可能需要转置")
            print(f"当前形状: {cross_attentions_tensor.shape}")
            # 尝试重新排列维度
            if cross_attentions_tensor.size(2) <= 20:  # 假设第3维是num_heads
                print(f"尝试将第2维和第3维交换")
                cross_attentions_tensor = cross_attentions_tensor.permute(0, 2, 1, 3)
                print(f"转置后形状: {cross_attentions_tensor.shape}")
        
        num_heads = cross_attentions_tensor.size(1)
        head_scores = {}

        print(f"\n🔍 分析所有{num_layers}层的{num_heads}个头部...")
        print("=" * 80)

        for l in range(num_layers):
            for h in range(num_heads):
                attn = cross_attentions_tensor[l, h]  # [seq_len, nmr_len]
                
                # 指标1: 最大注意力权重（峰值响应）
                max_response = attn.max().item()
                
                # 指标2: 注意力熵（低熵 = 集中对齐）
                attn_safe = attn + 1e-9
                entropy = -(attn_safe * torch.log(attn_safe)).sum(dim=-1).mean().item()
                
                # 指标3: 稀疏度（Gini系数）
                # 对每行进行排序并计算稀疏度
                sorted_attn, _ = attn.sort(dim=-1, descending=True)
                cumsum = sorted_attn.cumsum(dim=-1)
                weights = torch.arange(1, attn.size(-1) + 1, dtype=torch.float32, device=attn.device) / attn.size(-1)
                sparsity = (1 - 2 * (cumsum * weights).sum(dim=-1).mean()).item()
                
                # 综合评分
                score = max_response - 0.5 * entropy + 0.3 * sparsity
                
                head_scores[(l, h)] = {
                    'max_response': max_response,
                    'entropy': entropy,
                    'sparsity': sparsity,
                    'composite_score': score
                }
                
                # 显示每个头部的详细信息（可选，避免输出过多）
                if l == 4:  # 只显示第5层的详细信息
                    print(f"第{l}层头部{h}: 最大响应={max_response:.4f}, 熵={entropy:.4f}, "
                          f"稀疏度={sparsity:.4f}, 综合评分={score:.4f}")

        # 按综合评分排序
        sorted_heads = sorted(head_scores.items(), key=lambda x: x[1]['composite_score'], reverse=True)

        print("\n" + "=" * 80)
        print(f"🏆 Top-{min(top_k, len(sorted_heads))} 最佳头部:")
        
        for i, ((layer_idx, head_idx), scores) in enumerate(sorted_heads[:top_k]):
            print(f"  {i+1}. 第{layer_idx}层 - 头部{head_idx}:")
            print(f"     综合评分: {scores['composite_score']:.4f}")
            print(f"     最大响应: {scores['max_response']:.4f}")
            print(f"     熵值: {scores['entropy']:.4f}")
            print(f"     稀疏度: {scores['sparsity']:.4f}")

        best_score = sorted_heads[0][1]['composite_score']
        if best_score < threshold:
            print(f"\n⚠️  警告: 最佳头部评分 {best_score:.4f} 低于阈值 {threshold}")
            print(f"💡 建议: 考虑调整阈值或使用多层集成")

        # 返回top_k个最佳头部
        return [((layer_idx, head_idx), scores['composite_score']) 
                for (layer_idx, head_idx), scores in sorted_heads[:top_k]]

    def auto_select_best_head(self, cross_attentions, threshold: float = 0.1, prefer_layer: int = 4) -> tuple:
        """
        智能选择最佳头部 - 基于所有6层的全面分析
        
        Args:
            cross_attentions: 交叉注意力权重 (可能是tuple或tensor)
            threshold: 响应阈值
            prefer_layer: 优先选择的层 (默认第5层，索引4)

        Returns:
            (layer_idx, head_idx) 最佳头部的层和头部索引
        """
        # 调用新的全面分析函数，获取最佳头部
        top_heads = self.select_responsive_heads(cross_attentions, threshold, top_k=6)  # 获取top-6
        
        if not top_heads:
            raise ValueError("无法选择最佳头部")
        
        # 优先选择指定层的最佳头部，如果没有则选择全局最佳
        preferred_head = None
        for (layer_idx, head_idx), score in top_heads:
            if layer_idx == prefer_layer:
                preferred_head = (layer_idx, head_idx, score)
                break
        
        if preferred_head:
            layer_idx, head_idx, score = preferred_head
            print(f"\n🎯 最终选择: 第{layer_idx}层 (倒数第二层) - 头部{head_idx} (评分: {score:.4f})")
        else:
            (layer_idx, head_idx), score = top_heads[0]
            print(f"\n🎯 最终选择: 第{layer_idx}层 - 头部{head_idx} (评分: {score:.4f})")
            print(f"💡 注意: 第{prefer_layer}层没有高质量头部，选择了其他层的最佳头部")
        
        return (layer_idx, head_idx)

    def predict_and_analyze(self, nmr_shifts: List[float], target_smiles: str = None, num_returns: int = 1,
                          mode: str = "smiles", auto_select_head: bool = True) -> Dict:
        """
        执行完整的CMGNet预测和智能分析 - 直接使用forward计算loss

        Args:
            nmr_shifts: NMR化学位移列表
            target_smiles: 目标SMILES字符串，用于计算loss (可选)
            num_returns: 返回结果数量
            mode: 预测模式 ("frag", "smiles")
            auto_select_head: 是否自动选择最佳头部

        Returns:
            完整的分析结果字典
        """
        print(f"\n🎯 开始CMGNet预测和智能分析")
        print(f"   NMR峰数量: {len(nmr_shifts)}")
        print(f"   预测模式: {mode}")
        print(f"   返回Top-{num_returns}结果")
        if target_smiles:
            print(f"   目标SMILES: {target_smiles}")

        try:
            # 1. 处理NMR数据
            nmr_data = self.nmr_processor.process_nmr(nmr_shifts)

            # 2. 准备模型输入
            inputs = {
                "input_ids": torch.tensor([nmr_data["input_ids"]]).to(self.device),
                "attention_mask": torch.tensor([nmr_data["attention_mask"]]).to(self.device)
            }

            # 3. 设置token IDs
            if mode == "smiles":
                bos_token_id = 187
                eos_token_id = 188
                decoder_start_token_id = 187
            else:  # "frag" 模式
                bos_token_id = 183
                eos_token_id = 184
                decoder_start_token_id = 183

            # 4. 准备decoder输入 - 参照main_cmgnet.py
            if target_smiles:
                # 如果提供了目标SMILES，使用它来准备decoder输入
                smiles_result = self._smiles2token(target_smiles, phase="val")
                decoder_input_ids = torch.tensor([smiles_result["input_ids"]]).to(self.device)
                decoder_attention_mask = torch.tensor([smiles_result["attention_mask"]]).to(self.device)
                
                # 准备labels (用于计算loss) - 参照main_cmgnet.py的collate_fn
                labels = decoder_input_ids[:, 1:].clone()  # 去掉第一个token
                decoder_input_ids = decoder_input_ids[:, :-1]  # 去掉最后一个token
                decoder_attention_mask = decoder_attention_mask[:, :-1]
                
                print(f"   📝 目标SMILES长度: {len(target_smiles)}")
                print(f"   📊 Decoder输入长度: {decoder_input_ids.size(1)}")
            else:
                # 如果没有提供目标SMILES，创建简单的decoder输入用于生成
                decoder_input_ids = torch.full((1, 1), decoder_start_token_id, 
                                             dtype=torch.long, device=self.device)
                decoder_attention_mask = torch.ones((1, 1), dtype=torch.long, device=self.device)
                labels = None

            # 5. 执行模型forward - 参照main_cmgnet.py的training_step
            print(f"\n🧠 执行CMGNet forward pass...")
            with torch.no_grad():
                # 准备完整的输入 - 参照main_cmgnet.py的batch格式
                model_inputs = {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                    "decoder_input_ids": decoder_input_ids,
                    "output_attentions": True,  # 输出attention权重
                    "output_hidden_states": True,  # 输出hidden states
                    "return_dict": True
                }
                
                # 如果有labels，添加到输入中用于计算loss
                if labels is not None:
                    model_inputs["labels"] = labels
                
                # 执行forward pass - 获取loss和attentions
                outputs = self.model(**model_inputs)
                
                # 获取loss - 参照main_cmgnet.py的training_step处理
                if hasattr(outputs, 'loss'):
                    loss = outputs.loss
                    if loss is not None:
                        print(f"   📉 Model Loss: {loss.item():.4f}")
                    else:
                        loss = None
                        print(f"   ⚠️  Model loss is None")
                else:
                    loss = None
                    print(f"   ⚠️  Model output has no loss attribute")
                
                # 获取cross_attentions - 用于后续分析
                if hasattr(outputs, 'cross_attentions'):
                    cross_attentions = outputs.cross_attentions
                    print(f"   🎯 Cross-attentions获取成功:")
                    print(f"      - 层数: {len(cross_attentions)} (6层cross-attention)")
                    if len(cross_attentions) > 0:
                        layer_shape = cross_attentions[0].shape
                        print(f"      - 每层形状: {layer_shape} [batch, heads, seq_len, nmr_len]")
                        print(f"      - 注意力头数: {layer_shape[1]}")
                        print(f"      - 序列长度: {layer_shape[2]}")
                        print(f"      - NMR长度: {layer_shape[3]}")
                        
                        # 显示第5层(索引4)的attention信息
                        layer5_attn = cross_attentions[4]
                        print(f"      - 第5层attention范围: [{layer5_attn.min():.4f}, {layer5_attn.max():.4f}]")
                        print(f"      - 第5层attention均值: {layer5_attn.mean():.4f}")
                else:
                    cross_attentions = None
                    print(f"   ⚠️  Model output has no cross_attentions")

            # 6. 处理预测结果
            generated_sequences = []
            
            if target_smiles:
                # 使用目标SMILES作为预测结果
                generated_sequences.append(decoder_input_ids[0])
                pred_smiles = target_smiles
            else:
                # 如果没有提供目标SMILES，需要重新生成预测以获取cross_attentions
                print("   🔄 没有提供目标SMILES，重新生成预测...")
                with torch.no_grad():
                    # 逐步生成预测
                    current_decoder_input = torch.full((1, 1), decoder_start_token_id, 
                                                     dtype=torch.long, device=self.device)
                    
                    for step in range(100):  # 最大生成长度
                        outputs = self.model(
                            input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            decoder_input_ids=current_decoder_input,
                            output_attentions=True,
                            output_hidden_states=True,
                            return_dict=True
                        )
                        
                        logits = outputs.logits[:, -1, :]
                        next_token_id = torch.argmax(logits, dim=-1, keepdim=True)
                        
                        if next_token_id.item() == eos_token_id:
                            break
                            
                        current_decoder_input = torch.cat([current_decoder_input, next_token_id], dim=-1)
                    
                    # 更新cross_attentions为生成时的attentions
                    cross_attentions = outputs.cross_attentions
                    generated_sequences.append(current_decoder_input[0])
                    
                    # 解码预测结果
                    decoded = self.tokenizer.decode(current_decoder_input[0], skip_special_tokens=True)
                    pred_smiles = decoded.replace("<fragment>", "").replace("</fragment>", "").replace("<SMILES>", "").replace("</SMILES>", "").replace("<pad>", "").replace("</s>", "").strip()
                    
                    print(f"   🎯 生成预测SMILES: {pred_smiles}")

            # 7. 处理预测结果
            pred_sequences = [pred_smiles]
            
            # 8. 生成预测结果列表
            top_predictions = []
            for i, smiles in enumerate(pred_sequences):
                is_valid = self.validate_smiles(smiles)
                core_structure, has_core = self.query_core(smiles)

                prediction = {
                    'rank': i + 1,
                    'smiles': smiles,
                    'is_valid': is_valid,
                    'has_core_structure': bool(has_core),
                    'core_structure': core_structure
                }
                
                # 如果有loss，添加到结果中
                if loss is not None:
                    prediction['loss'] = float(loss.item())
                else:
                    prediction['loss'] = 0.0
                
                top_predictions.append(prediction)

            # 9. 智能头部选择和对齐分析
            head_analysis = {}
            if auto_select_head and cross_attentions is not None:
                print(f"\n🤖 执行智能头部选择和对齐分析...")
                
                # 详细解释attention结构
                self.explain_attention_structure(cross_attentions)

                # 自动选择最佳头部
                best_layer, best_head = self.auto_select_best_head(cross_attentions)

                # 使用最佳头部分析第一个预测结果的对齐
                if pred_sequences:
                    alignment_analysis = self.analyze_alignment(
                        cross_attentions, pred_sequences[0], nmr_shifts, best_head, best_layer
                    )
                    head_analysis = {
                        'best_head': best_head,
                        'best_layer': best_layer,
                        'alignment_analysis': alignment_analysis
                    }

            return {
                'top_predictions': top_predictions,
                'num_returns': len(top_predictions),
                'nmr_count': len(nmr_shifts),
                'mode': mode,
                'nmr_data': nmr_data,
                'head_analysis': head_analysis,
                'cross_attentions': cross_attentions,
                'loss': float(loss.item()) if loss is not None else None
            }

        except Exception as e:
            print(f"❌ 预测失败: {e}")
            import traceback
            traceback.print_exc()
            return {
                'top_predictions': [],
                'num_returns': 0,
                'error': str(e),
                'nmr_count': len(nmr_shifts)
            }

    def analyze_alignment(self, cross_attentions, smiles_output: str,
                         nmr_shifts: List[float], head_idx: int, layer_idx: int = 4) -> Dict:
        """
        分析NMR-SMILES对齐关系 - 使用智能选择的最佳头部

        Args:
            cross_attentions: 交叉注意力权重 (可能是tuple或tensor)
            smiles_output: SMILES字符串
            nmr_shifts: NMR化学位移列表
            head_idx: 智能选择的最佳头部索引
            layer_idx: 层索引 (默认第5层，索引4)

        Returns:
            对齐分析结果，包含智能匹配信息
        """
        print(f"🔍 使用智能选择的第{layer_idx}层 - 第{head_idx}号头部进行对齐分析")
        
        # 处理cross_attentions格式 - 需要正确处理batch维度
        if isinstance(cross_attentions, tuple):
            if len(cross_attentions) > 0:
                last_step_attention = cross_attentions[-1]
                if isinstance(last_step_attention, tuple):
                    # 格式: (layer0_attn, layer1_attn, ..., layer5_attn)
                    # 每个layer_attn形状: [batch_size, num_heads, seq_len, nmr_len]
                    layer_attention = last_step_attention[layer_idx]  # [batch_size, num_heads, seq_len, nmr_len]
                    print(f"   📊 layer_attention原始形状: {layer_attention.shape}")
                    
                    # 移除batch维度
                    if layer_attention.size(0) == 1:
                        layer_attention = layer_attention.squeeze(0)  # [num_heads, seq_len, nmr_len]
                    print(f"   📊 layer_attention处理后形状: {layer_attention.shape}")
                else:
                    # 已经是处理过的格式
                    layer_attention = last_step_attention
                    print(f"   📊 layer_attention直接使用形状: {layer_attention.shape}")
            else:
                raise ValueError("Empty cross_attentions tuple")
        else:
            # 直接是tensor的情况
            if cross_attentions.dim() == 4:  # [num_layers, batch, num_heads, seq_len, nmr_len]
                layer_attention = cross_attentions[layer_idx]
                if layer_attention.size(0) == 1:
                    layer_attention = layer_attention.squeeze(0)
            elif cross_attentions.dim() == 3:  # [num_heads, seq_len, nmr_len] (已经是单层)
                layer_attention = cross_attentions
            else:
                layer_attention = cross_attentions[layer_idx]
            
            print(f"   📊 tensor格式layer_attention形状: {layer_attention.shape}")

        # 检查维度和索引范围
        if layer_attention.dim() != 3:
            raise ValueError(f"layer_attention维度错误，期望3维，实际{layer_attention.dim()}维，形状: {layer_attention.shape}")
        
        num_heads = layer_attention.size(0)
        if head_idx >= num_heads:
            raise ValueError(f"头部索引{head_idx}超出范围 [0, {num_heads-1}]，layer_attention形状: {layer_attention.shape}")

        # 提取指定头部的注意力
        head_attention = layer_attention[head_idx]  # [seq_len, nmr_len] 或 [1, seq_len, nmr_len]
        
        # 移除可能的剩余batch维度
        if head_attention.dim() == 3 and head_attention.size(0) == 1:
            head_attention = head_attention.squeeze(0)
        
        attention_matrix = head_attention.cpu().numpy()  # [seq_len, nmr_len]
        
        print(f"   📊 头部{head_idx}的attention矩阵形状: {attention_matrix.shape}")
        print(f"   📈 Attention范围: [{attention_matrix.min():.4f}, {attention_matrix.max():.4f}]")
        print(f"   📊 Attention均值: {attention_matrix.mean():.4f}")

        # 准备SMILES tokens
        smiles_chars = list(smiles_output)

        # 处理长度不匹配
        seq_len, nmr_len = attention_matrix.shape
        if len(smiles_chars) != seq_len:
            if len(smiles_chars) < seq_len:
                # 截取有效长度
                attention_matrix = attention_matrix[:len(smiles_chars), :]
            else:
                # 截取SMILES
                smiles_chars = smiles_chars[:seq_len]

        # 分析对齐
        best_matches = attention_matrix.argmax(axis=1)
        best_confidence = attention_matrix.max(axis=1)

        alignment_results = []
        for i, (char, nmr_idx, conf) in enumerate(zip(smiles_chars, best_matches, best_confidence)):
            if nmr_idx < len(nmr_shifts):
                alignment_results.append({
                    'smiles_char': char,
                    'nmr_shift': nmr_shifts[nmr_idx],
                    'nmr_idx': int(nmr_idx),
                    'confidence': float(conf)
                })
        
        # 计算智能匹配统计信息
        confidences = [r['confidence'] for r in alignment_results]
        smart_stats = {
            'mean_confidence': float(np.mean(confidences)) if confidences else 0.0,
            'std_confidence': float(np.std(confidences)) if confidences else 0.0,
            'max_confidence': float(np.max(confidences)) if confidences else 0.0,
            'min_confidence': float(np.min(confidences)) if confidences else 0.0,
            'dynamic_threshold': float(np.mean(confidences) + 0.5 * np.std(confidences)) if confidences else 0.0
        }
        
        # 智能筛选的高置信度匹配
        smart_matches = [r for r in alignment_results if r['confidence'] >= smart_stats['dynamic_threshold']]
        
        print(f"   🎯 智能匹配统计:")
        print(f"      - 动态阈值: {smart_stats['dynamic_threshold']:.4f}")
        print(f"      - 平均置信度: {smart_stats['mean_confidence']:.4f}")
        print(f"      - 高置信度匹配数: {len(smart_matches)}/{len(alignment_results)}")

        return {
            'attention_matrix': attention_matrix,
            'alignment_results': alignment_results,
            'smiles_chars': smiles_chars,
            'nmr_shifts': nmr_shifts,
            'smart_stats': smart_stats,
            'smart_matches_count': len(smart_matches)
        }

    def visualize_complete_analysis(self, nmr_shifts: List[float], smiles_output: str,
                                  alignment_data: Dict, head_idx: int, layer_idx: int = 4, save_path: str = None):
        """
        完整的可视化分析 - 使用智能选择的最佳头部

        Args:
            nmr_shifts: NMR化学位移列表
            smiles_output: SMILES字符串
            alignment_data: 对齐数据
            head_idx: 智能选择的最佳头部索引
            layer_idx: 层索引 (默认第5层，索引4)
            save_path: 保存路径
        """
        attention_matrix = alignment_data['attention_matrix']
        smiles_chars = alignment_data['smiles_chars']
        alignment_results = alignment_data['alignment_results']
        
        layer_name = "第5层 (倒数第二层)" if layer_idx == 4 else f"第{layer_idx + 1}层"
        print(f"🎯 使用智能选择的{layer_name} - 第{head_idx}号头部进行可视化分析")
        print(f"📊 Attention矩阵信息:")
        print(f"   - 形状: {attention_matrix.shape} (SMILES长度 × NMR峰数量)")
        print(f"   - 范围: [{attention_matrix.min():.4f}, {attention_matrix.max():.4f}]")
        print(f"   - 均值: {attention_matrix.mean():.4f}")
        print(f"   - 标准差: {attention_matrix.std():.4f}")
        
        # 显示attention矩阵的部分内容
        print(f"   - 前3×3子矩阵示例:")
        print(f"     {attention_matrix[:3, :3]}")
        
        # 保存attention矩阵到文件
        if save_path:
            import json
            attention_file = save_path.replace('.png', '_attention_matrix.json')
            attention_data = {
                'matrix': attention_matrix.tolist(),
                'shape': list(attention_matrix.shape),
                'smiles_chars': smiles_chars,
                'nmr_shifts': nmr_shifts,
                'head_idx': int(head_idx),
                'statistics': {
                    'min': float(attention_matrix.min()),
                    'max': float(attention_matrix.max()),
                    'mean': float(attention_matrix.mean()),
                    'std': float(attention_matrix.std())
                }
            }
            with open(attention_file, 'w') as f:
                json.dump(attention_data, f, indent=2)
            print(f"💾 Attention矩阵已保存到: {attention_file}")

        # 创建可视化
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle(f'CMGNet {layer_name} - 头部{head_idx} (智能选择): 完整NMR-SMILES对齐分析',
                    fontsize=18, fontweight='bold')

        # 1. 主注意力热图
        ax1 = axes[0, 0]
        im1 = ax1.imshow(attention_matrix, cmap='hot', aspect='auto', origin='upper')

        ax1.set_xticks(range(len(nmr_shifts)))
        ax1.set_xticklabels([f"{shift:.1f}" for shift in nmr_shifts], rotation=90, fontsize=8)
        ax1.set_xlabel('NMR化学位移 (ppm)', fontsize=14, fontweight='bold')

        ax1.set_yticks(range(len(smiles_chars)))
        ax1.set_yticklabels(smiles_chars, fontsize=8)
        ax1.set_ylabel('SMILES字符', fontsize=14, fontweight='bold')

        ax1.set_title('注意力对齐矩阵', fontsize=16, fontweight='bold')
        plt.colorbar(im1, ax=ax1, label='Attention Weight')

        # 2. 最佳匹配路径
        ax2 = axes[0, 1]

        for result in alignment_results[:20]:  # 显示前20个
            char = result['smiles_char']
            nmr_shift = result['nmr_shift']
            confidence = result['confidence']

            # 找到字符位置
            char_idx = smiles_chars.index(char) if char in smiles_chars else len(smiles_chars) // 2
            # 找到NMR位置
            nmr_idx = result['nmr_idx']

            ax2.scatter(nmr_idx, char_idx, s=200, c=confidence, cmap='hot',
                      alpha=0.7, edgecolors='black', linewidths=1)
            ax2.annotate(f"{confidence:.2f}", (nmr_idx, char_idx),
                       textcoords="offset points", xytext=(0,10), ha='center',
                       fontsize=8, alpha=0.8)

        ax2.set_xlabel('NMR位置', fontsize=12, fontweight='bold')
        ax2.set_ylabel('SMILES位置', fontsize=12, fontweight='bold')
        ax2.set_title('最佳匹配路径', fontsize=14, fontweight='bold')
        ax2.grid(alpha=0.3)

        # 3. NMR峰统计
        ax3 = axes[1, 0]

        nmr_stats = {
            'mean_attention': attention_matrix.mean(axis=0),
            'max_attention': attention_matrix.max(axis=0)
        }

        # 处理维度不匹配：注意力矩阵可能有特殊标记
        nmr_stats_mean = nmr_stats['mean_attention']
        nmr_stats_max = nmr_stats['max_attention']

        # 初始化绘图用的NMR标签
        nmr_shifts_plot = nmr_shifts

        # 截取或调整到NMR峰数量
        if len(nmr_stats_mean) > len(nmr_shifts):
            # 假设前几个和后几个是特殊标记，取中间部分
            diff = len(nmr_stats_mean) - len(nmr_shifts)
            if diff % 2 == 0:
                # 对称的特殊标记
                start_idx = diff // 2
                end_idx = len(nmr_stats_mean) - diff // 2
                nmr_stats_mean = nmr_stats_mean[start_idx:end_idx]
                nmr_stats_max = nmr_stats_max[start_idx:end_idx]
            else:
                # 非对称，直接截取前len(nmr_shifts)个
                nmr_stats_mean = nmr_stats_mean[:len(nmr_shifts)]
                nmr_stats_max = nmr_stats_max[:len(nmr_shifts)]
        elif len(nmr_stats_mean) < len(nmr_shifts):
            # 注意力比NMR短，截取NMR
            nmr_shifts_plot = nmr_shifts[:len(nmr_stats_mean)]

        x_pos = np.arange(len(nmr_stats_mean))
        width = 0.35

        ax3.bar(x_pos - width/2, nmr_stats_mean, width,
               label='平均注意力', alpha=0.8, color='skyblue')
        ax3.bar(x_pos + width/2, nmr_stats_max, width,
               label='最大注意力', alpha=0.8, color='coral')

        ax3.set_xticks(x_pos)
        ax3.set_xticklabels([f"{shift:.0f}" for shift in nmr_shifts_plot], rotation=90, fontsize=9)
        ax3.set_xlabel('NMR化学位移 (ppm)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('注意力值', fontsize=12, fontweight='bold')
        ax3.set_title('NMR峰注意力统计', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(axis='y', alpha=0.3)

        # 4. 智能头部选择的最佳匹配
        ax4 = axes[1, 1]

        # 使用智能选择的头部进行分析，而非固定阈值
        # 按confidence排序，显示最佳匹配
        sorted_results = sorted(alignment_results, key=lambda x: x['confidence'], reverse=True)
        top_matches = sorted_results[:15]  # 显示top-15最佳匹配
        
        # 计算动态阈值，基于智能选择头部的统计特性
        if sorted_results:
            confidences = [r['confidence'] for r in sorted_results]
            mean_conf = np.mean(confidences)
            std_conf = np.std(confidences)
            dynamic_threshold = mean_conf + 0.5 * std_conf  # 动态阈值
            
            # 使用动态阈值筛选
            smart_matches = [r for r in sorted_results if r['confidence'] >= dynamic_threshold]
            if not smart_matches:  # 如果没有匹配，使用top匹配
                smart_matches = top_matches
        else:
            smart_matches = []
            dynamic_threshold = 0.0

        if smart_matches:
            for i, result in enumerate(smart_matches[:15]):  # 显示前15个
                char = result['smiles_char']
                nmr_shift = result['nmr_shift']
                confidence = result['confidence']

                colors = plt.cm.rainbow(np.linspace(0, 1, len(smart_matches)))
                ax4.scatter(i, confidence, s=150, c=[colors[i]], alpha=0.7,
                          edgecolors='black', linewidths=1)
                ax4.annotate(f"{char}", (i, confidence),
                           textcoords="offset points", xytext=(0,10), ha='center',
                           fontsize=10, fontweight='bold')

            # 添加NMR值标签
            for i, result in enumerate(smart_matches[:15]):
                nmr_shift = result['nmr_shift']
                confidence = result['confidence']
                ax4.annotate(f"{nmr_shift:.1f}ppm", (i, confidence),
                           textcoords="offset points", xytext=(0,-15), ha='center',
                           fontsize=8, alpha=0.8)

            ax4.set_xlabel('智能匹配序号', fontsize=12, fontweight='bold')
            ax4.set_ylabel('置信度', fontsize=12, fontweight='bold')
            ax4.set_title(f'智能选择 {layer_name} 头部{head_idx} 的最佳匹配 (阈值>{dynamic_threshold:.3f})', 
                         fontsize=14, fontweight='bold')
            ax4.grid(alpha=0.3)
            
            # 添加统计信息
            ax4.text(0.02, 0.98, f'{layer_name}\n智能选择头部: {head_idx}\n动态阈值: {dynamic_threshold:.3f}\n匹配数量: {len(smart_matches)}',
                    transform=ax4.transAxes, fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 可视化结果已保存到: {save_path}")

        plt.show()

    def validate_smiles(self, smiles: str) -> bool:
        """验证SMILES字符串"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False

    def query_core(self, smiles: str) -> Tuple[str, bool]:
        """检测分子是否包含母核结构"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            from rdkit.Chem import rdSubstructLibrary

            mols = rdSubstructLibrary.CachedSmilesMolHolder()
            fps = rdSubstructLibrary.TautomerPatternHolder(1024)
            mols.AddSmiles(smiles)
            fp = Chem.PatternFingerprint(mol, fpSize=1024, tautomerFingerprints=True)
            fps.AddFingerprint(fp)
            library = rdSubstructLibrary.SubstructLibrary(mols, fps)

            indices_1 = library.GetMatches(Chem.MolFromSmiles('O=C1C(C(O)=O)=CNC2=CC=CC=C12'), maxResults=50000)
            indices_2 = library.GetMatches(Chem.MolFromSmiles('O=C1C(C(O)=O)=CNC2=NC=CC=C12'), maxResults=50000)

            if len(indices_1) > 0:
                core_mol = Chem.MolFromSmiles('O=C(O)c1c[nH]c2ccccc2c1=O')
                canonical_core = Chem.MolToSmiles(core_mol)
                return canonical_core, True
            elif len(indices_2) > 0:
                core_mol = Chem.MolFromSmiles('O=C(O)c1c[nH]c2ncccc2c1=O')
                canonical_core = Chem.MolToSmiles(core_mol)
                return canonical_core, True
            else:
                return '', False
        except Exception as e:
            return '', False


def main():
    """主函数 - 完整的CMGNet分析演示"""

    print("=" * 80)
    print("🚀 完整CMGNet集成分析工具")
    print("=" * 80)

    try:
        # 1. 初始化CMGNet分析器
        print("步骤1: 初始化CMGNet模型...")
        analyzer = CompleteCMGNetAnalyzer()

        # 2. 您的具体NMR数据
        print("\n步骤2: 准备NMR数据...")
        nmr_shifts = [
            176.4, 166.1, 155.5, 146.2, 140.2, 132.1, 124.8, 119.6, 106.6, 103.3,
            68.0, 55.3, 54.8, 50.1, 46.0, 17.9
        ]
        
        # 用户提供的SMILES
        user_smiles = "CC1COc2c(N3CCN(C)CC3)c(F)cc3c(=O)c(C(=O)O)cn1c23"

        print(f"   NMR峰数量: {len(nmr_shifts)}")
        print(f"   化学位移范围: {min(nmr_shifts):.1f} - {max(nmr_shifts):.1f} ppm")
        print(f"   参考SMILES: {user_smiles}")

        # 3. 执行预测和智能分析
        print("\n步骤3: 执行CMGNet推理和智能分析...")
        result = analyzer.predict_and_analyze(
            nmr_shifts=nmr_shifts,
            target_smiles=user_smiles,  # 使用用户提供的SMILES作为目标
            num_returns=1,
            mode="smiles",
            auto_select_head=True
        )

        # 4. 打印预测结果
        print("\n步骤4: 预测结果...")
        if 'error' in result:
            print(f"   ❌ 预测失败: {result['error']}")
        elif not result.get('top_predictions'):
            print(f"   ❌ 没有生成预测结果")
        else:
            # 显示loss信息
            if result.get('loss') is not None:
                print(f"   📉 Forward Loss: {result['loss']:.4f}")
                
            print(f"   ✅ Top-{len(result['top_predictions'])} 预测结果:")
            for pred in result['top_predictions'][:3]:  # 显示前3个
                validity = "✅" if pred['is_valid'] else "❌"
                core = "🎯" if pred['has_core_structure'] else "❌"
                loss_str = f"Loss: {pred.get('loss', 0):.4f}" if pred.get('loss') else ""
                print(f"      #{pred['rank']} {validity} {core} {loss_str}")
                print(f"         SMILES: {pred['smiles']}")
                if pred['has_core_structure']:
                    print(f"         母核: {pred['core_structure']}")

        # 5. 智能头部选择结果
        print("\n步骤5: 智能头部选择结果...")
        if 'head_analysis' in result and result['head_analysis']:
            head_info = result['head_analysis']
            best_layer = head_info.get('best_layer', 4)
            layer_name = "第5层 (倒数第二层)" if best_layer == 4 else f"第{best_layer + 1}层"
            print(f"   🏆 最佳头部: {layer_name} - 头部{head_info['best_head']}")

        # 6. 可视化完整分析
        if 'head_analysis' in result and result['head_analysis']:
            print("\n步骤6: 生成可视化分析...")
            head_info = result['head_analysis']
            if 'alignment_analysis' in head_info and head_info['alignment_analysis']:
                analyzer.visualize_complete_analysis(
                    nmr_shifts=nmr_shifts,
                    smiles_output=result['top_predictions'][0]['smiles'] if result['top_predictions'] else "",
                    alignment_data=head_info['alignment_analysis'],
                    head_idx=head_info['best_head'],
                    layer_idx=head_info.get('best_layer', 4),
                    save_path="cmgnet_complete_analysis.png"
                )

        print("\n" + "=" * 80)
        print("✅ 完整CMGNet分析完成!")
        print("=" * 80)
        print("功能特点:")
        print("  🤖 真实CMGNet模型加载和推理")
        print("  🧠 智能头部选择算法")
        print("  📊 NMR-SMILES精确对齐分析")
        print("  🎯 多维度可视化")
        print("  ⚡ 自动化工作流程")
        print("=" * 80)

        return analyzer, result

    except Exception as e:
        print(f"\n❌ 程序执行失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    analyzer, result = main()
#!/usr/bin/env python3
"""
具体的NMR-SMILES对齐分析工具
分析给定的SMILES和NMR数据之间的对应关系
"""

import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, Descriptors
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import json

class NMRSMILESAnalyzer:
    """NMR-SMILES对齐分析器"""

    def __init__(self, smiles: str, nmr_shifts: List[float]):
        """
        初始化分析器

        Args:
            smiles: SMILES字符串
            nmr_shifts: NMR化学位移列表
        """
        self.smiles = smiles
        self.nmr_shifts = sorted(nmr_shifts, reverse=True)  # 按从大到小排序
        self.mol = Chem.MolFromSmiles(smiles)

        if self.mol is None:
            raise ValueError(f"无效的SMILES: {smiles}")

        print("=" * 80)
        print("NMR-SMILES对齐分析")
        print("=" * 80)
        print(f"SMILES: {smiles}")
        print(f"NMR峰数量: {len(nmr_shifts)}")
        print(f"NMR范围: {min(nmr_shifts):.1f} - {max(nmr_shifts):.1f} ppm")

    def analyze_molecular_structure(self) -> Dict:
        """分析分子结构"""
        print("\n分子结构分析:")
        print("-" * 80)

        analysis = {
            'molecular_formula': AllChem.CalcMolFormula(self.mol),
            'molecular_weight': Descriptors.MolWt(self.mol),
            'num_atoms': self.mol.GetNumAtoms(),
            'num_bonds': self.mol.GetNumBonds(),
            'num_rings': Descriptors.RingCount(self.mol),
            'num_aromatic_rings': Descriptors.NumAromaticRings(self.mol),
            'logp': Descriptors.MolLogP(self.mol),
        }

        # 分析原子类型
        atom_counts = {}
        for atom in self.mol.GetAtoms():
            symbol = atom.GetSymbol()
            atom_counts[symbol] = atom_counts.get(symbol, 0) + 1

        analysis['atom_composition'] = atom_counts

        # 分析功能团
        functional_groups = self._detect_functional_groups()
        analysis['functional_groups'] = functional_groups

        print(f"分子式: {analysis['molecular_formula']}")
        print(f"分子量: {analysis['molecular_weight']:.2f}")
        print(f"原子数: {analysis['num_atoms']}")
        print(f"环数: {analysis['num_rings']} (芳香环: {analysis['num_aromatic_rings']})")
        print(f"原子组成: {atom_counts}")

        return analysis

    def _detect_functional_groups(self) -> List[str]:
        """检测功能团"""
        groups = []

        # 酮基 (C=O)
        if self.mol.HasSubstructMatch(Chem.MolFromSmarts('[CX3]=[OX1]')):
            groups.append('酮基')

        # 酯基 (COO)
        if self.mol.HasSubstructMatch(Chem.MolFromSmarts('[CX3](=[OX1])[OX2]')):
            groups.append('酯基')

        # 芳香环
        if self.mol.HasSubstructMatch(Chem.MolFromSmarts('c1ccccc1')):
            groups.append('芳香环')

        # 吗啉环
        if self.mol.HasSubstructMatch(Chem.MolFromSmarts('O1CCNCC1')):
            groups.append('吗啉环')

        # 喹诺酮骨架
        if self.mol.HasSubstructMatch(Chem.MolFromSmarts('O=c1cc[nH]c2ccccc12')):
            groups.append('喹诺酮骨架')

        return groups

    def categorize_nmr_peaks(self) -> Dict:
        """分类NMR峰"""
        print("\nNMR峰分类:")
        print("-" * 80)

        categories = {
            'carbonyl': [],      # 羰基碳 (>160 ppm)
            'aromatic': [],      # 芳香碳 (100-160 ppm)
            'aliphatic': [],     # 脂肪碳 (0-100 ppm)
            'heteroatom': []     # 杂原子连接的碳
        }

        for shift in self.nmr_shifts:
            if shift > 160:
                categories['carbonyl'].append(shift)
            elif 100 <= shift <= 160:
                categories['aromatic'].append(shift)
            else:
                categories['aliphatic'].append(shift)

        print(f"羰基区域 (>160 ppm): {len(categories['carbonyl'])} 个峰")
        print(f"  {categories['carbonyl']}")
        print(f"芳香区域 (100-160 ppm): {len(categories['aromatic'])} 个峰")
        print(f"  {categories['aromatic']}")
        print(f"脂肪区域 (<100 ppm): {len(categories['aliphatic'])} 个峰")
        print(f"  {categories['aliphatic']}")

        return categories

    def predict_atom_nmr_assignment(self) -> List[Dict]:
        """预测原子-NMR峰的对应关系"""
        print("\n原子-NMR峰对应关系预测:")
        print("-" * 80)

        assignments = []

        # 获取所有原子的化学环境信息
        for atom in self.mol.GetAtoms():
            atom_idx = atom.GetIdx()
            symbol = atom.GetSymbol()
            hybridization = atom.GetHybridization().name
            is_aromatic = atom.GetIsAromatic()

            # 预测化学位移范围
            predicted_shift = self._predict_carbon_shift(atom)

            if symbol == 'C':  # 只分析碳原子
                # 找到最接近的实际NMR峰
                closest_shift = min(self.nmr_shifts,
                                   key=lambda x: abs(x - predicted_shift))

                assignment = {
                    'atom_idx': atom_idx,
                    'symbol': symbol,
                    'hybridization': hybridization,
                    'is_aromatic': is_aromatic,
                    'predicted_shift': predicted_shift,
                    'assigned_shift': closest_shift,
                    'environment': self._get_atom_environment(atom)
                }

                assignments.append(assignment)

                print(f"原子 {atom_idx} ({symbol}):")
                print(f"  环境: {assignment['environment']}")
                print(f"  预测位移: {predicted_shift:.1f} ppm")
                print(f"  分配位移: {closest_shift:.1f} ppm")
                print()

        return assignments

    def _predict_carbon_shift(self, atom) -> float:
        """预测碳原子的化学位移"""
        symbol = atom.GetSymbol()
        hybridization = atom.GetHybridization().name
        is_aromatic = atom.GetIsAromatic()

        # 基础位移值
        base_shift = {
            'SP': 80,      # sp杂化
            'SP2': 130,    # sp2杂化
            'SP3': 30,     # sp3杂化
            'UNSPECIFIED': 40
        }.get(hybridization, 40)

        # 调整因素
        adjustments = 0

        if symbol == 'C':
            # 羰基碳
            if atom.GetDegree() == 3:  # 三键
                adjustments += 100

            # 芳香碳
            if is_aromatic:
                adjustments += 50

            # 与杂原子相连
            neighbors = atom.GetNeighbors()
            for neighbor in neighbors:
                neighbor_symbol = neighbor.GetSymbol()
                if neighbor_symbol in ['O', 'N']:
                    adjustments += 30

            # 酮基
            if self._is_carbonyl_carbon(atom):
                adjustments += 80

        return base_shift + adjustments

    def _is_carbonyl_carbon(self, atom) -> bool:
        """判断是否为羰基碳"""
        if atom.GetSymbol() != 'C':
            return False

        neighbors = atom.GetNeighbors()
        for neighbor in neighbors:
            if neighbor.GetSymbol() == 'O' and neighbor.GetDegree() == 1:
                return True

        return False

    def _get_atom_environment(self, atom) -> str:
        """获取原子环境描述"""
        env_parts = []

        # 杂化状态
        hybridization = atom.GetHybridization().name
        env_parts.append(hybridization)

        # 芳香性
        if atom.GetIsAromatic():
            env_parts.append('aromatic')

        # 相邻原子
        neighbors = atom.GetNeighbors()
        neighbor_symbols = [n.GetSymbol() for n in neighbors]
        if 'O' in neighbor_symbols:
            env_parts.append('O-bonded')
        if 'N' in neighbor_symbols:
            env_parts.append('N-bonded')

        return ' '.join(env_parts)

    def create_attention_alignment_simulation(self) -> np.ndarray:
        """模拟注意力对齐矩阵"""
        print("\n生成模拟注意力对齐矩阵:")
        print("-" * 80)

        # 获取碳原子数量
        num_carbons = sum(1 for atom in self.mol.GetAtoms() if atom.GetSymbol() == 'C')
        num_nmr_peaks = len(self.nmr_shifts)

        print(f"碳原子数量: {num_carbons}")
        print(f"NMR峰数量: {num_nmr_peaks}")

        # 创建模拟对齐矩阵 [nmr_peaks, carbons]
        alignment_matrix = np.zeros((num_nmr_peaks, num_carbons))

        carbon_atoms = [atom for atom in self.mol.GetAtoms() if atom.GetSymbol() == 'C']

        for nmr_idx, nmr_shift in enumerate(self.nmr_shifts):
            for carbon_idx, carbon in enumerate(carbon_atoms):
                # 计算相似度 (基于预测位移)
                predicted_shift = self._predict_carbon_shift(carbon)
                similarity = np.exp(-0.1 * abs(nmr_shift - predicted_shift))
                alignment_matrix[nmr_idx, carbon_idx] = similarity

        # 归一化
        for nmr_idx in range(num_nmr_peaks):
            if alignment_matrix[nmr_idx].sum() > 0:
                alignment_matrix[nmr_idx] /= alignment_matrix[nmr_idx].sum()

        return alignment_matrix

    def visualize_alignment(self, alignment_matrix: np.ndarray, save_path: str = None):
        """可视化对齐矩阵"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # 左图: 对齐热图
        im = ax1.imshow(alignment_matrix, cmap='hot', aspect='auto', origin='upper')

        # 设置坐标轴
        carbon_atoms = [atom for atom in self.mol.GetAtoms() if atom.GetSymbol() == 'C']
        carbon_labels = [f"C{atom.GetIdx()}" for atom in carbon_atoms]

        ax1.set_xticks(range(len(carbon_labels)))
        ax1.set_xticklabels(carbon_labels, rotation=90, fontsize=8)
        ax1.set_yticks(range(len(self.nmr_shifts)))
        ax1.set_yticklabels([f"{shift:.1f}" for shift in self.nmr_shifts])

        ax1.set_xlabel('Carbon Atoms', fontsize=12)
        ax1.set_ylabel('NMR Chemical Shifts (ppm)', fontsize=12)
        ax1.set_title('NMR Peak - Carbon Atom Alignment', fontsize=14, fontweight='bold')

        plt.colorbar(im, ax=ax1, label='Attention Weight')

        # 右图: 分子结构
        if self.mol is not None:
            # 生成2D坐标
            AllChem.Compute2DCoords(self.mol)

            # 绘制分子
            img = Draw.MolToImage(self.mol, size=(600, 400))
            ax2.imshow(img)
            ax2.axis('off')
            ax2.set_title('Molecular Structure', fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"对齐可视化已保存到: {save_path}")

        plt.show()

    def analyze_carbon_nmr_correspondence(self) -> Dict:
        """分析碳-NMR对应关系"""
        print("\n详细碳-NMR对应分析:")
        print("=" * 80)

        correspondence = {
            'high_shift_carbons': [],      # 高位移碳 (>150 ppm)
            'medium_shift_carbons': [],    # 中等位移碳 (100-150 ppm)
            'low_shift_carbons': [],       # 低位移碳 (<100 ppm)
            'assignments': []
        }

        carbon_atoms = [atom for atom in self.mol.GetAtoms() if atom.GetSymbol() == 'C']

        for carbon in carbon_atoms:
            atom_idx = carbon.GetIdx()
            predicted_shift = self._predict_carbon_shift(carbon)

            # 找到最接近的NMR峰
            closest_shift = min(self.nmr_shifts,
                               key=lambda x: abs(x - predicted_shift))

            assignment = {
                'carbon_idx': atom_idx,
                'predicted_shift': predicted_shift,
                'assigned_peak': closest_shift,
                'environment': self._get_atom_environment(carbon),
                'is_aromatic': carbon.GetIsAromatic(),
                'is_carbonyl': self._is_carbonyl_carbon(carbon)
            }

            correspondence['assignments'].append(assignment)

            # 分类
            if predicted_shift > 150:
                correspondence['high_shift_carbons'].append(assignment)
            elif predicted_shift >= 100:
                correspondence['medium_shift_carbons'].append(assignment)
            else:
                correspondence['low_shift_carbons'].append(assignment)

        # 打印分析结果
        print(f"高位移碳 (>150 ppm): {len(correspondence['high_shift_carbons'])} 个")
        for item in correspondence['high_shift_carbons']:
            print(f"  C{item['carbon_idx']}: {item['assigned_peak']:.1f} ppm - {item['environment']}")

        print(f"\n中等位移碳 (100-150 ppm): {len(correspondence['medium_shift_carbons'])} 个")
        for item in correspondence['medium_shift_carbons']:
            print(f"  C{item['carbon_idx']}: {item['assigned_peak']:.1f} ppm - {item['environment']}")

        print(f"\n低位移碳 (<100 ppm): {len(correspondence['low_shift_carbons'])} 个")
        for item in correspondence['low_shift_carbons']:
            print(f"  C{item['carbon_idx']}: {item['assigned_peak']:.1f} ppm - {item['environment']}")

        return correspondence

    def generate_attention_analysis_report(self) -> Dict:
        """生成完整的注意力分析报告"""
        print("\n生成完整分析报告:")
        print("=" * 80)

        # 结构分析
        structure_analysis = self.analyze_molecular_structure()

        # NMR分类
        nmr_categories = self.categorize_nmr_peaks()

        # 模拟对齐矩阵
        alignment_matrix = self.create_attention_alignment_simulation()

        # 对应关系分析
        correspondence = self.analyze_carbon_nmr_correspondence()

        # 生成报告
        report = {
            'smiles': self.smiles,
            'structure_analysis': structure_analysis,
            'nmr_analysis': {
                'total_peaks': len(self.nmr_shifts),
                'peak_range': [min(self.nmr_shifts), max(self.nmr_shifts)],
                'categories': nmr_categories
            },
            'carbon_nmr_correspondence': correspondence,
            'alignment_matrix': alignment_matrix.tolist(),
            'recommendations': self._generate_alignment_recommendations(correspondence)
        }

        return report

    def _generate_alignment_recommendations(self, correspondence: Dict) -> Dict:
        """生成对齐建议"""
        total_carbons = len(correspondence['assignments'])
        high_shift_count = len(correspondence['high_shift_carbons'])
        medium_shift_count = len(correspondence['medium_shift_carbons'])
        low_shift_count = len(correspondence['low_shift_carbons'])

        # NMR峰分布
        nmr_high = sum(1 for shift in self.nmr_shifts if shift > 150)
        nmr_medium = sum(1 for shift in self.nmr_shifts if 100 <= shift <= 150)
        nmr_low = sum(1 for shift in self.nmr_shifts if shift < 100)

        recommendations = {
            'alignment_quality': 'good' if total_carbons <= len(self.nmr_shifts) * 2 else 'complex',
            'recommended_layer': 3,  # 基于之前的分析
            'attention_strategy': 'use_all_heads',
            'peak_carbon_ratio': len(self.nmr_shifts) / total_carbons if total_carbons > 0 else 0,
            'notes': []
        }

        # 添加分析注释
        if high_shift_count > nmr_high:
            recommendations['notes'].append(f"预测的高位移碳({high_shift_count})多于实际NMR峰({nmr_high})")

        if abs(total_carbons - len(self.nmr_shifts)) > 5:
            recommendations['notes'].append(f"碳原子数({total_carbons})与NMR峰数({len(self.nmr_shifts)})差异较大")

        # 建议使用的注意力层
        if high_shift_count > 3:  # 多个羰基
            recommendations['recommended_layer'] = 4  # 使用更高层
            recommendations['attention_strategy'] = 'use_later_heads'

        return recommendations

def analyze_specific_case():
    """分析具体的测试案例"""
    # 给定的数据
    smiles = "Fc4cc1c2N(/C=C(\\C1=O)C(=O)O)C(COc2c4N3CCN(C)CC3)C"
    nmr_shifts = [
        176.4, 166.1, 153.8, 146.2, 140.2, 132.1, 124.8, 119.8, 106.6, 103.2,
        68.0, 55.3, 54.8, 50.1, 46.0, 17.9
    ]

    print("分析具体案例:")
    print(f"SMILES: {smiles}")
    print(f"NMR峰: {len(nmr_shifts)} 个")

    try:
        # 创建分析器
        analyzer = NMRSMILESAnalyzer(smiles, nmr_shifts)

        # 生成完整报告
        report = analyzer.generate_attention_analysis_report()

        # 可视化
        alignment_matrix = np.array(report['alignment_matrix'])
        analyzer.visualize_alignment(alignment_matrix, 'nmr_smiles_alignment.png')

        # 保存报告
        with open('nmr_smiles_analysis_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print("\n分析完成!")
        print(f"报告已保存到: nmr_smiles_analysis_report.json")
        print(f"可视化已保存到: nmr_smiles_alignment.png")

        return report

    except Exception as e:
        print(f"分析失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def print_cmgnnet_recommendations():
    """打印CMGNet注意力建议"""
    print("\n" + "=" * 80)
    print("针对此分子的CMGNet注意力建议")
    print("=" * 80)

    recommendations = """
基于分子结构分析:

1. 分子特征:
   - 喹诺酮类骨架 (含羰基)
   - 芳香系统丰富
   - 吗啉环结构
   - 氟原子取代

2. NMR特征:
   - 16个NMR峰，涵盖宽范围 (17.9 - 176.4 ppm)
   - 高位移区域: 176.4, 166.1, 153.8 ppm (羰基+芳香碳)
   - 中位移区域: 146.2 - 103.2 ppm (芳香碳)
   - 低位移区域: 68.0 - 17.9 ppm (脂肪碳)

3. CMGNet注意力配置建议:

   推荐层级: 第3层 (主选) + 第4层 (备选)

   理由:
   - 第3层: 平衡局部对齐和全局语义，适合复杂芳香系统
   - 第4层: 处理高级语义特征，适合功能团识别

   注意力头选择:
   - 第3层: 使用所有头的加权平均
   - 第4层: 重点关注后3个头 (语义级对齐)

4. 实现策略:

   ```python
   # 提取第3层注意力 (主推)
   layer_3_attention = outputs.cross_attentions[2]
   alignment_matrix = layer_3_attention.mean(dim=1)[0]  # [smiles_len, nmr_len]

   # 提取第4层注意力 (备选)
   layer_4_attention = outputs.cross_attentions[3]
   alignment_matrix_4 = layer_4_attention[:, -3:, :, :].mean(dim=1)[0]  # 后3个头

   # 集成决策
   final_alignment = 0.6 * alignment_matrix + 0.4 * alignment_matrix_4
   ```

5. 验证要点:
   - 羰基碳 (176.4, 166.1 ppm) 应与SMILES中的C=O对应
   - 芳香碳应集中在100-150 ppm区域
   - 吗啉环碳应在50-70 ppm区域
   - 氟原子邻近碳应有显著去屏蔽效应

6. 对齐质量检查:
   - 化学环境匹配度
   - 位移范围合理性
   - 结构一致性
   """

    print(recommendations)
    print("=" * 80)

if __name__ == "__main__":
    # 运行具体案例分析
    report = analyze_specific_case()

    # 打印CMGNet建议
    print_cmgnnet_recommendations()

    # 基于分析结果的具体建议
    if report:
        print("\n具体分析结果:")
        print(f"- 碳原子数: {report['structure_analysis']['num_atoms']}")
        print(f"- NMR峰数: {len(report['nmr_analysis']['categories'])}")
        print(f"- 推荐注意力层: {report['recommendations']['recommended_layer']}")
        print(f"- 注意力策略: {report['recommendations']['attention_strategy']}")
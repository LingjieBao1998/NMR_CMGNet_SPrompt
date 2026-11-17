#!/usr/bin/env python
"""对比修改前后的行为"""

from rdkit import Chem
import copy

# 原始版本的RingSystemFinder (未修改COOH保护)
class OriginalRingSystemFinder:
    def __init__(self):
        self.ring_db_pat = Chem.MolFromSmarts("[#6R,#16R]=[OR0,SR0,CR0,NR0]")
        self.ring_atom_pat = Chem.MolFromSmarts("[R]")

    def tag_bonds_to_preserve(self, mol):
        for bnd in mol.GetBonds():
            bnd.SetBoolProp("protected", False)
        for match in mol.GetSubstructMatches(self.ring_db_pat):
            bgn, end = match
            bnd = mol.GetBondBetweenAtoms(bgn, end)
            bnd.SetBoolProp("protected", True)

    @staticmethod
    def cleave_linker_bonds(mol):
        frag_bond_list = []
        for bnd in mol.GetBonds():
            if not bnd.IsInRing() and not bnd.GetBoolProp("protected") and bnd.GetBondType() == Chem.BondType.SINGLE:
                frag_bond_list.append(bnd.GetIdx())

        if len(frag_bond_list):
            frag_mol = Chem.FragmentOnBonds(mol, frag_bond_list)
            Chem.SanitizeMol(frag_mol)
            return frag_mol
        else:
            return mol

    def cleanup_fragments(self, mol, keep_dummy=False):
        frag_list = Chem.GetMolFrags(mol, asMols=True)
        ring_system_list = []
        for frag in frag_list:
            if frag.HasSubstructMatch(self.ring_atom_pat):
                for atm in frag.GetAtoms():
                    if atm.GetAtomicNum() == 0:
                        if keep_dummy:
                            atm.SetProp("atomLabel", "R")
                        else:
                            atm.SetAtomicNum(1)
                    atm.SetIsotope(0)
                frag = Chem.RemoveAllHs(frag)
                ring_system_list.append(frag)
        return ring_system_list

    def find_ring_systems(self, mol, keep_dummy=False, as_mols=False):
        self.tag_bonds_to_preserve(mol)
        frag_mol = self.cleave_linker_bonds(mol)
        output_list = self.cleanup_fragments(frag_mol, keep_dummy=keep_dummy)
        if not as_mols:
            output_list = [Chem.MolToSmiles(x) for x in output_list]
        return output_list

# 修改后的版本
from new_RingSystemFinder import RingSystemFinder

def compare_versions():
    """对比原始版本和修改版本"""
    
    test_molecules = [
        ("环己基羧酸", "C1CCC(CC1)C(=O)O"),
        ("苯甲酸", "c1ccccc1C(=O)O"),
        ("乙酸+环己基羧酸", "CC(=O)O.C1CCC(CC1)C(=O)O"),
    ]
    
    original_finder = OriginalRingSystemFinder()
    modified_finder = RingSystemFinder()
    
    for name, smi in test_molecules:
        print(f"\n=== {name} ({smi}) ===")
        
        mol = Chem.MolFromSmiles(smi)
        
        # 原始版本
        original_systems = original_finder.find_ring_systems(mol, as_mols=True)
        print(f"原始版本 - 找到 {len(original_systems)} 个环系统:")
        for i, ring_mol in enumerate(original_systems, 1):
            ring_smi = Chem.MolToSmiles(ring_mol)
            atom_count = ring_mol.GetNumAtoms()
            print(f"  系统 {i}: {ring_smi} ({atom_count} 原子)")
        
        # 修改版本
        modified_systems = modified_finder.find_ring_systems(mol, as_mols=True)
        print(f"修改版本 - 找到 {len(modified_systems)} 个环系统:")
        for i, ring_mol in enumerate(modified_systems, 1):
            ring_smi = Chem.MolToSmiles(ring_mol)
            atom_count = ring_mol.GetNumAtoms()
            print(f"  系统 {i}: {ring_smi} ({atom_count} 原子)")
        
        # 对比结论
        if len(original_systems) != len(modified_systems):
            print("  🔄 环系统数量发生变化")
        elif original_systems and modified_systems:
            orig_atoms = original_systems[0].GetNumAtoms()
            mod_atoms = modified_systems[0].GetNumAtoms()
            if orig_atoms != mod_atoms:
                print(f"  🔄 原子数发生变化: {orig_atoms} -> {mod_atoms}")
                if mod_atoms > orig_atoms:
                    print("  ✅ COOH被保留在环系统中")
                else:
                    print("  ❌ COOH未被保留")
            else:
                print("  ➡️ 无变化")

if __name__ == "__main__":
    compare_versions()
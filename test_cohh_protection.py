#!/usr/bin/env python
"""Test script to verify COOH protection in RingSystemFinder"""

from rdkit import Chem
from new_RingSystemFinder import RingSystemFinder

def test_cohh_protection():
    """Test that COOH groups attached to rings are preserved"""
    
    # Test case 1: Simple COOH attached to benzene
    smi1 = "c1ccccc1C(=O)O"  # benzoic acid
    mol1 = Chem.MolFromSmiles(smi1)
    finder = RingSystemFinder()
    
    # Debug: Check if COOH pattern matches
    matches = mol1.GetSubstructMatches(finder.cooh_ring_pat)
    print(f"COOH pattern matches: {matches}")
    
    # Check original molecule structure
    original_cohh = Chem.MolFromSmarts("C(=O)[OX2H]")
    original_matches = mol1.GetSubstructMatches(original_cohh)
    print(f"Original COOH matches: {original_matches}")
    print("Original molecule atoms:")
    for atom in mol1.GetAtoms():
        print(f"  Atom {atom.GetIdx()}: {atom.GetSymbol()}, degree: {atom.GetDegree()}")
    
    # Debug: Check which bonds are protected
    finder.tag_bonds_to_preserve(mol1)
    protected_bonds = []
    for bond in mol1.GetBonds():
        if bond.GetBoolProp("protected"):
            begin_atom = bond.GetBeginAtomIdx()
            end_atom = bond.GetEndAtomIdx()
            protected_bonds.append(bond.GetIdx())
            print(f"  Protected bond {bond.GetIdx()}: atom {begin_atom} - atom {end_atom}")
    print(f"Total protected bonds: {protected_bonds}")
    
    # Debug: Show all bonds with their atom indices
    print("All bonds:")
    for bond in mol1.GetBonds():
        begin_atom = bond.GetBeginAtomIdx()
        end_atom = bond.GetEndAtomIdx()
        print(f"  Bond {bond.GetIdx()}: atom {begin_atom} - atom {end_atom}, protected: {bond.GetBoolProp('protected')}")
    
    ring_systems1 = finder.find_ring_systems(mol1, as_mols=True)
    smiles1 = [Chem.MolToSmiles(x) for x in ring_systems1]
    print(f"Test 1 - Benzoic acid: {smi1}")
    print(f"Ring systems found: {smiles1}")
    # Check if COOH is preserved by analyzing the molecular structure
    cooh_preserved = False
    if smiles1:
        test_mol = Chem.MolFromSmiles(smiles1[0])
        cooh_pattern = Chem.MolFromSmarts("C(=O)[OX2H]")
        cooh_matches = test_mol.GetSubstructMatches(cooh_pattern)
        print(f"COOH pattern matches in result: {cooh_matches}")
        
        # Try alternative pattern
        alt_pattern = Chem.MolFromSmarts("C(=O)O")
        alt_matches = test_mol.GetSubstructMatches(alt_pattern)
        print(f"Alternative COOH pattern matches: {alt_matches}")
        
        # Check atom types
        print("Atoms in result molecule:")
        for atom in test_mol.GetAtoms():
            print(f"  Atom {atom.GetIdx()}: {atom.GetSymbol()}, degree: {atom.GetDegree()}")
        
        cooh_preserved = len(cooh_matches) > 0 or len(alt_matches) > 0
    print(f"COOH preserved: {cooh_preserved}\n")
    
    # Test case 2: COOH attached to cyclohexane
    smi2 = "C1CCCCC1C(=O)O"  # cyclohexanecarboxylic acid
    mol2 = Chem.MolFromSmiles(smi2)
    ring_systems2 = finder.find_ring_systems(mol2, as_mols=True)
    smiles2 = [Chem.MolToSmiles(x) for x in ring_systems2]
    print(f"Test 2 - Cyclohexanecarboxylic acid: {smi2}")
    print(f"Ring systems found: {smiles2}")
    # Check if COOH is preserved
    cooh_preserved2 = False
    if smiles2:
        test_mol2 = Chem.MolFromSmiles(smiles2[0])
        cooh_preserved2 = test_mol2.HasSubstructMatch(Chem.MolFromSmarts("C(=O)[OX2H]"))
    print(f"COOH preserved: {cooh_preserved2}\n")
    
    # Test case 3: Multiple rings with COOH
    smi3 = "c1ccccc1C2CCCCC2C(=O)O"  # COOH attached to cyclohexane which is fused to benzene
    mol3 = Chem.MolFromSmiles(smi3)
    ring_systems3 = finder.find_ring_systems(mol3, as_mols=True)
    smiles3 = [Chem.MolToSmiles(x) for x in ring_systems3]
    print(f"Test 3 - Multi-ring system with COOH: {smi3}")
    print(f"Ring systems found: {smiles3}")
    for i, smi in enumerate(smiles3):
        test_mol3 = Chem.MolFromSmiles(smi)
        cooh_preserved3 = test_mol3.HasSubstructMatch(Chem.MolFromSmarts("C(=O)[OX2H]"))
        print(f"  System {i+1}: {smi} - COOH preserved: {cooh_preserved3}")
    print()
    
    # Test case 4: COOH not attached to ring (should be separated)
    smi4 = "C(=O)O.c1ccccc1"  # benzoic acid as separate fragments
    mol4 = Chem.MolFromSmiles(smi4)
    ring_systems4 = finder.find_ring_systems(mol4, as_mols=True)
    smiles4 = [Chem.MolToSmiles(x) for x in ring_systems4]
    print(f"Test 4 - Separate COOH and ring: {smi4}")
    print(f"Ring systems found: {smiles4}")
    print(f"Only ring preserved: {'c1ccccc1' in smiles4[0] if smiles4 else 'False'}")

if __name__ == "__main__":
    test_cohh_protection()
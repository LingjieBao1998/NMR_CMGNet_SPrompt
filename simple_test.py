#!/usr/bin/env python
"""Simple test to demonstrate COOH protection in RingSystemFinder"""

from rdkit import Chem
from new_RingSystemFinder import RingSystemFinder

def simple_test():
    """Simple test to show COOH is preserved with ring"""
    
    # Test case: Benzoic acid vs benzene with separate COOH
    smi1 = "c1ccccc1C(=O)O"  # benzoic acid (COOH attached to ring)
    smi2 = "c1ccccc1.C(=O)O"  # benzene + COOH (separate)
    
    finder = RingSystemFinder()
    
    print("Test 1: Benzoic acid (COOH attached to ring)")
    mol1 = Chem.MolFromSmiles(smi1)
    ring_systems1 = finder.find_ring_systems(mol1, as_mols=True)
    for i, ring_mol in enumerate(ring_systems1):
        smi_result = Chem.MolToSmiles(ring_mol)
        print(f"  Ring system {i+1}: {smi_result}")
        
        # Count atoms - COOH should increase atom count
        print(f"  Number of atoms: {ring_mol.GetNumAtoms()}")
    
    print(f"\nTest 2: Benzene + COOH (separate)")
    mol2 = Chem.MolFromSmiles(smi2)
    ring_systems2 = finder.find_ring_systems(mol2, as_mols=True)
    for i, ring_mol in enumerate(ring_systems2):
        smi_result = Chem.MolToSmiles(ring_mol)
        print(f"  Ring system {i+1}: {smi_result}")
        print(f"  Number of atoms: {ring_mol.GetNumAtoms()}")
    
    print("\nConclusion:")
    print("- Test 1 should have more atoms (COOH preserved with ring)")
    print("- Test 2 should have fewer atoms (only benzene ring)")

if __name__ == "__main__":
    simple_test()
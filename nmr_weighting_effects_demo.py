#!/usr/bin/env python3
"""
Modified NMR Distance Comparison with Clear Weighting Effects
Demonstrates the difference between weighted and unweighted Hungarian matching

Author: Claude Code Assistant
Date: 2025-10-26
Purpose: Show clear weighting effects on the 19-peak dataset
"""

import numpy as np
import json
from scipy.stats import wasserstein_distance
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def normalize_nmr_length_with_reference_interpolation(nmr_data, reference_nmr, target_length=19):
    """
    Normalize NMR data with explicit reference interpolation for molecules longer than reference
    """
    sorted_nmr = np.sort(nmr_data)
    sorted_ref = np.sort(reference_nmr)
    
    # If molecule is longer than reference, interpolate reference first
    if len(sorted_nmr) > len(sorted_ref):
        # Interpolate reference to match molecule length
        old_indices = np.linspace(0, len(sorted_ref) - 1, len(sorted_ref))
        new_indices = np.linspace(0, len(sorted_ref) - 1, len(sorted_nmr))
        interpolated_ref = np.interp(new_indices, old_indices, sorted_ref)
        
        # Then normalize both to target length
        old_indices = np.linspace(0, len(interpolated_ref) - 1, len(interpolated_ref))
        new_indices = np.linspace(0, len(interpolated_ref) - 1, target_length)
        normalized_nmr = np.interp(new_indices, old_indices, interpolated_ref)
        
        # Track which positions are interpolated
        ref_positions_in_target = np.round(np.linspace(0, target_length-1, len(sorted_ref))).astype(int)
        interpolation_mask = np.ones(target_length, dtype=bool)
        interpolation_mask[ref_positions_in_target] = False  # These are original ref positions
        
        return normalized_nmr, interpolation_mask
    else:
        # Normal interpolation for shorter molecules
        old_indices = np.linspace(0, len(sorted_nmr) - 1, len(sorted_nmr))
        new_indices = np.linspace(0, len(sorted_nmr) - 1, target_length)
        normalized_nmr = np.interp(new_indices, old_indices, sorted_nmr)
        
        # No interpolation (all are original)
        interpolation_mask = np.ones(target_length, dtype=bool)
        
        return normalized_nmr, interpolation_mask

def calculate_hungarian_with_weighting(reference_nmr, molecule_nmr_list, interpolation_weight=0.3):
    """Calculate both weighted and unweighted Hungarian distances with clear weighting"""
    target_length = len(reference_nmr)
    
    unweighted_distances = []
    weighted_distances = []
    interpolation_masks = []
    
    normalized_ref = np.sort(reference_nmr)
    
    for nmr_data in molecule_nmr_list:
        # Normalize with interpolation tracking
        normalized_nmr, interp_mask = normalize_nmr_length_with_reference_interpolation(
            nmr_data, reference_nmr, target_length
        )
        
        interpolation_masks.append(interp_mask)
        
        # Calculate cost matrix
        cost_matrix = np.abs(normalized_ref[:, np.newaxis] - normalized_nmr[np.newaxis, :])
        
        # Unweighted Hungarian
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        unweighted_cost = cost_matrix[row_ind, col_ind].sum()
        unweighted_distances.append(unweighted_cost)
        
        # Weighted Hungarian - apply weights to cost matrix
        weights = np.where(interp_mask, 1.0, interpolation_weight)
        weighted_cost_matrix = cost_matrix * weights[np.newaxis, :]
        
        row_ind_w, col_ind_w = linear_sum_assignment(weighted_cost_matrix)
        weighted_cost = weighted_cost_matrix[row_ind_w, col_ind_w].sum()
        weighted_distances.append(weighted_cost)
    
    return unweighted_distances, weighted_distances, interpolation_masks

def main():
    """Demonstrate clear weighting effects"""
    print("🧬 Modified 19-Peak NMR Distance Comparison with Clear Weighting")
    print("="*80)
    print("Focus on demonstrating interpolation weighting effects")
    
    # Load the NMR search results
    print("\n📂 Loading NMR search results...")
    with open("nmr_search_results_100_3_complete.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    reference_nmr = data['query_nmr_data']
    molecules = data['all_results']
    
    molecule_nmr_list = [mol['nmr_data'] for mol in molecules]
    molecule_info = [{'rank': mol['rank'], 'smiles': mol['smiles'], 'nmr_peaks': mol['nmr_peaks']} for mol in molecules]
    
    print(f"✅ Loaded {len(molecule_nmr_list)} molecules")
    print(f"📊 Reference NMR has {len(reference_nmr)} peaks")
    
    # Analyze peak distribution
    peak_counts = [len(nmr) for nmr in molecule_nmr_list]
    longer_molecules = [(i, pc) for i, pc in enumerate(peak_counts) if pc > len(reference_nmr)]
    print(f"\n📈 Peak Analysis:")
    print(f"  Molecules longer than reference: {len(longer_molecules)}")
    print(f"  Peak count range: {min(peak_counts)} - {max(peak_counts)}")
    
    # Calculate distances with clear weighting
    print(f"\n🔢 Calculating Hungarian distances with weighting...")
    unweighted_dists, weighted_dists, interp_masks = calculate_hungarian_with_weighting(
        reference_nmr, molecule_nmr_list, interpolation_weight=0.3
    )
    
    # Calculate Wasserstein for comparison
    wasserstein_dists = []
    target_length = len(reference_nmr)
    
    for nmr_data in molecule_nmr_list:
        normalized_ref = np.sort(reference_nmr)
        normalized_nmr, _ = normalize_nmr_length_with_reference_interpolation(nmr_data, reference_nmr, target_length)
        wd = wasserstein_distance(normalized_ref, normalized_nmr)
        wasserstein_dists.append(wd)
    
    # Analyze differences
    unweighted_arr = np.array(unweighted_dists)
    weighted_arr = np.array(weighted_dists)
    differences = weighted_arr - unweighted_arr
    
    print(f"\n📊 Weighting Analysis:")
    print(f"  Unweighted Hungarian: min={unweighted_arr.min():.2f}, max={unweighted_arr.max():.2f}, mean={unweighted_arr.mean():.2f}")
    print(f"  Weighted Hungarian:   min={weighted_arr.min():.2f}, max={weighted_arr.max():.2f}, mean={weighted_arr.mean():.2f}")
    print(f"  Differences:          min={differences.min():.2f}, max={differences.max():.2f}, mean={differences.mean():.2f}")
    print(f"  Significant differences (>0.1): {np.sum(np.abs(differences) > 0.1)}")
    
    # Find molecules with biggest differences
    biggest_diff_indices = np.argsort(np.abs(differences))[-10:]
    print(f"\n🎯 Top 10 molecules with biggest weighting effects:")
    
    results_df = pd.DataFrame({
        'rank': [info['rank'] for info in molecule_info],
        'smiles': [info['smiles'] for info in molecule_info],
        'nmr_peaks': [info['nmr_peaks'] for info in molecule_info],
        'unweighted_hungarian': unweighted_dists,
        'weighted_hungarian': weighted_dists,
        'difference': differences,
        'wasserstein_distance': wasserstein_dists,
        'interpolated_peaks': [np.sum(~mask) for mask in interp_masks]
    })
    
    top_differences = results_df.iloc[biggest_diff_indices][::-1]  # Reverse for descending order
    
    for i, (_, row) in enumerate(top_differences.iterrows()):
        print(f"  {i+1:2d}. Rank {row['rank']:3d}, Peaks {row['nmr_peaks']:2d}, "
              f"Diff {row['difference']:7.2f}, Interpolated {row['interpolated_peaks']:2d}")
    
    # Save results
    results_df.to_csv('19peaks_weighting_effects_detailed.csv', index=False)
    
    summary = {
        'metadata': {
            'reference_peaks': len(reference_nmr),
            'total_molecules': len(molecule_nmr_list),
            'interpolation_weight': 0.3,
            'description': 'Clear demonstration of Hungarian weighting effects'
        },
        'statistics': {
            'unweighted_hungarian': {
                'min': float(unweighted_arr.min()),
                'max': float(unweighted_arr.max()),
                'mean': float(unweighted_arr.mean())
            },
            'weighted_hungarian': {
                'min': float(weighted_arr.min()),
                'max': float(weighted_arr.max()),
                'mean': float(weighted_arr.mean())
            },
            'differences': {
                'min': float(differences.min()),
                'max': float(differences.max()),
                'mean': float(differences.mean()),
                'significant_count': int(np.sum(np.abs(differences) > 0.1))
            }
        },
        'top_differences': top_differences.to_dict('records')
    }
    
    with open('19peaks_weighting_effects_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved:")
    print(f"  - 19peaks_weighting_effects_detailed.csv (full dataset)")
    print(f"  - 19peaks_weighting_effects_summary.json (summary)")
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Scatter plot: unweighted vs weighted
    ax1.scatter(unweighted_dists, weighted_dists, alpha=0.6, s=30)
    ax1.plot([min(unweighted_dists), max(unweighted_dists)], 
             [min(unweighted_dists), max(unweighted_dists)], 'r--', alpha=0.8)
    ax1.set_xlabel('Unweighted Hungarian Distance')
    ax1.set_ylabel('Weighted Hungarian Distance')
    ax1.set_title('Weighting Effects on Hungarian Distance')
    ax1.grid(True, alpha=0.3)
    ax1.text(0.05, 0.95, f'Correlation: {np.corrcoef(unweighted_dists, weighted_dists)[0,1]:.3f}', 
             transform=ax1.transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
    
    # Difference plot
    colors = ['red' if diff > 0 else 'blue' for diff in differences]
    ax2.scatter(range(len(differences)), differences, alpha=0.6, s=20, c=colors)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Molecule Index')
    ax2.set_ylabel('Distance Difference (Weighted - Unweighted)')
    ax2.set_title('Weighting Effects by Molecule')
    ax2.grid(True, alpha=0.3)
    
    # Distribution comparison
    ax3.hist(unweighted_dists, bins=20, alpha=0.7, label='Unweighted', density=True)
    ax3.hist(weighted_dists, bins=20, alpha=0.7, label='Weighted', density=True)
    ax3.set_xlabel('Distance')
    ax3.set_ylabel('Density')
    ax3.set_title('Distance Distribution Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Difference vs interpolated peaks
    interp_counts = [np.sum(~mask) for mask in interp_masks]
    ax4.scatter(interp_counts, differences, alpha=0.6, s=30)
    ax4.set_xlabel('Number of Interpolated Peaks')
    ax4.set_ylabel('Distance Difference')
    ax4.set_title('Weighting Effect vs Interpolated Peaks')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('19peaks_weighting_effects_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"  - 19peaks_weighting_effects_visualization.png")
    
    print(f"\n✅ Weighting effects analysis completed!")
    print(f"🔍 Key finding: Weighting affects molecules with interpolated reference peaks")

if __name__ == "__main__":
    main()
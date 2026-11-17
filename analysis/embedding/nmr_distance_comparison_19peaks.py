#!/usr/bin/env python3
"""
NMR Distance Methods Comparison for nmr_search_results_100_3_complete.json
Analyze the specific dataset with 19-peak reference NMR

Author: Claude Code Assistant
Date: 2025-10-26
Purpose: Compare distance methods on the 19-peak NMR dataset
"""

import numpy as np
import json
from scipy.stats import wasserstein_distance
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd

def normalize_nmr_length(nmr_data, target_length=19, method='interpolation', reference_nmr=None):
    """Normalize NMR data to equal-length vectors with reference interpolation support"""
    sorted_nmr = np.sort(nmr_data)
    
    # Handle NMR data longer than reference by interpolating reference
    if reference_nmr is not None and len(sorted_nmr) > len(reference_nmr):
        sorted_ref = np.sort(reference_nmr)
        old_indices = np.linspace(0, len(sorted_ref) - 1, len(sorted_ref))
        new_indices = np.linspace(0, len(sorted_ref) - 1, len(sorted_nmr))
        normalized_nmr = np.interp(new_indices, old_indices, sorted_ref)
        
        if len(normalized_nmr) != target_length:
            old_indices = np.linspace(0, len(normalized_nmr) - 1, len(normalized_nmr))
            new_indices = np.linspace(0, len(normalized_nmr) - 1, target_length)
            normalized_nmr = np.interp(new_indices, old_indices, normalized_nmr)
    else:
        if method == 'interpolation':
            old_indices = np.linspace(0, len(sorted_nmr) - 1, len(sorted_nmr))
            new_indices = np.linspace(0, len(sorted_nmr) - 1, target_length)
            normalized_nmr = np.interp(new_indices, old_indices, sorted_nmr)
        elif method == 'padding':
            if len(sorted_nmr) >= target_length:
                normalized_nmr = sorted_nmr[:target_length]
            else:
                padding = np.full(target_length - len(sorted_nmr), sorted_nmr[-1])
                normalized_nmr = np.concatenate([sorted_nmr, padding])
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized_nmr

def calculate_wasserstein_distances(reference_nmr, molecule_nmr_list, target_length=19):
    """Calculate Wasserstein distances between reference and all molecules"""
    distances = []
    normalized_ref = normalize_nmr_length(reference_nmr, target_length, 'interpolation')
    
    for nmr_data in molecule_nmr_list:
        normalized_nmr = normalize_nmr_length(nmr_data, target_length, 'interpolation', reference_nmr)
        wd = wasserstein_distance(normalized_ref, normalized_nmr)
        distances.append(wd)
    
    return distances

def calculate_hungarian_distances(reference_nmr, molecule_nmr_list, target_length=19):
    """Calculate Hungarian matching distances between reference and all molecules"""
    distances = []
    normalized_ref = normalize_nmr_length(reference_nmr, target_length, 'interpolation')
    
    for nmr_data in molecule_nmr_list:
        normalized_nmr = normalize_nmr_length(nmr_data, target_length, 'interpolation', reference_nmr)
        
        # Calculate cost matrix
        cost_matrix = np.abs(normalized_ref[:, np.newaxis] - normalized_nmr[np.newaxis, :])
        
        # Apply Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        total_cost = cost_matrix[row_ind, col_ind].sum()
        
        distances.append(total_cost)
    
    return distances

def calculate_hungarian_distances_weighted(reference_nmr, molecule_nmr_list, target_length=19, interpolation_weight=0.5):
    """Calculate weighted Hungarian matching distances with interpolation weighting"""
    distances = []
    normalized_ref = normalize_nmr_length(reference_nmr, target_length, 'interpolation')
    
    for i, nmr_data in enumerate(molecule_nmr_list):
        # Determine if interpolation was needed
        original_ref_length = len(reference_nmr)
        if len(nmr_data) > original_ref_length and target_length >= original_ref_length:
            # Create interpolation mask
            orig_positions = np.round(np.linspace(0, target_length-1, original_ref_length)).astype(int)
            interp_mask = np.zeros(target_length, dtype=bool)
            interp_mask[orig_positions] = True
        else:
            interp_mask = np.ones(target_length, dtype=bool)
        
        normalized_nmr = normalize_nmr_length(nmr_data, target_length, 'interpolation', reference_nmr)
        
        # Calculate weighted cost matrix
        cost_matrix = np.abs(normalized_ref[:, np.newaxis] - normalized_nmr[np.newaxis, :])
        
        # Apply interpolation weights to columns
        weights = np.where(interp_mask, 1.0, interpolation_weight)
        weighted_cost_matrix = cost_matrix * weights[np.newaxis, :]
        
        # Apply Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(weighted_cost_matrix)
        total_cost = weighted_cost_matrix[row_ind, col_ind].sum()
        
        distances.append(total_cost)
    
    return distances

def analyze_distance_correlations(wasserstein_dists, hungarian_dists, hungarian_weighted_dists):
    """Analyze correlations between different distance methods"""
    # Convert to numpy arrays
    w_arr = np.array(wasserstein_dists)
    h_arr = np.array(hungarian_dists)
    hw_arr = np.array(hungarian_weighted_dists)
    
    # Calculate correlations
    corr_wh = np.corrcoef(w_arr, h_arr)[0, 1]
    corr_whw = np.corrcoef(w_arr, hw_arr)[0, 1]
    corr_hhw = np.corrcoef(h_arr, hw_arr)[0, 1]
    
    # Convert to similarity scores
    w_sim = 1.0 / (1.0 + w_arr)
    h_sim = 1.0 / (1.0 + h_arr)
    hw_sim = 1.0 / (1.0 + hw_arr)
    
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS FOR 100 MOLECULES (19-PEAK REFERENCE)")
    print("="*80)
    print(f"Wasserstein vs Hungarian correlation: {corr_wh:.4f}")
    print(f"Wasserstein vs Weighted Hungarian correlation: {corr_whw:.4f}")
    print(f"Hungarian vs Weighted Hungarian correlation: {corr_hhw:.4f}")
    
    print(f"\nDistance Statistics:")
    print(f"Wasserstein:     min={w_arr.min():.3f}, max={w_arr.max():.3f}, mean={w_arr.mean():.3f}")
    print(f"Hungarian:       min={h_arr.min():.3f}, max={h_arr.max():.3f}, mean={h_arr.mean():.3f}")
    print(f"Weighted Hungarian: min={hw_arr.min():.3f}, max={hw_arr.max():.3f}, mean={hw_arr.mean():.3f}")
    
    print(f"\nSimilarity Statistics:")
    print(f"Wasserstein:     min={w_sim.min():.3f}, max={w_sim.max():.3f}, mean={w_sim.mean():.3f}")
    print(f"Hungarian:       min={h_sim.min():.3f}, max={h_sim.max():.3f}, mean={h_sim.mean():.3f}")
    print(f"Weighted Hungarian: min={hw_sim.min():.3f}, max={hw_sim.max():.3f}, mean={hw_sim.mean():.3f}")
    
    # Find significant differences
    diff_hw_h = hw_arr - h_arr
    significant_diffs = np.abs(diff_hw_h) > 0.001
    print(f"\nWeighted Hungarian differences:")
    print(f"Molecules with significant differences (>0.001): {np.sum(significant_diffs)}")
    print(f"Max difference: {np.max(np.abs(diff_hw_h)):.4f}")
    print(f"Mean difference: {np.mean(np.abs(diff_hw_h)):.4f}")
    
    return {
        'correlations': {
            'wasserstein_hungarian': corr_wh,
            'wasserstein_hungarian_weighted': corr_whw,
            'hungarian_hungarian_weighted': corr_hhw
        },
        'distance_stats': {
            'wasserstein': {'min': float(w_arr.min()), 'max': float(w_arr.max()), 'mean': float(w_arr.mean())},
            'hungarian': {'min': float(h_arr.min()), 'max': float(h_arr.max()), 'mean': float(h_arr.mean())},
            'hungarian_weighted': {'min': float(hw_arr.min()), 'max': float(hw_arr.max()), 'mean': float(hw_arr.mean())}
        },
        'similarity_stats': {
            'wasserstein': {'min': float(w_sim.min()), 'max': float(w_sim.max()), 'mean': float(w_sim.mean())},
            'hungarian': {'min': float(h_sim.min()), 'max': float(h_sim.max()), 'mean': float(h_sim.mean())},
            'hungarian_weighted': {'min': float(hw_sim.min()), 'max': float(hw_sim.max()), 'mean': float(hw_sim.mean())}
        },
        'weighted_differences': {
            'significant_count': int(np.sum(significant_diffs)),
            'max_difference': float(np.max(np.abs(diff_hw_h))),
            'mean_difference': float(np.mean(np.abs(diff_hw_h)))
        }
    }

def create_visualization_19peaks(wasserstein_dists, hungarian_dists, hungarian_weighted_dists, save_file="19peaks_100_molecules_comparison.png"):
    """Create comprehensive visualization for 19-peak reference comparison"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Scatter plot: Wasserstein vs Hungarian
    ax1.scatter(wasserstein_dists, hungarian_dists, alpha=0.6, s=30)
    ax1.set_xlabel('Wasserstein Distance')
    ax1.set_ylabel('Hungarian Distance')
    ax1.set_title('Wasserstein vs Hungarian Distance (19-peak Reference)')
    ax1.grid(True, alpha=0.3)
    
    # Add correlation line
    z = np.polyfit(wasserstein_dists, hungarian_dists, 1)
    p = np.poly1d(z)
    ax1.plot(wasserstein_dists, p(wasserstein_dists), "r--", alpha=0.8)
    ax1.text(0.05, 0.95, f'Correlation: {np.corrcoef(wasserstein_dists, hungarian_dists)[0,1]:.3f}', 
             transform=ax1.transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
    
    # Scatter plot: Wasserstein vs Weighted Hungarian
    ax2.scatter(wasserstein_dists, hungarian_weighted_dists, alpha=0.6, s=30, color='green')
    ax2.set_xlabel('Wasserstein Distance')
    ax2.set_ylabel('Weighted Hungarian Distance')
    ax2.set_title('Wasserstein vs Weighted Hungarian Distance (19-peak Reference)')
    ax2.grid(True, alpha=0.3)
    
    # Add correlation line
    z = np.polyfit(wasserstein_dists, hungarian_weighted_dists, 1)
    p = np.poly1d(z)
    ax2.plot(wasserstein_dists, p(wasserstein_dists), "r--", alpha=0.8)
    ax2.text(0.05, 0.95, f'Correlation: {np.corrcoef(wasserstein_dists, hungarian_weighted_dists)[0,1]:.3f}', 
             transform=ax2.transAxes, bbox=dict(boxstyle="round", facecolor='lightgreen'))
    
    # Distribution comparison
    ax3.hist(wasserstein_dists, bins=20, alpha=0.7, label='Wasserstein', density=True)
    ax3.hist(hungarian_dists, bins=20, alpha=0.7, label='Hungarian', density=True)
    ax3.hist(hungarian_weighted_dists, bins=20, alpha=0.7, label='Weighted Hungarian', density=True)
    ax3.set_xlabel('Distance')
    ax3.set_ylabel('Density')
    ax3.set_title('Distance Distribution Comparison (19-peak Reference)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Difference plot: Weighted - Original Hungarian
    diff_hw_h = np.array(hungarian_weighted_dists) - np.array(hungarian_dists)
    
    ax4.scatter(range(len(diff_hw_h)), diff_hw_h, alpha=0.6, s=20, color='red')
    ax4.set_xlabel('Molecule Index')
    ax4.set_ylabel('Distance Difference (Weighted - Original)')
    ax4.set_title('Hungarian Weighting Effect (19-peak Reference)')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='blue', linestyle='--', alpha=0.7)
    
    # Highlight significant differences
    significant_diffs = np.abs(diff_hw_h) > 0.001
    if np.sum(significant_diffs) > 0:
        sig_indices = np.where(significant_diffs)[0]
        ax4.scatter(sig_indices, diff_hw_h[sig_indices], color='orange', s=30, alpha=0.8, 
                   label=f'Significant diffs: {np.sum(significant_diffs)}')
        ax4.legend()
    
    plt.tight_layout()
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"19-peak reference comparison visualization saved to: {save_file}")

def load_nmr_results(filename):
    """Load NMR search results from JSON file"""
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    reference_nmr = data['query_nmr_data']
    molecules = data['all_results']
    
    # Extract NMR data for each molecule
    molecule_nmr_list = []
    molecule_info = []
    
    for mol in molecules:
        molecule_nmr_list.append(mol['nmr_data'])
        molecule_info.append({
            'rank': mol['rank'],
            'key': mol['key'],
            'smiles': mol['smiles'],
            'original_wasserstein': mol['wasserstein_distance'],
            'nmr_peaks': mol['nmr_peaks']
        })
    
    return reference_nmr, molecule_nmr_list, molecule_info

def main():
    """Main function for 19-peak NMR comparison"""
    print("🧬 19-Peak NMR Distance Methods Comparison")
    print("="*80)
    print("Analyzing nmr_search_results_100_3_complete.json with 19-peak reference NMR")
    
    # Load the NMR search results
    print("\n📂 Loading NMR search results...")
    try:
        reference_nmr, molecule_nmr_list, molecule_info = load_nmr_results("nmr_search_results_100_3_complete.json")
        print(f"✅ Loaded {len(molecule_nmr_list)} molecules")
        print(f"📊 Reference NMR has {len(reference_nmr)} peaks: {reference_nmr}")
    except FileNotFoundError:
        print("❌ Error: nmr_search_results_100_3_complete.json not found!")
        print("Please ensure the file exists in the current directory.")
        return
    
    # Analyze NMR peak distribution
    peak_counts = [len(nmr) for nmr in molecule_nmr_list]
    print(f"\n📈 NMR peak statistics:")
    print(f"  Reference: {len(reference_nmr)} peaks")
    print(f"  Molecules: min={min(peak_counts)}, max={max(peak_counts)}, mean={np.mean(peak_counts):.1f}")
    print(f"  Molecules longer than reference: {sum(1 for pc in peak_counts if pc > len(reference_nmr))}")
    
    # Calculate distances using all three methods
    print("\n🔢 Calculating Wasserstein distances...")
    wasserstein_dists = calculate_wasserstein_distances(reference_nmr, molecule_nmr_list, target_length=19)
    
    print("🔢 Calculating Hungarian matching distances...")
    hungarian_dists = calculate_hungarian_distances(reference_nmr, molecule_nmr_list, target_length=19)
    
    print("🔢 Calculating Weighted Hungarian matching distances...")
    hungarian_weighted_dists = calculate_hungarian_distances_weighted(reference_nmr, molecule_nmr_list, target_length=19)
    
    # Analyze correlations
    analysis_results = analyze_distance_correlations(wasserstein_dists, hungarian_dists, hungarian_weighted_dists)
    
    # Create detailed results dataframe
    results_df = pd.DataFrame({
        'rank': [info['rank'] for info in molecule_info],
        'key': [info['key'] for info in molecule_info],
        'smiles': [info['smiles'] for info in molecule_info],
        'nmr_peaks': [info['nmr_peaks'] for info in molecule_info],
        'original_wasserstein': [info['original_wasserstein'] for info in molecule_info],
        'calculated_wasserstein': wasserstein_dists,
        'hungarian_distance': hungarian_dists,
        'hungarian_weighted_distance': hungarian_weighted_dists,
        'hungarian_difference': np.array(hungarian_weighted_dists) - np.array(hungarian_dists),
        'wasserstein_similarity': 1.0 / (1.0 + np.array(wasserstein_dists)),
        'hungarian_similarity': 1.0 / (1.0 + np.array(hungarian_dists)),
        'hungarian_weighted_similarity': 1.0 / (1.0 + np.array(hungarian_weighted_dists))
    })
    
    # Save detailed results
    output_csv = '19peaks_100_molecules_distance_comparison.csv'
    results_df.to_csv(output_csv, index=False)
    print(f"\n💾 Detailed results saved to: {output_csv}")
    
    # Save summary results
    summary_results = {
        'metadata': {
            'total_molecules': len(molecule_nmr_list),
            'reference_nmr_peaks': len(reference_nmr),
            'reference_nmr_data': reference_nmr,
            'description': '19-peak NMR distance methods comparison on 100 molecules'
        },
        'peak_statistics': {
            'molecule_peaks': {
                'min': min(peak_counts),
                'max': max(peak_counts),
                'mean': float(np.mean(peak_counts)),
                'longer_than_reference': sum(1 for pc in peak_counts if pc > len(reference_nmr))
            }
        },
        'analysis': analysis_results,
        'top_10_molecules': {
            'by_wasserstein': results_df.nsmallest(10, 'calculated_wasserstein')[['rank', 'smiles', 'nmr_peaks', 'calculated_wasserstein', 'wasserstein_similarity']].to_dict('records'),
            'by_hungarian': results_df.nsmallest(10, 'hungarian_distance')[['rank', 'smiles', 'nmr_peaks', 'hungarian_distance', 'hungarian_similarity']].to_dict('records'),
            'by_hungarian_weighted': results_df.nsmallest(10, 'hungarian_weighted_distance')[['rank', 'smiles', 'nmr_peaks', 'hungarian_weighted_distance', 'hungarian_weighted_similarity']].to_dict('records')
        },
        'molecules_with_weighting_effect': results_df[results_df['hungarian_difference'].abs() > 0.001][['rank', 'smiles', 'nmr_peaks', 'hungarian_distance', 'hungarian_weighted_distance', 'hungarian_difference']].to_dict('records')
    }
    
    output_json = '19peaks_100_molecules_comparison_summary.json'
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(summary_results, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Summary results saved to: {output_json}")
    
    # Create visualization
    print("\n📊 Creating visualization...")
    try:
        create_visualization_19peaks(wasserstein_dists, hungarian_dists, hungarian_weighted_dists)
    except ImportError:
        print("⚠️  matplotlib/seaborn not available, skipping visualization")
    
    print("\n✅ 19-peak NMR comparison completed!")
    print("📁 Output files:")
    print(f"  - {output_csv} (detailed results)")
    print(f"  - {output_json} (summary statistics)")
    print("  - 19peaks_100_molecules_comparison.png (visualization)")

if __name__ == "__main__":
    main()
"""
Calculate Top-N metrics for CMGNet test results
For each sample, check if has_core_structure=True in any of the top-N predictions
"""

import json
import numpy as np
from typing import List, Dict


def calculate_top_n_metrics(data: List[Dict], top_ns: List[int] = [1, 3, 5, 10]) -> Dict:
    """
    Calculate Top-N metrics for core structure detection

    Args:
        data: List of test samples with predictions
        top_ns: List of N values for Top-N calculation

    Returns:
        Dictionary with metrics for each Top-N
    """
    results = {}

    # Get true labels (reference has_core_structure)
    true_positives_ref = sum(1 for item in data if item['reference_analysis']['has_core_structure'])
    true_negatives_ref = len(data) - true_positives_ref

    print(f"Total samples: {len(data)}")
    print(f"True positives (reference has core): {true_positives_ref}")
    print(f"True negatives (reference no core): {true_negatives_ref}")
    print()

    for top_n in top_ns:
        tp = 0  # True positives: correctly predicted as having core
        fp = 0  # False positives: predicted as having core but actually doesn't
        fn = 0  # False negatives: predicted as no core but actually has core
        tn = 0  # True negatives: correctly predicted as not having core

        for item in data:
            # Get reference label
            reference_has_core = item['reference_analysis']['has_core_structure']

            # Get predictions up to top_n
            predictions = item['predictions'][:top_n]

            # Check if any prediction in top_n has core structure
            predicted_has_core = any(
                pred.get('has_core_structure', False) for pred in predictions
            )

            # Calculate confusion matrix
            if reference_has_core and predicted_has_core:
                tp += 1
            elif not reference_has_core and predicted_has_core:
                fp += 1
            elif reference_has_core and not predicted_has_core:
                fn += 1
            else:
                tn += 1

        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

        results[f"top_{top_n}"] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "specificity": specificity,
            "accuracy": accuracy,
            "confusion_matrix": (int(tp), int(fp), int(fn), int(tn)),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn)
        }

    return results


def print_metrics_table(metrics: Dict):
    """Print metrics in a formatted table"""
    print("\n" + "=" * 120)
    print("TOP-N METRICS FOR CORE STRUCTURE DETECTION")
    print("=" * 120)
    print("\n| Top-N  | Precision | Recall  | F1 Score | Specificity | Accuracy | Confusion Matrix (TP, FP, FN, TN) |")
    print("|--------|-----------|---------|----------|-------------|----------|-------------------------------|")

    for top_n in [1, 3, 5, 10]:
        key = f"top_{top_n}"
        if key in metrics:
            m = metrics[key]
            cm = m["confusion_matrix"]
            print(f"| Top-{top_n:<4} | {m['precision']:.4f}    | {m['recall']:.4f} | {m['f1_score']:.4f}   | "
                  f"{m['specificity']:.4f}      | {m['accuracy']:.4f}   | "
                  f"({cm[0]}, {cm[1]}, {cm[2]}, {cm[3]})        |")

    print("\n" + "=" * 120)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Calculate Top-N metrics for CMGNet test results')
    parser.add_argument('file_path', nargs='?',
                       default="output/test_1218_cmgnet_add_positive/test_smiles_beams10_samples6410.json",
                       help='Path to JSON file with CMGNet test results')
    args = parser.parse_args()

    # Load data
    file_path = args.file_path

    print(f"Loading data from: {file_path}")
    with open(file_path, 'r') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples")

    # Calculate metrics
    metrics = calculate_top_n_metrics(data, top_ns=[1, 3, 5, 10])

    # Print results
    print_metrics_table(metrics)

    # Save results to JSON
    output_path = "cmgnet_top_n_metrics.json"
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    # Print detailed analysis
    print("\n" + "=" * 120)
    print("DETAILED ANALYSIS")
    print("=" * 120)

    for top_n in [1, 3, 5, 10]:
        key = f"top_{top_n}"
        m = metrics[key]
        print(f"\nTop-{top_n}:")
        print(f"  True Positives (TP):  {m['tp']:4d} - Correctly identified as having core structure")
        print(f"  False Positives (FP): {m['fp']:4d} - Incorrectly identified as having core structure")
        print(f"  False Negatives (FN): {m['fn']:4d} - Failed to identify core structure")
        print(f"  True Negatives (TN):  {m['tn']:4d} - Correctly identified as not having core structure")
        print(f"  ---")
        print(f"  Precision: {m['precision']:.4f} - TP / (TP + FP)")
        print(f"  Recall:    {m['recall']:.4f} - TP / (TP + FN)")
        print(f"  F1 Score:  {m['f1_score']:.4f} - 2 * Precision * Recall / (Precision + Recall)")
        print(f"  Specificity: {m['specificity']:.4f} - TN / (TN + FP)")
        print(f"  Accuracy:   {m['accuracy']:.4f} - (TP + TN) / Total")


if __name__ == "__main__":
    main()

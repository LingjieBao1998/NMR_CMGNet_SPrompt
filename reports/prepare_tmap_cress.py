#!/usr/bin/env python3
"""
Prepare TMAP visualization for CRESS embeddings of test data.
Labels: 1 for positive (core molecules), 0 for negative.

Output files for TMAP web viewer:
- test_data.lmhdb: TMAP database file
- test_data_metadata.txt: Metadata for coloring
"""
import os
import json
import numpy as np
import pickle
from tqdm import tqdm


def load_cress_embeddings():
    """Load CRESS embeddings and labels from saved predictions"""
    print("Loading XGBoost predictions...")
    with open('xgb_cress_1220_predictions_sorted.json', 'r') as f:
        data = json.load(f)

    predictions = data['predictions']

    # Also load the actual CRESS embeddings from pickle cache
    embeddings_cache = 'output/cress_1217/test_4503_embeddings_cache.pkl'

    if os.path.exists(embeddings_cache):
        print(f"Loading embeddings from cache: {embeddings_cache}")
        with open(embeddings_cache, 'rb') as f:
            cache_data = pickle.load(f)
            embeddings = cache_data['embeddings']
    else:
        print("Warning: Embeddings cache not found.")
        print("Please run CRESS embedding generation first or provide embedding file.")
        return None, None

    # Prepare data
    smiles_list = []
    labels = []
    indices = []

    # Sort by sample_index to ensure correct order
    predictions_sorted = sorted(predictions, key=lambda x: x['sample_index'])

    for pred in predictions_sorted:
        smiles_list.append(pred['cano_smiles'])
        labels.append(pred['true_label'])
        indices.append(pred['sample_index'])

    labels = np.array(labels)

    print(f"Loaded {len(smiles_list)} samples")
    print(f"  Positive (label=1): {np.sum(labels)}")
    print(f"  Negative (label=0): {np.sum(labels == 0)}")

    return embeddings, smiles_list, labels, indices


def create_tmap_files(embeddings, smiles_list, labels, indices, output_dir='output/tmap_cress'):
    """Create TMAP input files"""

    os.makedirs(output_dir, exist_ok=True)

    # 1. Create .lmhdb file (TMAP database format)
    lmhdb_file = os.path.join(output_dir, 'test_data.lmhdb')
    print(f"\nCreating TMAP database file: {lmhdb_file}")

    with open(lmhdb_file, 'w') as f:
        for i, (smiles, label, idx, emb) in enumerate(tqdm(zip(smiles_list, labels, indices, embeddings),
                                                           total=len(smiles_list),
                                                           desc="Writing LMHDB")):
            # Format: id\t{smiles}\t{label}\t{properties}
            f.write(f"{idx}\t{smiles}\t{label}\t")
            f.write("{\"cress_embedding\": [")
            # Write embedding as comma-separated values
            emb_str = ",".join([f"{x:.6f}" for x in emb])
            f.write(emb_str)
            f.write("]}\n")

    print(f"✓ Created {lmhdb_file}")

    # 2. Create metadata file for coloring
    metadata_file = os.path.join(output_dir, 'test_data_metadata.txt')
    print(f"\nCreating metadata file: {metadata_file}")

    with open(metadata_file, 'w') as f:
        # Header
        f.write("id\tlabel\n")
        for idx, label in zip(indices, labels):
            f.write(f"{idx}\t{label}\n")

    print(f"✓ Created {metadata_file}")

    # 3. Create Python script for TMAP generation
    script_file = os.path.join(output_dir, 'generate_tmap.py')
    print(f"\nCreating TMAP generation script: {script_file}")

    script_content = '''#!/usr/bin/env python3
"""
Generate TMAP visualization from CRESS embeddings.
Requires: pip install tmap
"""
from tmap import generate_tmap
from config import read_config

# Generate TMAP
print("Loading data...")
config = read_config()

print("Generating TMAP...")
generate_tmap(
    database_file=config['database_file'],
    metadata_file=config['metadata_file'],
    output_file=config['output_file']
)

print(f"✓ TMAP saved to: {config['output_file']}")
print("\nUpload the .html file to https://tmap.gdb.tools/#ex-dsstox")
'''

    with open(script_file, 'w') as f:
        f.write(script_content)

    print(f"✓ Created {script_file}")

    # 4. Create a simplified version using Python tmap library directly
    simple_script = os.path.join(output_dir, 'create_tmap_simple.py')
    print(f"\nCreating simple TMAP script: {simple_script}")

    simple_script_content = '''#!/usr/bin/env python3
"""
Simple TMAP generation script for CRESS embeddings.
"""
import numpy as np
from tmap import TMAP, mainl
from faerun import Faerun

# Load embeddings and labels
print("Loading data...")

# Read from lmhdb file
smiles_list = []
labels = []
embeddings = []

with open('test_data.lmhdb', 'r') as f:
    for line in f:
        parts = line.strip().split('\\t')
        idx = parts[0]
        smiles = parts[1]
        label = int(parts[2])

        # Parse embedding from JSON-like format
        props_str = '\\t'.join(parts[3:])
        emb_str = props_str.split('[')[1].rstrip(']}')
        embedding = np.array([float(x) for x in emb_str.split(',')])

        smiles_list.append(smiles)
        labels.append(label)
        embeddings.append(embedding)

embeddings = np.array(embeddings)
labels = np.array(labels)

print(f"Loaded {len(smiles_list)} samples")
print(f"  Positive (label=1): {np.sum(labels)}")
print(f"  Negative (label=0): {np.sum(labels == 0)}")
print(f"  Embedding shape: {embeddings.shape}")

# Create TMAP
print("\\nCreating TMAP...")

# Use Mainland algorithm to create graph
cfg = {
    "k": 10,
    "epsilon": 0.1
}

graph, _ = mainl(embeddings, **cfg)

# Create layout with TMAP
x, y, s, t = TMAP(
    graph,
    steps=500,
    keep_knn=True
)

# Save to file
print("\\nSaving to file...")

# Prepare data for visualization
data_dict = {
    "x": x.tolist(),
    "y": y.tolist(),
    "size": s.tolist(),
    "labels": labels.tolist(),
    "smiles": smiles_list
}

import json
with open('tmap_cress_data.json', 'w') as f:
    json.dump(data_dict, f)

print("✓ Saved to: tmap_cress_data.json")
print("\\nNext steps:")
print("1. Use a TMAP viewer to load the data")
print("2. Or visit https://tmap.gdb.tools/#ex-dsstox and upload the .lmhdb file")
'''

    with open(simple_script, 'w') as f:
        f.write(simple_script_content)

    print(f"✓ Created {simple_script}")

    # 5. Create a README with instructions
    readme_file = os.path.join(output_dir, 'README.md')
    readme_content = '''# TMAP Visualization for CRESS Embeddings

## Files Generated

1. **test_data.lmhdb** - TMAP database file with CRESS embeddings
2. **test_data_metadata.txt** - Metadata file (labels)
3. **create_tmap_simple.py** - Python script to generate TMAP

## Quick Start

### Option 1: Use TMAP Web Viewer (Recommended)

1. Visit https://tmap.gdb.tools/#ex-dsstox
2. Upload `test_data.lmhdb`
3. The viewer will automatically visualize the data

### Option 2: Use Python TMAP Library

```bash
pip install tmap faerun
python create_tmap_simple.py
```

### Option 3: Use Command Line

```bash
pip install tmap
tmap create -i test_data.lmhdb -o tmap_output.html
```

## Data Information

- **Total samples**: 4503
- **Positive samples (label=1)**: 65 core molecules
- **Negative samples (label=0)**: 4438 other molecules
- **Embedding**: CRESS (768 dimensions)
- **Objective**: Visualize if core molecules cluster separately

## Expected Results

If the model has learned meaningful representations:
- Core molecules (label=1) should form distinct clusters
- Non-core molecules (label=0) should be separated or form different clusters
- Clear separation indicates good feature learning

## Color Coding

- **Red/Orange**: Label=1 (Positive/Core molecules)
- **Blue/Green**: Label=0 (Negative/Other molecules)

## Troubleshooting

If TMAP doesn't load:
1. Check that .lmhdb file format is correct
2. Ensure embeddings are properly normalized
3. Try reducing the number of samples for testing
'''

    with open(readme_file, 'w') as f:
        f.write(readme_content)

    print(f"✓ Created {readme_file}")

    print(f"\n{'='*80}")
    print(f"✓ All files created in: {output_dir}/")
    print(f"{'='*80}")
    print(f"\nGenerated files:")
    print(f"  1. test_data.lmhdb           - TMAP database")
    print(f"  2. test_data_metadata.txt   - Metadata (labels)")
    print(f"  3. create_tmap_simple.py    - Generation script")
    print(f"  4. README.md                - Instructions")
    print(f"\nNext steps:")
    print(f"  1. Visit https://tmap.gdb.tools/#ex-dsstox")
    print(f"  2. Upload: {output_dir}/test_data.lmhdb")
    print(f"  3. The viewer will color by label (1=red, 0=blue)")
    print(f"{'='*80}")


def main():
    print("="*80)
    print("TMAP Visualization for CRESS Embeddings")
    print("="*80)

    # Load data
    embeddings, smiles_list, labels, indices = load_cress_embeddings()

    if embeddings is None:
        print("\nError: Could not load embeddings.")
        print("Please ensure the embeddings cache exists:")
        print("  output/cress_1217/test_4503_embeddings_cache.pkl")
        return

    # Create TMAP files
    create_tmap_files(embeddings, smiles_list, labels, indices)

    print("\n✓ Done! Ready for TMAP visualization.")


if __name__ == "__main__":
    main()

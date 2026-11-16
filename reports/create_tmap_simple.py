#!/usr/bin/env python3
"""
Create simple TMAP input files using true labels only.
For use with https://tmap.gdb.tools/#ex-dsstox

Input: xgb_cress_1220_predictions_sorted.json
Output: tmap_input.smi (SMILES + true label)
"""
import json
import os


def create_tmap_input():
    """Create simple SMILES file with true labels for TMAP"""

    print("Loading predictions...")
    with open('xgb_cress_1220_predictions_sorted.json', 'r') as f:
        data = json.load(f)

    predictions = data['predictions']

    print(f"Total samples: {len(predictions)}")

    output_dir = 'output/tmap_cress'
    os.makedirs(output_dir, exist_ok=True)

    # Create simple SMILES file with labels
    output_file = os.path.join(output_dir, 'tmap_input.smi')
    print(f"\nCreating TMAP input file: {output_file}")

    with open(output_file, 'w') as f:
        # Header comment
        f.write("# TMAP input file for CRESS embeddings\n")
        f.write("# Format: SMILES <space> Label\n")
        f.write("# Label: 1 = positive (core molecule), 0 = negative\n")
        f.write("# Total: 65 positive + 4438 negative = 4503 samples\n")
        f.write("\n")

        for pred in predictions:
            # Format: SMILES <space> Label
            # Use true_label, not predicted_label
            smiles = pred['cano_smiles']
            label = pred['true_label']
            f.write(f"{smiles} {label}\n")

    print(f"✓ Created {output_file}")

    # Statistics
    positive_count = sum(1 for p in predictions if p['true_label'] == 1)
    negative_count = sum(1 for p in predictions if p['true_label'] == 0)

    print(f"\nFile statistics:")
    print(f"  Positive (label=1): {positive_count}")
    print(f"  Negative (label=0): {negative_count}")
    print(f"  Total: {len(predictions)}")

    # Also create a version with metadata for reference
    metadata_file = os.path.join(output_dir, 'tmap_metadata_with_index.txt')
    print(f"\nCreating metadata file: {metadata_file}")

    with open(metadata_file, 'w') as f:
        f.write("ID\tSMILES\tLabel\n")
        for pred in predictions:
            f.write(f"{pred['sample_index']}\t{pred['cano_smiles']}\t{pred['true_label']}\n")

    print(f"✓ Created {metadata_file}")

    # Create README
    readme_file = os.path.join(output_dir, 'README_TMAP.md')
    readme_content = '''# TMAP Visualization for CRESS Embeddings

## Files

1. **tmap_input.smi** - Main input file for TMAP
   - Format: SMILES <space> Label
   - Label: 1 = positive (core molecule), 0 = negative
   - 65 positive + 4438 negative samples

2. **tmap_metadata_with_index.txt** - Reference file with sample IDs

## How to Use with tmap.gdb.tools

### Step 1: Prepare CRESS Embeddings

You need to compute CRESS embeddings for these molecules first.
Use the model: `output/cress_1217/model_xgb_cress_1220.pkl`

```python
# Example: Generate CRESS embeddings
from get_cress_embedding import get_single_cress_embedding
import json

with open('xgb_cress_1220_predictions_sorted.json', 'r') as f:
    data = json.load(f)

embeddings = []
for pred in data['predictions']:
    emb = get_single_cress_embedding(pred['cano_smiles'])
    embeddings.append(emb)

# Save embeddings
import numpy as np
np.save('cress_embeddings_4503.npy', np.array(embeddings))
```

### Step 2: Create TMAP with Python

```python
import numpy as np
from tmap import TMAP, mainl
from faerun import Faerun

# Load data
smiles = []
labels = []
with open('tmap_input.smi', 'r') as f:
    for line in f:
        if line.startswith('#'):
            continue
        parts = line.strip().split()
        if len(parts) >= 2:
            smiles.append(parts[0])
            labels.append(int(parts[1]))

embeddings = np.load('cress_embeddings_4503.npy')

# Create TMAP
graph, _ = mainl(embeddings, k=10, epsilon=0.1)
x, y, s, t = TMAP(graph, steps=500)

# Visualize
faerun = Faerun(view="front", clear_bg=True)
faerun.add_scatter("cress", {
    "x": x, "y": y, "s": s, "t": t,
    "c": labels,
    "c_map": "jet",
    "labels": labels
})
faerun.plot('tmap_cress')
```

### Step 3: View Results

1. Open the generated HTML file in browser
2. Or visit https://tmap.gdb.tools/#ex-dsstox
3. Upload your data file

## Expected Results

If CRESS embeddings are good:
- Red points (label=1, core molecules) should cluster together
- Blue points (label=0, other molecules) should be separated
- Clear separation indicates good feature learning

## Color Coding

- **Label 1 (Positive)**: Core molecules (65 samples)
- **Label 0 (Negative)**: Other molecules (4,438 samples)
'''

    with open(readme_file, 'w') as f:
        f.write(readme_content)

    print(f"✓ Created {readme_file}")

    print(f"\n{'='*80}")
    print("✓ TMAP input files created!")
    print(f"{'='*80}")
    print(f"\nOutput directory: {output_dir}/")
    print(f"\nMain file: tmap_input.smi")
    print(f"  - Format: SMILES <space> TRUE_LABEL")
    print(f"  - Uses true labels (1=positive, 0=negative)")
    print(f"  - Ready for TMAP visualization with CRESS embeddings")
    print(f"\nNext steps:")
    print(f"  1. Generate CRESS embeddings for these SMILES")
    print(f"  2. Use Python tmap library to create visualization")
    print(f"  3. View results in browser or upload to tmap.gdb.tools")
    print(f"{'='*80}")


if __name__ == "__main__":
    create_tmap_input()

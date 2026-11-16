# Fig. S-vb — ring-system map — caption (SI)

Files:
- `reports/fig_ring_system_map.pdf` / `.png` (600 dpi)
- analysis/figure: `reports/ring_system_map.py`
- numbers: `reports/ring_system_map.json`

## Full caption

**Figure S-vb.** Ring-system space of the external test set. Every external
molecule was reduced to its ring-only Bemis–Murcko scaffold; equal scaffolds
were grouped (2,053 groups, of which 210 with n ≥ 3, a group maximum score
≥ 0.40, or at least one true positive are shown). Scaffolds were embedded with
the CReSS SMILES encoder (cosine space; no fingerprint similarity) and
projected to two dimensions with UMAP (cosine metric, 15 neighbours,
min_dist = 0.25). Point area scales with the number of spectra in the group and
colour encodes the highest XGBoost score observed in that group (log colour
scale). The quinolone/quinolinone region is the only high-score island (red,
upper right), whereas benzene, pyridine, indole, pyrrolidine, pyrazole and
other ordinary carbocyclic/heterocyclic systems occupy the low-score blue
region. Red circles mark the five ring-system groups that contain a false
positive at τ_op = 0.40; representative ring systems are drawn in the side
margins and connected to their coordinates. Group counts: benzene n = 640,
pyridine n = 55, indole n = 25, pyrrolidine n = 23, pyrazole n = 21, quinolone
n = 13, N-phenyl quinolinone n = 2, azo-thiazole n = 1.

## Short caption

**Figure S-vb.** UMAP map of ring-only Murcko scaffolds from the external test
set (CReSS SMILES embeddings). Point size = number of spectra in the group;
colour = maximum XGBoost score in the group; red circles = groups containing a
false positive at τ_op = 0.40. High scores are confined to the
quinolone/quinolinone island; ordinary carbocycles and N-heterocycles remain in
the low-score region.

## Notes

- Scaffolds are ring-only (Bemis–Murcko), so exocyclic carbonyls are retained
  for quinolinone-type rings but flexible substituents are removed.
- The map is descriptive: positions are UMAP coordinates of CReSS SMILES
  embeddings, not a classifier decision surface.

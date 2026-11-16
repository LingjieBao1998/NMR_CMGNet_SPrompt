import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from matplotlib import cm

import os as _os
B = _os.environ.get('NMR_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
B = B.rstrip('/') + '/'
D={o["idx"]:o for o in json.load(open(B+"reports/last10_models_summary.json"))}
ids=list(range(54,64))
models=[("CMGNet 0910","orig0910"),("Frag","Frag"),("SMILES","SMILES"),
        ("SPrompt","SPrompt"),("NMRMind","nmrmind"),("NMR-Solver","solver")]
# load final table for frag/smiles/sprompt strings
F={o["idx"]:o for o in json.load(open(B+"reports/last10_final_table.json"))}
def mean(s): return float(str(s).split("±")[0])
Z=np.zeros((len(ids),len(models)))
for i,idx in enumerate(ids):
    for j,(lab,key) in enumerate(models):
        if key=="orig0910": Z[i,j]=D[idx][key]
        elif key=="nmrmind": Z[i,j]=D[idx][key]
        elif key=="solver": Z[i,j]=D[idx][key]
        elif key=="Frag": Z[i,j]=mean(F[idx]["frag"])
        elif key=="SMILES": Z[i,j]=mean(F[idx]["smiles"])
        elif key=="SPrompt": Z[i,j]=mean(F[idx]["sprompt"])

fig=plt.figure(figsize=(9.2,6.2))
ax=fig.add_subplot(111,projection="3d")
dx=dy=0.62
cmap=matplotlib.colormaps["RdYlGn"]
for i in range(len(ids)):
    for j in range(len(models)):
        z=Z[i,j]
        color=cmap(z/10)
        ax.bar3d(j, i, 0, dx, dy, max(z,0.05), color=color, edgecolor="white", linewidth=0.3)
ax.set_xticks(np.arange(len(models))+dx/2)
ax.set_xticklabels([m[0] for m in models], rotation=25, ha="right", fontsize=7)
ax.set_yticks(np.arange(len(ids))+dy/2)
ax.set_yticklabels(ids, fontsize=6)
ax.set_zlim(0,10)
ax.set_zlabel("top-10 candidates with core (/10)", fontsize=8, labelpad=6)
ax.set_xlabel("Model", fontsize=8, labelpad=4)
ax.set_ylabel("External positive idx", fontsize=8, labelpad=4)
ax.view_init(elev=24, azim=-58)
ax.set_title("3D hit-count matrix (last 10 positives)", loc="left", fontweight="bold")
fig.tight_layout()
for e in ("png","pdf"):
    fig.savefig(B+f"reports/fig_hit_rate_3d.{e}", dpi=600 if e=="png" else None)
print("saved")
